#!/usr/bin/env python3
"""
significance.py — Multi-seed replication + significance testing pipeline.

SEEDING FIX (the core purpose of this script):
  The original codebase fixes seed=42 in three interdependent places. This
  script varies all three consistently per replication seed:
    (1) torch.manual_seed(seed)                   → weight initialisation RNG
    (2) seed= kwarg passed to get_data()           → input X, labels y
    (3) torch.Generator().manual_seed(seed) used   → train/val split membership
        inside random_split
  All three receive the same value for a given run. Different seeds therefore
  produce genuinely different replications (different weights AND different
  data splits). Seed divergence is verified on startup.

SIGNIFICANCE TESTS — one-to-one mapping to thesis claims:

  Claim A: Dropout hurts NID interaction recovery
    → Wilcoxon signed-rank (paired by seed × function, averaged over noise),
      dropout vs base. Reports median Δ + rank-biserial r.

  Claim B: Dropout + weight_decay hurts more than base
    → Same paired Wilcoxon, dropout+weight_decay vs base.

  Claim C: Weight decay alone is benign
    → Same paired Wilcoxon, weight_decay vs base.
      Framed as non-inferiority: 95 % bootstrap CI on median difference;
      CI upper bound < 0.05 AUPRC is the non-inferiority criterion.

  Claim D: Noise causes an accelerating decline in AUPRC
    → (i)  Friedman test across noise levels within the base condition.
      (ii) Targeted contrast: is the second noise-segment drop larger than
           the first? (one-sided Wilcoxon on drop2 − drop1, alt='greater')
      Core mode (η ∈ {0, 0.5, 1.0}): segments [0→0.5] and [0.5→1.0].
      Full mode: uses first and middle noise level as boundary.

  Claim E: AUPRC tracks val_loss (Spearman ρ, bootstrap 95 % CI on ρ).
    → p-value reported but NOT Holm-corrected (uninformative at large n).

  Holm–Bonferroni correction applied jointly to Claims A, B, C,
  D-Friedman, D-contrast (5 raw p-values). Claim E excluded from family.
  n (paired observation count) is reported for every test.
"""

from __future__ import annotations

import argparse
import math
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon
from torch.utils.data import DataLoader, TensorDataset, random_split

# ── Project imports — these modules are NEVER modified ──────────────────────
from multilayer_perceptron import MLP, get_weights
import NID as _nid
from analysis import (
    get_ground_truth,
    normalize_nid_output,
    _metric_triplet,
)
from synth import get_data, functions as synth_functions
from config import EXPERIMENTS

# ── Constants mirrored from train.py / stats_analysis.py ───────────────────
_L1_CONST = 1e-4   # L1 penalty coefficient on interaction_mlp weights
_L2_CONST = 1e-4   # weight_decay magnitude when weight_decay=True
_DROPOUT  = 0.2    # dropout rate for dropout conditions
_LR       = 0.01
_N_EPOCHS = 100
_BATCH    = 64
_VAL_RATIO   = 0.2
_N_SAMPLES   = 30_000
_NUM_FEATURES = 10
_MIN_CONV_REDUCTION = 0.03  # ≥3 % val-loss drop required for convergence

MODEL_ARCH = {"small": [64, 64], "big": [140, 100, 60, 20]}
FUNCTION_NAMES = [f.__name__ for f in synth_functions]

_DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ── Regularization state label ───────────────────────────────────────────────

def _reg_state(dropout: float, weight_decay: bool) -> str:
    if dropout > 0 and weight_decay:
        return "dropout+weight_decay"
    if dropout > 0:
        return "dropout"
    if weight_decay:
        return "weight_decay"
    return "base"


# ── Condition builders ───────────────────────────────────────────────────────

def _build_core_conditions() -> list[dict[str, Any]]:
    """12 conditions: 4 regularisation states × 3 noise levels, adam, small."""
    conds: list[dict[str, Any]] = []
    for noise in [0.0, 0.5, 1.0]:
        tag = str(noise).replace(".", "p")
        conds.append(dict(name=f"ms_base_n{tag}", noise=noise,
                          optimizer="adam", dropout=0.0, weight_decay=False,
                          reg_state="base"))
        conds.append(dict(name=f"ms_dp_n{tag}", noise=noise,
                          optimizer="adam", dropout=_DROPOUT, weight_decay=False,
                          reg_state="dropout"))
        conds.append(dict(name=f"ms_wd_n{tag}", noise=noise,
                          optimizer="adam", dropout=0.0, weight_decay=True,
                          reg_state="weight_decay"))
        conds.append(dict(name=f"ms_dpwd_n{tag}", noise=noise,
                          optimizer="adam", dropout=_DROPOUT, weight_decay=True,
                          reg_state="dropout+weight_decay"))
    return conds


def _build_full_conditions() -> list[dict[str, Any]]:
    """All deduplicated config.EXPERIMENTS, both model sizes."""
    seen: set[str] = set()
    conds: list[dict[str, Any]] = []
    for e in EXPERIMENTS:
        name = str(e.get("name", ""))
        if name in seen:
            continue
        seen.add(name)
        dp = float(e.get("dropout", 0.0))
        wd = bool(e.get("weight_decay", False))
        conds.append({
            "name": f"ms_{name}",
            "noise": float(e["noise"]),
            "optimizer": str(e.get("optimizer", "adam")),
            "dropout": dp,
            "weight_decay": wd,
            "reg_state": _reg_state(dp, wd),
        })
    return conds


# ── Seed-divergence verification ────────────────────────────────────────────

def _verify_seeds_differ(seed_a: int, seed_b: int) -> None:
    """Warn if two seeds produce identical initial weights (seeding is broken)."""
    def _first_weights(s: int) -> np.ndarray:
        rng_state = torch.get_rng_state()
        torch.manual_seed(s)
        net = MLP(num_features=_NUM_FEATURES, hidden_units=[64, 64],
                  use_main_effect_nets=False, dropout=0.0)
        w = get_weights(net)[0].flatten()
        torch.set_rng_state(rng_state)
        return w

    wa, wb = _first_weights(seed_a), _first_weights(seed_b)
    dist = float(np.linalg.norm(wa - wb))
    if np.allclose(wa, wb):
        warnings.warn(
            f"Seeds {seed_a} and {seed_b} produced identical initial weights "
            "(L2 distance = 0). torch.manual_seed may not be taking effect.",
            RuntimeWarning, stacklevel=2,
        )
    else:
        print(f"  [seed check] seeds {seed_a}/{seed_b} differ  "
              f"(weight L2 distance = {dist:.4f})  OK")


# ── Data loaders ─────────────────────────────────────────────────────────────

def _make_loaders(fn, noise: float, seed: int) -> dict[str, DataLoader]:
    """
    Mirrors make_data_loaders() from train.py with all three seed locations
    controlled by the caller:
      seed (2): get_data(seed=seed)
      seed (3): torch.Generator().manual_seed(seed) in random_split
    seed (1) (torch.manual_seed) is set by the caller before MLP construction.
    """
    X, y, _ = get_data(fn, num_samples=_N_SAMPLES, noise=noise, seed=seed)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    ds = TensorDataset(X_t, y_t)
    n_val   = int(_VAL_RATIO * len(ds))
    n_train = len(ds) - n_val
    gen = torch.Generator().manual_seed(seed)  # seed (3): split membership
    tr, va = random_split(ds, [n_train, n_val], generator=gen)
    return {
        "train": DataLoader(tr, batch_size=_BATCH, shuffle=True),
        "val":   DataLoader(va, batch_size=_BATCH, shuffle=False),
    }


# ── Minimal training loop ────────────────────────────────────────────────────

def _evaluate(net: nn.Module, loader: DataLoader, criterion, dev) -> float:
    net.eval()
    losses: list[float] = []
    with torch.no_grad():
        for X_b, y_b in loader:
            losses.append(criterion(net(X_b.to(dev)), y_b.to(dev)).item())
    return float(np.mean(losses)) if losses else math.nan


def _train_one(
    net: nn.Module,
    loaders: dict[str, DataLoader],
    opt_name: str,
    l2_const: float,
    dev,
) -> tuple[nn.Module, float, int, float, float]:
    """
    Replicates train.train() logic without any file I/O:
      - MSELoss, Adam/SGD with weight_decay=l2_const
      - L1 penalty (_L1_CONST) on interaction_mlp weights every step
      - Tracks best-val-loss epoch; restores those weights at the end

    Returns: (net, best_val_loss, best_epoch, first_epoch_val_loss, last_epoch_val_loss)
    first/last val_loss are used for the convergence filter (≥3 % reduction).
    """
    criterion = nn.MSELoss(reduction="mean")
    opt_cls = torch.optim.SGD if opt_name.lower() == "sgd" else torch.optim.Adam
    opt = opt_cls(net.parameters(), lr=_LR, weight_decay=l2_const)

    net.to(dev)
    best_loss   = math.inf
    best_state: dict | None = None
    best_epoch  = 0
    first_vl    = math.nan
    last_vl     = math.nan

    for epoch in range(_N_EPOCHS):
        net.train()
        for X_b, y_b in loaders["train"]:
            X_b, y_b = X_b.to(dev), y_b.to(dev)
            opt.zero_grad()
            loss = criterion(net(X_b), y_b)
            reg = sum(
                torch.sum(torch.abs(p))
                for nm, p in net.named_parameters()
                if "interaction_mlp" in nm and "weight" in nm
            )
            (loss + reg * _L1_CONST).backward()
            opt.step()

        vl = _evaluate(net, loaders["val"], criterion, dev)
        if epoch == 0:
            first_vl = vl
        last_vl = vl

        if vl < best_loss:
            best_loss  = vl
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}

    if best_state is not None:
        net.load_state_dict(best_state)
    return net, best_loss, best_epoch, first_vl, last_vl


# ── Checkpoint path helpers ───────────────────────────────────────────────────

def _ckpt_dir(snap_root: Path, model_size: str, fn_name: str,
              cond_name: str, seed: int) -> Path:
    return snap_root / model_size / fn_name / cond_name / f"seed_{seed}"


def _find_best_ckpt(snap_root: Path, model_size: str, fn_name: str,
                    cond_name: str, seed: int) -> Path | None:
    hits = sorted(_ckpt_dir(snap_root, model_size, fn_name, cond_name, seed)
                  .glob("best_epoch_*.pt"))
    return hits[0] if hits else None


# ── Training phase ────────────────────────────────────────────────────────────

def run_training(
    seeds: list[int],
    conditions: list[dict[str, Any]],
    model_sizes: list[str],
    snap_root: Path,
    dev,
    dry_run: bool = False,
) -> None:
    fn_map = {f.__name__: f for f in synth_functions}
    total  = len(seeds) * len(FUNCTION_NAMES) * len(conditions) * len(model_sizes)
    done   = 0
    t0     = time.time()

    for seed in seeds:
        for fn_name in FUNCTION_NAMES:
            fn = fn_map[fn_name]
            for cond in conditions:
                for model_size in model_sizes:
                    done += 1
                    elapsed = time.time() - t0
                    eta_str = (
                        f"  ETA {elapsed / done * (total - done) / 60:.1f} min"
                        if done > 1 else ""
                    )
                    tag = (f"[{done:4d}/{total}] seed={seed} "
                           f"{fn_name}/{cond['name']} ({model_size})")

                    existing = _find_best_ckpt(snap_root, model_size,
                                               fn_name, cond["name"], seed)
                    if existing:
                        print(f"  {tag}  SKIP (exists)")
                        continue

                    if dry_run:
                        print(f"  {tag}  DRY-RUN{eta_str}")
                        continue

                    print(f"  {tag}  training…{eta_str}", flush=True)

                    # seed (1): weight-initialisation RNG — set immediately
                    # before MLP construction so it is not advanced by anything
                    # else in the loop (data loading uses a separate Generator).
                    torch.manual_seed(seed)

                    loaders = _make_loaders(fn, cond["noise"], seed)

                    net = MLP(
                        num_features=_NUM_FEATURES,
                        hidden_units=MODEL_ARCH[model_size],
                        use_main_effect_nets=False,
                        dropout=cond["dropout"],
                    )
                    l2 = _L2_CONST if cond["weight_decay"] else 0.0
                    net, best_val, best_epoch, fvl, lvl = _train_one(
                        net, loaders, cond["optimizer"], l2, dev
                    )

                    converged = (
                        math.isfinite(fvl) and fvl > 0
                        and math.isfinite(lvl)
                        and (fvl - lvl) / fvl >= _MIN_CONV_REDUCTION
                    )

                    ckpt_d = _ckpt_dir(snap_root, model_size, fn_name,
                                       cond["name"], seed)
                    ckpt_d.mkdir(parents=True, exist_ok=True)
                    ckpt_file = ckpt_d / f"best_epoch_{best_epoch:04d}.pt"
                    torch.save({
                        "epoch":           best_epoch,
                        "function":        fn_name,
                        "experiment":      cond["name"],
                        "model_state_dict": {k: v.cpu()
                                             for k, v in net.state_dict().items()},
                        "val_loss":        best_val,
                        "first_val_loss":  fvl,
                        "last_val_loss":   lvl,
                        "seed":            seed,
                        "model_size":      model_size,
                        "condition":       cond,
                    }, ckpt_file)

                    conv_tag = "converged" if converged else "NOT converged"
                    print(f"    → best_epoch={best_epoch}  "
                          f"val={best_val:.5f}  {conv_tag}")


# ── Analysis phase ────────────────────────────────────────────────────────────

def run_analysis(
    seeds: list[int],
    conditions: list[dict[str, Any]],
    model_sizes: list[str],
    snap_root: Path,
) -> pd.DataFrame:
    """
    Loads each best checkpoint, runs NID, computes auprc_full via _metric_triplet
    (identical to analysis.py's best-epoch path).  Reads first_val_loss and
    last_val_loss from the checkpoint to apply the convergence filter inline.
    """
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        for fn_name in FUNCTION_NAMES:
            gt = get_ground_truth(fn_name)

            for cond in conditions:
                for model_size in model_sizes:
                    base_row: dict[str, Any] = {
                        "seed":           seed,
                        "function_name":  fn_name,
                        "condition":      cond["name"],
                        "noise":          cond["noise"],
                        "optimizer":      cond["optimizer"],
                        "dropout":        cond["dropout"],
                        "weight_decay":   cond["weight_decay"],
                        "reg_state":      cond["reg_state"],
                        "model_size":     model_size,
                        "auprc_full":     math.nan,
                        "val_loss":       math.nan,
                        "first_val_loss": math.nan,
                        "last_val_loss":  math.nan,
                        "best_epoch":     math.nan,
                        "converged":      False,
                    }

                    ckpt = _find_best_ckpt(snap_root, model_size,
                                           fn_name, cond["name"], seed)
                    if ckpt is None:
                        rows.append(base_row)
                        continue

                    try:
                        obj = torch.load(ckpt, map_location="cpu",
                                         weights_only=False)
                    except Exception as exc:
                        base_row["converged"] = False
                        rows.append(base_row)
                        print(f"  [warn] failed to load {ckpt}: {exc}")
                        continue

                    fvl = float(obj.get("first_val_loss", math.nan))
                    lvl = float(obj.get("last_val_loss",  math.nan))
                    bvl = float(obj.get("val_loss",        math.nan))
                    ep  = obj.get("epoch", math.nan)
                    base_row.update({
                        "val_loss":       bvl,
                        "first_val_loss": fvl,
                        "last_val_loss":  lvl,
                        "best_epoch":     int(ep) if ep == ep else math.nan,
                        "converged": (
                            math.isfinite(fvl) and fvl > 0
                            and math.isfinite(lvl)
                            and (fvl - lvl) / fvl >= _MIN_CONV_REDUCTION
                        ),
                    })

                    try:
                        sd = obj["model_state_dict"]
                        net = MLP(
                            num_features=_NUM_FEATURES,
                            hidden_units=MODEL_ARCH[model_size],
                            use_main_effect_nets=False,
                            dropout=cond["dropout"],
                        )
                        net.load_state_dict(sd, strict=False)
                        net.eval()
                        interactions = _nid.get_interactions(
                            get_weights(net), one_indexed=True
                        )
                        nid_map = normalize_nid_output(interactions)
                        _, auprc_full, _, _ = _metric_triplet(gt, nid_map)
                        base_row["auprc_full"] = auprc_full
                    except Exception as exc:
                        print(f"  [warn] NID failed for {ckpt}: {exc}")

                    rows.append(base_row)

    return pd.DataFrame(rows)


# ── Convergence filter ────────────────────────────────────────────────────────

def apply_convergence_filter(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int]:
    """
    Drop runs that did not achieve ≥3 % val-loss reduction (epoch 1 → epoch N).
    This mirrors stats_analysis._filter_convergent_runs but applied inline from
    first_val_loss / last_val_loss stored in each checkpoint — no trajectory
    CSV required.
    """
    n_before = len(df)
    df_f     = df[df["converged"]].copy()
    n_after  = len(df_f)
    n_rm     = n_before - n_after
    pct      = n_rm / max(n_before, 1) * 100
    print(f"  [convergence filter] removed {n_rm} runs ({pct:.1f}%); "
          f"{n_after} retained.")
    return df_f, n_before, n_after


# ── Aggregation ───────────────────────────────────────────────────────────────

def _stage1_per_seed_function(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str = "auprc_full",
) -> pd.DataFrame:
    """
    Stage 1 of two-stage aggregation: mean within (seed, function_name,
    *group_cols) across noise levels. This collapses within-seed, within-
    function variance, leaving one estimate per (seed, function, group).

    SD is later computed ACROSS SEEDS within each function before pooling —
    this is the genuine stability estimate that the single-seed pipeline
    (stats_analysis.py) cannot produce; that script conflates experimental
    variance (across noise/optimizer conditions) with sampling variance.
    Here we isolate the latter by averaging out noise within each seed first.
    """
    return (
        df.groupby(["seed", "function_name"] + group_cols, as_index=False)
        [value_col].mean()
    )


# ── Statistical helpers ───────────────────────────────────────────────────────

def _rank_biserial(diffs: np.ndarray) -> float:
    """Rank-biserial correlation r for a one-sample Wilcoxon signed-rank test."""
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(diffs))) + 1.0
    w_plus  = float(np.sum(ranks[diffs > 0]))
    w_minus = float(np.sum(ranks[diffs < 0]))
    return (w_plus - w_minus) / (len(diffs) * (len(diffs) + 1) / 2)


def _wilcoxon_paired(
    treatment: np.ndarray,
    control: np.ndarray,
    alternative: str = "two-sided",
) -> tuple[float, float, float, float]:
    """
    Returns (median_diff, rank_biserial_r, statistic, p).
    median_diff = median(treatment − control); negative means treatment < control.
    """
    diffs    = treatment - control
    med_diff = float(np.median(diffs))
    r        = _rank_biserial(diffs)
    try:
        stat, p = wilcoxon(diffs, alternative=alternative)
    except Exception:
        stat, p = math.nan, math.nan
    return med_diff, r, float(stat), float(p)


def _bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    rng_seed: int = 0,
) -> tuple[float, float, float]:
    """Returns (rho_observed, ci_lo, ci_hi) with percentile bootstrap CI."""
    rng = np.random.default_rng(rng_seed)
    n   = len(x)
    rhos: list[float] = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try:
            r, _ = spearmanr(x[idx], y[idx])
            rhos.append(float(r))
        except Exception:
            pass
    rho_obs, _ = spearmanr(x, y)
    alpha = (1 - ci) / 2
    lo = float(np.nanpercentile(rhos, alpha * 100))
    hi = float(np.nanpercentile(rhos, (1 - alpha) * 100))
    return float(rho_obs), lo, hi


def _bootstrap_median_ci(
    diffs: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    rng_seed: int = 1,
) -> tuple[float, float]:
    """Bootstrap 95 % CI on the median of diffs."""
    rng = np.random.default_rng(rng_seed)
    n   = len(diffs)
    meds = [float(np.median(rng.choice(diffs, n, replace=True)))
            for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return float(np.percentile(meds, alpha * 100)), float(np.percentile(meds, (1 - alpha) * 100))


def _holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm–Bonferroni correction. Returns adjusted p-values in original order."""
    n       = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adj     = [0.0] * n
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        a = p * (n - rank)
        running_max = max(running_max, a)
        adj[orig_idx] = min(running_max, 1.0)
    return adj


# ── Significance tests ────────────────────────────────────────────────────────

def run_significance_tests(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Runs all thesis significance tests and returns a list of result dicts.
    Each dict: claim, comparison, n, effect_size_str, raw_p, adj_p, verdict, extra.
    adj_p and verdict are filled after Holm–Bonferroni is applied.
    """
    VALUE = "auprc_full"
    results: list[dict[str, Any]] = []

    # Stage-1 aggregate: one value per (seed, function_name, reg_state),
    # averaged over noise levels — the paired unit for Claims A/B/C.
    sf = _stage1_per_seed_function(df, ["reg_state"], VALUE)

    def _paired_arrays(trt_state: str, ctrl_state: str):
        ctrl_idx = sf[sf["reg_state"] == ctrl_state].set_index(["seed", "function_name"])[VALUE]
        trt_idx  = sf[sf["reg_state"] == trt_state].set_index(["seed", "function_name"])[VALUE]
        common   = ctrl_idx.index.intersection(trt_idx.index)
        return trt_idx.loc[common].values, ctrl_idx.loc[common].values, len(common)

    # ── Claim A: Dropout vs Base ─────────────────────────────────────────────
    trt_a, ctrl_a, n_a = _paired_arrays("dropout", "base")
    med_a, r_a, _, p_a = _wilcoxon_paired(trt_a, ctrl_a)
    results.append({
        "claim": "A",
        "comparison": "Dropout vs Base",
        "n": n_a,
        "effect_size_str": (rf"$\Delta\tilde{{m}}={med_a:+.3f}$, "
                            rf"$r={r_a:+.3f}$"),
        "raw_p": p_a, "adj_p": math.nan, "verdict": "",
        "extra": {"med": med_a, "r": r_a},
    })

    # ── Claim B: Dropout+WD vs Base ──────────────────────────────────────────
    trt_b, ctrl_b, n_b = _paired_arrays("dropout+weight_decay", "base")
    med_b, r_b, _, p_b = _wilcoxon_paired(trt_b, ctrl_b)
    results.append({
        "claim": "B",
        "comparison": "Dropout+WD vs Base",
        "n": n_b,
        "effect_size_str": (rf"$\Delta\tilde{{m}}={med_b:+.3f}$, "
                            rf"$r={r_b:+.3f}$"),
        "raw_p": p_b, "adj_p": math.nan, "verdict": "",
        "extra": {"med": med_b, "r": r_b},
    })

    # ── Claim C: Weight Decay vs Base (non-inferiority) ──────────────────────
    trt_c, ctrl_c, n_c = _paired_arrays("weight_decay", "base")
    diffs_c = trt_c - ctrl_c
    med_c, r_c, _, p_c = _wilcoxon_paired(trt_c, ctrl_c)
    ci_lo_c, ci_hi_c = _bootstrap_median_ci(diffs_c)
    results.append({
        "claim": "C",
        "comparison": "Weight Decay vs Base",
        "n": n_c,
        "effect_size_str": (rf"$\Delta\tilde{{m}}={med_c:+.3f}$, "
                            rf"95\,\% CI $[{ci_lo_c:+.3f},\,{ci_hi_c:+.3f}]$"),
        "raw_p": p_c, "adj_p": math.nan, "verdict": "",
        "extra": {"med": med_c, "r": r_c, "ci_lo": ci_lo_c, "ci_hi": ci_hi_c},
    })

    # ── Claim D: Noise Friedman + targeted contrast ───────────────────────────
    base_df     = df[df["reg_state"] == "base"].copy()
    noise_levels = sorted(base_df["noise"].unique())

    # Pivot: rows = (seed, function), columns = noise levels
    piv = base_df.pivot_table(
        index=["seed", "function_name"],
        columns="noise",
        values=VALUE,
        aggfunc="mean",
    ).dropna()
    n_d = len(piv)

    friedman_stat = math.nan
    friedman_p    = math.nan
    contrast_med  = math.nan
    contrast_p    = math.nan
    eta0 = eta_mid = eta_max = math.nan

    if len(noise_levels) >= 2 and n_d > 0:
        groups = [piv[nl].values for nl in noise_levels]
        if len(groups) >= 3:
            try:
                friedman_stat, friedman_p = friedmanchisquare(*groups)
            except Exception:
                pass

        # Targeted contrast: does the second noise segment drop more than the first?
        # Core mode  (η={0, 0.5, 1.0}): [0→0.5] vs [0.5→1.0]
        # Full mode (≥4 levels): [η_0→η_mid] vs [η_mid→η_max]
        # This tests the "accelerating decline" claim.
        eta0    = noise_levels[0]
        eta_mid = noise_levels[len(noise_levels) // 2]
        eta_max = noise_levels[-1]

        if all(nl in piv.columns for nl in [eta0, eta_mid, eta_max]):
            drop1 = piv[eta0].values - piv[eta_mid].values   # first-segment drop
            drop2 = piv[eta_mid].values - piv[eta_max].values  # second-segment drop
            accel = drop2 - drop1   # > 0 means acceleration
            contrast_med = float(np.median(accel))
            try:
                # one-sided: H1 = decline accelerates (drop2 > drop1)
                _, contrast_p = wilcoxon(accel, alternative="greater")
            except TypeError:
                # scipy < 1.7 does not support alternative=; fall back to two-sided / 2
                _, p2 = wilcoxon(accel)
                contrast_p = float(p2) / 2

    noise_tag = "{" + ", ".join(str(nl) for nl in noise_levels) + "}"
    results.append({
        "claim": "D-Friedman",
        "comparison": rf"Noise Friedman ($\eta\in{noise_tag}$, base cond.)",
        "n": n_d,
        "effect_size_str": (rf"$\chi^2_F={friedman_stat:.2f}$"
                            if math.isfinite(friedman_stat) else "--"),
        "raw_p": friedman_p, "adj_p": math.nan, "verdict": "",
        "extra": {"stat": friedman_stat, "p": friedman_p},
    })

    mid_lo  = f"{eta0:g}"
    mid_hi  = f"{eta_mid:g}"
    mid_max = f"{eta_max:g}"
    results.append({
        "claim": "D-contrast",
        "comparison": (rf"Noise acceleration "
                       rf"($\eta$: {mid_lo}$\to${mid_hi} vs {mid_hi}$\to${mid_max})"),
        "n": n_d,
        "effect_size_str": (rf"$\Delta\tilde{{m}}_\mathrm{{accel}}={contrast_med:+.3f}$"
                            if math.isfinite(contrast_med) else "--"),
        "raw_p": contrast_p, "adj_p": math.nan, "verdict": "",
        "extra": {"med": contrast_med, "p": contrast_p},
    })

    # ── Claim E: AUPRC vs val_loss (Spearman ρ) ──────────────────────────────
    valid = df[[VALUE, "val_loss"]].dropna()
    n_e   = len(valid)
    rho_obs, ci_lo_e, ci_hi_e = _bootstrap_spearman_ci(
        valid[VALUE].values, valid["val_loss"].values
    )
    _, rho_p = spearmanr(valid[VALUE].values, valid["val_loss"].values)
    results.append({
        "claim": "E",
        "comparison": r"AUPRC vs val\_loss (Spearman $\rho$)",
        "n": n_e,
        "effect_size_str": (rf"$\rho={rho_obs:.3f}$, "
                            rf"95\,\% CI $[{ci_lo_e:.3f},\,{ci_hi_e:.3f}]$"),
        "raw_p": rho_p,
        "adj_p": rho_p,   # NOT Holm-corrected — see module docstring
        "verdict": "",
        "extra": {"rho": rho_obs, "ci_lo": ci_lo_e, "ci_hi": ci_hi_e, "p": rho_p},
    })

    # ── Holm–Bonferroni on A, B, C, D-Friedman, D-contrast ──────────────────
    holm_claims = {"A", "B", "C", "D-Friedman", "D-contrast"}
    holm_idx    = [i for i, r in enumerate(results) if r["claim"] in holm_claims]
    raw_ps      = [results[i]["raw_p"] for i in holm_idx]
    adj_ps      = _holm_bonferroni(raw_ps)
    for list_pos, orig_idx in enumerate(holm_idx):
        results[orig_idx]["adj_p"] = adj_ps[list_pos]

    # ── Verdicts ─────────────────────────────────────────────────────────────
    ALPHA = 0.05

    def _p_str(p: float) -> str:
        if not math.isfinite(p):
            return "n/a"
        return r"$<0.001$" if p < 0.001 else f"{p:.3f}"

    for res in results:
        p   = res["adj_p"]
        sig = math.isfinite(p) and p < ALPHA
        c   = res["claim"]
        ex  = res["extra"]

        if c == "A":
            res["verdict"] = (
                rf"Sig.\ drop ($p_\mathrm{{adj}}={_p_str(p)}$)" if sig
                else rf"No sig.\ effect ($p_\mathrm{{adj}}={_p_str(p)}$)"
            )
        elif c == "B":
            res["verdict"] = (
                rf"Sig.\ drop ($p_\mathrm{{adj}}={_p_str(p)}$)" if sig
                else rf"No sig.\ effect ($p_\mathrm{{adj}}={_p_str(p)}$)"
            )
        elif c == "C":
            ci_hi = ex.get("ci_hi", math.nan)
            # Non-inferiority: CI upper bound < 0.05 AUPRC is the criterion
            if math.isfinite(ci_hi) and ci_hi < 0.05:
                res["verdict"] = (rf"Non-inferior "
                                  rf"(95\,\% CI $\leq{ci_hi:+.3f}$ AUPRC)")
            elif not sig:
                res["verdict"] = (rf"Fail to reject "
                                  rf"($p_\mathrm{{adj}}={_p_str(p)}$); "
                                  rf"CI $[{ex.get('ci_lo', math.nan):+.3f},"
                                  rf"\,{ci_hi:+.3f}]$")
            else:
                res["verdict"] = (rf"Sig.\ harm "
                                  rf"($p_\mathrm{{adj}}={_p_str(p)}$)")
        elif c == "D-Friedman":
            res["verdict"] = (
                rf"Sig.\ noise effect ($p_\mathrm{{adj}}={_p_str(p)}$)" if sig
                else rf"No sig.\ noise effect ($p_\mathrm{{adj}}={_p_str(p)}$)"
            )
        elif c == "D-contrast":
            res["verdict"] = (
                rf"Decline accelerates ($p_\mathrm{{adj}}={_p_str(p)}$)" if sig
                else rf"No sig.\ acceleration ($p_\mathrm{{adj}}={_p_str(p)}$)"
            )
        elif c == "E":
            res["verdict"] = (
                r"Strong tracking ($p<0.001$, uncorrected)" if rho_p < 0.001
                else rf"Tracking ($p={_p_str(rho_p)}$, uncorrected)" if rho_p < ALPHA
                else r"No sig.\ tracking (uncorrected)"
            )

    return results


# ── Output writers ────────────────────────────────────────────────────────────

def _fmt_p(p: float) -> str:
    if not math.isfinite(p):
        return "--"
    return r"$<0.001$" if p < 0.001 else f"{p:.3f}"


def write_significance_table(results: list[dict[str, Any]], out_dir: Path) -> Path:
    """booktabs LaTeX table: one row per claim, both raw and Holm-adj p."""
    col_spec = "l l r l l l p{3.8cm}"
    header = [
        "Claim", "Comparison", r"$n$", "Effect size",
        r"$p$ (raw)", r"$p$ (Holm)", "Verdict",
    ]
    lines = [
        r"\begin{table}[ht]",
        r"\small\centering",
        r"\caption{%",
        r"  Significance tests for headline thesis claims (multi-seed replication,",
        r"  $N_\mathrm{seeds}$ varied across all three seed locations).",
        r"  Claims A--D are Holm--Bonferroni corrected jointly.",
        r"  Claim E (Spearman $\rho$) is not corrected (uninformative at large $n$).%",
        r"}",
        r"\label{tab:significance_multiseed}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for res in results:
        p_raw_str = _fmt_p(res["raw_p"])
        p_adj_str = _fmt_p(res["adj_p"]) if res["claim"] != "E" else "--"
        row = [
            res["claim"],
            res["comparison"],
            str(res["n"]),
            res["effect_size_str"],
            p_raw_str,
            p_adj_str,
            res["verdict"],
        ]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out = out_dir / "significance_table.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def write_inline_stats(
    results: list[dict[str, Any]],
    df_valid: pd.DataFrame,
    n_before: int,
    n_after: int,
    seeds: list[int],
    out_dir: Path,
) -> Path:
    """
    \\newcommand macros in the same style as stats_analysis.py inline_stats.tex.
    All macros use the 'Seeded' suffix to avoid clashing with existing macros.
    """
    macros: dict[str, str] = {}

    def _p(v: float) -> str:
        if not math.isfinite(v):
            return "--"
        return r"$<0.001$" if v < 0.001 else f"{v:.3f}"

    def _f(v: float, fmt: str = ".3f") -> str:
        return format(v, fmt) if math.isfinite(v) else "--"

    macros["statSeededNSeeds"]       = str(len(seeds))
    macros["statSeededNFunctions"]   = str(len(FUNCTION_NAMES))
    macros["statSeededTotalRuns"]    = str(n_before)
    macros["statSeededRetainedRuns"] = str(n_after)
    pct = n_after / max(n_before, 1) * 100
    macros["statSeededRetainedPct"]  = rf"{pct:.1f}\%"

    for res in results:
        c  = res["claim"]
        ex = res["extra"]

        if c == "A":
            macros["statDropoutDropSeeded"]    = _f(ex["med"], "+.3f")
            macros["statDropoutRBCSeeded"]     = _f(ex["r"],   "+.3f")
            macros["statDropoutPvalSeeded"]    = _p(res["raw_p"])
            macros["statDropoutPvalAdjSeeded"] = _p(res["adj_p"])
        elif c == "B":
            macros["statDropoutWDDropSeeded"]    = _f(ex["med"], "+.3f")
            macros["statDropoutWDRBCSeeded"]     = _f(ex["r"],   "+.3f")
            macros["statDropoutWDPvalSeeded"]    = _p(res["raw_p"])
            macros["statDropoutWDPvalAdjSeeded"] = _p(res["adj_p"])
        elif c == "C":
            macros["statWDDropSeeded"]    = _f(ex["med"],   "+.3f")
            macros["statWDRBCSeeded"]     = _f(ex["r"],     "+.3f")
            macros["statWDCILoSeeded"]    = _f(ex["ci_lo"], "+.3f")
            macros["statWDCIHiSeeded"]    = _f(ex["ci_hi"], "+.3f")
            macros["statWDPvalSeeded"]    = _p(res["raw_p"])
            macros["statWDPvalAdjSeeded"] = _p(res["adj_p"])
        elif c == "D-Friedman":
            macros["statFriedmanStatSeeded"]    = _f(ex["stat"], ".2f")
            macros["statFriedmanPvalSeeded"]    = _p(ex["p"])
            macros["statFriedmanPvalAdjSeeded"] = _p(res["adj_p"])
        elif c == "D-contrast":
            macros["statNoiseAccelMedSeeded"]    = _f(ex["med"], "+.3f")
            macros["statNoiseAccelPvalSeeded"]   = _p(ex["p"])
            macros["statNoiseAccelPvalAdjSeeded"]= _p(res["adj_p"])
        elif c == "E":
            macros["statSpearmanRhoSeeded"]   = _f(ex["rho"],   ".3f")
            macros["statSpearmanCILoSeeded"]  = _f(ex["ci_lo"], ".3f")
            macros["statSpearmanCIHiSeeded"]  = _f(ex["ci_hi"], ".3f")
            macros["statSpearmanPvalSeeded"]  = _p(ex["p"])

    lines = [
        r"% Auto-generated by significance.py — do not edit manually.",
        r"% Multi-seed replication inline stats. Suffix 'Seeded' avoids",
        r"% conflicts with stats_analysis.py macros. \input this file and",
        r"% drop macros directly into thesis prose.",
        "",
    ]
    for k in sorted(macros):
        v = macros[k]
        lines.append(rf"\newcommand{{\{k}}}{{{v}}}")

    out = out_dir / "inline_stats_multiseed.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


# ── Human-readable summary ────────────────────────────────────────────────────

def print_summary(
    results: list[dict[str, Any]],
    n_before: int,
    n_after: int,
    seeds: list[int],
) -> None:
    ALPHA = 0.05
    bar   = "═" * 68
    print(f"\n{bar}")
    print("MULTI-SEED SIGNIFICANCE SUMMARY")
    print(f"  Seeds:     {seeds}  ({len(seeds)} replications)")
    print(f"  Runs:      {n_before} total → {n_after} retained after convergence filter")
    print(bar)

    def _strip_latex(s: str) -> str:
        for old, new in [
            (r"\Delta\tilde{m}", "Δm̃"), (r"\chi^2_F", "χ²_F"),
            (r"\rho", "ρ"), (r"\$", "$"), (r"$", ""),
            (r"\,", " "), (r"\%", "%"), (r"\ ", " "),
            (r"\mathrm{adj}", "adj"), (r"\mathrm{accel}", "accel"),
            (r"\leq", "≤"), (r"\geq", "≥"), (r"\in", "∈"),
            (r"\eta", "η"), (r"_{", "("), (r"^{", "^("),
            ("}", ")"), ("{", "("),
        ]:
            s = s.replace(old, new)
        return s

    for res in results:
        p_adj = res["adj_p"]
        p_raw = res["raw_p"]
        sig_marker = " *" if (math.isfinite(p_adj) and p_adj < ALPHA) else "  "
        p_raw_s = f"{p_raw:.4f}" if math.isfinite(p_raw) else "  n/a  "
        p_adj_s = f"{p_adj:.4f}" if math.isfinite(p_adj) else "  n/a  "
        print(
            f"  [{res['claim']:<10s}] n={res['n']:4d}  "
            f"p_raw={p_raw_s}  p_holm={p_adj_s}{sig_marker}  "
            f"{res['comparison']}"
        )
        print(f"             effect: {_strip_latex(res['effect_size_str'])}")
        print(f"             verdict: {_strip_latex(res['verdict'])}")
    print(bar)
    print("  * = significant after Holm–Bonferroni correction (α=0.05)")
    print(bar)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--seeds", default="42,43,44,45,46",
        help="Comma-separated seed integers (default: 42,43,44,45,46).",
    )
    p.add_argument(
        "--conditions", choices=["core", "full"], default="core",
        help=(
            "'core': 4 reg states × 3 noise levels × adam × small [default]. "
            "'full': all deduplicated EXPERIMENTS × both model sizes."
        ),
    )
    p.add_argument(
        "--full", action="store_true",
        help="Alias for --conditions full.",
    )
    p.add_argument(
        "--snap-root", default="snapshots_multiseed", type=Path,
        help="Snapshot root (default: snapshots_multiseed/).",
    )
    p.add_argument(
        "--results-root", default="results_multiseed", type=Path,
        help="Output root (default: results_multiseed/).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print full plan and run count without training.",
    )
    p.add_argument(
        "--analysis-only", action="store_true",
        help="Skip training; load existing checkpoints and run analysis/stats.",
    )
    p.add_argument(
        "--no-mps", action="store_true",
        help="Force CPU even if MPS is available.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    seeds     = [int(s.strip()) for s in args.seeds.split(",")]
    use_full  = args.full or (args.conditions == "full")
    dev       = _DEVICE
    if args.no_mps and dev.type == "mps":
        dev = torch.device("cpu")

    conditions  = _build_full_conditions() if use_full else _build_core_conditions()
    model_sizes = ["small", "big"] if use_full else ["small"]

    snap_root    = args.snap_root
    results_root = args.results_root
    stats_dir    = results_root / "stats"
    total_runs   = len(seeds) * len(FUNCTION_NAMES) * len(conditions) * len(model_sizes)

    print(f"\nDevice:      {dev}")
    print(f"Mode:        {'full' if use_full else 'core'}")
    print(f"Seeds:       {seeds}")
    print(f"Functions:   {len(FUNCTION_NAMES)}")
    print(f"Conditions:  {len(conditions)}")
    print(f"Model sizes: {model_sizes}")
    print(f"Total runs:  {total_runs}")
    print(f"Snap root:   {snap_root}/")
    print(f"Results:     {results_root}/")

    if len(seeds) >= 2:
        print("\nVerifying seed divergence …")
        _verify_seeds_differ(seeds[0], seeds[1])

    if args.dry_run:
        print("\n[dry-run] Training plan:")
        run_training(seeds, conditions, model_sizes, snap_root, dev, dry_run=True)
        print(f"\n[dry-run] {total_runs} runs planned. No files written.")
        return

    # ── Training ──────────────────────────────────────────────────────────────
    if not args.analysis_only:
        print(f"\n{'─'*68}")
        print("TRAINING PHASE")
        print(f"{'─'*68}")
        run_training(seeds, conditions, model_sizes, snap_root, dev)

    # ── Analysis ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print("ANALYSIS PHASE")
    print(f"{'─'*68}")
    df = run_analysis(seeds, conditions, model_sizes, snap_root)
    print(f"  Analysed {len(df)} checkpoints.")

    stats_dir.mkdir(parents=True, exist_ok=True)
    csv_path = stats_dir / "multiseed_runs.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Wrote {csv_path}")

    df_conv, n_before, n_after = apply_convergence_filter(df)
    df_valid = df_conv[df_conv["auprc_full"].notna()].copy()
    print(f"  {len(df_valid)} runs have valid auprc_full.")

    if df_valid.empty:
        print("  ERROR: no valid runs after convergence filter. Exiting.")
        return

    # ── Significance tests ────────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print("SIGNIFICANCE TESTS")
    print(f"{'─'*68}")
    test_results = run_significance_tests(df_valid)

    # ── Write outputs ─────────────────────────────────────────────────────────
    tbl_path    = write_significance_table(test_results, stats_dir)
    inline_path = write_inline_stats(test_results, df_valid,
                                     n_before, n_after, seeds, stats_dir)
    print(f"  Wrote {tbl_path}")
    print(f"  Wrote {inline_path}")

    print_summary(test_results, n_before, n_after, seeds)


if __name__ == "__main__":
    main()
