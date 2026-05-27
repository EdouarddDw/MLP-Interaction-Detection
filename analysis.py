#!/usr/bin/env python3
"""Run NID on saved checkpoints and export thesis-compatible analysis CSVs."""

from __future__ import annotations

import argparse
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

import NID
from config import EXPERIMENTS
from multilayer_perceptron import MLP
from synth import functions as synth_functions


MODEL_ARCH = {
    "small": [64, 64],
    "big": [140, 100, 60, 20],
}
NUM_FEATURES = 10


def _all_interactions_one_based(num_features: int = NUM_FEATURES) -> set[tuple[int, ...]]:
    all_sets: set[tuple[int, ...]] = set()
    for r in range(2, num_features + 1):
        all_sets.update(combinations(range(1, num_features + 1), r))
    return all_sets


ALL_CANDIDATES = _all_interactions_one_based(NUM_FEATURES)


def _normalize_interaction_key(raw: Any) -> tuple[int, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("(") and stripped.endswith(")"):
            stripped = stripped[1:-1]
        if stripped.startswith("{") and stripped.endswith("}"):
            stripped = stripped[1:-1]
        if not stripped:
            return None
        parts = [p.strip() for p in stripped.split(",") if p.strip()]
        try:
            vals = [int(p) for p in parts]
        except ValueError:
            return None
        if len(vals) < 2:
            return None
        return tuple(sorted(vals))
    if isinstance(raw, (list, tuple, set, frozenset, np.ndarray)):
        try:
            vals = [int(x) for x in list(raw)]
        except (TypeError, ValueError):
            return None
        if len(vals) < 2:
            return None
        return tuple(sorted(vals))
    return None


def _looks_zero_indexed(interactions: Iterable[tuple[int, ...]]) -> bool:
    seen = list(interactions)
    if not seen:
        return False
    min_val = min(min(i) for i in seen)
    max_val = max(max(i) for i in seen)
    return min_val == 0 and max_val <= NUM_FEATURES - 1


def _normalize_ground_truth(raw_gt: Any) -> set[tuple[int, ...]]:
    norm: set[tuple[int, ...]] = set()
    if raw_gt is None:
        return norm
    if isinstance(raw_gt, dict):
        iterable = raw_gt.keys()
    else:
        iterable = raw_gt
    for item in iterable:
        key = _normalize_interaction_key(item)
        if key is not None and len(key) >= 2:
            norm.add(key)
    if _looks_zero_indexed(norm):
        norm = {tuple(v + 1 for v in inter) for inter in norm}
    return norm


def get_ground_truth_interactions(function, num_samples: int = 30000, noise: float = 0.0, seed: int = 42) -> set[frozenset[int]]:
    _, _, gt = function(num_samples=num_samples, seed=seed)
    normalized = _normalize_ground_truth(gt)
    return {frozenset(i) for i in normalized}


def get_ground_truth(function_name: str, num_samples: int = 30000, seed: int = 42) -> set[tuple[int, ...]]:
    function_map = {f.__name__: f for f in synth_functions}
    fn = function_map[function_name]
    _, _, gt = fn(num_samples=num_samples, seed=seed)
    return _normalize_ground_truth(gt)


def normalize_nid_output(nid_output: Any) -> dict[tuple[int, ...], float]:
    mapping: dict[tuple[int, ...], float] = {}

    if nid_output is None:
        return mapping

    if isinstance(nid_output, pd.DataFrame):
        cols = {c.lower(): c for c in nid_output.columns}
        i_col = cols.get("interaction") or cols.get("interactions")
        s_col = cols.get("score") or cols.get("strength") or cols.get("weight")
        if i_col is None or s_col is None:
            return mapping
        for _, row in nid_output.iterrows():
            k = _normalize_interaction_key(row[i_col])
            if k is not None and len(k) >= 2:
                mapping[k] = float(row[s_col])
    elif isinstance(nid_output, dict):
        for k_raw, v in nid_output.items():
            k = _normalize_interaction_key(k_raw)
            if k is not None and len(k) >= 2:
                mapping[k] = float(v)
    else:
        for item in nid_output:
            if isinstance(item, dict):
                k_raw = item.get("interaction", item.get("interactions"))
                s_raw = item.get("score", item.get("strength", item.get("weight", 0.0)))
                k = _normalize_interaction_key(k_raw)
                if k is not None and len(k) >= 2:
                    mapping[k] = float(s_raw)
                continue
            if isinstance(item, (tuple, list)) and len(item) == 2:
                k = _normalize_interaction_key(item[0])
                if k is not None and len(k) >= 2:
                    mapping[k] = float(item[1])

    if _looks_zero_indexed(mapping.keys()):
        mapping = {tuple(v + 1 for v in inter): s for inter, s in mapping.items()}

    return {tuple(sorted(k)): float(v) for k, v in mapping.items() if len(k) >= 2}


def _extract_state_dict(ckpt_obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        if "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
            return ckpt_obj["model_state_dict"]
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"]
        if ckpt_obj and all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj
    raise ValueError("Unsupported checkpoint format for state_dict.")


def _extract_epoch_val_loss(ckpt_obj: Any, checkpoint_path: Path) -> tuple[int | float, float]:
    epoch = math.nan
    val_loss = math.nan
    if isinstance(ckpt_obj, dict):
        if "epoch" in ckpt_obj:
            try:
                epoch = int(ckpt_obj["epoch"])
            except Exception:
                epoch = math.nan
        if "val_loss" in ckpt_obj:
            try:
                val_loss = float(ckpt_obj["val_loss"])
            except Exception:
                val_loss = math.nan
    if isinstance(epoch, float) and math.isnan(epoch):
        stem = checkpoint_path.stem
        if stem.startswith("epoch_") or stem.startswith("best_epoch_"):
            token = stem.split("_")[-1]
            if token.isdigit():
                epoch = int(token)
    return epoch, val_loss


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def load_model_and_interactions(
    checkpoint_path: Path,
    dropout: float = 0.0,
    model_size: str | None = None,
) -> tuple[MLP | None, list[tuple[tuple[int, ...], float]] | None, float, int | float]:
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return None, None, math.nan, math.nan

    if model_size is None:
        parts = {p.lower() for p in checkpoint_path.parts}
        model_size = "big" if "big" in parts else "small"
    hidden_units = MODEL_ARCH[model_size]

    try:
        state_dict = _strip_module_prefix(_extract_state_dict(ckpt))
        model = MLP(
            num_features=NUM_FEATURES,
            hidden_units=hidden_units,
            use_main_effect_nets=False,
            main_effect_net_units=[10, 10, 10],
            dropout=float(dropout),
        )
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        interactions = NID.get_interactions(NID.get_weights(model), one_indexed=True)
        val_epoch, val_loss = _extract_epoch_val_loss(ckpt, checkpoint_path)
        return model, interactions, val_loss, val_epoch
    except Exception:
        return None, None, math.nan, math.nan


def _is_subset_of_any_gt(candidate: tuple[int, ...], gt_set: set[tuple[int, ...]]) -> bool:
    s = set(candidate)
    for gt in gt_set:
        gt_s = set(gt)
        if s < gt_s:
            return True
    return False


def compute_auroc_data(
    gt_interactions: set[frozenset[int]] | set[tuple[int, ...]],
    nid_interactions: Any,
    full_space: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    gt = {tuple(sorted(map(int, g))) for g in gt_interactions}
    nid_map = normalize_nid_output(nid_interactions)
    detected = set(nid_map.keys())

    if full_space:
        candidates = sorted(ALL_CANDIDATES)
    else:
        candidates = sorted(gt | detected)
        candidates = [c for c in candidates if not (c not in gt and _is_subset_of_any_gt(c, gt))]

    if not candidates:
        return None, None

    y_true = np.array([1 if c in gt else 0 for c in candidates], dtype=int)
    y_score = np.array([float(nid_map.get(c, 0.0)) for c in candidates], dtype=float)
    return y_score, y_true


def compute_metrics(scores: np.ndarray | None, labels: np.ndarray | None) -> dict[str, float]:
    if scores is None or labels is None or len(scores) == 0:
        return {"auroc": math.nan, "precision": math.nan, "recall": math.nan, "auprc": math.nan}

    if len(np.unique(labels)) < 2:
        auroc = math.nan
    else:
        auroc = float(roc_auc_score(labels, scores))

    if np.any(labels == 1):
        auprc = float(average_precision_score(labels, scores))
    else:
        auprc = math.nan

    preds = (scores > 0).astype(int)
    precision = float(precision_score(labels, preds, zero_division=0))
    recall = float(recall_score(labels, preds, zero_division=0))
    return {"auroc": auroc, "precision": precision, "recall": recall, "auprc": auprc}


def _metric_triplet(gt: set[tuple[int, ...]], nid_map: dict[tuple[int, ...], float]) -> tuple[float, float, float, float]:
    s_restricted, y_restricted = compute_auroc_data(gt, nid_map, full_space=False)
    m_restricted = compute_metrics(s_restricted, y_restricted)

    s_full, y_full = compute_auroc_data(gt, nid_map, full_space=True)
    m_full = compute_metrics(s_full, y_full)
    return m_restricted["auprc"], m_full["auprc"], m_restricted["auroc"], m_full["auroc"]


def _safe_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in {"1", "true", "yes"}
    return bool(v)


def _dedup_experiments_by_name(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for exp in experiments:
        name = str(exp.get("name", ""))
        if name in seen:
            continue
        seen.add(name)
        out.append(exp)
    return out


def _sorted_epoch_checkpoints(exp_dir: Path) -> list[Path]:
    paths = list(exp_dir.glob("epoch_*.pt"))
    def key_fn(path: Path) -> int:
        stem = path.stem
        token = stem.split("_")[-1]
        return int(token) if token.isdigit() else -1
    return sorted(paths, key=key_fn)


def _best_checkpoint(exp_dir: Path) -> Path | None:
    best = sorted(exp_dir.glob("best_epoch_*.pt"))
    if best:
        return best[0]
    epochs = _sorted_epoch_checkpoints(exp_dir)
    return epochs[-1] if epochs else None


def _ensure_dirs(results_root: Path) -> None:
    (results_root / "small").mkdir(parents=True, exist_ok=True)
    (results_root / "big").mkdir(parents=True, exist_ok=True)
    (results_root / "trajectories").mkdir(parents=True, exist_ok=True)


def _base_best_row(
    function_name: str,
    experiment: dict[str, Any],
    model_size: str,
) -> dict[str, Any]:
    return {
        "success": False,
        "function_name": function_name,
        "function": function_name,
        "experiment": experiment["name"],
        "noise": float(experiment["noise"]),
        "optimizer": experiment["optimizer"],
        "dropout": float(experiment["dropout"]),
        "weight_decay": _safe_bool(experiment["weight_decay"]),
        "model_size": model_size,
        "best_epoch": math.nan,
        "checkpoint_path": "",
        "val_loss": math.nan,
        "auprc_full": math.nan,
        "auroc": math.nan,
        "auroc_full": math.nan,
        "num_gt": math.nan,
        "num_detected": math.nan,
        "num_candidates": len(ALL_CANDIDATES),
        "error": "",
    }


def _base_traj_row(
    function_name: str,
    experiment: dict[str, Any],
    model_size: str,
) -> dict[str, Any]:
    return {
        "epoch": math.nan,
        "function_name": function_name,
        "function": function_name,
        "experiment": experiment["name"],
        "noise": float(experiment["noise"]),
        "optimizer": experiment["optimizer"],
        "dropout": float(experiment["dropout"]),
        "weight_decay": _safe_bool(experiment["weight_decay"]),
        "model_size": model_size,
        "checkpoint_path": "",
        "val_loss": math.nan,
        "auprc": math.nan,
        "auprc_full": math.nan,
        "auroc": math.nan,
        "auroc_full": math.nan,
        "success": False,
        "error": "",
    }


def run_analysis(args: argparse.Namespace) -> None:
    snapshots_root = Path(args.snapshots_root)
    results_root = Path(args.results_root)
    selected_model_sizes = [args.model_size] if args.model_size else ["small", "big"]
    selected_functions = [args.function] if args.function else [f"f{i}" for i in range(1, 11)]

    experiments = _dedup_experiments_by_name(EXPERIMENTS)
    if args.experiment:
        experiments = [e for e in experiments if e["name"] == args.experiment]
        if not experiments:
            raise SystemExit(f"Experiment not found in config.EXPERIMENTS: {args.experiment}")

    if not args.dry_run:
        _ensure_dirs(results_root)

    all_epoch_rows: list[dict[str, Any]] = []

    for model_size in selected_model_sizes:
        for function_name in selected_functions:
            print(f"[analysis] {model_size}/{function_name}")
            gt = get_ground_truth(function_name)
            best_rows: list[dict[str, Any]] = []

            for exp in experiments:
                exp_name = exp["name"]
                exp_dir = snapshots_root / model_size / function_name / exp_name

                best_row = _base_best_row(function_name, exp, model_size)
                traj_rows: list[dict[str, Any]] = []

                if not exp_dir.exists():
                    best_row["error"] = f"Missing experiment directory: {exp_dir}"
                    if not args.best_only:
                        error_row = _base_traj_row(function_name, exp, model_size)
                        error_row["error"] = best_row["error"]
                        traj_rows.append(error_row)
                else:
                    if args.best_only:
                        checkpoints = [_best_checkpoint(exp_dir)] if _best_checkpoint(exp_dir) else []
                    else:
                        checkpoints = _sorted_epoch_checkpoints(exp_dir)
                        if not checkpoints:
                            b = _best_checkpoint(exp_dir)
                            checkpoints = [b] if b else []

                    if not checkpoints:
                        best_row["error"] = f"No checkpoints found in {exp_dir}"
                        if not args.best_only:
                            error_row = _base_traj_row(function_name, exp, model_size)
                            error_row["error"] = best_row["error"]
                            traj_rows.append(error_row)
                    else:
                        for ckpt in checkpoints:
                            model, nid_interactions, val_loss, epoch = load_model_and_interactions(
                                ckpt,
                                dropout=float(exp["dropout"]),
                                model_size=model_size,
                            )
                            row = _base_traj_row(function_name, exp, model_size)
                            row["checkpoint_path"] = str(ckpt)
                            row["epoch"] = epoch
                            row["val_loss"] = val_loss

                            if model is None or nid_interactions is None:
                                row["error"] = f"Failed to load or parse checkpoint: {ckpt}"
                            else:
                                nid_map = normalize_nid_output(nid_interactions)
                                auprc, auprc_full, auroc, auroc_full = _metric_triplet(gt, nid_map)
                                row.update(
                                    {
                                        "success": True,
                                        "auprc": auprc,
                                        "auprc_full": auprc_full,
                                        "auroc": auroc,
                                        "auroc_full": auroc_full,
                                    }
                                )
                            traj_rows.append(row)

                        best_ckpt = _best_checkpoint(exp_dir)
                        if best_ckpt is None:
                            best_row["error"] = f"No best checkpoint found in {exp_dir}"
                        else:
                            model, nid_interactions, val_loss, epoch = load_model_and_interactions(
                                best_ckpt,
                                dropout=float(exp["dropout"]),
                                model_size=model_size,
                            )
                            best_row["checkpoint_path"] = str(best_ckpt)
                            best_row["best_epoch"] = epoch
                            best_row["val_loss"] = val_loss
                            if model is None or nid_interactions is None:
                                best_row["error"] = f"Failed to load or parse checkpoint: {best_ckpt}"
                            else:
                                nid_map = normalize_nid_output(nid_interactions)
                                _, auprc_full, auroc, auroc_full = _metric_triplet(gt, nid_map)
                                best_row.update(
                                    {
                                        "success": True,
                                        "auprc_full": auprc_full,
                                        "auroc": auroc,
                                        "auroc_full": auroc_full,
                                        "num_gt": len(gt),
                                        "num_detected": len(nid_map),
                                        "num_candidates": len(ALL_CANDIDATES),
                                        "error": "",
                                    }
                                )

                best_rows.append(best_row)

                if not args.best_only:
                    traj_path = results_root / "trajectories" / f"{model_size}_{function_name}_{exp_name}_trajectory.csv"
                    traj_df = pd.DataFrame(traj_rows)
                    all_epoch_rows.extend(traj_rows)
                    if args.dry_run:
                        print(f"  [dry-run] would write {traj_path} ({len(traj_df)} rows)")
                    else:
                        if traj_path.exists() and not args.overwrite:
                            print(f"  [skip] trajectory exists (use --overwrite): {traj_path}")
                        else:
                            traj_df.to_csv(traj_path, index=False)

            if not args.trajectories_only:
                best_path = results_root / model_size / f"{function_name}_analysis.csv"
                best_df = pd.DataFrame(best_rows)
                if args.dry_run:
                    print(f"  [dry-run] would write {best_path} ({len(best_df)} rows)")
                else:
                    best_df.to_csv(best_path, index=False)

    if not args.best_only:
        epoch_path = results_root / "auroc_by_epoch.csv"
        epoch_df = pd.DataFrame(all_epoch_rows)
        if not epoch_df.empty:
            keep_cols = [
                "epoch",
                "function_name",
                "experiment",
                "noise",
                "optimizer",
                "dropout",
                "weight_decay",
                "model_size",
                "val_loss",
                "auprc",
                "auprc_full",
                "auroc",
                "auroc_full",
                "success",
                "error",
                "checkpoint_path",
            ]
            epoch_df = epoch_df[[c for c in keep_cols if c in epoch_df.columns]]
        if args.dry_run:
            print(f"[dry-run] would write {epoch_path} ({len(epoch_df)} rows)")
        else:
            epoch_df.to_csv(epoch_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze checkpoints with NID and export CSV metrics.")
    parser.add_argument("--snapshots-root", default="snapshots_clean", help="Root directory of saved snapshots.")
    parser.add_argument("--results-root", default="results", help="Root output directory for CSVs.")
    parser.add_argument("--model-size", choices=["small", "big"], help="Optional model size filter.")
    parser.add_argument("--function", choices=[f"f{i}" for i in range(1, 11)], help="Optional function filter.")
    parser.add_argument("--experiment", help="Optional experiment-name filter from config.EXPERIMENTS.")
    parser.add_argument("--best-only", action="store_true", help="Only analyze and save best checkpoints.")
    parser.add_argument("--trajectories-only", action="store_true", help="Only generate trajectory outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without writing files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing trajectory CSVs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
