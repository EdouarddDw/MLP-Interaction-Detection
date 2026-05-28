#!/usr/bin/env python3
"""Statistical analysis for thesis.

Loads best-epoch results, applies the trajectory-based convergence filter,
and writes LaTeX tables + inline \\newcommand stats to results/stats/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

AUPRC_COLUMN = "auprc_full"
NOISE_ORDER = [0.0, 0.1, 0.2, 0.5, 1.0]
REGULARIZATION_ORDER = ["base", "dropout", "weight_decay", "dropout+weight_decay"]
REGULARIZATION_LABELS = {
    "base": "Base",
    "dropout": "Dropout",
    "weight_decay": "Weight decay",
    "dropout+weight_decay": "Dropout + weight decay",
}
FUNCTION_ORDER = [f"f{i}" for i in range(1, 11)]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _regularization_state(dropout: float, weight_decay: bool) -> str:
    if dropout > 0.0 and weight_decay:
        return "dropout+weight_decay"
    if dropout > 0.0:
        return "dropout"
    if weight_decay:
        return "weight_decay"
    return "base"


def _read_analysis_csvs(folder: Path, model_size: str) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    if not folder.exists():
        return frames
    for path in sorted(folder.glob("*_analysis.csv")):
        if path.name.endswith("_trajectory.csv"):
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["model_size"] = model_size
        frame["source_file"] = path.name
        frames.append(frame)
    return frames


def _filter_convergent_runs(
    df: pd.DataFrame, trajectories_dir: Path, min_reduction: float = 0.03
) -> pd.DataFrame:
    trajectory_files = (
        sorted(trajectories_dir.glob("*_trajectory.csv"))
        if trajectories_dir.exists()
        else []
    )
    frames: list[pd.DataFrame] = []
    for path in trajectory_files:
        model_size = path.name.split("_")[0]
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        if "model_size" not in frame.columns:
            frame["model_size"] = model_size
        frames.append(frame)

    if not frames:
        return df

    traj_df = pd.concat(frames, ignore_index=True)
    traj_df["epoch"] = pd.to_numeric(traj_df["epoch"], errors="coerce")
    traj_df["val_loss"] = pd.to_numeric(traj_df["val_loss"], errors="coerce")
    traj_df = traj_df.dropna(subset=["epoch", "val_loss"])

    convergent_keys: set[tuple] = set()
    for (function_name, experiment), group in traj_df.groupby(
        ["function_name", "experiment"]
    ):
        start_rows = group.loc[group["epoch"] == group["epoch"].min(), "val_loss"]
        end_rows = group.loc[group["epoch"] == group["epoch"].max(), "val_loss"]
        if start_rows.empty or end_rows.empty:
            continue
        start_loss = float(start_rows.iloc[0])
        end_loss = float(end_rows.iloc[0])
        if start_loss <= 0 or (start_loss - end_loss) / start_loss < min_reduction:
            continue
        convergent_keys.add((function_name, experiment))

    group_col = "experiment_name" if "experiment_name" in df.columns else "experiment"
    keep = df.apply(
        lambda row: (
            True
            if pd.isna(row.get("function_name")) or pd.isna(row.get(group_col))
            else (row["function_name"], row[group_col]) in convergent_keys
        ),
        axis=1,
    )
    n_removed = int((~keep).sum())
    print(
        f"  [convergence filter] removed {n_removed} runs "
        f"({n_removed / len(df) * 100:.1f}%) based on trajectory val_loss"
    )
    return df[keep].copy()


def load_data(
    results_root: Path, trajectories_dir: Path
) -> tuple[pd.DataFrame, int, int]:
    """Returns (filtered_df, n_before_filter, n_after_filter)."""
    frames: list[pd.DataFrame] = []
    frames.extend(_read_analysis_csvs(results_root / "small", "small"))
    frames.extend(_read_analysis_csvs(results_root / "big", "big"))

    if not frames:
        for path in sorted(results_root.glob("small_*_analysis.csv")):
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            frame = frame.copy()
            frame["model_size"] = "small"
            frame["source_file"] = path.name
            frames.append(frame)

    if not frames:
        raise RuntimeError(f"No analysis CSVs found under {results_root}")

    df = pd.concat(frames, ignore_index=True)
    df = df[df["success"].fillna(False)].copy()
    df = df[df["auroc"].notna()].copy()

    if AUPRC_COLUMN not in df.columns:
        df[AUPRC_COLUMN] = None
    if "auroc_full" not in df.columns:
        df["auroc_full"] = None

    df["noise"] = df["noise"].astype(float)
    df["dropout"] = df["dropout"].astype(float)
    df["weight_decay"] = df["weight_decay"].astype(bool)
    df["regularization_state"] = [
        _regularization_state(d, w)
        for d, w in zip(df["dropout"], df["weight_decay"])
    ]

    total_before = len(df)
    df = _filter_convergent_runs(df, trajectories_dir)
    total_after = len(df)
    return df, total_before, total_after


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _two_stage_mean(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Average within (function_name, *group_cols), then average across functions."""
    func_level = df.groupby(
        ["function_name", *group_cols], as_index=False
    )[AUPRC_COLUMN].mean()
    summary = func_level.groupby(group_cols, as_index=False)[AUPRC_COLUMN].agg(
        mean="mean", std="std"
    )
    summary["std"] = summary["std"].fillna(0.0)
    return summary


def _noise_series(df: pd.DataFrame, model_size: str | None = None) -> pd.Series:
    """Two-stage mean AUPRC indexed by noise, optionally filtered by model_size."""
    sub = df if model_size is None else df[df["model_size"] == model_size]
    cols = ["function_name", "noise"]
    func_level = sub.groupby(cols, as_index=False)[AUPRC_COLUMN].mean()
    result = func_level.groupby("noise")[AUPRC_COLUMN].mean()
    return result.reindex(NOISE_ORDER)


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------


def _latex_table(
    header: list[str], rows: list[list[str]], caption: str, label: str
) -> str:
    col_spec = "l" + "r" * (len(header) - 1)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------


def make_table_auprc_by_function(df: pd.DataFrame) -> str:
    rows = []
    for fn in FUNCTION_ORDER:
        sub = df[df["function_name"] == fn][AUPRC_COLUMN].dropna()
        if sub.empty:
            rows.append([fn, "--", "--", "--", "--"])
        else:
            rows.append(
                [
                    fn,
                    f"{sub.mean():.3f}",
                    f"{sub.std():.3f}",
                    f"{sub.min():.3f}",
                    f"{sub.max():.3f}",
                ]
            )
    caption = (
        r"Mean AUPRC ($2^{[10]}$) per synthetic function across all experimental conditions."
    )
    return _latex_table(
        ["Function", "Mean AUPRC", "SD", "Min", "Max"],
        rows,
        caption,
        "tab:auprc_by_function",
    )


def make_table_auprc_by_regularization(df: pd.DataFrame) -> str:
    summary = _two_stage_mean(df, ["regularization_state"]).set_index(
        "regularization_state"
    )
    base_mean = (
        float(summary.loc["base", "mean"]) if "base" in summary.index else float("nan")
    )

    rows = []
    for state in REGULARIZATION_ORDER:
        label = REGULARIZATION_LABELS[state]
        if state not in summary.index:
            rows.append([label, "--", "--", "--"])
            continue
        m = float(summary.loc[state, "mean"])
        s = float(summary.loc[state, "std"])
        if state == "base":
            drop_str = "--"
        else:
            pct = (m - base_mean) / base_mean * 100
            drop_str = rf"{pct:.1f}\%"
        rows.append([label, f"{m:.3f}", f"{s:.3f}", drop_str])

    caption = (
        r"Mean AUPRC ($2^{[10]}$) by regularisation condition, averaged across all "
        r"functions, noise levels, and architectures."
    )
    return _latex_table(
        ["Regularisation", "Mean AUPRC", "SD", r"Drop vs Base (\%)"],
        rows,
        caption,
        "tab:auprc_by_regularization",
    )


def make_table_auprc_by_noise(df: pd.DataFrame) -> str:
    pooled = _noise_series(df)
    small = _noise_series(df, "small")
    big = _noise_series(df, "big")
    noise0 = float(pooled[0.0]) if not np.isnan(pooled[0.0]) else float("nan")

    rows = []
    for noise in NOISE_ORDER:
        sm = float(small[noise]) if not np.isnan(small[noise]) else float("nan")
        bg = float(big[noise]) if not np.isnan(big[noise]) else float("nan")
        pl = float(pooled[noise]) if not np.isnan(pooled[noise]) else float("nan")

        sm_str = f"{sm:.3f}" if not np.isnan(sm) else "--"
        bg_str = f"{bg:.3f}" if not np.isnan(bg) else "--"
        pl_str = f"{pl:.3f}" if not np.isnan(pl) else "--"

        if noise == 0.0 or np.isnan(noise0) or np.isnan(pl):
            drop_str = "--"
        else:
            pct = (pl - noise0) / noise0 * 100
            drop_str = rf"{pct:.1f}\%"

        rows.append([f"{noise:g}", sm_str, bg_str, pl_str, drop_str])

    caption = r"Mean AUPRC ($2^{[10]}$) by noise level and architecture."
    return _latex_table(
        ["Noise", "Small Mean", "Big Mean", "Pooled Mean", r"Drop vs Noise=0 (\%)"],
        rows,
        caption,
        "tab:auprc_by_noise",
    )


def make_table_auprc_by_optimizer(df: pd.DataFrame) -> str:
    if "optimizer" not in df.columns:
        return "% optimizer column not found\n"
    summary = _two_stage_mean(df, ["optimizer"]).set_index("optimizer")
    rows = []
    for opt in ["adam", "sgd"]:
        label = opt.capitalize()
        if opt not in summary.index:
            rows.append([label, "--", "--"])
            continue
        m = float(summary.loc[opt, "mean"])
        s = float(summary.loc[opt, "std"])
        rows.append([label, f"{m:.3f}", f"{s:.3f}"])

    caption = (
        r"Mean AUPRC ($2^{[10]}$) by optimizer, averaged across all other conditions."
    )
    return _latex_table(
        ["Optimizer", "Mean AUPRC", "SD"],
        rows,
        caption,
        "tab:auprc_by_optimizer",
    )


def make_table_architecture_comparison(df: pd.DataFrame) -> str:
    arch = (
        df.groupby(["function_name", "model_size"])[AUPRC_COLUMN]
        .mean()
        .unstack("model_size")
    )

    rows = []
    for fn in FUNCTION_ORDER:
        if fn not in arch.index:
            rows.append([fn, "--", "--", "--", "--"])
            continue
        sm = float(arch.loc[fn, "small"]) if "small" in arch.columns else float("nan")
        bg = float(arch.loc[fn, "big"]) if "big" in arch.columns else float("nan")

        sm_str = f"{sm:.3f}" if not np.isnan(sm) else "--"
        bg_str = f"{bg:.3f}" if not np.isnan(bg) else "--"

        if np.isnan(sm) or np.isnan(bg):
            delta_str, winner = "--", "--"
        else:
            delta = bg - sm
            delta_str = f"{delta:+.3f}"
            if abs(delta) < 0.01:
                winner = "Tie"
            elif delta > 0:
                winner = "Big"
            else:
                winner = "Small"

        rows.append([fn, sm_str, bg_str, delta_str, winner])

    caption = (
        r"Per-function AUPRC comparison between small and big architectures."
    )
    return _latex_table(
        [
            "Function",
            "Small Mean AUPRC",
            "Big Mean AUPRC",
            r"$\Delta$ (Big $-$ Small)",
            "Winner",
        ],
        rows,
        caption,
        "tab:architecture_comparison",
    )


# ---------------------------------------------------------------------------
# Inline stats
# ---------------------------------------------------------------------------


def compute_inline_stats(
    df: pd.DataFrame, total_before: int, total_after: int
) -> dict[str, str]:
    s: dict[str, str] = {}

    # Overall mean (two-stage: within function, then across)
    func_means = df.groupby("function_name")[AUPRC_COLUMN].mean()
    overall_mean = float(func_means.mean())
    s["statOverallMeanAUPRC"] = f"{overall_mean:.3f}"

    # By regularization
    reg = _two_stage_mean(df, ["regularization_state"]).set_index("regularization_state")
    base_mean = float(reg.loc["base", "mean"]) if "base" in reg.index else float("nan")

    def _reg_mean(state: str) -> float:
        return float(reg.loc[state, "mean"]) if state in reg.index else float("nan")

    s["statBaselineMeanAUPRC"] = f"{_reg_mean('base'):.3f}"
    s["statDropoutMeanAUPRC"] = f"{_reg_mean('dropout'):.3f}"
    s["statWeightDecayMeanAUPRC"] = f"{_reg_mean('weight_decay'):.3f}"
    s["statDropoutWDMeanAUPRC"] = f"{_reg_mean('dropout+weight_decay'):.3f}"

    def _drop_from_base(state: str) -> str:
        m = _reg_mean(state)
        if np.isnan(m) or np.isnan(base_mean) or base_mean == 0:
            return "--"
        pct = (base_mean - m) / base_mean * 100
        return rf"{pct:.1f}\%"

    s["statDropoutDrop"] = _drop_from_base("dropout")
    s["statDropoutWDDrop"] = _drop_from_base("dropout+weight_decay")

    # By noise
    noise_means = _noise_series(df)
    noise0 = float(noise_means[0.0]) if not np.isnan(noise_means[0.0]) else float("nan")
    noise1 = float(noise_means[1.0]) if not np.isnan(noise_means[1.0]) else float("nan")
    s["statNoiseZeroMean"] = f"{noise0:.3f}" if not np.isnan(noise0) else "--"
    s["statNoiseOneMean"] = f"{noise1:.3f}" if not np.isnan(noise1) else "--"
    if not np.isnan(noise0) and not np.isnan(noise1) and noise0 != 0:
        s["statNoiseDrop"] = rf"{(noise0 - noise1) / noise0 * 100:.1f}\%"
    else:
        s["statNoiseDrop"] = "--"

    # By optimizer
    if "optimizer" in df.columns:
        opt = _two_stage_mean(df, ["optimizer"]).set_index("optimizer")
        s["statAdamMean"] = (
            f"{float(opt.loc['adam', 'mean']):.3f}" if "adam" in opt.index else "--"
        )
        s["statSGDMean"] = (
            f"{float(opt.loc['sgd', 'mean']):.3f}" if "sgd" in opt.index else "--"
        )
    else:
        s["statAdamMean"] = "--"
        s["statSGDMean"] = "--"

    # Run counts
    s["statTotalRuns"] = str(total_before)
    s["statRetainedRuns"] = str(total_after)
    retained_pct = total_after / total_before * 100 if total_before > 0 else 0.0
    s["statRetainedPct"] = rf"{retained_pct:.1f}\%"

    # Spearman ρ between AUPRC and val_loss
    valid = df[[AUPRC_COLUMN, "val_loss"]].dropna()
    if len(valid) >= 3:
        rho, pval = spearmanr(valid[AUPRC_COLUMN], valid["val_loss"])
        s["statSpearmanRho"] = f"{rho:.2f}"
        s["statSpearmanP"] = r"$p < 0.001$" if pval < 0.001 else f"{pval:.3f}"
    else:
        s["statSpearmanRho"] = "--"
        s["statSpearmanP"] = "--"

    # Random baseline = mean(num_gt) / 1013
    if "num_gt" in df.columns and df["num_gt"].notna().any():
        s["statRandomBaseline"] = f"{float(df['num_gt'].mean()) / 1013:.3f}"
    else:
        s["statRandomBaseline"] = "--"

    return s


def make_inline_stats_tex(stats: dict[str, str]) -> str:
    lines = [
        r"% Auto-generated inline statistics — regenerate with stats_analysis.py",
        r"% Do not edit manually.",
        "",
    ]
    for key, val in stats.items():
        lines.append(rf"\newcommand{{\{key}}}{{{val}}}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute statistical tables and inline stats for thesis."
    )
    parser.add_argument("--results-root", default="results", type=Path)
    parser.add_argument(
        "--trajectories-dir", default="results/trajectories", type=Path
    )
    args = parser.parse_args()

    results_root: Path = args.results_root
    trajectories_dir: Path = args.trajectories_dir
    output_dir = results_root / "stats"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df, total_before, total_after = load_data(results_root, trajectories_dir)
    print(
        f"  {total_before} runs loaded; {total_after} retained after convergence filter."
    )

    df_valid = df[df[AUPRC_COLUMN].notna()].copy()
    print(f"  {len(df_valid)} runs have non-null {AUPRC_COLUMN}.")

    tables = {
        "table_auprc_by_function.tex": make_table_auprc_by_function(df_valid),
        "table_auprc_by_regularization.tex": make_table_auprc_by_regularization(
            df_valid
        ),
        "table_auprc_by_noise.tex": make_table_auprc_by_noise(df_valid),
        "table_auprc_by_optimizer.tex": make_table_auprc_by_optimizer(df_valid),
        "table_architecture_comparison.tex": make_table_architecture_comparison(
            df_valid
        ),
    }

    print("\nWriting tables...")
    for filename, content in tables.items():
        path = output_dir / filename
        path.write_text(content)
        print(f"  {path}")

    print("\nComputing inline stats...")
    stats = compute_inline_stats(df_valid, total_before, total_after)
    inline_path = output_dir / "inline_stats.tex"
    inline_path.write_text(make_inline_stats_tex(stats))
    print(f"  {inline_path}")

    print("\n--- Summary ---")
    for key, val in stats.items():
        print(f"  \\{key:<30} {val}")


if __name__ == "__main__":
    main()
