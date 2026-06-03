#!/usr/bin/env python3
"""Generate thesis-ready summary plots from best-epoch analysis CSVs.

This script reads the existing `results/big/` and `results/small/` analysis
outputs, aggregates AUPRC values across optimizers, and creates:
- noise effect plots
- regularisation effect plots
- architecture comparison plots (big vs small)
- AUROC vs validation loss scatter plot
- per-function 5x4 heatmaps

The inputs are expected to already come from `best_epoch_*.pt` analyses.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from synth import functions as synth_functions


# Primary metric: full-space AUPRC over all 1013 candidates in 2^[10].
# See compute_auroc_data in analysis.py.
AUPRC_COLUMN = "auprc_full"

NOISE_ORDER = [0.0, 0.1, 0.2, 0.5, 1.0]
REGULARIZATION_ORDER = [
    "base",
    "dropout",
    "weight_decay",
    "dropout+weight_decay",
]
MODEL_SIZE_ORDER = ["small", "big"]
COMPARISON_FUNCTIONS = [f"f{i}" for i in range(1, 11)]

_CONVERGENCE_STATS: pd.DataFrame = pd.DataFrame()

# Functions used in the regularisation-strength probe sweep.
PROBE_FUNCTIONS = ["f1", "f7", "f8"]


def _filter_convergent_runs(df: pd.DataFrame, trajectories_dir: Path, min_reduction: float = 0.03) -> pd.DataFrame:
    trajectory_files = sorted(trajectories_dir.glob("*_trajectory.csv")) if trajectories_dir.exists() else []
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
    for (function_name, experiment), group in traj_df.groupby(["function_name", "experiment"]):
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
    print(f"  [convergence filter] removed {n_removed} runs ({n_removed/len(df)*100:.1f}%) based on trajectory val_loss")
    return df[keep].copy()


def _regularization_state(dropout: float, weight_decay: bool) -> str:
    if dropout > 0.0 and weight_decay:
        return "dropout+weight_decay"
    if dropout > 0.0:
        return "dropout"
    if weight_decay:
        return "weight_decay"
    return "base"


def _pretty_regularization_label(state: str) -> str:
    mapping = {
        "base": "Base",
        "dropout": "Dropout",
        "weight_decay": "Weight decay",
        "dropout+weight_decay": "Dropout + weight decay",
    }
    return mapping.get(state, state)


def _pretty_noise_label(noise: float) -> str:
    return f"{noise:g}"


def _get_interaction_order(function_name: str) -> int:
    """
    Get the maximum interaction order for a given function.
    
    The interaction order is the size of the largest interaction in the function's
    ground truth interaction set.
    
    Args:
        function_name: Name like "f1", "f2", etc.
    
    Returns:
        int: Maximum interaction order (number of features in largest interaction)
    """
    # Find the function object by name
    func = None
    for f in synth_functions:
        if f.__name__ == function_name:
            func = f
            break
    
    if func is None:
        # Fallback if function not found
        return 2
    
    # Call function to get ground truth interactions
    try:
        _, _, gt_interactions = func(num_samples=1000, seed=42)
        if not gt_interactions:
            return 2
        # Find maximum size of interactions
        max_order = max(len(interaction) for interaction in gt_interactions)
        return max_order
    except Exception:
        # Fallback on any error
        return 2


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


def compute_convergence_stats(df: pd.DataFrame, trajectories_dir: Path, min_reduction: float = 0.03) -> pd.DataFrame:
    """
    Given the raw (unfiltered) best-epoch dataframe, return a summary of which
    experiments were flagged as non-convergent based on trajectory val_loss.
    Returns a DataFrame with columns:
      noise, regularization_state, model_size, total, removed, pct_removed
    """
    _empty = pd.DataFrame(columns=["noise", "regularization_state", "model_size", "total", "removed", "pct_removed"])
    if "regularization_state" not in df.columns:
        return _empty

    trajectory_files = sorted(trajectories_dir.glob("*_trajectory.csv")) if trajectories_dir.exists() else []
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
        return _empty

    traj_df = pd.concat(frames, ignore_index=True)
    traj_df["epoch"] = pd.to_numeric(traj_df["epoch"], errors="coerce")
    traj_df["val_loss"] = pd.to_numeric(traj_df["val_loss"], errors="coerce")
    traj_df = traj_df.dropna(subset=["epoch", "val_loss"])

    convergent_keys: set[tuple] = set()
    for (function_name, experiment), group in traj_df.groupby(["function_name", "experiment"]):
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
    tmp = df[["noise", "regularization_state", "model_size"]].copy()
    tmp["_keep"] = keep
    summary = (
        tmp.groupby(["noise", "regularization_state", "model_size"])
        .agg(total=("_keep", "count"), removed=("_keep", lambda x: (~x).sum()))
        .reset_index()
    )
    summary["pct_removed"] = summary["removed"] / summary["total"] * 100
    return summary.sort_values("pct_removed", ascending=False).reset_index(drop=True)


def load_best_epoch_results(results_root: Path, trajectories_dir: Path) -> pd.DataFrame:
    """Load best-epoch analysis CSVs from results/big and results/small.

    Prefer the structured subdirectories when available. Fall back to top-level
    `small_*.csv` files if the folder is empty.
    """
    frames: list[pd.DataFrame] = []

    frames.extend(_read_analysis_csvs(results_root / "small", "small"))
    frames.extend(_read_analysis_csvs(results_root / "big", "big"))

    if not frames:
        # Fallback for the older top-level small_* outputs.
        for path in sorted(results_root.glob("small_*_analysis.csv")):
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            frame = frame.copy()
            frame["model_size"] = "small"
            frame["source_file"] = path.name
            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df[df["success"].fillna(False)].copy()
    df = df[df["auroc"].notna()].copy()

    if "auroc_full" not in df.columns:
        df["auroc_full"] = None

    if "auprc_full" not in df.columns:
        df["auprc_full"] = None
    if df["auprc_full"].notna().sum() == 0:
        print(
            "WARNING: auprc_full has no non-null values — rerun analysis.py to generate it"
        )

    df["noise"] = df["noise"].astype(float)
    df["dropout"] = df["dropout"].astype(float)
    df["weight_decay"] = df["weight_decay"].astype(bool)
    df["regularization_state"] = [
        _regularization_state(dropout, weight_decay)
        for dropout, weight_decay in zip(df["dropout"], df["weight_decay"])
    ]
    df["regularization_label"] = df["regularization_state"].map(_pretty_regularization_label)
    df["noise_label"] = df["noise"].map(_pretty_noise_label)
    df["is_best_epoch_source"] = True
    global _CONVERGENCE_STATS
    _CONVERGENCE_STATS = compute_convergence_stats(df, trajectories_dir)
    df = _filter_convergent_runs(df, trajectories_dir)
    return df


def _function_level_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Aggregate across optimizers first, then average across functions.

    This prevents one function or optimizer from dominating the thesis plots.
    """
    if df.empty:
        return pd.DataFrame()

    # First aggregate within each function.
    function_level = (
        df.groupby(["function_name", *group_cols], as_index=False)[AUPRC_COLUMN]
        .mean()
        .rename(columns={AUPRC_COLUMN: "function_mean_auroc"})
    )

    # Then average those function means across functions.
    summary = (
        function_level.groupby(group_cols, as_index=False)["function_mean_auroc"]
        .agg(mean="mean", std="std", count="count")
    )
    summary["std"] = summary["std"].fillna(0.0)
    summary["lower"] = summary["mean"] - summary["std"]
    summary["upper"] = summary["mean"] + summary["std"]
    return summary


def _global_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    summary = df.groupby(group_cols, as_index=False)[AUPRC_COLUMN].agg(mean="mean", std="std", count="count")
    summary["std"] = summary["std"].fillna(0.0)
    summary["lower"] = summary["mean"] - summary["std"]
    summary["upper"] = summary["mean"] + summary["std"]
    return summary


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _save_pgf(fig, output_path: Path) -> None:
    pgf_path = output_path.with_suffix(".pgf")
    png_path = output_path.with_suffix(".png")
    _ensure_parent(pgf_path)
    fig.savefig(pgf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02)


def _setup_thesis_style() -> None:
    # Prefer SciencePlots styles for academic-quality figures if available.
    try:
        import scienceplots  # noqa: F401

        # common science plots styles: 'science', with variants like 'ieee'
        plt.style.use(["science", "ieee"])
    except Exception:
        # Fallback to seaborn whitegrid when SciencePlots isn't available
        warnings.warn("SciencePlots not available; falling back to seaborn-v0_8-whitegrid")
        plt.style.use("seaborn-v0_8-whitegrid")

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 250,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
            "text.usetex": False,
            "pgf.rcfonts": False,
            "pgf.texsystem": "pdflatex",
        }
    )


def _plot_grouped_bars(
    summary: pd.DataFrame,
    x_col: str,
    title: str,
    output_path: Path,
    group_col: str = "model_size",
    x_order: Iterable | None = None,
    ylabel: str = "Average AUPRC",
    auprc_random_baseline: float = 0.0,
) -> None:
    if summary.empty:
        return

    _ensure_parent(output_path)

    if x_order is None:
        x_values = list(summary[x_col].dropna().unique())
    else:
        x_values = list(x_order)

    group_values = [g for g in MODEL_SIZE_ORDER if g in summary[group_col].unique()]
    if not group_values:
        group_values = sorted(summary[group_col].dropna().unique())

    x_positions = np.arange(len(x_values))
    bar_width = 0.35 if len(group_values) == 2 else 0.8 / max(len(group_values), 1)

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    colors = {"small": "#2A6F97", "big": "#E76F51"}

    for index, group_value in enumerate(group_values):
        group_df = summary[summary[group_col] == group_value].copy()
        means = []
        errors = []
        for value in x_values:
            match = group_df[group_df[x_col] == value]
            if match.empty:
                means.append(np.nan)
                errors.append(np.nan)
            else:
                means.append(float(match.iloc[0]["mean"]))
                errors.append(float(match.iloc[0]["std"]))

        offset = (index - (len(group_values) - 1) / 2) * bar_width
        positions = x_positions + offset
        ax.bar(
            positions,
            means,
            width=bar_width * 0.92,
            yerr=errors,
            capsize=4,
            color=colors.get(group_value, None),
            label=group_value,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.8,
        )

    max_val = summary["mean"].max() if not summary.empty else 0.0
    ax.set_title(title)
    ax.title.set_fontsize(10)
    ax.set_ylabel("Average AUPRC (±1 SD across functions)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([
        _pretty_regularization_label(str(v)) if x_col == "regularization_state"
        else _pretty_noise_label(v) if x_col == "noise"
        else str(v)
        for v in x_values
    ], rotation=15, ha="right")
    ax.set_ylim(0, 1)
    #ax.set_ylim(top=max_val * 1.15)
    if auprc_random_baseline >= 0.25:
        ax.axhline(auprc_random_baseline, color="gray", linestyle=":", linewidth=0.8, label="random baseline")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.tight_layout()
    _save_pgf(fig, output_path)
    plt.close(fig)


def _plot_architecture_x_regularization(results_df: pd.DataFrame, output_path: Path, auprc_random_baseline: float = 0.0) -> None:
    if results_df.empty:
        return

    # Stage 1: mean AUPRC within each (function_name, model_size, regularization_state)
    function_level = (
        results_df.groupby(["function_name", "model_size", "regularization_state"], as_index=False)[AUPRC_COLUMN]
        .mean()
        .rename(columns={AUPRC_COLUMN: "function_mean_auroc"})
    )

    # Stage 2: average across functions, grouped by (model_size, regularization_state)
    summary = (
        function_level.groupby(["model_size", "regularization_state"], as_index=False)["function_mean_auroc"]
        .agg(mean="mean", std="std")
    )
    summary["std"] = summary["std"].fillna(0.0)

    if summary.empty:
        return

    _ensure_parent(output_path)

    x_values = REGULARIZATION_ORDER
    group_values = [g for g in MODEL_SIZE_ORDER if g in summary["model_size"].unique()]
    x_positions = np.arange(len(x_values))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    colors = {"small": "#2A6F97", "big": "#E76F51"}

    for index, model_size in enumerate(group_values):
        group_df = summary[summary["model_size"] == model_size].copy()
        means = []
        errors = []
        for reg_state in x_values:
            match = group_df[group_df["regularization_state"] == reg_state]
            if match.empty:
                means.append(np.nan)
                errors.append(np.nan)
            else:
                means.append(float(match.iloc[0]["mean"]))
                errors.append(float(match.iloc[0]["std"]))

        offset = (index - (len(group_values) - 1) / 2) * bar_width
        positions = x_positions + offset
        ax.bar(
            positions,
            means,
            width=bar_width * 0.92,
            yerr=errors,
            capsize=4,
            color=colors.get(model_size, None),
            label=model_size,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.8,
        )

    max_val = summary["mean"].max() if not summary.empty else 0.0
    ax.set_title("Regularisation Effect by Architecture")
    ax.title.set_fontsize(10)
    ax.set_ylabel("Average AUPRC (±1 SD across functions)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [_pretty_regularization_label(s) for s in x_values],
        rotation=15,
        ha="right",
    )
    ax.set_ylim(0, 1)
   #ax.set_ylim(top=max_val * 1.15)
    if auprc_random_baseline >= 0.25:
        ax.axhline(auprc_random_baseline, color="gray", linestyle=":", linewidth=0.8, label="random baseline")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.tight_layout()
    _save_pgf(fig, output_path)
    plt.close(fig)


# Uses restricted AUROC (G∪D) — see compute_auroc_data in analysis.py
def _plot_dumbbell_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return

    _ensure_parent(output_path)

    fig, ax = plt.subplots(figsize=(7.0, 3.5))

    y_positions = np.arange(len(summary))[::-1]
    small = summary["small_mean"].to_numpy(dtype=float)
    big = summary["big_mean"].to_numpy(dtype=float)
    delta = big - small

    for y, small_value, big_value, diff in zip(y_positions, small, big, delta):
        line_color = "#2A9D8F" if diff >= 0 else "#E76F51"
        ax.plot([small_value, big_value], [y, y], color=line_color, linewidth=2.2, alpha=0.85)
        ax.scatter(small_value, y, color="#2A6F97", s=70, zorder=3, label="small" if y == y_positions[0] else None)
        ax.scatter(big_value, y, color="#E76F51", s=70, zorder=3, label="big" if y == y_positions[0] else None)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(summary["function_name"].tolist())
    ax.set_xlabel("Mean AUPRC")
    ax.set_title("Big vs Small Model Comparison by Function")
    max_x = max(np.nanmax(small), np.nanmax(big)) if len(small) > 0 and len(big) > 0 else 0.2
    ax.set_xlim(0.0, max_x * 1.15)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _save_pgf(fig, output_path)
    plt.close(fig)


def _plot_auprc_vs_val_loss_scatter(results_df: pd.DataFrame, output_path: Path, auprc_random_baseline: float = 0.0) -> None:
    if results_df.empty or "val_loss" not in results_df.columns:
        return

    plot_df = results_df.dropna(subset=[AUPRC_COLUMN, "val_loss"]).copy()
    if plot_df.empty:
        return

    # Aggregate to one point per (function_name, model_size) pair for the scatter
    agg_df = (
        plot_df.groupby(["function_name", "model_size"], as_index=False)[[AUPRC_COLUMN, "val_loss"]]
        .mean()
    )
    if agg_df.empty:
        return

    _ensure_parent(output_path)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    colors  = {"small": "#2A6F97", "big": "#E76F51"}
    markers = {"small": "o",       "big": "^"}

    for model_size in MODEL_SIZE_ORDER:
        subset = agg_df[agg_df["model_size"] == model_size]
        if subset.empty:
            continue
        ax.scatter(
            subset["val_loss"],
            subset[AUPRC_COLUMN],
            s=16,
            alpha=0.75,
            color=colors[model_size],
            marker=markers[model_size],
            label=model_size,
            edgecolors="none",
        )

    # Correlation and trend line use raw per-run rows, not aggregated points
    x_all = plot_df["val_loss"].to_numpy(dtype=float)
    y_all = plot_df[AUPRC_COLUMN].to_numpy(dtype=float)
    valid_mask = x_all > 0
    x = x_all[valid_mask]
    y = y_all[valid_mask]

    if len(x) >= 2 and np.std(np.log10(x)) > 0:
        rho, pval = spearmanr(x, y)

        # Fit on log10 space so trend line appears straight on log x-axis
        x_log = np.log10(x)
        slope, intercept = np.polyfit(x_log, y, 1)
        # Evaluate on logspace
        x_line = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
        y_line = slope * np.log10(x_line) + intercept
        ax.plot(x_line, y_line, color="black", linewidth=1.0, linestyle="--", label="linear trend")

        p_str = f"{pval:.3f}" if pval >= 0.001 else "$p < 0.001$"
        ax.set_title(rf"AUPRC vs Validation Loss ($\rho$={rho:.2f}, {p_str})")
    else:
        ax.set_title("AUPRC vs Validation Loss")

    ax.title.set_fontsize(9)
    ax.set_xscale("log")
    ax.set_xlabel("Validation loss (log scale)")
    ax.set_ylabel("AUPRC")
    ax.axhline(auprc_random_baseline, color="gray", linestyle=":", linewidth=0.8, label="random baseline")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    _save_pgf(fig, output_path)
    plt.close(fig)


def _plot_auprc_by_interaction_order(results_df: pd.DataFrame, output_path: Path, auprc_random_baseline: float = 0.0) -> None:
    """
    Plot AUROC grouped by interaction order and regularization state.
    
    Derives interaction order for each function from ground truth, then aggregates
    using the two-stage summary pattern (within function, then across functions).
    """
    if results_df.empty or "function_name" not in results_df.columns:
        return
    
    plot_df = results_df.copy()
    
    # Add interaction_order column by mapping each function to its max interaction order
    plot_df["interaction_order"] = plot_df["function_name"].map(
        lambda fn: _get_interaction_order(fn)
    )
    
    # Filter out any rows where function_name is NaN or interaction_order is NaN
    plot_df = plot_df.dropna(subset=["function_name", "interaction_order"])
    
    if plot_df.empty:
        return
    
    # Aggregate using two-stage pattern: within function, then across functions
    interaction_summary = _function_level_summary(
        plot_df, ["model_size", "interaction_order", "regularization_state"]
    )
    
    if interaction_summary.empty:
        return
    
    # Now aggregate across model_size to get interaction_order × regularization_state
    final_summary = (
        interaction_summary.groupby(["interaction_order", "regularization_state"], as_index=False)
        .agg({"mean": "mean", "std": "mean", "count": "sum"})
        .reset_index(drop=True)
    )
    
    _ensure_parent(output_path)
    
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    
    # Get unique interaction orders in sorted order
    interaction_orders = sorted(final_summary["interaction_order"].dropna().unique())
    reg_states = sorted(final_summary["regularization_state"].dropna().unique())
    
    if not interaction_orders or not reg_states:
        plt.close(fig)
        return
    
    x_positions = np.arange(len(interaction_orders))
    bar_width = 0.35 if len(reg_states) == 2 else 0.8 / max(len(reg_states), 1)
    
    colors = {
        "base": "#1f77b4",
        "dropout": "#ff7f0e",
        "weight_decay": "#2ca02c",
        "dropout+weight_decay": "#d62728",
    }
    
    for index, reg_state in enumerate(reg_states):
        state_df = final_summary[final_summary["regularization_state"] == reg_state].copy()
        means = []
        errors = []
        for order in interaction_orders:
            match = state_df[state_df["interaction_order"] == order]
            if match.empty:
                means.append(np.nan)
                errors.append(np.nan)
            else:
                means.append(float(match.iloc[0]["mean"]))
                errors.append(float(match.iloc[0]["std"]))
        
        offset = (index - (len(reg_states) - 1) / 2) * bar_width
        positions = x_positions + offset
        ax.bar(
            positions,
            means,
            width=bar_width * 0.92,
            yerr=errors,
            capsize=4,
            color=colors.get(reg_state, None),
            label=_pretty_regularization_label(reg_state),
            alpha=0.9,
            edgecolor="white",
            linewidth=0.8,
        )
    
    max_val = final_summary["mean"].max() if not final_summary.empty else 0.0
    ax.set_title("Effect of Interaction Order on Average AUPRC")
    ax.title.set_fontsize(10)
    ax.set_ylabel("Average AUPRC (±1 SD across functions)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{int(order)}" for order in interaction_orders], rotation=0)
    ax.set_xlabel("Maximum Interaction Order")
    ax.set_ylim(0.0, None)
    ax.set_ylim(top=max_val * 1.15)
    ax.axhline(auprc_random_baseline, color="gray", linestyle=":", linewidth=0.8, label="random baseline")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.tight_layout()
    _save_pgf(fig, output_path)
    plt.close(fig)


def _plot_interaction_recovery_trajectories(trajectories_dir: Path, output_path: Path) -> None:
    """Plot AUPRC recovery trajectories across epochs.

    Uses full-space AUPRC over all 1013 candidates in $2^{[10]}$, not restricted G∪D.
    """
    if not trajectories_dir.exists() or not trajectories_dir.is_dir():
        return

    # Collect trajectory files from flat directory layout
    trajectory_files_with_size: list[tuple[str, Path]] = []
    trajectory_files = sorted(trajectories_dir.glob("*_trajectory.csv"))
    for path in trajectory_files:
        model_size = path.name.split("_")[0]  # "small" or "big"
        trajectory_files_with_size.append((model_size, path))
    print(f"  [trajectories] {len(trajectory_files_with_size)} file(s)")
    if not trajectory_files_with_size:
        return

    frames: list[pd.DataFrame] = []
    for model_size, path in trajectory_files_with_size:
        # auprc column in trajectory CSVs is restricted (G∪D) — see analyze_trajectory in analysis.py
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        if "model_size" not in frame.columns:
            frame["model_size"] = model_size
        frames.append(frame)

    if not frames:
        return

    plot_df = pd.concat(frames, ignore_index=True)
    if plot_df.empty:
        return

    required_columns = {"epoch", "auprc_full", "function_name", "noise", "dropout", "weight_decay"}
    if not required_columns.issubset(plot_df.columns):
        return

    plot_df = plot_df.copy()
    plot_df["epoch"] = pd.to_numeric(plot_df["epoch"], errors="coerce")
    plot_df["auprc_full"] = pd.to_numeric(plot_df["auprc_full"], errors="coerce")
    plot_df["noise"] = pd.to_numeric(plot_df["noise"], errors="coerce")
    plot_df["dropout"] = pd.to_numeric(plot_df["dropout"], errors="coerce")
    plot_df["weight_decay"] = plot_df["weight_decay"].astype(bool)
    plot_df = plot_df.dropna(subset=["epoch", "auprc_full", "function_name", "noise", "dropout"])

    # Exclude trajectories where val_loss did not decrease by >= 3% from start to end epoch
    if "val_loss" in plot_df.columns:
        plot_df["val_loss"] = pd.to_numeric(plot_df["val_loss"], errors="coerce")
        convergent_keys: set[tuple] = set()
        for (fn, exp), group in plot_df.groupby(["function_name", "experiment"]):
            start_rows = group.loc[group["epoch"] == group["epoch"].min(), "val_loss"]
            end_rows = group.loc[group["epoch"] == group["epoch"].max(), "val_loss"]
            if start_rows.empty or end_rows.empty:
                continue
            start_loss = float(start_rows.iloc[0])
            end_loss = float(end_rows.iloc[0])
            if start_loss <= 0 or (start_loss - end_loss) / start_loss < 0.03:
                continue
            convergent_keys.add((fn, exp))
        plot_df = plot_df[
            plot_df.apply(lambda r: (r["function_name"], r["experiment"]) in convergent_keys, axis=1)
        ].copy()

    plot_df["regularization_state"] = [
        _regularization_state(dropout, weight_decay)
        for dropout, weight_decay in zip(plot_df["dropout"], plot_df["weight_decay"])
    ]
    plot_df = plot_df[plot_df["regularization_state"].isin(REGULARIZATION_ORDER)].copy()

    if plot_df.empty:
        return

    def _function_sort_key(name: str):
        if str(name).startswith("f") and str(name)[1:].isdigit():
            return int(str(name)[1:])
        return str(name)

    _ensure_parent(output_path)

    fig, axes = plt.subplots(
        len(NOISE_ORDER),
        len(REGULARIZATION_ORDER),
        figsize=(10, 7),
        sharex=True,
        sharey=True,
    )

    if len(NOISE_ORDER) == 1 and len(REGULARIZATION_ORDER) == 1:
        axes = np.array([[axes]])
    elif len(NOISE_ORDER) == 1:
        axes = np.array([axes])
    elif len(REGULARIZATION_ORDER) == 1:
        axes = np.array([[ax] for ax in axes])

    all_epochs = np.sort(plot_df["epoch"].dropna().unique())
    if all_epochs.size == 0:
        plt.close(fig)
        return

    for row_index, noise_value in enumerate(NOISE_ORDER):
        for col_index, reg_state in enumerate(REGULARIZATION_ORDER):
            ax = axes[row_index, col_index]
            cell_df = plot_df[
                (plot_df["noise"] == noise_value)
                & (plot_df["regularization_state"] == reg_state)
            ].copy()

            if not cell_df.empty:
                cell_df = cell_df[cell_df["epoch"] > 1].copy()
                function_epoch = (
                    cell_df.groupby(["function_name", "epoch"], as_index=False)["auprc_full"]
                    .mean()
                )
                if not function_epoch.empty:
                    function_names = sorted(function_epoch["function_name"].dropna().unique(), key=_function_sort_key)
                    pivot = (
                        function_epoch.pivot(index="epoch", columns="function_name", values="auprc_full")
                        .reindex(all_epochs)
                    )

                    for function_name in function_names:
                        if function_name not in pivot.columns:
                            continue
                        ax.plot(
                            pivot.index,
                            pivot[function_name],
                            color="0.6",
                            linewidth=0.6,
                            alpha=0.35,
                            zorder=1,
                        )

                    mean_series = pivot.mean(axis=1)
                    std_series = pivot.std(axis=1).fillna(0.0)
                    valid = mean_series.notna()
                    if valid.any():
                        x_values = mean_series.index.to_numpy(dtype=float)[valid.to_numpy()]
                        mean_values = mean_series.to_numpy(dtype=float)[valid.to_numpy()]
                        std_values = std_series.to_numpy(dtype=float)[valid.to_numpy()]
                        ax.fill_between(
                            x_values,
                            mean_values - std_values,
                            mean_values + std_values,
                            color="#2A6F97",
                            alpha=0.15,
                            linewidth=0,
                            zorder=2,
                        )
                        ax.plot(
                            x_values,
                            mean_values,
                            color="#2A6F97",
                            linewidth=2.2,
                            zorder=5,
                        )

            ax.set_ylim(0.0, 1.0)
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            if row_index == 0:
                ax.set_title(_pretty_regularization_label(reg_state), fontsize=10)
            if row_index < len(NOISE_ORDER) - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Epoch")
            if col_index > 0:
                ax.tick_params(labelleft=False)

            if col_index == len(REGULARIZATION_ORDER) - 1:
                ax.text(
                    1.03,
                    0.5,
                    f"noise = {_pretty_noise_label(noise_value)}",
                    transform=ax.transAxes,
                    ha="left",
                    va="center",
                    fontsize=10,
                )

    fig.supylabel(r"AUPRC ($2^{[10]}$)", fontsize=9)
    fig.tight_layout()
    _save_pgf(fig, output_path)
    plt.close(fig)


def _plot_trajectory_highlight(trajectories_dir: Path, output_path: Path) -> None:
    if not trajectories_dir.exists() or not trajectories_dir.is_dir():
        return

    trajectory_files_with_size: list[tuple[str, Path]] = []
    for path in sorted(trajectories_dir.glob("*_trajectory.csv")):
        model_size = path.name.split("_")[0]
        trajectory_files_with_size.append((model_size, path))
    if not trajectory_files_with_size:
        return

    frames: list[pd.DataFrame] = []
    for model_size, path in trajectory_files_with_size:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        if "model_size" not in frame.columns:
            frame["model_size"] = model_size
        frames.append(frame)

    if not frames:
        return

    plot_df = pd.concat(frames, ignore_index=True)
    if plot_df.empty:
        return

    required_columns = {"epoch", "auprc_full", "function_name", "noise", "dropout", "weight_decay"}
    if not required_columns.issubset(plot_df.columns):
        return

    plot_df = plot_df.copy()
    plot_df["epoch"] = pd.to_numeric(plot_df["epoch"], errors="coerce")
    plot_df["auprc_full"] = pd.to_numeric(plot_df["auprc_full"], errors="coerce")
    plot_df["noise"] = pd.to_numeric(plot_df["noise"], errors="coerce")
    plot_df["dropout"] = pd.to_numeric(plot_df["dropout"], errors="coerce")
    plot_df["weight_decay"] = plot_df["weight_decay"].astype(bool)
    plot_df = plot_df.dropna(subset=["epoch", "auprc_full", "function_name", "noise", "dropout"])

    if "val_loss" in plot_df.columns:
        plot_df["val_loss"] = pd.to_numeric(plot_df["val_loss"], errors="coerce")
        convergent_keys: set[tuple] = set()
        for (fn, exp), group in plot_df.groupby(["function_name", "experiment"]):
            start_rows = group.loc[group["epoch"] == group["epoch"].min(), "val_loss"]
            end_rows = group.loc[group["epoch"] == group["epoch"].max(), "val_loss"]
            if start_rows.empty or end_rows.empty:
                continue
            start_loss = float(start_rows.iloc[0])
            end_loss = float(end_rows.iloc[0])
            if start_loss <= 0 or (start_loss - end_loss) / start_loss < 0.03:
                continue
            convergent_keys.add((fn, exp))
        plot_df = plot_df[
            plot_df.apply(lambda r: (r["function_name"], r["experiment"]) in convergent_keys, axis=1)
        ].copy()

    plot_df["regularization_state"] = [
        _regularization_state(dropout, weight_decay)
        for dropout, weight_decay in zip(plot_df["dropout"], plot_df["weight_decay"])
    ]
    plot_df = plot_df[plot_df["regularization_state"].isin(REGULARIZATION_ORDER)].copy()

    if plot_df.empty:
        return

    def _function_sort_key(name: str):
        if str(name).startswith("f") and str(name)[1:].isdigit():
            return int(str(name)[1:])
        return str(name)

    _ensure_parent(output_path)

    highlight_cells = [
        (0, 0, 0.0, "base"),
        (0, 1, 0.0, "dropout+weight_decay"),
        (1, 0, 1.0, "base"),
        (1, 1, 1.0, "dropout+weight_decay"),
    ]
    col_titles = ["Base", "Dropout + weight decay"]
    row_labels_text = ["noise = 0", "noise = 1"]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(7.0, 5.0),
        sharex=True,
        sharey=True,
    )

    all_epochs = np.sort(plot_df["epoch"].dropna().unique())
    if all_epochs.size == 0:
        plt.close(fig)
        return

    for row_index, col_index, noise_value, reg_state in highlight_cells:
        ax = axes[row_index, col_index]
        cell_df = plot_df[
            (plot_df["noise"] == noise_value)
            & (plot_df["regularization_state"] == reg_state)
        ].copy()

        if not cell_df.empty:
            cell_df = cell_df[cell_df["epoch"] > 1].copy()
            function_epoch = (
                cell_df.groupby(["function_name", "epoch"], as_index=False)["auprc_full"]
                .mean()
            )
            if not function_epoch.empty:
                function_names = sorted(
                    function_epoch["function_name"].dropna().unique(),
                    key=_function_sort_key,
                )
                pivot = (
                    function_epoch.pivot(index="epoch", columns="function_name", values="auprc_full")
                    .reindex(all_epochs)
                )

                for function_name in function_names:
                    if function_name not in pivot.columns:
                        continue
                    ax.plot(
                        pivot.index,
                        pivot[function_name],
                        color="0.6",
                        linewidth=0.6,
                        alpha=0.35,
                        zorder=1,
                    )

                mean_series = pivot.mean(axis=1)
                std_series = pivot.std(axis=1).fillna(0.0)
                valid = mean_series.notna()
                if valid.any():
                    x_vals = mean_series.index.to_numpy(dtype=float)[valid.to_numpy()]
                    mean_vals = mean_series.to_numpy(dtype=float)[valid.to_numpy()]
                    std_vals = std_series.to_numpy(dtype=float)[valid.to_numpy()]
                    ax.fill_between(
                        x_vals,
                        mean_vals - std_vals,
                        mean_vals + std_vals,
                        color="#2A6F97",
                        alpha=0.15,
                        linewidth=0,
                        zorder=2,
                    )
                    ax.plot(
                        x_vals,
                        mean_vals,
                        color="#2A6F97",
                        linewidth=2.2,
                        zorder=5,
                    )

        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

        if row_index == 0:
            ax.set_title(col_titles[col_index], fontsize=10)

        if row_index < 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Epoch")

        if col_index > 0:
            ax.tick_params(labelleft=False)

        if col_index == 1:
            ax.text(
                1.03,
                0.5,
                row_labels_text[row_index],
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=10,
            )

    fig.supylabel(r"AUPRC ($2^{[10]}$)", fontsize=9)
    fig.tight_layout()
    _save_pgf(fig, output_path)
    plt.close(fig)


def compute_peak_auroc_epoch_summary(trajectories_dir: Path) -> pd.DataFrame:
    _EMPTY_COLS = ["function", "experiment", "noise", "regularization_state", "peak_auprc_epoch", "peak_auprc", "model_size"]
    if not trajectories_dir.exists() or not trajectories_dir.is_dir():
        return pd.DataFrame(columns=_EMPTY_COLS)

    # Collect trajectory files from flat directory layout
    trajectory_files_with_size: list[tuple[str, Path]] = []
    trajectory_files = sorted(trajectories_dir.glob("*_trajectory.csv"))
    for path in trajectory_files:
        model_size = path.name.split("_")[0]  # "small" or "big"
        trajectory_files_with_size.append((model_size, path))
    print(f"  [trajectories] {len(trajectory_files_with_size)} file(s)")
    if not trajectory_files_with_size:
        return pd.DataFrame(columns=_EMPTY_COLS)

    rows = []
    for model_size, path in trajectory_files_with_size:
        frame = pd.read_csv(path)
        if frame.empty or "auprc_full" not in frame.columns or "epoch" not in frame.columns:
            continue

        required_columns = {"function_name", "experiment", "noise", "dropout", "weight_decay"}
        if not required_columns.issubset(frame.columns):
            continue

        frame = frame.copy()
        frame["epoch"] = pd.to_numeric(frame["epoch"], errors="coerce")
        frame["auprc_full"] = pd.to_numeric(frame["auprc_full"], errors="coerce")
        frame["noise"] = pd.to_numeric(frame["noise"], errors="coerce")
        frame["dropout"] = pd.to_numeric(frame["dropout"], errors="coerce")
        if frame["weight_decay"].dtype == bool:
            weight_decay = frame["weight_decay"]
        else:
            weight_decay = frame["weight_decay"].astype(str).str.lower().isin(["true", "1", "yes"])
        frame["weight_decay"] = weight_decay
        frame = frame.dropna(subset=["epoch", "auprc_full", "function_name", "experiment", "noise", "dropout"])
        if frame.empty:
            continue

        if "val_loss" not in frame.columns:
            continue
        frame["val_loss"] = pd.to_numeric(frame["val_loss"], errors="coerce")
        start_rows = frame.loc[frame["epoch"] == frame["epoch"].min(), "val_loss"].dropna()
        end_rows = frame.loc[frame["epoch"] == frame["epoch"].max(), "val_loss"].dropna()
        if start_rows.empty or end_rows.empty:
            continue
        start_loss = float(start_rows.iloc[0])
        end_loss = float(end_rows.iloc[0])
        if start_loss <= 0 or (start_loss - end_loss) / start_loss < 0.03:
            continue

        peak_index = frame["auprc_full"].idxmax()
        peak_row = frame.loc[peak_index]
        regularization_state = _regularization_state(float(peak_row["dropout"]), bool(peak_row["weight_decay"]))
        if regularization_state not in REGULARIZATION_ORDER:
            continue

        # Use model_size from the CSV when present; otherwise fall back to the
        # subdir name (e.g. "small" or "big") that was recorded when the file was collected above.
        row_model_size = peak_row["model_size"] if "model_size" in frame.columns else model_size
        rows.append(
            {
                "function": peak_row["function_name"],
                "experiment": peak_row["experiment"],
                "noise": float(peak_row["noise"]),
                "regularization_state": regularization_state,
                "peak_auprc_epoch": int(float(peak_row["epoch"])),
                "peak_auprc": float(peak_row["auprc_full"]),
                "model_size": row_model_size,
            }
        )

    if not rows:
        return pd.DataFrame(columns=_EMPTY_COLS)

    return pd.DataFrame(rows)


def _plot_peak_epoch_vs_noise(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        return

    summary_df = summary_df.copy()
    summary_df = summary_df[summary_df["regularization_state"].isin(REGULARIZATION_ORDER)]
    if summary_df.empty:
        return

    function_level = (
        summary_df.groupby(["function", "noise", "regularization_state"], as_index=False)["peak_auprc_epoch"]
        .mean()
        .rename(columns={"peak_auprc_epoch": "function_mean_peak_epoch"})
    )
    if function_level.empty:
        return

    stats = (
        function_level.groupby(["noise", "regularization_state"], as_index=False)["function_mean_peak_epoch"]
        .agg(mean="mean", std="std", count="count")
    )
    if stats.empty:
        return

    _ensure_parent(output_path)
    _setup_thesis_style()

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    style_map = {
        "base": {"color": "#2A6F97", "linestyle": "-"},
        "dropout": {"color": "#E76F51", "linestyle": "--"},
        "weight_decay": {"color": "#2A9D8F", "linestyle": "-.",},
        "dropout+weight_decay": {"color": "#7A5195", "linestyle": ":"},
    }

    for reg_state in REGULARIZATION_ORDER:
        state_stats = stats[stats["regularization_state"] == reg_state].copy()
        if state_stats.empty:
            continue
        state_stats = state_stats.set_index("noise").reindex(NOISE_ORDER)
        x_values = np.array(NOISE_ORDER, dtype=float)
        mean_values = state_stats["mean"].to_numpy(dtype=float)
        std_values = state_stats["std"].fillna(0.0).to_numpy(dtype=float)

        valid = ~np.isnan(mean_values)
        if not valid.any():
            continue

        style = style_map.get(reg_state, {"color": "0.4", "linestyle": "-"})
        ax.plot(
            x_values[valid],
            mean_values[valid],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.6,
            label=_pretty_regularization_label(reg_state),
        )
        ax.fill_between(
            x_values[valid],
            mean_values[valid] - std_values[valid],
            mean_values[valid] + std_values[valid],
            color=style["color"],
            alpha=0.15,
            linewidth=0,
        )

    ax.set_xlabel("Noise level")
    ax.set_ylabel("Mean epoch of peak AUPRC")
    ax.set_xticks(NOISE_ORDER)
    ax.set_xticklabels([_pretty_noise_label(noise) for noise in NOISE_ORDER])
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    _save_pgf(fig, output_path)
    plt.close(fig)


def generate_peak_epoch_table(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        return

    summary_df = summary_df.copy()
    summary_df = summary_df[summary_df["regularization_state"].isin(REGULARIZATION_ORDER)]
    if summary_df.empty:
        return

    function_level = (
        summary_df.groupby(["function", "noise", "regularization_state"], as_index=False)["peak_auprc_epoch"]
        .mean()
        .rename(columns={"peak_auprc_epoch": "function_mean_peak_epoch"})
    )
    if function_level.empty:
        return

    stats = (
        function_level.groupby(["noise", "regularization_state"], as_index=False)["function_mean_peak_epoch"]
        .agg(mean="mean", std="std", count="count")
    )
    if stats.empty:
        return

    mean_pivot = stats.pivot(index="noise", columns="regularization_state", values="mean").reindex(index=NOISE_ORDER, columns=REGULARIZATION_ORDER)
    std_pivot = stats.pivot(index="noise", columns="regularization_state", values="std").reindex(index=NOISE_ORDER, columns=REGULARIZATION_ORDER)

    _ensure_parent(output_path)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{tabular}{l" + "c" * len(REGULARIZATION_ORDER) + r"}",
        r"\toprule",
        "Noise level & " + " & ".join(_pretty_regularization_label(state) for state in REGULARIZATION_ORDER) + r" \\",
        r"\midrule",
    ]

    for noise in NOISE_ORDER:
        row_cells = [_pretty_noise_label(noise)]
        for reg_state in REGULARIZATION_ORDER:
            mean_value = mean_pivot.loc[noise, reg_state] if noise in mean_pivot.index and reg_state in mean_pivot.columns else np.nan
            std_value = std_pivot.loc[noise, reg_state] if noise in std_pivot.index and reg_state in std_pivot.columns else np.nan
            if pd.isna(mean_value):
                cell = r"--"
            else:
                std_value = 0.0 if pd.isna(std_value) else float(std_value)
                cell = f"{float(mean_value):.1f} $\\pm$ {std_value:.1f}"
            row_cells.append(cell)
        lines.append(" & ".join(row_cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Mean epoch of peak AUPRC across functions by noise level and regularization state.}",
        r"\end{table}",
        "",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _heatmap_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(index=NOISE_ORDER, columns=REGULARIZATION_ORDER, dtype=float)

    pivot = (
        df.groupby(["noise", "regularization_state"], as_index=False)[AUPRC_COLUMN]
        .mean()
        .pivot(index="noise", columns="regularization_state", values=AUPRC_COLUMN)
    )
    pivot = pivot.reindex(index=NOISE_ORDER, columns=REGULARIZATION_ORDER)
    return pivot


def _plot_heatmap(ax, matrix: pd.DataFrame, title: str):
    data = matrix.to_numpy(dtype=float)
    valid = data[~np.isnan(data)]
    vmin = float(np.min(valid)) if len(valid) > 0 else 0.0
    vmax = float(np.max(valid)) if len(valid) > 0 else 1.0
    im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap="cividis", aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels([_pretty_regularization_label(col) for col in matrix.columns], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels([f"{noise:g}" for noise in matrix.index])
    ax.set_xlabel("Regularisation")
    ax.set_ylabel("Noise")
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = data[i, j]
            if np.isnan(value):
                label = "N/A"
                color = "white"
            else:
                label = f"{value:.2f}"
                # at vmax=0.2, value < 0.10 is the crossover for cividis luminance
                color = "white" if value < 0.10 else "black"
            ax.text(j, i, label, ha="center", va="center", fontsize=9, color=color)

    return im


def _plot_loss_vs_auprc_divergence(epoch_results_path: Path, output_path: Path, model_size: str = "small") -> None:
    if not epoch_results_path.exists():
        warnings.warn(
            f"Epoch results file not found: {epoch_results_path}. "
            "Run `python plot_analysis.py --snapshot-root snapshots_clean` first to generate it."
        )
        return

    df = pd.read_csv(epoch_results_path)
    if df.empty:
        warnings.warn(f"Epoch results file is empty: {epoch_results_path}")
        return

    required_cols = {"epoch", "auprc_full", "val_loss", "function_name", "noise", "dropout", "weight_decay", "model_size"}
    if not required_cols.issubset(df.columns):
        warnings.warn(f"Missing columns in epoch results: {required_cols - set(df.columns)}")
        return

    df = df.copy()
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df["auprc_full"] = pd.to_numeric(df["auprc_full"], errors="coerce")
    df["val_loss"] = pd.to_numeric(df["val_loss"], errors="coerce")
    df["noise"] = pd.to_numeric(df["noise"], errors="coerce")
    df["dropout"] = pd.to_numeric(df["dropout"], errors="coerce")
    df["weight_decay"] = df["weight_decay"].astype(str).str.lower().isin(["true", "1", "yes"])
    df["regularization_state"] = [
        _regularization_state(d, w) for d, w in zip(df["dropout"], df["weight_decay"])
    ]

    df = df[df["regularization_state"] == "base"].copy()
    df = df[df["model_size"] == model_size].copy()
    df = df[df["noise"].isin([0.0, 1.0])].copy()
    df = df.dropna(subset=["epoch", "auprc_full", "val_loss", "function_name"])

    if df.empty:
        warnings.warn("No data after filtering for noise=0/1 and base regularization.")
        return

    color_loss = "#E76F51"
    color_auroc = "#2A6F97"

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # Proxy artists for shared legend — empty so invisible in the plot itself
    loss_handle, = axes[0].plot([], [], color=color_loss, linewidth=1.6, label="Validation loss")
    auprc_handle, = axes[0].plot([], [], color=color_auroc, linewidth=1.6, label=r"AUPRC ($2^{[10]}$)")

    for ax, noise_val in zip(axes, [0.0, 1.0]):
        ax.set_title(f"Noise = {_pretty_noise_label(noise_val)}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation loss", color=color_loss)
        ax.tick_params(axis="y", labelcolor=color_loss)

        noise_df = df[df["noise"] == noise_val].copy()
        if noise_df.empty:
            continue

        # Stage 1: average within (function_name, epoch) across experiments/optimisers
        func_epoch = (
            noise_df.groupby(["function_name", "epoch"], as_index=False)[["auprc_full", "val_loss"]]
            .mean()
        )

        # Keep only epochs where at least 3 distinct functions have valid data
        func_count = func_epoch.groupby("epoch")["function_name"].nunique()
        valid_epochs = func_count[func_count >= 3].index
        func_epoch = func_epoch[func_epoch["epoch"].isin(valid_epochs)].copy()
        if func_epoch.empty:
            continue

        # Stage 2: aggregate across functions per epoch
        auprc_agg = func_epoch.groupby("epoch")["auprc_full"].agg(mean="mean", std="std")
        loss_agg = func_epoch.groupby("epoch")["val_loss"].agg(mean="mean", std="std")
        epoch_stats = pd.DataFrame({
            "auprc_mean": auprc_agg["mean"],
            "auprc_std": auprc_agg["std"].fillna(0.0),
            "loss_mean": loss_agg["mean"],
            "loss_std": loss_agg["std"].fillna(0.0),
        }).reset_index().sort_values("epoch")

        x = epoch_stats["epoch"].to_numpy(dtype=float)
        auprc_mean = epoch_stats["auprc_mean"].to_numpy(dtype=float)
        auprc_std = epoch_stats["auprc_std"].to_numpy(dtype=float)
        loss_mean = epoch_stats["loss_mean"].to_numpy(dtype=float)
        loss_std = epoch_stats["loss_std"].to_numpy(dtype=float)

        # Find first epoch where AUPRC reaches 95 % of its series maximum
        plateau_epoch = None
        valid_auroc = auprc_mean[~np.isnan(auprc_mean)]
        if len(valid_auroc) > 0:
            threshold = 0.95 * np.max(valid_auroc)
            plateau_mask = auprc_mean >= threshold
            if plateau_mask.any():
                plateau_epoch = x[plateau_mask][0]

        # Left axis: validation loss
        ax.plot(x, loss_mean, color=color_loss, linewidth=1.6)
        ax.fill_between(x, loss_mean - loss_std, loss_mean + loss_std,
                        color=color_loss, alpha=0.15, linewidth=0)

        # Right axis: AUPRC (twin)
        ax2 = ax.twinx()
        ax2.plot(x, auprc_mean, color=color_auroc, linewidth=1.6)
        ax2.fill_between(x, auprc_mean - auprc_std, auprc_mean + auprc_std,
                         color=color_auroc, alpha=0.15, linewidth=0)
        ax2.set_ylabel(r"AUPRC ($2^{[10]}$)", color=color_auroc)
        ax2.tick_params(axis="y", labelcolor=color_auroc)
        ax2.set_ylim(0.0, 1.0)

        if plateau_epoch is not None:
            ax.axvline(plateau_epoch, color="gray", linestyle="--", linewidth=1.0)
            ax.text(
                plateau_epoch,
                0.97,
                "AUPRC plateau",
                transform=ax.get_xaxis_transform(),
                ha="right",
                va="top",
                fontsize=7,
                color="gray",
                rotation=90,
            )

    fig.legend(
        handles=[loss_handle, auprc_handle],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.05),
        fontsize=9,
    )
    fig.tight_layout()
    _save_pgf(fig, output_path)
    plt.close(fig)


def _plot_summary_heatmap(results_df: pd.DataFrame, output_path: Path) -> None:
    if results_df.empty:
        return

    # Stage 1: mean within each (function_name, noise, regularization_state)
    function_level = (
        results_df.groupby(["function_name", "noise", "regularization_state"], as_index=False)[AUPRC_COLUMN]
        .mean()
        .rename(columns={AUPRC_COLUMN: "function_mean_auroc"})
    )

    # Stage 2: mean across functions, grouped by (noise, regularization_state)
    cross_function = (
        function_level.groupby(["noise", "regularization_state"], as_index=False)["function_mean_auroc"]
        .mean()
    )

    matrix = (
        cross_function.pivot(index="noise", columns="regularization_state", values="function_mean_auroc")
        .reindex(index=NOISE_ORDER, columns=REGULARIZATION_ORDER)
    )

    _ensure_parent(output_path)

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    im = _plot_heatmap(ax, matrix, "Mean AUPRC by Noise and Regularisation (all functions)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean AUPRC")
    fig.tight_layout()
    _save_pgf(fig, output_path)
    plt.close(fig)


def _plot_convergence_removal(stats_df: pd.DataFrame, output_path: Path) -> None:
    if stats_df.empty:
        return

    _ensure_parent(output_path)

    reg_colors = {
        "base": "#2A6F97",
        "dropout": "#E76F51",
        "weight_decay": "#2A9D8F",
        "dropout+weight_decay": "#7A5195",
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), sharey=True)

    bar_height = 0.18
    n_reg = len(REGULARIZATION_ORDER)

    for ax, model_size in zip(axes, MODEL_SIZE_ORDER):
        size_df = stats_df[stats_df["model_size"] == model_size].copy()
        y_positions = np.arange(len(NOISE_ORDER), dtype=float)

        for reg_idx, reg_state in enumerate(REGULARIZATION_ORDER):
            reg_df = size_df[size_df["regularization_state"] == reg_state]
            widths = []
            for noise_val in NOISE_ORDER:
                match = reg_df[reg_df["noise"] == noise_val]
                widths.append(float(match["pct_removed"].iloc[0]) if not match.empty else 0.0)

            offset = (reg_idx - (n_reg - 1) / 2) * bar_height
            ax.barh(
                y_positions + offset,
                widths,
                height=bar_height * 0.9,
                color=reg_colors.get(reg_state, "0.5"),
                label=_pretty_regularization_label(reg_state),
                alpha=0.9,
            )

        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([_pretty_noise_label(n) for n in NOISE_ORDER])
        ax.set_xlabel("Runs removed (%)")
        ax.set_title("Small model" if model_size == "small" else "Big model", fontsize=10)
        ax.set_xlim(left=0)

    axes[0].set_ylabel("Noise")
    axes[-1].legend(
        frameon=False,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        fontsize=8,
    )
    fig.tight_layout()
    _save_pgf(fig, output_path)
    plt.close(fig)


def generate_convergence_report(stats_df: pd.DataFrame, output_path: Path) -> None:
    if stats_df.empty:
        return

    _ensure_parent(output_path)

    size_order_map = {s: i for i, s in enumerate(MODEL_SIZE_ORDER)}
    noise_order_map = {n: i for i, n in enumerate(NOISE_ORDER)}
    reg_order_map = {r: i for i, r in enumerate(REGULARIZATION_ORDER)}

    sorted_df = stats_df.copy()
    sorted_df["_size_ord"] = sorted_df["model_size"].map(size_order_map).fillna(99)
    sorted_df["_noise_ord"] = sorted_df["noise"].map(noise_order_map).fillna(99)
    sorted_df["_reg_ord"] = sorted_df["regularization_state"].map(reg_order_map).fillna(99)
    sorted_df = sorted_df.sort_values(["_size_ord", "_noise_ord", "_reg_ord"]).reset_index(drop=True)

    lines = [
        r"\begin{longtable}{llllcc}",
        r"\caption{Runs removed by convergence filter (validation loss did not decrease by $\geq 5\%$).}",
        r"\label{tab:convergence_removed} \\",
        r"\toprule",
        r"Model & Noise & Regularisation & Total runs & Removed & Removed (\%) \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Model & Noise & Regularisation & Total runs & Removed & Removed (\%) \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]

    for _, row in sorted_df.iterrows():
        model = str(row["model_size"])
        noise = _pretty_noise_label(float(row["noise"]))
        reg = _pretty_regularization_label(str(row["regularization_state"]))
        total = int(row["total"])
        removed = int(row["removed"])
        pct = float(row["pct_removed"])

        if pct > 50:
            parts = [
                r"\textbf{" + model + "}",
                r"\textbf{" + noise + "}",
                r"\textbf{" + reg + "}",
                r"\textbf{" + str(total) + "}",
                r"\textbf{" + str(removed) + "}",
                r"\textbf{" + f"{pct:.1f}" + "}",
            ]
        else:
            parts = [model, noise, reg, str(total), str(removed), f"{pct:.1f}"]
        lines.append(" & ".join(parts) + r" \\")

    lines.extend([r"\end{longtable}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _plot_convergence_removal_dots(trajectories_dir: Path, output_path: Path) -> None:
    trajectory_files = sorted(trajectories_dir.glob("*_trajectory.csv")) if trajectories_dir.exists() else []
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
        return

    traj_df = pd.concat(frames, ignore_index=True)
    traj_df["epoch"] = pd.to_numeric(traj_df["epoch"], errors="coerce")
    traj_df["val_loss"] = pd.to_numeric(traj_df["val_loss"], errors="coerce")
    traj_df = traj_df.dropna(subset=["epoch", "val_loss"])

    removed_set: set[tuple[str, str, str]] = set()
    for (fn, exp, ms), group in traj_df.groupby(["function_name", "experiment", "model_size"]):
        start_rows = group.loc[group["epoch"] == group["epoch"].min(), "val_loss"]
        end_rows = group.loc[group["epoch"] == group["epoch"].max(), "val_loss"]
        if start_rows.empty or end_rows.empty:
            continue
        start_loss = float(start_rows.iloc[0])
        end_loss = float(end_rows.iloc[0])
        if start_loss <= 0 or (start_loss - end_loss) / start_loss < 0.03:
            removed_set.add((str(fn), str(exp), str(ms)))

    exp_meta: dict[str, dict] = {}
    for exp, group in traj_df.groupby("experiment"):
        row = group.iloc[0]
        dropout = float(row["dropout"]) if "dropout" in traj_df.columns else 0.0
        weight_decay = str(row.get("weight_decay", "False")).lower() in ["true", "1", "yes"]
        exp_meta[str(exp)] = {
            "noise": float(row["noise"]) if "noise" in traj_df.columns else 0.0,
            "regularization_state": _regularization_state(dropout, weight_decay),
            "optimizer": str(row["optimizer"]) if "optimizer" in traj_df.columns else "unknown",
        }

    noise_ord = {n: i for i, n in enumerate(NOISE_ORDER)}
    reg_ord = {r: i for i, r in enumerate(REGULARIZATION_ORDER)}

    all_experiments = sorted(
        exp_meta.keys(),
        key=lambda e: (
            noise_ord.get(exp_meta[e]["noise"], 99),
            reg_ord.get(exp_meta[e]["regularization_state"], 99),
            exp_meta[e]["optimizer"],
        ),
    )

    functions = [f"f{i}" for i in range(1, 11)]
    n_rows = len(all_experiments)
    n_cols = len(functions)

    noise_groups: dict[float, list[int]] = {}
    for row_i, exp in enumerate(all_experiments):
        noise_groups.setdefault(exp_meta[exp]["noise"], []).append(row_i)

    def _reg_short(state: str) -> str:
        if state == "dropout+weight_decay":
            return "Dropout+W"
        return _pretty_regularization_label(state).split()[0]

    row_labels = [
        f"{exp_meta[e]['optimizer']} · {_reg_short(exp_meta[e]['regularization_state'])}"
        for e in all_experiments
    ]

    _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(10, 12))

    small_handle = ax.scatter([], [], marker="o", color="#2A6F97", s=60, label="Small (removed)")
    big_handle = ax.scatter([], [], marker="s", color="#E76F51", s=60, label="Big (removed)")

    for row_i, exp in enumerate(all_experiments):
        for col_i, fn in enumerate(functions):
            small_removed = (fn, exp, "small") in removed_set
            big_removed = (fn, exp, "big") in removed_set
            if small_removed and big_removed:
                ax.scatter(col_i - 0.15, row_i, marker="o", color="#2A6F97", s=60, zorder=3)
                ax.scatter(col_i + 0.15, row_i, marker="s", color="#E76F51", s=60, zorder=3)
            elif small_removed:
                ax.scatter(col_i, row_i, marker="o", color="#2A6F97", s=60, zorder=3)
            elif big_removed:
                ax.scatter(col_i, row_i, marker="s", color="#E76F51", s=60, zorder=3)

    for row_i in range(n_rows):
        ax.axhline(row_i, color="gray", alpha=0.15, linewidth=0.5, zorder=0)

    sorted_noise_groups = sorted(noise_groups.items(), key=lambda x: noise_ord.get(x[0], 99))
    for noise_val, row_indices in sorted_noise_groups:
        max_row = max(row_indices)
        if max_row < n_rows - 1:
            ax.axhline(max_row + 0.5, color="black", linewidth=0.8, zorder=1)
        mid = (min(row_indices) + max(row_indices)) / 2.0
        ax.text(
            -1.2, mid,
            f"noise = {_pretty_noise_label(noise_val)}",
            ha="right", va="center", fontsize=9,
            transform=ax.transData,
            clip_on=False,
        )

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(functions)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        handles=[small_handle, big_handle],
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        frameon=False,
        fontsize=9,
        borderaxespad=0,
    )

    fig.tight_layout()
    _save_pgf(fig, output_path)
    plt.close(fig)


def generate_thesis_plots(results_df: pd.DataFrame, output_dir: Path, trajectories_dir: Path, results_root: Path) -> None:
    if results_df.empty:
        raise ValueError("No best-epoch results found in the provided results directory")

    output_dir.mkdir(parents=True, exist_ok=True)
    # Set up style once at the beginning for all plots
    _setup_thesis_style()

    if not _CONVERGENCE_STATS.empty:
        _plot_convergence_removal(_CONVERGENCE_STATS, output_dir / "convergence_removal.pgf")
        generate_convergence_report(_CONVERGENCE_STATS, output_dir / "convergence_removal.tex")
        _plot_convergence_removal_dots(trajectories_dir, output_dir / "convergence_removal_dots.pgf")

    auprc_random_baseline = (
        float(results_df["num_gt"].mean()) / 1013
        if "num_gt" in results_df.columns and results_df["num_gt"].notna().any()
        else 0.0
    )

    noise_summary = _function_level_summary(results_df, ["model_size", "noise"])
    _plot_grouped_bars(
        noise_summary,
        x_col="noise",
        x_order=NOISE_ORDER,
        title="Effect of Noise on Average AUPRC",
        output_path=output_dir / "noise_effect_by_model_size.pgf",
        auprc_random_baseline=auprc_random_baseline,
    )

    reg_summary = _function_level_summary(results_df, ["model_size", "regularization_state"])
    _plot_grouped_bars(
        reg_summary,
        x_col="regularization_state",
        x_order=REGULARIZATION_ORDER,
        title="Effect of Regularisation on Average AUPRC",
        output_path=output_dir / "regularization_effect_by_model_size.pgf",
        auprc_random_baseline=auprc_random_baseline,
    )

    _plot_architecture_x_regularization(results_df, output_dir / "architecture_x_regularization.pgf", auprc_random_baseline=auprc_random_baseline)

    _plot_auprc_vs_val_loss_scatter(results_df, output_dir / "auprc_vs_validation_loss.pgf", auprc_random_baseline=auprc_random_baseline)

    _plot_auprc_by_interaction_order(results_df, output_dir / "auprc_by_interaction_order.pgf", auprc_random_baseline=auprc_random_baseline)

    _plot_interaction_recovery_trajectories(
        trajectories_dir,
        output_dir / "interaction_recovery_trajectories.pgf",
    )

    _plot_trajectory_highlight(trajectories_dir, output_dir / "interaction_recovery_highlight.pgf")

    peak_summary_df = compute_peak_auroc_epoch_summary(trajectories_dir)
    _plot_peak_epoch_vs_noise(peak_summary_df, output_dir / "peak_epoch_vs_noise.pgf")
    generate_peak_epoch_table(peak_summary_df, output_dir / "peak_epoch_table.tex")

    _plot_loss_vs_auprc_divergence(
        epoch_results_path=results_root / "auroc_by_epoch.csv",
        output_path=output_dir / "loss_vs_auprc_divergence.pgf",
        model_size="small",
    )

    def _function_sort_key(name: str):
        if str(name).startswith("f") and str(name)[1:].isdigit():
            return int(str(name)[1:])
        return str(name)

    comparison_df = results_df[results_df["function_name"].isin(COMPARISON_FUNCTIONS)].copy()
    if not comparison_df.empty:
        comparison_summary = (
            comparison_df.groupby(["function_name", "model_size"], as_index=False)[AUPRC_COLUMN]
            .mean()
            .pivot(index="function_name", columns="model_size", values=AUPRC_COLUMN)
            .reset_index()
        )
        comparison_summary = comparison_summary.dropna(subset=["small", "big"], how="any")
        if not comparison_summary.empty:
            comparison_summary = comparison_summary.rename(columns={"small": "small_mean", "big": "big_mean"})
            comparison_summary["delta"] = comparison_summary["big_mean"] - comparison_summary["small_mean"]
            comparison_summary = comparison_summary.sort_values(
                "function_name",
                key=lambda col: col.map(_function_sort_key),
                ascending=True,
            ).reset_index(drop=True)
            _plot_dumbbell_comparison(comparison_summary, output_dir / "architecture_comparison_big_vs_small.pgf")

    _plot_summary_heatmap(results_df, output_dir / "summary_heatmap_noise_x_regularization.pgf")

    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    for function_name in sorted(results_df["function_name"].dropna().unique(), key=_function_sort_key):
        function_df = results_df[results_df["function_name"] == function_name]
        small_df = function_df[function_df["model_size"] == "small"]
        big_df = function_df[function_df["model_size"] == "big"]

        small_matrix = _heatmap_matrix(small_df)
        big_matrix = _heatmap_matrix(big_df)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        im = _plot_heatmap(axes[0], small_matrix, f"{function_name} - Small")
        _plot_heatmap(axes[1], big_matrix, f"{function_name} - Big")
        cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
        cbar.set_label("Mean AUPRC")
        fig.suptitle(f"{function_name}: AUPRC Heatmap by Noise and Regularisation", y=0.98)
        _save_pgf(fig, heatmap_dir / f"{function_name}_heatmap_big_vs_small.pgf")
        plt.close(fig)


# ─── Probe / dose-response analysis ────────────────────────────────────────


def _parse_wd_lambda(exp_name: str) -> float | None:
    """Parse L2 lambda from wd_lam* experiment names (wd_lam0 → 0.0, wd_lam1e-5 → 1e-5).

    The numeric weight_decay column in probe CSVs is boolean and cannot distinguish
    different lambda values, so the experiment name is the authoritative source here.
    """
    if not exp_name.startswith("wd_lam"):
        return None
    try:
        return float(exp_name[len("wd_lam"):])
    except ValueError:
        return None


def _load_probe_results(probe_root: Path) -> pd.DataFrame:
    """Load successful probe rows for PROBE_FUNCTIONS from probe_root/small/."""
    frames: list[pd.DataFrame] = []
    for fn in PROBE_FUNCTIONS:
        path = probe_root / "small" / f"{fn}_analysis.csv"
        if not path.exists():
            print(f"  [probe] WARNING: {path} not found — skipping {fn}")
            continue
        df = pd.read_csv(path)
        df = df[df["success"].fillna(False)].copy()
        if df.empty:
            print(f"  [probe] WARNING: no successful rows in {path}")
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out[AUPRC_COLUMN] = pd.to_numeric(out[AUPRC_COLUMN], errors="coerce")
    out["dropout"] = pd.to_numeric(out["dropout"], errors="coerce")
    return out


def _probe_two_stage_summary(probe_df: pd.DataFrame, sweep: str) -> pd.DataFrame:
    """Two-stage aggregation over probe data.

    sweep=="dropout": selects dropout_p* experiments; x_value from numeric dropout column.
    sweep=="wd": selects wd_lam* experiments (including lam0); x_value parsed from name.

    Stage 1: mean AUPRC within each (function_name, x_value) — always one row in the probe.
    Stage 2: mean ± SD of those function means across PROBE_FUNCTIONS.

    Returns DataFrame: x_value | mean | std | count, sorted by x_value.
    """
    if sweep == "dropout":
        mask = probe_df["experiment"].str.startswith("dropout_p")
        sub = probe_df[mask].copy()
        sub["x_value"] = sub["dropout"].astype(float)
    else:
        mask = probe_df["experiment"].str.startswith("wd_lam")
        sub = probe_df[mask].copy()
        sub["x_value"] = sub["experiment"].map(_parse_wd_lambda)

    sub = sub.dropna(subset=["x_value", AUPRC_COLUMN])

    fn_level = (
        sub.groupby(["function_name", "x_value"], as_index=False)[AUPRC_COLUMN]
        .mean()
        .rename(columns={AUPRC_COLUMN: "fn_mean"})
    )
    summary = (
        fn_level.groupby("x_value", as_index=False)["fn_mean"]
        .agg(mean="mean", std="std", count="count")
    )
    summary["std"] = summary["std"].fillna(0.0)
    return summary.sort_values("x_value").reset_index(drop=True)


# Functions whose anchor AUPRC falls below this threshold are excluded from the
# per-function % normalisation to avoid a tiny denominator inflating the mean/SD.
# All probe anchors in the current data are well above this (f1: 0.29, f7: 0.25,
# f8: 0.68+), so in practice no function is excluded — the guard is a safety net.
_PROBE_ANCHOR_THRESHOLD = 0.15


def _compute_pct_change(
    probe_df: pd.DataFrame, sweep: str, threshold: float = _PROBE_ANCHOR_THRESHOLD
) -> tuple[pd.DataFrame, list[str]]:
    """Per-function % change in AUPRC relative to that function's own zero-strength anchor.

    Normalising per function before aggregating removes the between-function level
    offset (e.g. f8 being high doesn't inflate the mean or the SD).

    sweep=="dropout": anchor = dropout_p0.0 AUPRC; x_value from numeric dropout column.
    sweep=="wd":      anchor = wd_lam0 AUPRC;      x_value parsed from experiment name.

    Returns:
      pct_df  — function_name | x_value | pct_change | anchor_auprc | auprc_full
      excluded — functions dropped because anchor_auprc < threshold (printed to stdout)
    """
    if sweep == "dropout":
        mask = probe_df["experiment"].str.startswith("dropout_p")
        anchor_exp = "dropout_p0.0"
    else:
        mask = probe_df["experiment"].str.startswith("wd_lam")
        anchor_exp = "wd_lam0"

    sub = probe_df[mask].copy()
    if sweep == "dropout":
        sub["x_value"] = sub["dropout"].astype(float)
    else:
        sub["x_value"] = sub["experiment"].map(_parse_wd_lambda)

    sub = sub.dropna(subset=["x_value", AUPRC_COLUMN])

    anchor_map: dict[str, float] = (
        probe_df[probe_df["experiment"] == anchor_exp]
        .set_index("function_name")[AUPRC_COLUMN]
        .to_dict()
    )

    excluded: list[str] = []
    rows = []
    for fn in PROBE_FUNCTIONS:
        anchor_val = anchor_map.get(fn, np.nan)
        if pd.isna(anchor_val) or float(anchor_val) < threshold:
            excluded.append(fn)
            print(
                f"  [probe/pct] {sweep} panel: excluding {fn} "
                f"(anchor AUPRC={anchor_val:.4f} < threshold {threshold})"
            )
            continue
        for _, r in sub[sub["function_name"] == fn].iterrows():
            rows.append({
                "function_name": fn,
                "x_value": float(r["x_value"]),
                "pct_change": (float(r[AUPRC_COLUMN]) - float(anchor_val)) / float(anchor_val) * 100.0,
                "anchor_auprc": float(anchor_val),
                AUPRC_COLUMN: float(r[AUPRC_COLUMN]),
            })

    if not excluded:
        print(f"  [probe/pct] {sweep} panel: all {len(PROBE_FUNCTIONS)} functions pass anchor threshold.")

    return pd.DataFrame(rows), excluded


def _probe_pct_summary(pct_df: pd.DataFrame) -> pd.DataFrame:
    """Two-stage aggregation of per-function % changes.

    Stage 1: mean within (function_name, x_value) — always one probe row per cell.
    Stage 2: mean ± SD across functions.

    Returns: x_value | mean_pct | sd_pct | n_valid, sorted by x_value.
    """
    if pct_df.empty:
        return pd.DataFrame(columns=["x_value", "mean_pct", "sd_pct", "n_valid"])
    fn_level = (
        pct_df.groupby(["function_name", "x_value"], as_index=False)["pct_change"]
        .mean()
        .rename(columns={"pct_change": "fn_mean_pct"})
    )
    summary = (
        fn_level.groupby("x_value", as_index=False)["fn_mean_pct"]
        .agg(mean_pct="mean", sd_pct="std", n_valid="count")
    )
    summary["sd_pct"] = summary["sd_pct"].fillna(0.0)
    return summary.sort_values("x_value").reset_index(drop=True)


def _shape_verdict(pct_array: np.ndarray) -> str:
    """Short data-driven verdict for the shape of a % change series (non-anchor points)."""
    if len(pct_array) == 0:
        return "unknown"
    max_abs = float(np.max(np.abs(pct_array)))
    last = float(pct_array[-1])
    if max_abs < 10:
        return "no clear effect"
    mid = pct_array[:-1]
    if last < -20 and len(mid) > 0 and float(np.max(np.abs(mid))) < 15:
        return "flat then collapse"
    if last < -20:
        return "declining with strength"
    if last > 20:
        return "improving with strength"
    return "mixed response"


def _plot_dose_response_pct(probe_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot 2 — Two-panel % change figure (7.0 × 3.5, double-column).

    Left:  mean % change in AUPRC vs dropout rate p  (linear x-axis).
           Trend guide: thin grey dashed linear fit (the dropout effect is ~linear).
    Right: mean % change in AUPRC vs L2 lambda       (log x-axis; lambda=0 = 0% reference).
           No trend line drawn: with n=3 single-seed probe functions the cross-function
           % change shows high variance at all tested lambda values and no consistent
           flat-then-collapse shape across functions (f7 actually improves at lambda=1e-2),
           so a guide line would misrepresent the shape certainty.

    Both panels: points = mean pct change; error bars = ±1 SD across valid functions;
    horizontal reference at 0%; no per-function spaghetti.
    Saved as regularization_dose_response_pct.{pgf,png}.
    """
    if probe_df.empty:
        return

    color_main = "#2A6F97"

    dp_pct, _ = _compute_pct_change(probe_df, "dropout")
    wd_pct, _ = _compute_pct_change(probe_df, "wd")

    dp_sum = _probe_pct_summary(dp_pct)
    wd_sum = _probe_pct_summary(wd_pct)

    # Exclude lambda=0 from WD plotted points (it IS the 0% reference line)
    wd_plot = wd_sum[wd_sum["x_value"] > 0.0].copy()

    fig, (ax_dp, ax_wd) = plt.subplots(1, 2, figsize=(7.0, 3.5))

    # ── Left panel: dropout ──────────────────────────────────────────────────
    dp_x = dp_sum["x_value"].to_numpy(dtype=float)
    dp_m = dp_sum["mean_pct"].to_numpy(dtype=float)
    dp_s = dp_sum["sd_pct"].to_numpy(dtype=float)

    ax_dp.axhline(0.0, color="0.5", linestyle="-", linewidth=0.7, zorder=1)

    if len(dp_x) >= 2:
        coeffs = np.polyfit(dp_x, dp_m, 1)
        x_fine = np.linspace(dp_x.min(), dp_x.max(), 100)
        ax_dp.plot(x_fine, np.polyval(coeffs, x_fine),
                   color="0.6", linestyle="--", linewidth=0.8, zorder=2)

    ax_dp.errorbar(dp_x, dp_m, yerr=dp_s, fmt="o", color=color_main,
                   capsize=4, markersize=5, linewidth=1.2, zorder=4)

    ax_dp.set_xlabel("Dropout rate $p$")
    ax_dp.set_ylabel("AUPRC change vs anchor (%,  ±1 SD)")
    ax_dp.set_title("Dropout dose-response")
    ax_dp.title.set_fontsize(10)
    ax_dp.set_xlim(-0.02, 0.55)
    ax_dp.set_xticks([0.0, 0.1, 0.2, 0.3, 0.5])

    # ── Right panel: L2 weight decay ─────────────────────────────────────────
    wd_x = wd_plot["x_value"].to_numpy(dtype=float)
    wd_m = wd_plot["mean_pct"].to_numpy(dtype=float)
    wd_s = wd_plot["sd_pct"].to_numpy(dtype=float)

    ax_wd.axhline(0.0, color="0.5", linestyle="-", linewidth=0.7, zorder=1)

    ax_wd.errorbar(wd_x, wd_m, yerr=wd_s, fmt="o", color=color_main,
                   capsize=4, markersize=5, linewidth=1.2, zorder=4)

    ax_wd.set_xscale("log")
    ax_wd.set_xlabel(r"Weight-decay $\lambda$")
    ax_wd.set_ylabel("AUPRC change vs anchor (%,  ±1 SD)")
    ax_wd.set_title("L2 weight-decay dose-response")
    ax_wd.title.set_fontsize(10)

    fig.tight_layout()
    _save_pgf(fig, output_dir / "regularization_dose_response_pct.pgf")
    plt.close(fig)


def _plot_dose_response_overlay(probe_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot 3 — Single-axis overlay: dropout vs L2 weight decay (3.5 × 2.6, single-column).

    X-axis: rank index 0..4, where rank 0 = zero-strength anchor (p=0 / λ=0) for both
    sweeps.  Both series include the anchor as a plotted point so both start at 0%.
    The two sweeps are aligned by ordered rank, not by identical physical units; the
    axis label makes this explicit.

    SD shown as shaded bands (not cap bars) to keep two overlaid series readable.
    No trend guides — the shape contrast between the two series is the deliverable.
    Saved as regularization_dose_response_overlay.{pgf,png}.

    Rank → value mapping:
      Dropout: rank 0=p=0.0, 1=p=0.1, 2=p=0.2, 3=p=0.3, 4=p=0.5
      WD:      rank 0=λ=0,   1=λ=1e-5, 2=λ=1e-4, 3=λ=1e-3, 4=λ=1e-2
    """
    if probe_df.empty:
        return

    color_dp = "#2A6F97"   # small-blue
    color_wd = "#E76F51"   # big-orange

    dp_pct, _ = _compute_pct_change(probe_df, "dropout")
    wd_pct, _ = _compute_pct_change(probe_df, "wd")

    dp_sum = _probe_pct_summary(dp_pct).sort_values("x_value").reset_index(drop=True)
    wd_sum = _probe_pct_summary(wd_pct).sort_values("x_value").reset_index(drop=True)

    # Assign integer rank (0 = anchor) — sorted ascending by strength value
    dp_sum["rank"] = np.arange(len(dp_sum), dtype=float)
    wd_sum["rank"] = np.arange(len(wd_sum), dtype=float)

    dp_r = dp_sum["rank"].to_numpy(dtype=float)
    dp_m = dp_sum["mean_pct"].to_numpy(dtype=float)
    dp_s = dp_sum["sd_pct"].to_numpy(dtype=float)

    wd_r = wd_sum["rank"].to_numpy(dtype=float)
    wd_m = wd_sum["mean_pct"].to_numpy(dtype=float)
    wd_s = wd_sum["sd_pct"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    ax.axhline(0.0, color="0.5", linestyle="-", linewidth=0.7, zorder=1)

    ax.fill_between(dp_r, dp_m - dp_s, dp_m + dp_s,
                    color=color_dp, alpha=0.15, linewidth=0, zorder=2)
    ax.plot(dp_r, dp_m, "o-", color=color_dp, linewidth=1.4, markersize=4,
            zorder=4, label="Dropout")

    ax.fill_between(wd_r, wd_m - wd_s, wd_m + wd_s,
                    color=color_wd, alpha=0.15, linewidth=0, zorder=2)
    ax.plot(wd_r, wd_m, "s-", color=color_wd, linewidth=1.4, markersize=4,
            zorder=4, label="L2 weight decay")

    n_ranks = max(len(dp_r), len(wd_r))
    ax.set_xticks(np.arange(n_ranks))
    ax.set_xlabel(
        "Regularisation rank  (0 = off, 4 = strongest tested)\n"
        r"Dropout: $p{=}0,\,0.1,\,0.2,\,0.3,\,0.5$"
        r"  ·  WD: $\lambda{=}0,\,10^{-5},\,10^{-4},\,10^{-3},\,10^{-2}$",
        fontsize=7,
    )
    ax.set_ylabel("AUPRC change vs anchor (%)")
    ax.set_title("Dropout vs L2 weight decay")
    ax.title.set_fontsize(10)
    ax.legend(frameon=False, fontsize=7, loc="lower left")

    fig.tight_layout()
    _save_pgf(fig, output_dir / "regularization_dose_response_overlay.pgf")
    plt.close(fig)


def _save_dose_response_table(
    probe_df: pd.DataFrame, csv_path: Path, tex_path: Path
) -> None:
    """Print dose-response table to stdout, save CSV, and write a booktabs LaTeX table.

    Table columns: constant value | mean AUPRC (abs) | SD AUPRC | mean % change vs anchor.
    One section per regulariser. Caption notes: single-seed, 3 probe functions (f1/f7/f8),
    small arch, η=0, plus a data-driven shape verdict per regulariser.
    """
    dp_abs = _probe_two_stage_summary(probe_df, "dropout")
    wd_abs = _probe_two_stage_summary(probe_df, "wd")

    dp_pct, _ = _compute_pct_change(probe_df, "dropout")
    wd_pct, _ = _compute_pct_change(probe_df, "wd")
    dp_pct_sum = _probe_pct_summary(dp_pct)
    wd_pct_sum = _probe_pct_summary(wd_pct)

    def _merge_rows(abs_df: pd.DataFrame, pct_df: pd.DataFrame, sweep_label: str) -> list[dict]:
        out = []
        for _, row in abs_df.iterrows():
            xv = row["x_value"]
            match = pct_df[pct_df["x_value"] == xv]
            mean_pct = float(match["mean_pct"].iloc[0]) if not match.empty else np.nan
            out.append({
                "sweep": sweep_label,
                "constant_value": xv,
                "mean_auprc": round(float(row["mean"]), 5),
                "sd_auprc": round(float(row["std"]), 5),
                "n_functions": int(row["count"]),
                "mean_pct_change": round(mean_pct, 2) if not np.isnan(mean_pct) else np.nan,
            })
        return out

    rows = _merge_rows(dp_abs, dp_pct_sum, "dropout") + _merge_rows(wd_abs, wd_pct_sum, "weight_decay")
    table_df = pd.DataFrame(rows)

    print("\n" + "=" * 72)
    print("DOSE-RESPONSE TABLE  (f1, f7, f8 | small arch | η=0 | Adam)")
    print("=" * 72)
    print(table_df.to_string(index=False))
    print("=" * 72)

    _ensure_parent(csv_path)
    table_df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV  → {csv_path}")

    # ── LaTeX table ──────────────────────────────────────────────────────────
    dp_non_anchor = dp_pct_sum.loc[dp_pct_sum["x_value"] > 0, "mean_pct"].to_numpy(dtype=float)
    wd_non_anchor = wd_pct_sum.loc[wd_pct_sum["x_value"] > 0, "mean_pct"].to_numpy(dtype=float)
    dp_verdict = _shape_verdict(dp_non_anchor)
    wd_verdict = _shape_verdict(wd_non_anchor)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            r"\caption{Best-epoch AUPRC dose-response for single-seed probe sweep "
            r"(functions f1, f7, f8; small architecture; $\eta=0$; Adam). "
            r"Mean \% change is per-function normalised before averaging across functions. "
            f"Dropout shape: \\textit{{{dp_verdict}}}. "
            f"L2 weight-decay shape: \\textit{{{wd_verdict}}}.}}"
        ),
        r"\label{tab:dose_response}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Constant & Mean AUPRC & SD & Mean \% change \\",
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Dropout (anchor: $p=0$)}} \\",
    ]

    for _, r in table_df[table_df["sweep"] == "dropout"].iterrows():
        val = f"$p = {r['constant_value']:g}$"
        pct_str = f"{r['mean_pct_change']:+.1f}" if not pd.isna(r["mean_pct_change"]) else "--"
        lines.append(f"\\quad {val} & {r['mean_auprc']:.3f} & {r['sd_auprc']:.3f} & {pct_str} \\\\")

    lines += [
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{L2 weight decay (anchor: $\lambda=0$)}} \\",
    ]

    for _, r in table_df[table_df["sweep"] == "weight_decay"].iterrows():
        xv = r["constant_value"]
        if xv == 0.0:
            val = r"$\lambda = 0$ (anchor)"
        else:
            exp = int(round(np.log10(xv)))
            val = f"$\\lambda = 10^{{{exp}}}$"
        pct_str = f"{r['mean_pct_change']:+.1f}" if not pd.isna(r["mean_pct_change"]) else "--"
        lines.append(f"\\quad {val} & {r['mean_auprc']:.3f} & {r['sd_auprc']:.3f} & {pct_str} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]

    _ensure_parent(tex_path)
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved LaTeX → {tex_path}")


def _print_reproduction_check(probe_df: pd.DataFrame, main_results_root: Path) -> None:
    """Compare probe anchors and swept conditions against the matched main-results slice.

    Matched slice: f1, f7, f8 | small | noise=0 | Adam.
    Reports dropout_p0.0 vs main base_adam and wd_lam0 vs main base_adam as anchor checks.
    Reports dropout_p0.2 and wd_lam1e-4 vs main slice if that slice exists; states
    explicitly when it does not (so as not to compare against a wrong baseline).
    """
    print("\n" + "=" * 72)
    print("REPRODUCTION CHECK  (main results for f1/f7/f8 | small | noise=0 | Adam)")
    print("=" * 72)

    main_frames = []
    for fn in PROBE_FUNCTIONS:
        path = main_results_root / "small" / f"{fn}_analysis.csv"
        if path.exists():
            df = pd.read_csv(path)
            df = df[df["success"].fillna(False)].copy()
            main_frames.append(df)

    if not main_frames:
        print("  Main results not found — skipping.")
        print("=" * 72 + "\n")
        return

    main_df = pd.concat(main_frames, ignore_index=True)
    main_df["noise"] = pd.to_numeric(main_df["noise"], errors="coerce")
    main_df["dropout"] = pd.to_numeric(main_df["dropout"], errors="coerce")
    main_df["weight_decay"] = main_df["weight_decay"].astype(str).str.lower().isin(["true", "1", "yes"])
    main_df[AUPRC_COLUMN] = pd.to_numeric(main_df[AUPRC_COLUMN], errors="coerce")

    main_slice = main_df[(main_df["noise"] == 0.0) & (main_df["optimizer"] == "adam")].copy()
    main_base = main_slice[(main_slice["dropout"] == 0.0) & (~main_slice["weight_decay"])].copy()

    main_anchor_mean = np.nan
    if not main_base.empty:
        main_fn_means = main_base.groupby("function_name")[AUPRC_COLUMN].mean()
        main_anchor_mean = float(main_fn_means.mean())
        main_anchor_std = float(main_fn_means.std()) if len(main_fn_means) > 1 else 0.0
        print(f"\nMain base_adam (dropout=0, no WD, noise=0)  n_fn={len(main_fn_means)}")
        for fn, v in main_fn_means.items():
            print(f"  {fn}: {v:.4f}")
        print(f"  Mean {main_anchor_mean:.4f}  SD {main_anchor_std:.4f}")
    else:
        print("\nMain base_adam at noise=0: not found")

    def _report_probe_condition(label: str, exp_name: str, anchor: float) -> None:
        rows = probe_df[probe_df["experiment"] == exp_name]
        if rows.empty:
            print(f"\n{label}: experiment not found in probe data")
            return
        fn_vals = rows.set_index("function_name")[AUPRC_COLUMN]
        print(f"\n{label}")
        for fn in PROBE_FUNCTIONS:
            v = fn_vals.get(fn, np.nan)
            print(f"  {fn}: {v:.4f}" if not pd.isna(v) else f"  {fn}: N/A")
        probe_mean = float(fn_vals.reindex(PROBE_FUNCTIONS).mean())
        probe_std = float(fn_vals.reindex(PROBE_FUNCTIONS).std())
        print(f"  Mean {probe_mean:.4f}  SD {probe_std:.4f}")
        if not np.isnan(anchor):
            diff = abs(probe_mean - anchor)
            pct = diff / anchor * 100
            flag = "  *** >10% divergence (single-seed variance)" if pct > 10 else "  ok (<10%)"
            print(f"  vs main anchor: |Δ|={diff:.4f}  ({pct:.1f}%){flag}")

    dp_summary = _probe_two_stage_summary(probe_df, "dropout")
    wd_summary = _probe_two_stage_summary(probe_df, "wd")
    dp_anchor_mean = float(dp_summary.loc[dp_summary["x_value"] == 0.0, "mean"].iloc[0]) \
        if not dp_summary[dp_summary["x_value"] == 0.0].empty else np.nan
    wd_anchor_mean = float(wd_summary.loc[wd_summary["x_value"] == 0.0, "mean"].iloc[0]) \
        if not wd_summary[wd_summary["x_value"] == 0.0].empty else np.nan

    _report_probe_condition("Probe dropout_p0.0  (p=0, noise=0, Adam)", "dropout_p0.0", main_anchor_mean)
    _report_probe_condition("Probe wd_lam0  (λ=0, noise=0, Adam)", "wd_lam0", main_anchor_mean)

    # dropout_p0.2 vs main
    print("\nProbe dropout_p0.2  (p=0.2, noise=0, Adam)")
    dp02_rows = probe_df[probe_df["experiment"] == "dropout_p0.2"].set_index("function_name")[AUPRC_COLUMN]
    dp02_vals = [dp02_rows.get(fn, np.nan) for fn in PROBE_FUNCTIONS]
    for fn, v in zip(PROBE_FUNCTIONS, dp02_vals):
        print(f"  {fn}: {v:.4f}" if not pd.isna(v) else f"  {fn}: N/A")
    dp02_mean = float(np.nanmean(dp02_vals))
    print(f"  Mean {dp02_mean:.4f}")
    main_dp02 = main_slice[(main_slice["dropout"] == 0.2) & (~main_slice["weight_decay"])]
    if main_dp02.empty:
        print("  Main results for noise=0+adam+dropout=0.2: NOT FOUND "
              "(main experiment did not run this condition — probe is filling the gap).")
    else:
        m = float(main_dp02.groupby("function_name")[AUPRC_COLUMN].mean().mean())
        pct = abs(dp02_mean - m) / m * 100 if m > 0 else np.nan
        flag = "  *** >10% divergence" if pct > 10 else "  ok"
        print(f"  Main slice mean: {m:.4f} | |Δ|={abs(dp02_mean - m):.4f}  ({pct:.1f}%){flag}")

    # wd_lam1e-4 vs main
    print("\nProbe wd_lam1e-4  (λ=1e-4, noise=0, Adam)")
    wd4_rows = probe_df[probe_df["experiment"] == "wd_lam1e-4"].set_index("function_name")[AUPRC_COLUMN]
    wd4_vals = [wd4_rows.get(fn, np.nan) for fn in PROBE_FUNCTIONS]
    for fn, v in zip(PROBE_FUNCTIONS, wd4_vals):
        print(f"  {fn}: {v:.4f}" if not pd.isna(v) else f"  {fn}: N/A")
    wd4_mean = float(np.nanmean(wd4_vals))
    print(f"  Mean {wd4_mean:.4f}")
    main_wd = main_slice[(main_slice["dropout"] == 0.0) & main_slice["weight_decay"]]
    if main_wd.empty:
        print("  Main results for noise=0+adam+weight_decay=True: NOT FOUND "
              "(main experiment did not run this condition — probe is filling the gap).")
    else:
        m = float(main_wd.groupby("function_name")[AUPRC_COLUMN].mean().mean())
        pct = abs(wd4_mean - m) / m * 100 if m > 0 else np.nan
        flag = "  *** >10% divergence" if pct > 10 else "  ok"
        print(f"  Main slice mean (boolean WD, any lambda): {m:.4f} | |Δ|={abs(wd4_mean - m):.4f}  ({pct:.1f}%){flag}")

    print("=" * 72 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis plots from best-epoch results")
    parser.add_argument("--results-root", default="results", help="Root folder containing big/small analysis CSVs")
    parser.add_argument("--output-dir", default="results/thesis_plots", help="Directory for generated thesis plots")
    parser.add_argument("--trajectories-dir", default="results/trajectories", help="Directory containing trajectory CSVs")
    parser.add_argument(
        "--prob",
        action="store_true",
        help="Generate regularisation dose-response figures from probe results; skip full thesis suite.",
    )
    parser.add_argument(
        "--probe-results-root",
        default="results_probe",
        help="Root folder for probe analysis CSVs (default: results_probe)",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    trajectories_dir = Path(args.trajectories_dir)

    if args.prob:
        probe_root = Path(args.probe_results_root)
        _setup_thesis_style()
        probe_df = _load_probe_results(probe_root)
        if probe_df.empty:
            raise SystemExit(
                f"No probe results found under {probe_root}/small/ — "
                "check that dropout_p* and wd_lam* experiments succeeded."
            )
        print(f"Loaded {len(probe_df)} successful probe rows "
              f"from {probe_df['function_name'].nunique()} functions.")
        output_dir.mkdir(parents=True, exist_ok=True)
        _plot_dose_response_pct(probe_df, output_dir)
        _plot_dose_response_overlay(probe_df, output_dir)
        _save_dose_response_table(
            probe_df,
            probe_root / "dose_response.csv",
            probe_root / "dose_response.tex",
        )
        _print_reproduction_check(probe_df, results_root)
        print(f"Saved: {output_dir}/regularization_dose_response_pct.{{pgf,png}}")
        print(f"       {output_dir}/regularization_dose_response_overlay.{{pgf,png}}")
        return

    results_df = load_best_epoch_results(results_root, trajectories_dir)
    if results_df.empty:
        raise SystemExit(f"No best-epoch analysis CSVs found under {results_root}")

    generate_thesis_plots(results_df, output_dir, trajectories_dir, results_root)
    print(f"Saved thesis plots to {output_dir}")


if __name__ == "__main__":
    main()
