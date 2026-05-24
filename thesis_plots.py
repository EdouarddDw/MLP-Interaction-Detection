#!/usr/bin/env python3
"""Generate thesis-ready summary plots from best-epoch analysis CSVs.

This script reads the existing `results/big/` and `results/small/` analysis
outputs, aggregates AUROC values across optimizers, and creates:
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
from scipy.stats import spearmanr

from synth import functions as synth_functions


NOISE_ORDER = [0.0, 0.1, 0.2, 0.5, 1.0]
REGULARIZATION_ORDER = [
    "base",
    "dropout",
    "weight_decay",
    "dropout+weight_decay",
]
MODEL_SIZE_ORDER = ["small", "big"]
COMPARISON_FUNCTIONS = [f"f{i}" for i in range(1, 11)]


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


def load_best_epoch_results(results_root: Path) -> pd.DataFrame:
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
    return df


def _function_level_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Aggregate across optimizers first, then average across functions.

    This prevents one function or optimizer from dominating the thesis plots.
    """
    if df.empty:
        return pd.DataFrame()

    # First aggregate within each function.
    function_level = (
        df.groupby(["function_name", *group_cols], as_index=False)["auroc"]
        .mean()
        .rename(columns={"auroc": "function_mean_auroc"})
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
    summary = df.groupby(group_cols, as_index=False)["auroc"].agg(mean="mean", std="std", count="count")
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
    ylabel: str = "Average AUROC",
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

    ax.set_title(title)
    ax.title.set_fontsize(10)
    ax.set_ylabel("Average AUROC (±1 SD across functions)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([
        _pretty_regularization_label(str(v)) if x_col == "regularization_state"
        else _pretty_noise_label(v) if x_col == "noise"
        else str(v)
        for v in x_values
    ], rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.tight_layout()
    _save_pgf(fig, output_path)
    plt.close(fig)


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
    ax.set_xlabel("Mean AUROC")
    ax.set_title("Big vs Small Model Comparison by Function")
    ax.set_xlim(0.0, 1.0)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _save_pgf(fig, output_path)
    plt.close(fig)


def _plot_auroc_vs_val_loss_scatter(results_df: pd.DataFrame, output_path: Path) -> None:
    if results_df.empty or "val_loss" not in results_df.columns:
        return

    plot_df = results_df.dropna(subset=["auroc", "val_loss"]).copy()
    if plot_df.empty:
        return

    # Aggregate to one point per (function_name, model_size) pair
    agg_df = (
        plot_df.groupby(["function_name", "model_size"], as_index=False)[["auroc", "val_loss"]]
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
            subset["auroc"],
            s=16,
            alpha=0.75,
            color=colors[model_size],
            marker=markers[model_size],
            label=model_size,
            edgecolors="none",
        )

    # Filter to positive val_loss to avoid log-scale issues
    x_all = agg_df["val_loss"].to_numpy(dtype=float)
    y_all = agg_df["auroc"].to_numpy(dtype=float)
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
        ax.set_title(rf"AUROC vs Validation Loss ($\rho$={rho:.2f}, {p_str})")
    else:
        ax.set_title("AUROC vs Validation Loss")

    ax.title.set_fontsize(9)
    ax.set_xscale("log")
    ax.set_xlabel("Validation loss (log scale)")
    ax.set_ylabel("AUROC")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    _save_pgf(fig, output_path)
    plt.close(fig)


def _plot_auroc_by_interaction_order(results_df: pd.DataFrame, output_path: Path) -> None:
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
    
    ax.set_title("Effect of Interaction Order on Average AUROC")
    ax.title.set_fontsize(10)
    ax.set_ylabel("Average AUROC (±1 SD across functions)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{int(order)}" for order in interaction_orders], rotation=0)
    ax.set_xlabel("Maximum Interaction Order")
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.tight_layout()
    _save_pgf(fig, output_path)
    plt.close(fig)


def _plot_interaction_recovery_trajectories(trajectories_dir: Path, output_path: Path) -> None:
    if not trajectories_dir.exists() or not trajectories_dir.is_dir():
        return

    trajectory_files = sorted(trajectories_dir.glob("*_trajectory.csv"))
    if not trajectory_files:
        return

    frames: list[pd.DataFrame] = []
    for path in trajectory_files:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        if "function_name" not in frame.columns and "function" in frame.columns:
            frame = frame.rename(columns={"function": "function_name"})
        if "experiment_name" not in frame.columns and "experiment" in frame.columns:
            frame = frame.rename(columns={"experiment": "experiment_name"})
        frames.append(frame)

    if not frames:
        return

    plot_df = pd.concat(frames, ignore_index=True)
    if plot_df.empty:
        return

    required_columns = {"epoch", "auroc", "function_name", "noise", "dropout", "weight_decay"}
    if not required_columns.issubset(plot_df.columns):
        return

    plot_df = plot_df.copy()
    plot_df["epoch"] = pd.to_numeric(plot_df["epoch"], errors="coerce")
    plot_df["auroc"] = pd.to_numeric(plot_df["auroc"], errors="coerce")
    plot_df["noise"] = pd.to_numeric(plot_df["noise"], errors="coerce")
    plot_df["dropout"] = pd.to_numeric(plot_df["dropout"], errors="coerce")
    plot_df["weight_decay"] = plot_df["weight_decay"].astype(bool)
    plot_df = plot_df.dropna(subset=["epoch", "auroc", "function_name", "noise", "dropout"])

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
                function_epoch = (
                    cell_df.groupby(["function_name", "epoch"], as_index=False)["auroc"]
                    .mean()
                )
                if not function_epoch.empty:
                    function_names = sorted(function_epoch["function_name"].dropna().unique(), key=_function_sort_key)
                    pivot = (
                        function_epoch.pivot(index="epoch", columns="function_name", values="auroc")
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
                            linewidth=1.8,
                            zorder=3,
                        )

            ax.set_ylim(0.0, 1.0)
            if row_index == 0:
                ax.set_title(_pretty_regularization_label(reg_state), fontsize=10)
            if row_index < len(NOISE_ORDER) - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Epoch")
            if col_index > 0:
                ax.tick_params(labelleft=False)
                ax.set_ylabel("")
            else:
                ax.set_ylabel("AUROC")

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

    fig.tight_layout()
    _save_pgf(fig, output_path)
    plt.close(fig)


def compute_peak_auroc_epoch_summary(trajectories_dir: Path) -> pd.DataFrame:
    if not trajectories_dir.exists() or not trajectories_dir.is_dir():
        return pd.DataFrame(
            columns=["function", "experiment", "noise", "regularization_state", "peak_auroc_epoch", "peak_auroc"]
        )

    trajectory_files = sorted(trajectories_dir.glob("*_trajectory.csv"))
    if not trajectory_files:
        return pd.DataFrame(
            columns=["function", "experiment", "noise", "regularization_state", "peak_auroc_epoch", "peak_auroc"]
        )

    rows = []
    for path in trajectory_files:
        frame = pd.read_csv(path)
        if frame.empty or "auroc" not in frame.columns or "epoch" not in frame.columns:
            continue

        if "function" not in frame.columns and "function_name" in frame.columns:
            frame = frame.rename(columns={"function_name": "function"})
        if "experiment" not in frame.columns and "experiment_name" in frame.columns:
            frame = frame.rename(columns={"experiment_name": "experiment"})

        required_columns = {"function", "experiment", "noise", "dropout", "weight_decay"}
        if not required_columns.issubset(frame.columns):
            continue

        frame = frame.copy()
        frame["epoch"] = pd.to_numeric(frame["epoch"], errors="coerce")
        frame["auroc"] = pd.to_numeric(frame["auroc"], errors="coerce")
        frame["noise"] = pd.to_numeric(frame["noise"], errors="coerce")
        frame["dropout"] = pd.to_numeric(frame["dropout"], errors="coerce")
        if frame["weight_decay"].dtype == bool:
            weight_decay = frame["weight_decay"]
        else:
            weight_decay = frame["weight_decay"].astype(str).str.lower().isin(["true", "1", "yes"])
        frame["weight_decay"] = weight_decay
        frame = frame.dropna(subset=["epoch", "auroc", "function", "experiment", "noise", "dropout"])
        if frame.empty:
            continue

        peak_index = frame["auroc"].idxmax()
        peak_row = frame.loc[peak_index]
        regularization_state = _regularization_state(float(peak_row["dropout"]), bool(peak_row["weight_decay"]))
        if regularization_state not in REGULARIZATION_ORDER:
            continue

        rows.append(
            {
                "function": peak_row["function"],
                "experiment": peak_row["experiment"],
                "noise": float(peak_row["noise"]),
                "regularization_state": regularization_state,
                "peak_auroc_epoch": int(float(peak_row["epoch"])),
                "peak_auroc": float(peak_row["auroc"]),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["function", "experiment", "noise", "regularization_state", "peak_auroc_epoch", "peak_auroc"]
        )

    return pd.DataFrame(rows)


def _plot_peak_epoch_vs_noise(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        return

    summary_df = summary_df.copy()
    summary_df = summary_df[summary_df["regularization_state"].isin(REGULARIZATION_ORDER)]
    if summary_df.empty:
        return

    function_level = (
        summary_df.groupby(["function", "noise", "regularization_state"], as_index=False)["peak_auroc_epoch"]
        .mean()
        .rename(columns={"peak_auroc_epoch": "function_mean_peak_epoch"})
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
    ax.set_ylabel("Mean epoch of peak AUROC")
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
        summary_df.groupby(["function", "noise", "regularization_state"], as_index=False)["peak_auroc_epoch"]
        .mean()
        .rename(columns={"peak_auroc_epoch": "function_mean_peak_epoch"})
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
        r"\caption{Mean epoch of peak AUROC across functions by noise level and regularization state.}",
        r"\end{table}",
        "",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _heatmap_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(index=NOISE_ORDER, columns=REGULARIZATION_ORDER, dtype=float)

    pivot = (
        df.groupby(["noise", "regularization_state"], as_index=False)["auroc"]
        .mean()
        .pivot(index="noise", columns="regularization_state", values="auroc")
    )
    pivot = pivot.reindex(index=NOISE_ORDER, columns=REGULARIZATION_ORDER)
    return pivot


def _plot_heatmap(ax, matrix: pd.DataFrame, title: str):
    data = matrix.to_numpy(dtype=float)
    im = ax.imshow(data, vmin=0.0, vmax=1.0, cmap="cividis", aspect="auto")

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
                # 0.55 is the luminance crossover point for cividis colormap
                color = "white" if value < 0.55 else "black"
            ax.text(j, i, label, ha="center", va="center", fontsize=9, color=color)

    return im


def _plot_loss_vs_auroc_divergence(epoch_results_path: Path, output_path: Path) -> None:
    if not epoch_results_path.exists():
        warnings.warn(f"Epoch results file not found: {epoch_results_path}")
        return

    df = pd.read_csv(epoch_results_path)
    if df.empty:
        warnings.warn(f"Epoch results file is empty: {epoch_results_path}")
        return

    required_cols = {"epoch", "auroc", "val_loss", "function_name", "noise", "dropout", "weight_decay"}
    if not required_cols.issubset(df.columns):
        warnings.warn(f"Missing columns in epoch results: {required_cols - set(df.columns)}")
        return

    df = df.copy()
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df["auroc"] = pd.to_numeric(df["auroc"], errors="coerce")
    df["val_loss"] = pd.to_numeric(df["val_loss"], errors="coerce")
    df["noise"] = pd.to_numeric(df["noise"], errors="coerce")
    df["dropout"] = pd.to_numeric(df["dropout"], errors="coerce")
    df["weight_decay"] = df["weight_decay"].astype(str).str.lower().isin(["true", "1", "yes"])
    df["regularization_state"] = [
        _regularization_state(d, w) for d, w in zip(df["dropout"], df["weight_decay"])
    ]

    df = df[df["regularization_state"] == "base"].copy()
    df = df[df["noise"].isin([0.0, 1.0])].copy()
    df = df.dropna(subset=["epoch", "auroc", "val_loss", "function_name"])

    if df.empty:
        warnings.warn("No data after filtering for noise=0/1 and base regularization.")
        return

    color_loss = "#E76F51"
    color_auroc = "#2A6F97"

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # Proxy artists for shared legend — empty so invisible in the plot itself
    loss_handle, = axes[0].plot([], [], color=color_loss, linewidth=1.6, label="Validation loss")
    auroc_handle, = axes[0].plot([], [], color=color_auroc, linewidth=1.6, label="AUROC")

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
            noise_df.groupby(["function_name", "epoch"], as_index=False)[["auroc", "val_loss"]]
            .mean()
        )

        # Keep only epochs where at least 3 distinct functions have valid data
        func_count = func_epoch.groupby("epoch")["function_name"].nunique()
        valid_epochs = func_count[func_count >= 3].index
        func_epoch = func_epoch[func_epoch["epoch"].isin(valid_epochs)].copy()
        if func_epoch.empty:
            continue

        # Stage 2: aggregate across functions per epoch
        auroc_agg = func_epoch.groupby("epoch")["auroc"].agg(mean="mean", std="std")
        loss_agg = func_epoch.groupby("epoch")["val_loss"].agg(mean="mean", std="std")
        epoch_stats = pd.DataFrame({
            "auroc_mean": auroc_agg["mean"],
            "auroc_std": auroc_agg["std"].fillna(0.0),
            "loss_mean": loss_agg["mean"],
            "loss_std": loss_agg["std"].fillna(0.0),
        }).reset_index().sort_values("epoch")

        x = epoch_stats["epoch"].to_numpy(dtype=float)
        auroc_mean = epoch_stats["auroc_mean"].to_numpy(dtype=float)
        auroc_std = epoch_stats["auroc_std"].to_numpy(dtype=float)
        loss_mean = epoch_stats["loss_mean"].to_numpy(dtype=float)
        loss_std = epoch_stats["loss_std"].to_numpy(dtype=float)

        # Find first epoch where AUROC reaches 95 % of its series maximum
        plateau_epoch = None
        valid_auroc = auroc_mean[~np.isnan(auroc_mean)]
        if len(valid_auroc) > 0:
            threshold = 0.95 * np.max(valid_auroc)
            plateau_mask = auroc_mean >= threshold
            if plateau_mask.any():
                plateau_epoch = x[plateau_mask][0]

        # Left axis: validation loss
        ax.plot(x, loss_mean, color=color_loss, linewidth=1.6)
        ax.fill_between(x, loss_mean - loss_std, loss_mean + loss_std,
                        color=color_loss, alpha=0.15, linewidth=0)

        # Right axis: AUROC (twin)
        ax2 = ax.twinx()
        ax2.plot(x, auroc_mean, color=color_auroc, linewidth=1.6)
        ax2.fill_between(x, auroc_mean - auroc_std, auroc_mean + auroc_std,
                         color=color_auroc, alpha=0.15, linewidth=0)
        ax2.set_ylabel("AUROC", color=color_auroc)
        ax2.tick_params(axis="y", labelcolor=color_auroc)
        ax2.set_ylim(0.0, 1.0)

        if plateau_epoch is not None:
            ax.axvline(plateau_epoch, color="gray", linestyle="--", linewidth=1.0)
            ax.text(
                plateau_epoch,
                0.97,
                "AUROC plateau",
                transform=ax.get_xaxis_transform(),
                ha="right",
                va="top",
                fontsize=7,
                color="gray",
                rotation=90,
            )

    fig.legend(
        handles=[loss_handle, auroc_handle],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.05),
        fontsize=9,
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

    noise_summary = _function_level_summary(results_df, ["model_size", "noise"])
    _plot_grouped_bars(
        noise_summary,
        x_col="noise",
        x_order=NOISE_ORDER,
        title="Effect of Noise on Average AUROC",
        output_path=output_dir / "noise_effect_by_model_size.pgf",
    )

    reg_summary = _function_level_summary(results_df, ["model_size", "regularization_state"])
    _plot_grouped_bars(
        reg_summary,
        x_col="regularization_state",
        x_order=REGULARIZATION_ORDER,
        title="Effect of Regularisation on Average AUROC",
        output_path=output_dir / "regularization_effect_by_model_size.pgf",
    )

    _plot_auroc_vs_val_loss_scatter(results_df, output_dir / "auroc_vs_validation_loss.pgf")

    _plot_auroc_by_interaction_order(results_df, output_dir / "auroc_by_interaction_order.pgf")

    _plot_interaction_recovery_trajectories(
        trajectories_dir,
        output_dir / "interaction_recovery_trajectories.pgf",
    )

    peak_summary_df = compute_peak_auroc_epoch_summary(trajectories_dir)
    _plot_peak_epoch_vs_noise(peak_summary_df, output_dir / "peak_epoch_vs_noise.pgf")
    generate_peak_epoch_table(peak_summary_df, output_dir / "peak_epoch_table.tex")

    _plot_loss_vs_auroc_divergence(
        epoch_results_path=results_root / "auroc_by_epoch.csv",
        output_path=output_dir / "loss_vs_auroc_divergence.pgf",
    )

    def _function_sort_key(name: str):
        if str(name).startswith("f") and str(name)[1:].isdigit():
            return int(str(name)[1:])
        return str(name)

    comparison_df = results_df[results_df["function_name"].isin(COMPARISON_FUNCTIONS)].copy()
    if not comparison_df.empty:
        comparison_summary = (
            comparison_df.groupby(["function_name", "model_size"], as_index=False)["auroc"]
            .mean()
            .pivot(index="function_name", columns="model_size", values="auroc")
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
        cbar.set_label("Mean AUROC")
        fig.suptitle(f"{function_name}: AUROC Heatmap by Noise and Regularisation", y=0.98)
        _save_pgf(fig, heatmap_dir / f"{function_name}_heatmap_big_vs_small.pgf")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis plots from best-epoch results")
    parser.add_argument("--results-root", default="results", help="Root folder containing big/small analysis CSVs")
    parser.add_argument("--output-dir", default="results/thesis_plots", help="Directory for generated thesis plots")
    parser.add_argument("--trajectories-dir", default="results/trajectories", help="Directory containing trajectory CSVs")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    trajectories_dir = Path(args.trajectories_dir)

    results_df = load_best_epoch_results(results_root)
    if results_df.empty:
        raise SystemExit(f"No best-epoch analysis CSVs found under {results_root}")

    generate_thesis_plots(results_df, output_dir, trajectories_dir, results_root)
    print(f"Saved thesis plots to {output_dir}")


if __name__ == "__main__":
    main()
