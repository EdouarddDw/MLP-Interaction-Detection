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


NOISE_ORDER = [0.0, 0.1, 0.2, 0.5, 1.0]
REGULARIZATION_ORDER = [
    "base",
    "dropout",
    "weight_decay",
    "dropout+weight_decay",
]
MODEL_SIZE_ORDER = ["small", "big"]
COMPARISON_FUNCTIONS = [f"f{i}" for i in range(1, 10)]


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

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
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
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(v) for v in x_values])
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
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


def generate_thesis_plots(results_df: pd.DataFrame, output_dir: Path) -> None:
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
            comparison_summary = comparison_summary.sort_values("delta", ascending=False).reset_index(drop=True)
            _plot_dumbbell_comparison(comparison_summary, output_dir / "architecture_comparison_big_vs_small.pgf")

    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    def _function_sort_key(name: str):
        if str(name).startswith("f") and str(name)[1:].isdigit():
            return int(str(name)[1:])
        return str(name)

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
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)

    results_df = load_best_epoch_results(results_root)
    if results_df.empty:
        raise SystemExit(f"No best-epoch analysis CSVs found under {results_root}")

    generate_thesis_plots(results_df, output_dir)
    print(f"Saved thesis plots to {output_dir}")


if __name__ == "__main__":
    main()
