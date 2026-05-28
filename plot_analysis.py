#!/usr/bin/env python
"""Generate epoch-level AUROC summary plots from results/auroc_by_epoch.csv."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from thesis_plots import _save_pgf, _setup_thesis_style


RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
EPOCH_RESULTS_CSV = RESULTS_DIR / "auroc_by_epoch.csv"

_LINE_STYLES = [
    ("#1f77b4", "-"), ("#d62728", "--"), ("#2ca02c", "-."),
    ("#ff7f0e", ":"), ("#9467bd", (0, (3, 1, 1, 1))),
    ("#8c564b", (0, (5, 2))), ("#e377c2", (0, (1, 1))),
    ("#7f7f7f", (0, (3, 5, 1, 5))), ("#bcbd22", "-"),
    ("#17becf", "--"),
]


def _regularization_category(dropout: float, weight_decay: bool) -> str:
    if dropout > 0.0 and weight_decay:
        return "Dropout + weight decay"
    if dropout > 0.0:
        return "Dropout only"
    if weight_decay:
        return "Weight decay only"
    return "No regularization"


def _group_sort_key(group_column: str, value):
    if group_column == "function_name" and isinstance(value, str) and value.startswith("f"):
        try:
            return (0, int(value[1:]))
        except ValueError:
            return (1, str(value))
    if group_column == "regularization_category":
        order = {
            "No regularization": 0,
            "Dropout only": 1,
            "Weight decay only": 2,
            "Dropout + weight decay": 3,
        }
        return (0, order.get(str(value), 99), str(value))
    if group_column == "optimizer":
        order = {"adam": 0, "sgd": 1}
        return (0, order.get(str(value).lower(), 99), str(value))
    if group_column == "noise":
        try:
            return (0, float(value))
        except (TypeError, ValueError):
            return (1, str(value))
    return (0, str(value))


def collect_epoch_level_results(results_path: Path = Path("results/auroc_by_epoch.csv")) -> pd.DataFrame:
    if not results_path.exists():
        raise FileNotFoundError(f"auroc_by_epoch.csv not found at {results_path} — run analysis.py first")
    results_df = pd.read_csv(results_path)
    if "val_loss" in results_df.columns:
        results_df["val_loss"] = pd.to_numeric(results_df["val_loss"], errors="coerce")
        traj_max = results_df.groupby(["function_name", "experiment"])["val_loss"].transform("max")
        traj_min = results_df.groupby(["function_name", "experiment"])["val_loss"].transform("min")
        converged = (traj_max - traj_min) / traj_max.replace(0, np.nan) >= 0.05
        n_removed = (~converged.fillna(True)).sum()
        if n_removed > 0:
            print(f"  [convergence filter] removed {n_removed} epoch rows from non-convergent runs")
        results_df = results_df[converged.fillna(True)].copy()
    return results_df


def _summary_by_epoch(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    clean = df.dropna(subset=["auprc_full"])
    function_level = (
        clean.groupby(["function_name", group_column, "epoch"], as_index=False)["auprc_full"]
        .mean()
        .rename(columns={"auprc_full": "function_mean_auprc"})
    )
    summary = (
        function_level.groupby([group_column, "epoch"], as_index=False)["function_mean_auprc"]
        .agg(mean="mean", std="std", count="count")
    )
    summary["std"] = summary["std"].fillna(0.0)
    summary["lower"] = summary["mean"] - summary["std"]
    summary["upper"] = summary["mean"] + summary["std"]
    return summary


def _plot_grouped_lines(
    summary_df: pd.DataFrame,
    group_column: str,
    title: str,
    output_path: Path,
    ylabel: str = r"AUPRC ($2^{[10]}$)",
):
    """Plot one line per unique value of *group_column* from *summary_df*.

    *summary_df* must contain columns: group_column, epoch, mean, lower, upper.
    *group_column* controls which dimension is split into separate lines (e.g.
    "function_name", "noise", "regularization_category", "optimizer").
    """
    if summary_df.empty:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    groups = list(summary_df[group_column].dropna().unique())
    groups.sort(key=lambda value: _group_sort_key(group_column, value))

    fig, ax = plt.subplots(figsize=(7.0, 3.5))

    for index, group_name in enumerate(groups):
        group_df = summary_df[summary_df[group_column] == group_name].sort_values("epoch")
        if group_df.empty:
            continue

        epochs = group_df["epoch"].to_numpy(dtype=float)
        means = group_df["mean"].to_numpy(dtype=float)
        lower = group_df["lower"].to_numpy(dtype=float)
        upper = group_df["upper"].to_numpy(dtype=float)

        color, linestyle = _LINE_STYLES[index % len(_LINE_STYLES)]
        ax.plot(epochs, means, label=str(group_name), color=color, linestyle=linestyle, linewidth=2)
        ax.fill_between(epochs, lower, upper, color=color, alpha=0.18, linewidth=0)

    ax.set_title(title)
    ax.title.set_fontsize(10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.tight_layout()
    _save_pgf(fig, output_path)
    plt.close(fig)


def generate_plots(results_df: pd.DataFrame, plots_dir: Path = PLOTS_DIR) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    _setup_thesis_style()

    function_summary = _summary_by_epoch(results_df, "function_name")
    _plot_grouped_lines(
        function_summary,
        "function_name",
        "AUPRC Through Epochs by Synthetic Function",
        plots_dir / "auprc_by_function.pgf",
        ylabel=r"AUPRC ($2^{[10]}$)",
    )

    optimizer_summary = _summary_by_epoch(results_df, "optimizer")
    _plot_grouped_lines(
        optimizer_summary,
        "optimizer",
        "AUPRC Through Epochs by Optimizer",
        plots_dir / "auprc_by_optimizer.pgf",
        ylabel=r"AUPRC ($2^{[10]}$)",
    )

    noise_summary = _summary_by_epoch(results_df, "noise")
    _plot_grouped_lines(
        noise_summary,
        "noise",
        "AUPRC Through Epochs by Noise Level",
        plots_dir / "auprc_by_noise.pgf",
        ylabel=r"AUPRC ($2^{[10]}$)",
    )

    regularization_summary = _summary_by_epoch(results_df, "regularization_category")
    _plot_grouped_lines(
        regularization_summary,
        "regularization_category",
        "AUPRC Through Epochs by Regularization Technique",
        plots_dir / "auprc_by_regularization.pgf",
        ylabel=r"AUPRC ($2^{[10]}$)",
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate epoch-level AUPRC summary plots")
    parser.add_argument(
        "--results-root",
        default="results",
        help="Root folder containing auroc_by_epoch.csv (default: results)",
    )
    parser.add_argument(
        "--model-size",
        default="small",
        choices=["small", "big"],
        help="Model size to plot (default: small)",
    )
    args = parser.parse_args()
    results_root = Path(args.results_root)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading epoch-level AUPRC results from {results_root / 'auroc_by_epoch.csv'}...")
    results_df = collect_epoch_level_results(results_path=results_root / "auroc_by_epoch.csv")
    print(f"Loaded {len(results_df)} rows.")

    if "model_size" in results_df.columns:
        results_df = results_df[results_df["model_size"] == args.model_size]

    if "regularization_category" not in results_df.columns and "dropout" in results_df.columns:
        results_df["regularization_category"] = [
            _regularization_category(float(d), bool(w))
            for d, w in zip(results_df["dropout"], results_df["weight_decay"])
        ]

    print("Generating plots...")
    generate_plots(results_df)
    print(f"Saved plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
