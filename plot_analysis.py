#!/usr/bin/env python
"""Compute epoch-level AUROC results and generate summary plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis import (
    compute_auroc_data,
    compute_metrics,
    get_ground_truth_interactions,
    load_model_and_interactions,
)
from config import EXPERIMENTS
from synth import functions
from thesis_plots import _save_pgf, _setup_thesis_style


SNAPSHOT_ROOT = Path("snapshots")
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


def _sorted_epoch_checkpoints(snapshot_dir: Path) -> list[Path]:
    checkpoint_paths = list(snapshot_dir.glob("epoch_*.pt"))

    def _epoch_key(path: Path) -> int:
        stem = path.stem
        try:
            return int(stem.split("_")[-1])
        except ValueError:
            return -1

    return sorted(checkpoint_paths, key=_epoch_key)


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


def _get_ground_truth(function, noise: float, num_samples: int, seed: int):
    return get_ground_truth_interactions(function, num_samples=num_samples, noise=noise, seed=seed)


def collect_epoch_level_results(snapshot_root=SNAPSHOT_ROOT, num_samples: int = 30000, seed: int = 42) -> pd.DataFrame:
    rows = []
    gt_cache: dict[tuple[str, float, int, int], set[frozenset[int]]] = {}

    for function in functions:
        function_name = function.__name__

        for experiment in EXPERIMENTS:
            experiment_name = experiment["name"]
            noise = experiment["noise"]
            optimizer = experiment["optimizer"]
            dropout = float(experiment["dropout"])
            weight_decay = bool(experiment["weight_decay"])

            snapshot_dir = Path(snapshot_root) / function_name / experiment_name
            checkpoint_paths = _sorted_epoch_checkpoints(snapshot_dir)
            if not checkpoint_paths:
                continue

            gt_key = (function_name, noise, num_samples, seed)
            if gt_key not in gt_cache:
                gt_cache[gt_key] = _get_ground_truth(function, noise, num_samples, seed)
            gt_interactions = gt_cache[gt_key]

            for checkpoint_path in checkpoint_paths:
                model, nid_interactions, val_loss, epoch = load_model_and_interactions(
                    checkpoint_path,
                    dropout=dropout,
                )

                row = {
                    "function_name": function_name,
                    "experiment_name": experiment_name,
                    "optimizer": optimizer,
                    "noise": noise,
                    "dropout": dropout,
                    "weight_decay": weight_decay,
                    "regularization_category": _regularization_category(dropout, weight_decay),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "snapshot_path": str(checkpoint_path),
                    "auroc": None,
                    "precision": None,
                    "recall": None,
                    "success": False,
                }

                if model is None or nid_interactions is None:
                    rows.append(row)
                    continue

                # Epoch-level trajectories use restricted AUROC only; auroc_full is not
                # written to auroc_by_epoch.csv and is not needed by this module.
                scores, labels = compute_auroc_data(gt_interactions, nid_interactions)
                if scores is None:
                    rows.append({**row, "success": True})
                    continue

                metrics = compute_metrics(scores, labels)
                row.update(
                    {
                        "auroc": metrics["auroc"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "success": True,
                    }
                )
                rows.append(row)

    results_df = pd.DataFrame(rows)
    if not results_df.empty:
        results_df = results_df.sort_values(
            ["function_name", "experiment_name", "epoch"],
            kind="mergesort",
        ).reset_index(drop=True)
    return results_df


def save_epoch_results(results_df: pd.DataFrame, output_path: Path = EPOCH_RESULTS_CSV) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)


def _summary_by_epoch(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    clean = df.dropna(subset=["auroc"])
    function_level = (
        clean.groupby(["function_name", group_column, "epoch"], as_index=False)["auroc"]
        .mean()
        .rename(columns={"auroc": "function_mean_auroc"})
    )
    summary = (
        function_level.groupby([group_column, "epoch"], as_index=False)["function_mean_auroc"]
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
    ylabel: str = "AUROC",
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
        "AUROC Through Epochs by Synthetic Function",
        plots_dir / "auroc_by_function.pgf",
    )

    optimizer_summary = _summary_by_epoch(results_df, "optimizer")
    _plot_grouped_lines(
        optimizer_summary,
        "optimizer",
        "AUROC Through Epochs by Optimizer",
        plots_dir / "auroc_by_optimizer.pgf",
    )

    noise_summary = _summary_by_epoch(results_df, "noise")
    _plot_grouped_lines(
        noise_summary,
        "noise",
        "AUROC Through Epochs by Noise Level",
        plots_dir / "auroc_by_noise.pgf",
    )

    regularization_summary = _summary_by_epoch(results_df, "regularization_category")
    _plot_grouped_lines(
        regularization_summary,
        "regularization_category",
        "AUROC Through Epochs by Regularization Technique",
        plots_dir / "auroc_by_regularization.pgf",
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Collecting epoch-level AUROC results...")
    results_df = collect_epoch_level_results()
    print(f"Collected {len(results_df)} checkpoint rows.")

    save_epoch_results(results_df)
    print(f"Saved epoch-level results to {EPOCH_RESULTS_CSV}")

    print("Generating plots...")
    generate_plots(results_df)
    print(f"Saved plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()