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


SNAPSHOT_ROOT = Path("snapshots")
RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
EPOCH_RESULTS_CSV = RESULTS_DIR / "auroc_by_epoch.csv"


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
    summary = (
        df.dropna(subset=["auroc"])
        .groupby([group_column, "epoch"], as_index=False)["auroc"]
        .agg(mean="mean", std="std", min="min", max="max", count="count")
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
    if summary_df.empty:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    groups = list(summary_df[group_column].dropna().unique())
    groups.sort(key=lambda value: _group_sort_key(group_column, value))

    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.get_cmap("tab10")

    for index, group_name in enumerate(groups):
        group_df = summary_df[summary_df[group_column] == group_name].sort_values("epoch")
        if group_df.empty:
            continue

        epochs = group_df["epoch"].to_numpy(dtype=float)
        means = group_df["mean"].to_numpy(dtype=float)
        lower = group_df["lower"].to_numpy(dtype=float)
        upper = group_df["upper"].to_numpy(dtype=float)

        color = cmap(index % cmap.N)
        ax.plot(epochs, means, label=str(group_name), color=color, linewidth=2)
        ax.fill_between(epochs, lower, upper, color=color, alpha=0.18, linewidth=0)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_plots(results_df: pd.DataFrame, plots_dir: Path = PLOTS_DIR) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    function_summary = _summary_by_epoch(results_df, "function_name")
    _plot_grouped_lines(
        function_summary,
        "function_name",
        "AUROC Through Epochs by Synthetic Function",
        plots_dir / "auroc_by_function.png",
    )

    optimizer_summary = _summary_by_epoch(results_df, "optimizer")
    _plot_grouped_lines(
        optimizer_summary,
        "optimizer",
        "AUROC Through Epochs by Optimizer",
        plots_dir / "auroc_by_optimizer.png",
    )

    noise_summary = _summary_by_epoch(results_df, "noise")
    _plot_grouped_lines(
        noise_summary,
        "noise",
        "AUROC Through Epochs by Noise Level",
        plots_dir / "auroc_by_noise.png",
    )

    regularization_summary = _summary_by_epoch(results_df, "regularization_category")
    _plot_grouped_lines(
        regularization_summary,
        "regularization_category",
        "AUROC Through Epochs by Regularization Technique",
        plots_dir / "auroc_by_regularization.png",
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