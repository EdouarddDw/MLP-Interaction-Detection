import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def load_analysis_results(results_root: Path) -> pd.DataFrame:
    """
    Loads all *_analysis.csv files from results/small and results/big.
    """

    all_files = list(results_root.glob("small/*_analysis.csv")) + list(
        results_root.glob("big/*_analysis.csv")
    )

    if not all_files:
        raise FileNotFoundError(
            f"No analysis CSV files found in {results_root}/small or {results_root}/big"
        )

    dfs = []
    for file in all_files:
        df = pd.read_csv(file)
        df["source_file"] = str(file)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def normalize_weight_decay_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Makes weight_decay comparable even if stored as True/False, yes/no, 0/1,
    or numeric values.
    """

    df = df.copy()

    def parse_wd(x):
        if pd.isna(x):
            return False

        if isinstance(x, bool):
            return x

        x_str = str(x).strip().lower()

        if x_str in ["true", "yes", "enabled", "wd", "1"]:
            return True

        if x_str in ["false", "no", "disabled", "none", "0", "0.0"]:
            return False

        try:
            return float(x_str) > 0
        except ValueError:
            return False

    df["weight_decay_bool"] = df["weight_decay"].apply(parse_wd)
    return df


def compare_weight_decay(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compares each weight decay run against the matching base run.

    Matching keys:
    function_name, noise, optimizer, dropout, model_size
    """

    df = df.copy()

    if "success" in df.columns:
        df = df[df["success"] == True].copy()

    df = normalize_weight_decay_column(df)

    required_cols = [
        "function_name",
        "noise",
        "optimizer",
        "dropout",
        "model_size",
        "weight_decay_bool",
        "auprc_full",
        "auroc",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "val_loss" not in df.columns:
        df["val_loss"] = np.nan

    group_keys = [
        "function_name",
        "noise",
        "optimizer",
        "dropout",
        "model_size",
    ]

    base = df[df["weight_decay_bool"] == False].copy()
    wd = df[df["weight_decay_bool"] == True].copy()

    base = base.rename(
        columns={
            "experiment": "base_experiment",
            "auprc_full": "base_auprc_full",
            "auroc": "base_auroc",
            "val_loss": "base_val_loss",
        }
    )

    wd = wd.rename(
        columns={
            "experiment": "wd_experiment",
            "auprc_full": "wd_auprc_full",
            "auroc": "wd_auroc",
            "val_loss": "wd_val_loss",
        }
    )

    comparison = wd.merge(
        base[
            group_keys
            + [
                "base_experiment",
                "base_auprc_full",
                "base_auroc",
                "base_val_loss",
            ]
        ],
        on=group_keys,
        how="inner",
    )

    comparison["delta_auprc_full"] = (
        comparison["wd_auprc_full"] - comparison["base_auprc_full"]
    )

    comparison["delta_auroc"] = comparison["wd_auroc"] - comparison["base_auroc"]

    comparison["delta_val_loss"] = (
        comparison["wd_val_loss"] - comparison["base_val_loss"]
    )

    comparison["wd_better_auprc"] = comparison["delta_auprc_full"] > 0
    comparison["wd_better_auroc"] = comparison["delta_auroc"] > 0

    # For validation loss, lower is better
    comparison["wd_better_val_loss"] = comparison["delta_val_loss"] < 0

    comparison["wd_meaningful_auprc_gain"] = comparison["delta_auprc_full"] >= 0.01
    comparison["wd_meaningful_auprc_loss"] = comparison["delta_auprc_full"] <= -0.01

    return comparison


def summarize_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    """
    Creates aggregate summary by model size, optimizer, dropout, and noise.
    """

    group_cols = ["model_size", "optimizer", "dropout", "noise"]

    summary = (
        comparison.groupby(group_cols)
        .agg(
            n_comparisons=("delta_auprc_full", "count"),
            mean_delta_auprc=("delta_auprc_full", "mean"),
            std_delta_auprc=("delta_auprc_full", "std"),
            median_delta_auprc=("delta_auprc_full", "median"),
            pct_wd_better_auprc=("wd_better_auprc", "mean"),
            pct_meaningful_gain=("wd_meaningful_auprc_gain", "mean"),
            pct_meaningful_loss=("wd_meaningful_auprc_loss", "mean"),
            mean_delta_auroc=("delta_auroc", "mean"),
            pct_wd_better_auroc=("wd_better_auroc", "mean"),
            mean_delta_val_loss=("delta_val_loss", "mean"),
            pct_wd_better_val_loss=("wd_better_val_loss", "mean"),
        )
        .reset_index()
    )

    percentage_cols = [
        "pct_wd_better_auprc",
        "pct_meaningful_gain",
        "pct_meaningful_loss",
        "pct_wd_better_auroc",
        "pct_wd_better_val_loss",
    ]

    for col in percentage_cols:
        summary[col] = 100 * summary[col]

    return summary


def global_summary(comparison: pd.DataFrame) -> None:
    print("\nGlobal weight decay comparison")
    print("=" * 40)

    n = len(comparison)

    print(f"Number of direct matched comparisons: {n}")

    if n == 0:
        print("No matching base vs weight decay pairs found.")
        return

    mean_delta = comparison["delta_auprc_full"].mean()
    median_delta = comparison["delta_auprc_full"].median()
    pct_better = 100 * comparison["wd_better_auprc"].mean()
    pct_gain = 100 * comparison["wd_meaningful_auprc_gain"].mean()
    pct_loss = 100 * comparison["wd_meaningful_auprc_loss"].mean()

    print(f"Mean AUPRC difference, WD minus base: {mean_delta:.4f}")
    print(f"Median AUPRC difference, WD minus base: {median_delta:.4f}")
    print(f"WD better than base in AUPRC: {pct_better:.1f}% of comparisons")
    print(f"WD has meaningful AUPRC gain >= 0.01: {pct_gain:.1f}%")
    print(f"WD has meaningful AUPRC loss <= -0.01: {pct_loss:.1f}%")

    if "delta_val_loss" in comparison.columns:
        mean_loss_delta = comparison["delta_val_loss"].mean()
        pct_loss_better = 100 * comparison["wd_better_val_loss"].mean()

        print(f"Mean validation loss difference, WD minus base: {mean_loss_delta:.6f}")
        print(f"WD better than base in validation loss: {pct_loss_better:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=str,
        default="results",
        help="Folder containing small/ and big/ analysis CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/weight_decay_comparison",
        help="Where to save comparison CSV files.",
    )

    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_analysis_results(results_root)
    comparison = compare_weight_decay(df)
    summary = summarize_comparison(comparison)

    comparison_path = output_dir / "weight_decay_direct_comparisons.csv"
    summary_path = output_dir / "weight_decay_summary.csv"

    comparison.to_csv(comparison_path, index=False)
    summary.to_csv(summary_path, index=False)

    global_summary(comparison)

    print("\nSaved files:")
    print(f"Direct comparisons: {comparison_path}")
    print(f"Summary:            {summary_path}")

    print("\nBest positive AUPRC gains:")
    cols_to_show = [
        "function_name",
        "model_size",
        "noise",
        "optimizer",
        "dropout",
        "base_experiment",
        "wd_experiment",
        "base_auprc_full",
        "wd_auprc_full",
        "delta_auprc_full",
    ]

    print(
        comparison.sort_values("delta_auprc_full", ascending=False)[cols_to_show]
        .head(10)
        .to_string(index=False)
    )

    print("\nWorst AUPRC drops:")
    print(
        comparison.sort_values("delta_auprc_full", ascending=True)[cols_to_show]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
