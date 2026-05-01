#!/usr/bin/env python
"""Generate summary statistics from AUROC analysis results"""

import pandas as pd
import numpy as np
from pathlib import Path

results_dir = Path("results")
csv_files = sorted(results_dir.glob("f*_analysis.csv"))

print("="*80)
print("AUROC ANALYSIS SUMMARY - NID Interaction Detection")
print("="*80)
print()

# Aggregate all results
all_results = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    all_results.append(df)

combined_df = pd.concat(all_results, ignore_index=True)

# Overall statistics
total_experiments = len(combined_df)
successful = combined_df['success'].sum()
auroc_values = combined_df['auroc'].dropna()

print(f"Total experiments analyzed: {total_experiments}")
print(f"Successful analyses: {successful} ({100*successful/total_experiments:.1f}%)")
print()

print("AUROC STATISTICS:")
print(f"  Samples with AUROC computed: {len(auroc_values)} ({100*len(auroc_values)/total_experiments:.1f}%)")
if len(auroc_values) > 0:
    print(f"  Mean AUROC:     {auroc_values.mean():.4f}")
    print(f"  Median AUROC:   {auroc_values.median():.4f}")
    print(f"  Std Dev:        {auroc_values.std():.4f}")
    print(f"  Min AUROC:      {auroc_values.min():.4f}")
    print(f"  Max AUROC:      {auroc_values.max():.4f}")
    print()

# Statistics by function
print("AUROC BY FUNCTION:")
for func in combined_df['function_name'].unique():
    func_df = combined_df[combined_df['function_name'] == func]
    func_aurocs = func_df['auroc'].dropna()
    if len(func_aurocs) > 0:
        mean_auroc = func_aurocs.mean()
        print(f"  {func}: {mean_auroc:.4f} (mean, n={len(func_aurocs)})")

print()

# Statistics by noise level
print("AUROC BY NOISE LEVEL:")
for noise in sorted(combined_df['noise'].unique()):
    noise_df = combined_df[combined_df['noise'] == noise]
    noise_aurocs = noise_df['auroc'].dropna()
    if len(noise_aurocs) > 0:
        mean_auroc = noise_aurocs.mean()
        print(f"  Noise {noise}: {mean_auroc:.4f} (mean, n={len(noise_aurocs)})")

print()

# Statistics by optimizer
print("AUROC BY OPTIMIZER:")
for opt in combined_df['optimizer'].unique():
    opt_df = combined_df[combined_df['optimizer'] == opt]
    opt_aurocs = opt_df['auroc'].dropna()
    if len(opt_aurocs) > 0:
        mean_auroc = opt_aurocs.mean()
        print(f"  {opt.upper()}: {mean_auroc:.4f} (mean, n={len(opt_aurocs)})")

print()

# Top performers
print("TOP 15 AUROC PERFORMANCES:")
top_df = combined_df.nlargest(15, 'auroc')[['function_name', 'experiment_name', 'auroc', 'num_gt', 'num_detected', 'num_matched']]
for idx, row in enumerate(top_df.itertuples(), 1):
    print(f"  {idx:2d}. {row.function_name}/{row.experiment_name:20s} AUROC={row.auroc:.4f} (GT:{row.num_gt} Detected:{row.num_detected} Matched:{row.num_matched})")

print()

# Worst performers (with valid AUROC)
print("BOTTOM 15 AUROC PERFORMANCES (with valid scores):")
bottom_df = combined_df[combined_df['auroc'].notna()].nsmallest(15, 'auroc')[['function_name', 'experiment_name', 'auroc', 'num_gt', 'num_detected', 'num_matched']]
for idx, row in enumerate(bottom_df.itertuples(), 1):
    print(f"  {idx:2d}. {row.function_name}/{row.experiment_name:20s} AUROC={row.auroc:.4f} (GT:{row.num_gt} Detected:{row.num_detected} Matched:{row.num_matched})")

print()
print("="*80)
print(f"Detailed results saved in: results/")
print("  - f1_analysis.csv through f10_analysis.csv (one file per function)")
print("="*80)
