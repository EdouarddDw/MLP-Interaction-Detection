import pandas as pd
import numpy as np
from pathlib import Path

results_root = Path("results")
frames = []

for model_size, folder in [("small", results_root / "small"), ("big", results_root / "big")]:
    for path in sorted(folder.glob("*_analysis.csv")):
        df = pd.read_csv(path)
        df["model_size"] = model_size
        frames.append(df)

df = pd.concat(frames, ignore_index=True)
df = df[df["success"].fillna(False)].copy()

# Prevalence baseline per row (restricted evaluation)
# n_pos = num_gt, n_neg = num_detected - num_exact_matched (approx)
# Simpler: just report the metric distribution

for col, label in [("auprc", "Restricted AUPRC (G∪D)"), ("auprc_full", "Full-space AUPRC (2^[10])")]:
    if col not in df.columns:
        print(f"{col} not found in CSVs — rerun analysis.py to generate it")
        continue
    vals = df[col].dropna()
    print(f"\n{label}")
    print(f"  Count:   {len(vals)}")
    print(f"  Mean:    {vals.mean():.4f}")
    print(f"  Median:  {vals.median():.4f}")
    print(f"  Std:     {vals.std():.4f}")
    print(f"  Min:     {vals.min():.4f}")
    print(f"  Max:     {vals.max():.4f}")
    print(f"  25th pct: {vals.quantile(0.25):.4f}")
    print(f"  75th pct: {vals.quantile(0.75):.4f}")

# Breakdown by regularization state
print("\n── Restricted AUPRC by regularization ──")
if "auprc" in df.columns and "regularization_state" in df.columns:
    print(df.groupby("regularization_state")["auprc"].agg(
        mean="mean", std="std", median="median", count="count"
    ).round(4).to_string())

print("\n── Restricted AUPRC by noise level ──")
if "auprc" in df.columns and "noise" in df.columns:
    print(df.groupby("noise")["auprc"].agg(
        mean="mean", std="std", median="median", count="count"
    ).round(4).to_string())
