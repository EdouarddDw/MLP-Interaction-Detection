# MLP Interaction Detection

![Python](https://img.shields.io/badge/python-3.9%2B-blue)

Empirical study of how regularization and input noise affect interaction recovery in multilayer perceptrons, evaluated against synthetic ground truth using NID.

## Background

Neural Interaction Detection (NID) is a post-hoc method that infers feature interactions from a trained MLP by analysing the product of weight matrices across layers. Interaction recovery refers to how faithfully NID can reconstruct the ground-truth interactions embedded in a synthetic function. This repository is the codebase for a Bachelor's thesis at Maastricht University, School of Business and Economics, investigating how regularization techniques (dropout, weight decay) and input noise jointly affect the fidelity and trajectory of interaction recovery across training.

## Repository structure

```
MLP-Interaction-Detection/
│
├── train.py                   # Training loop, snapshot saving, provenance logging
├── analysis.py                # Load checkpoints, run NID, compute AUROC/AUPRC metrics
├── plot_analysis.py           # Epoch-level AUROC collection and trajectory plots
├── thesis_plots.py            # All publication figures and tables for the thesis
├── config.py                  # Experiment grid (noise × optimizer × dropout × weight decay)
├── synth.py                   # Synthetic functions f1–f10 with known ground truth
├── NID.py                     # Neural Interaction Detection implementation
├── multilayer_perceptron.py   # MLP model definition
├── missing_exp.py             # Utility to identify incomplete experimental runs
│
├── snapshots/                 # Trained model checkpoints (per function / experiment / run)
├── results/
│   ├── small/                 # Per-function analysis CSVs for small architecture
│   ├── big/                   # Per-function analysis CSVs for big architecture
│   ├── trajectories/          # Per-epoch AUROC trajectories
│   ├── auroc_by_epoch.csv     # Aggregated epoch-level results
│   └── thesis_plots/          # Generated figures (.pgf + .png)
└── requirements.txt
```

## Setup

Requires Python 3.9 or later.

```bash
pip install -r requirements.txt
```

[SciencePlots](https://github.com/garrettj403/SciencePlots) is optional but recommended for publication-quality figures. Install it with:

```bash
pip install SciencePlots
```

If SciencePlots is not available, `thesis_plots.py` falls back to `seaborn-v0_8-whitegrid` automatically.

## Reproducing the experiments

### 1. Train all models

```bash
# Small architecture [64, 64] — default
python train.py

# Big architecture [140, 100, 60, 20]
python train.py --hidden_units "140,100,60,20"

# Optional flags
#   --num_samples N         training set size (default: 30000)
#   --weight-decay-only     re-run only weight-decay experiments
#   --l2-const FLOAT        override L2 regularization constant (default: 1e-4)
```

### 2. Run analysis

```bash
python analysis.py

# Optional flags
#   --snapshot-root PATH    root folder for snapshots (default: snapshots/)
#   --num-samples N         samples used for GT generation (default: 30000)
#   --seed INT              random seed (default: 42)
#   --run-id ID             select a specific training run by ID
#   --clean                 delete and recreate results/ subdirs before running
```

### 3. Collect epoch-level results

```bash
python plot_analysis.py
```

### 4. Generate thesis plots

```bash
python thesis_plots.py

# Optional flags
#   --results-root PATH       root folder with big/small CSVs (default: results/)
#   --output-dir PATH         output directory for figures (default: results/thesis_plots/)
#   --trajectories-dir PATH   directory with trajectory CSVs (default: results/trajectories/)
```

## Experiment design

The experiment grid crosses 5 noise levels (σ ∈ {0, 0.1, 0.2, 0.5, 1.0}, as a fraction of the target standard deviation), 2 optimizers (Adam, SGD), 2 dropout settings (none, p = 0.2), and 2 weight-decay settings (disabled, enabled), giving 40 conditions per function. Applied across 10 synthetic functions and 2 architectures (small: [64, 64], big: [140, 100, 60, 20]), this yields 800 total runs. The L2 regularization constant defaults to 1e-4 and can be overridden via `--l2-const`.

## Evaluation

AUROC and AUPRC are computed over the restricted evaluation space G ∪ D — the union of ground-truth interactions (G) and NID-detected interactions (D) — rather than the full 2^10 power set. This avoids inflating scores with the large number of trivially absent interactions. Subsets of ground-truth interactions that are proper subsets of a detected interaction are excluded from negatives. Full-space metrics are also stored as `auroc_full` and `auprc_full` in the result CSVs for comparison.

## Key results

- Unregularized base conditions consistently achieve higher AUROC than regularized conditions across all noise levels and both architectures.
- Weight decay suppresses interaction recovery, with the effect intensifying at larger L2 constants; the trajectory of recovery is slowed rather than merely capped.
- Input noise lowers the performance ceiling but does not qualitatively alter the relative ranking of regularization conditions.

## AI-assisted development

A `CLAUDE.md` file is included in the repository with project conventions and codebase context used during AI-assisted development with Claude Code.


