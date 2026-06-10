# MLP Interaction Detection

![Python](https://img.shields.io/badge/python-3.9%2B-blue)

Empirical study of how regularization and input noise affect interaction recovery in multilayer perceptrons, evaluated against synthetic ground truth using NID.

## Background

Neural Interaction Detection (NID) is a post-hoc method that infers feature interactions from a trained MLP by analysing the product of weight matrices across layers. Interaction recovery refers to how faithfully NID can reconstruct the ground-truth interactions embedded in a synthetic function. This repository is the codebase for a Bachelor's thesis at Maastricht University, School of Business and Economics, investigating how regularization techniques (dropout, weight decay) and input noise jointly affect the fidelity and trajectory of interaction recovery across training.

## Repository structure

```
MLP-Interaction-Detection/
│
├── train.py                   # Training loop, snapshot saving, probe sweep mode
├── analysis.py                # Load checkpoints, run NID, compute AUROC/AUPRC metrics
├── plot_analysis.py           # Epoch-level AUROC summary plots from auroc_by_epoch.csv
├── thesis_plots.py            # All publication figures and probe dose-response plots
├── stats_analysis.py          # LaTeX tables + inline \newcommand stats → results/stats/
├── significance.py            # Multi-seed replication + significance testing (Wilcoxon, Friedman, Spearman)
├── config.py                  # Experiment grid (noise × optimizer × dropout × weight decay)
├── synth.py                   # Synthetic functions f1–f10 with known ground truth
├── NID.py                     # Neural Interaction Detection implementation
├── multilayer_perceptron.py   # MLP model definition
├── migrate_snapshots.py       # One-off migration of snapshots/ → snapshots_clean/
├── missing_exp.py             # Lists known-missing experiment conditions
├── list_runs.py               # Utility: list available run_id subdirs for an experiment
├── print_summary.py           # Quick AUROC distribution summary to stdout
├── stats_AUPRC.py             # Standalone AUPRC distribution stats to stdout
├── wd_effective.py            # Weight-decay effective-regularization analysis
│
├── snapshots_clean/           # Canonical checkpoints — model_size/function/experiment/run_id/
├── snapshots_probe/           # Checkpoints from dropout/L2 probe sweeps
├── results/
│   ├── small/                 # Per-function analysis CSVs for small architecture
│   ├── big/                   # Per-function analysis CSVs for big architecture
│   ├── trajectories/          # Per-epoch AUROC/loss trajectory CSVs
│   ├── plots/                 # Epoch-level summary plots (plot_analysis.py output)
│   ├── stats/                 # LaTeX tables and \newcommand stats (stats_analysis.py)
│   ├── auroc_by_epoch.csv     # Aggregated epoch-level results
│   └── thesis_plots/          # Generated figures (.pgf + .png)
├── results_probe/             # Analysis CSVs for probe sweep experiments
├── results_multiseed/
│   └── stats/                 # Significance test outputs (Wilcoxon, Friedman, Spearman)
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

#### Probe sweep mode (regularization dose-response)

```bash
# Dropout sweep — default grid (p ∈ {0, 0.1, 0.2, 0.3, 0.5})
python train.py --dropout-sweep

# L2 sweep — default grid (λ ∈ {0, 1e-5, 1e-4, 1e-3, 1e-2})
python train.py --l2-sweep

# Custom grids
python train.py --dropout-sweep "0.0,0.1,0.4"
python train.py --l2-sweep "0,1e-4,1e-2"

# Additional sweep flags
#   --functions F1,F2,...   functions to sweep (default: f1,f7,f8)
#   --snapshot-root PATH    snapshot output directory (default: snapshots_probe)
#   --no-mps                force CPU even if MPS is available
```

Sweep checkpoints land in `snapshots_probe/` and a `sweep_experiments.json` file is written for use with `analysis.py --extra-experiments`.

### 2. Run analysis

```bash
python analysis.py

# Optional flags
#   --snapshots-root PATH    root folder for snapshots (default: snapshots_clean)
#   --results-root PATH      output directory for CSVs (default: results)
#   --model-size {small,big} filter to one architecture
#   --function fN            filter to one function (f1–f10)
#   --experiment NAME        filter to one experiment name from config.EXPERIMENTS
#   --best-only              only analyze best-epoch checkpoints
#   --trajectories-only      only generate trajectory CSVs
#   --dry-run                print planned work without writing files
#   --overwrite              overwrite existing trajectory CSVs
#   --extra-experiments FILE JSON file of additional experiments (e.g. sweep_experiments.json)
```

To analyze probe sweep results, point `--snapshots-root` and `--results-root` at the probe directories:

```bash
python analysis.py --snapshots-root snapshots_probe --results-root results_probe \
  --extra-experiments sweep_experiments.json
```

### 3. Collect epoch-level results

```bash
python plot_analysis.py
```

### 4. Generate thesis plots

```bash
python thesis_plots.py

# Optional flags
#   --results-root PATH         root folder with big/small CSVs (default: results/)
#   --output-dir PATH           output directory for figures (default: results/thesis_plots/)
#   --trajectories-dir PATH     directory with trajectory CSVs (default: results/trajectories/)
#   --prob                      generate probe dose-response figures instead of full thesis suite
#   --probe-results-root PATH   root folder for probe CSVs (default: results_probe)
```

### 5. Generate statistical tables

```bash
python stats_analysis.py

# Optional flags
#   --results-root PATH         root folder with big/small CSVs (default: results)
#   --trajectories-dir PATH     trajectories directory for convergence filter (default: results/trajectories)
```

Outputs LaTeX tables and inline `\newcommand` stats to `results/stats/`.

### 6. Run significance tests (multi-seed)

```bash
python significance.py

# Optional flags
#   --seeds N1,N2,...           seeds to replicate over (default: 42,1,2,3,4)
#   --results-root PATH         root folder with big/small CSVs (default: results)
#   --output-dir PATH           output directory for test results (default: results_multiseed/stats)
#   --noise-levels N1,N2,...    noise levels to test (default: 0,0.1,0.2,0.5,1.0)
#   --full                      use all five noise levels for Claim D contrast (default: core mode η ∈ {0,0.5,1.0})
```

Runs paired Wilcoxon signed-rank tests for Claims A–C (dropout, dropout+weight decay, weight decay vs base), a Friedman test and directional noise contrast for Claim D, and Spearman ρ for Claim E (AUPRC vs validation loss). Applies Holm–Bonferroni correction to Claims A–C jointly. Outputs go to `results_multiseed/stats/`.

## Experiment design

The experiment grid crosses 5 noise levels (σ ∈ {0, 0.1, 0.2, 0.5, 1.0}, as a fraction of the target standard deviation), 2 optimizers (Adam, SGD), 2 dropout settings (none, p = 0.2), and 2 weight-decay settings (disabled, enabled), giving 40 conditions per function. Applied across 10 synthetic functions and 2 architectures (small: [64, 64], big: [140, 100, 60, 20]), this yields 800 total runs. The L2 regularization constant defaults to 1e-4 and can be overridden via `--l2-const`.

## Evaluation

AUROC and AUPRC are computed over the restricted evaluation space G ∪ D — the union of ground-truth interactions (G) and NID-detected interactions (D) — rather than the full 2^10 power set. This avoids inflating scores with the large number of trivially absent interactions. Subsets of ground-truth interactions that are proper subsets of a detected interaction are excluded from negatives. Full-space metrics are also stored as `auroc_full` and `auprc_full` in the result CSVs for comparison.

## Key results

- **Dropout is the primary threat to interaction recovery.** Averaged across all noise levels, functions, and architectures, dropout reduces mean AUPRC by 22.9% relative to the unregularised baseline. Weight decay is considerably more benign at −7.1%. Standard regularisation practice therefore transfers poorly to the interaction detection setting.
- **The dropout + weight decay combination is more harmful than either alone**, particularly at noise levels 0 and 1 in multi-seed analysis — suggesting the effects are at least partially cumulative rather than redundant.
- **Output noise lowers the recovery plateau without altering the early trajectory shape.** Interaction structure is encoded during the first 10–20 epochs of training, before noise fitting begins to compete for representational capacity. Recovery declines gradually through η = 0.2 and then more steeply, with the largest single drop between η = 0.2 and η = 0.5.
- **Validation loss is a weak proxy for interaction recovery quality** (Spearman ρ, p < 0.001). Standard loss-based model selection does not reliably identify models with strong interaction structure.
- **Smaller architectures match or outperform the larger model** on the majority of benchmark functions. The larger model's marginal advantage in the noise-free setting disappears at every positive noise level, with the largest gaps appearing under adverse conditions.

## AI-assisted development

A `CLAUDE.md` file is included in the repository with project conventions and codebase context used during AI-assisted development with Claude Code.
