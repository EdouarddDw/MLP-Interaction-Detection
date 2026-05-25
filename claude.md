# Project context

This is a thesis project studying interaction recovery in neural networks
using NID (Neural Interaction Detection) across synthetic functions with
known ground truth interactions. The core question is how regularization
(dropout, weight decay) and input noise jointly affect the fidelity and
trajectory of interaction recovery.

# Codebase structure

- `analysis.py` — loads model checkpoints, runs NID, computes AUROC metrics
- `epoch_results.py` — collects per-epoch AUROC across all checkpoints
- `thesis_plots.py` — all plotting and table generation for the thesis
- `synth.py` — synthetic functions f1–f10 with known ground truth interactions
- `train.py` — model training loop
- `config.py` — experiment definitions (noise, optimizer, dropout, weight_decay)
- `NID.py` — Neural Interaction Detection implementation
- `multilayer_perceptron.py` — MLP model definition

# Key data structures

- Ground truth interactions: `set` of `frozenset` of 1-indexed feature ints
- NID interactions: list of `(tuple_of_0indexed_features, strength_float)` pairs
- Results CSVs: one per function under `results/small/` and `results/big/`
- Trajectory CSVs: per epoch under `results/trajectories/`
- Epoch-level results: `results/auroc_by_epoch.csv`
- Snapshots: `snapshots/{function_name}/{experiment_name}/best_epoch_XXXX.pt`

# Conventions to always follow

- AUROC is evaluated over G ∪ D (ground truth union NID detections), not
  all 2^10 subsets, unless --full-space flag is passed
- Subsets of ground truth interactions detected by NID are excluded from
  negatives in AUROC computation (they are neither TP nor FP)
- Two-stage aggregation: always average within function first, then across
  functions, to prevent any single function dominating results
- Feature indices: GT interactions are 1-indexed, NID outputs are 0-indexed;
  always convert NID to 1-indexed before comparing with GT
- Model sizes: "small" = [64, 64] hidden units, "big" = [140, 100, 60, 20]
- Regularization states: "base", "dropout", "weight_decay",
  "dropout+weight_decay" — always use these exact strings internally and
  _pretty_regularization_label() for display

# Plot conventions

- Always use _setup_thesis_style() once before any plotting, not per function
- Figure sizes: (3.5, 2.6) single column, (7.0, 3.5) double column
- Colors: small=#2A6F97, big=#E76F51, mean lines=#2A6F97
- Always save via _save_pgf() which writes both .pgf and .png
- Never hardcode dpi in savefig calls — let rcParams control it
- Legends outside the plot area where possible

# What not to change

- Never modify compute_metrics(), match_interactions_one_to_one(),
  or load_model_and_interactions() without explicit instruction
- Never change the default behaviour of compute_auroc_data()
  (full_space=False by default)
- Never change existing CSV column names — downstream plots depend on them
