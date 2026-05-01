import csv
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from NID import get_interactions
from multilayer_perceptron import MLP, get_weights
from synth import functions
from train import make_data_loaders
from config import EXPERIMENTS


device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


def load_model_and_interactions(snapshot_path, num_features=10, dropout=0.0):
    """
    Load a trained model from checkpoint and extract NID interactions.
    
    Args:
        snapshot_path: Path to best_epoch_XXXX.pt checkpoint
        num_features: Number of input features (default 10)
        dropout: Dropout rate (should match the training configuration)
    
    Returns:
        tuple: (model, nid_interactions, best_loss, epoch)
            - model: loaded MLP model
            - nid_interactions: list of ((feature_indices), strength) tuples (0-indexed)
            - best_loss: validation loss at best epoch
            - epoch: epoch number
    """
    if not Path(snapshot_path).exists():
        return None, None, None, None
    
    checkpoint = torch.load(snapshot_path, map_location=device)
    
    # Reconstruct model with correct dropout value to match training
    model = MLP(
        num_features=num_features,
        hidden_units=[64, 64],
        use_main_effect_nets=False,
        main_effect_net_units=[10, 10, 10],
        dropout=dropout,  # Use the dropout from experiment settings
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Extract interactions using NID
    weights = get_weights(model)
    if not weights:  # Empty weights means no interaction_mlp layer
        return model, [], checkpoint.get("val_loss", float("inf")), checkpoint.get("epoch", -1)
    
    nid_interactions = get_interactions(weights, pairwise=False, one_indexed=False)
    
    return model, nid_interactions, checkpoint.get("val_loss", float("inf")), checkpoint.get("epoch", -1)


def get_ground_truth_interactions(function, num_samples=30000, noise=0.0, seed=42):
    """
    Get ground truth interactions for a given function.
    
    Args:
        function: Function from synth.py (e.g., f1, f2, ...)
        num_samples: Number of samples to generate
        noise: Noise level (for data generation, but GT is noise-independent)
        seed: Random seed
    
    Returns:
        set of frozensets: Ground truth interactions (1-indexed, as frozensets for hashability)
    """
    _, _, gt_interactions = function(num_samples=num_samples, seed=seed)
    # Convert list of sets to set of frozensets for hashability
    return set(frozenset(inter) for inter in gt_interactions)


def _is_numeric(value):
    return isinstance(value, (int, float, np.integer, np.floating))


def _normalize_gt_interactions(gt_interactions):
    normalized = []
    for interaction in gt_interactions:
        normalized.append(frozenset(int(feature) for feature in interaction))
    return normalized


def _normalize_nid_interactions(nid_interactions):
    normalized = []
    for interaction in nid_interactions:
        if isinstance(interaction, tuple) and len(interaction) == 2 and _is_numeric(interaction[1]):
            features, strength = interaction
            feature_set = frozenset(int(feature) + 1 for feature in features)
            normalized.append((feature_set, float(strength)))
        else:
            feature_set = frozenset(int(feature) for feature in interaction)
            normalized.append((feature_set, 1.0))
    return normalized


def is_exact_or_superset(nid_tuple, gt_set):
    """
    Check if NID-detected interaction is exact match or superset of GT interaction.
    NID indices are 0-indexed; GT sets are 1-indexed.
    
    Args:
        nid_tuple: 0-indexed tuple from NID (e.g., (0, 1, 2))
        gt_set: 1-indexed frozenset from GT (e.g., frozenset({1, 2, 3}))
    
    Returns:
        bool: True if NID detection is exact match or superset of GT
    """
    # Convert NID 0-indexed tuple to 1-indexed set
    nid_set = frozenset(i + 1 for i in nid_tuple)
    return nid_set >= gt_set  # Superset or equal


def match_interactions_one_to_one(gt_interactions, nid_interactions):
    """
    Match ground truth interactions to NID-detected interactions using one-to-one matching.

    Each GT interaction and each NID interaction can be used at most once.
    Exact matches are assigned first.
    Superset matches are assigned second, preferring the smallest valid superset.

    Parameters
    ----------
    gt_interactions : iterable of sets or frozensets
        Ground truth feature interactions, using 1-indexed feature ids.
    nid_interactions : iterable
        Detected NID interactions. This may be either:
        - iterable of sets/frozensets using 1-indexed feature ids
        - iterable of tuples like ((0, 1), strength), where detected features are 0-indexed

    Returns
    -------
    dict with:
        num_exact_matched
        num_superset_matched_unique
        num_matched
        matched_pairs
    """
    normalized_gt = _normalize_gt_interactions(gt_interactions)
    normalized_nid = _normalize_nid_interactions(nid_interactions)

    gt_used = [False] * len(normalized_gt)
    nid_used = [False] * len(normalized_nid)
    matched_pairs = []

    for gt_index, gt_set in enumerate(normalized_gt):
        if gt_used[gt_index]:
            continue
        for nid_index, (nid_set, strength) in enumerate(normalized_nid):
            if nid_used[nid_index]:
                continue
            if nid_set == gt_set:
                gt_used[gt_index] = True
                nid_used[nid_index] = True
                matched_pairs.append(
                    {
                        "gt": gt_set,
                        "nid": nid_set,
                        "match_type": "exact",
                        "extra_features": 0,
                        "strength": strength,
                    }
                )
                break

    candidate_pairs = []
    for gt_index, gt_set in enumerate(normalized_gt):
        if gt_used[gt_index]:
            continue
        for nid_index, (nid_set, strength) in enumerate(normalized_nid):
            if nid_used[nid_index]:
                continue
            if nid_set > gt_set:
                extra_features = len(nid_set) - len(gt_set)
                candidate_pairs.append(
                    (
                        extra_features,
                        -float(strength),
                        len(nid_set),
                        gt_index,
                        nid_index,
                    )
                )

    candidate_pairs.sort()

    for extra_features, neg_strength, _, gt_index, nid_index in candidate_pairs:
        if gt_used[gt_index] or nid_used[nid_index]:
            continue

        gt_set = normalized_gt[gt_index]
        nid_set, strength = normalized_nid[nid_index]

        if nid_set > gt_set:
            gt_used[gt_index] = True
            nid_used[nid_index] = True
            matched_pairs.append(
                {
                    "gt": gt_set,
                    "nid": nid_set,
                    "match_type": "superset",
                    "extra_features": extra_features,
                    "strength": strength,
                }
            )

    num_exact_matched = sum(1 for pair in matched_pairs if pair["match_type"] == "exact")
    num_superset_matched_unique = sum(1 for pair in matched_pairs if pair["match_type"] == "superset")
    num_matched = len(matched_pairs)

    return {
        "num_exact_matched": num_exact_matched,
        "num_superset_matched_unique": num_superset_matched_unique,
        "num_matched": num_matched,
        "matched_pairs": matched_pairs,
    }


def compute_auroc_data(gt_interactions, nid_interactions):
    """
    Create binary labels and scores for AUROC computation.
    
    For each GT interaction:
      - Label = 1 (ground truth positive)
      - Score = NID strength if exact/superset match found, else 0
    
    For each unmatched NID detection (not a superset of any GT):
      - Label = 0 (false positive candidate)
      - Score = NID strength
    
    Args:
        gt_interactions: set of frozensets (1-indexed)
        nid_interactions: list of (tuple, strength) pairs (0-indexed)
    
    Returns:
        tuple: (scores, labels) - arrays suitable for sklearn.metrics.roc_auc_score
               Returns None, None if insufficient samples for AUROC
    """
    scores = []
    labels = []
    
    # Track which NID detections matched a GT interaction
    matched_nid_indices = set()
    
    # Process each GT interaction
    for gt_inter in gt_interactions:
        best_score = 0.0
        best_idx = None
        
        # Find best matching NID detection (highest strength among supersets/exact matches)
        for idx, (nid_tuple, strength) in enumerate(nid_interactions):
            if is_exact_or_superset(nid_tuple, gt_inter):
                if strength > best_score:
                    best_score = strength
                    best_idx = idx
        
        # Add this GT as a positive sample
        scores.append(best_score)
        labels.append(1)
        
        # Mark matched NID detection
        if best_idx is not None:
            matched_nid_indices.add(best_idx)
    
    # Process unmatched NID detections
    for idx, (nid_tuple, strength) in enumerate(nid_interactions):
        if idx not in matched_nid_indices:
            # Check if this NID detection is a subset of any GT interaction
            # (if so, it's "covered" and should be considered a false positive attempt)
            nid_set = frozenset(i + 1 for i in nid_tuple)
            is_subset_of_gt = any(nid_set < gt_inter for gt_inter in gt_interactions)
            
            if not is_subset_of_gt:
                # This is a detection that doesn't match any GT (false positive candidate)
                scores.append(strength)
                labels.append(0)
    
    if len(labels) < 2:
        # Need at least 2 samples for AUROC
        return None, None
    
    return np.array(scores), np.array(labels)


def compute_metrics(scores, labels):
    """
    Compute AUROC, precision, and recall using numpy (no sklearn dependency).
    
    Args:
        scores: array of model scores/probabilities
        labels: array of binary labels (0 or 1)
    
    Returns:
        dict: {'auroc': float, 'precision': float, 'recall': float}
              Returns None values if metrics cannot be computed
    """
    if scores is None or len(np.unique(labels)) < 2:
        return {"auroc": None, "precision": None, "recall": None}
    
    metrics = {}
    
    # Compute AUROC using concordant pairs method (rank-based)
    try:
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int32)
        
        n_pos = np.sum(labels == 1)
        n_neg = np.sum(labels == 0)
        
        if n_pos == 0 or n_neg == 0:
            metrics["auroc"] = None
        else:
            # Count pairs where positive score > negative score
            pos_scores = scores[labels == 1]
            neg_scores = scores[labels == 0]
            
            concordant = 0.0
            for pos_score in pos_scores:
                concordant += np.sum(pos_score > neg_scores)
            
            # Handle ties: give half credit for ties
            for pos_score in pos_scores:
                concordant += 0.5 * np.sum(pos_score == neg_scores)
            
            # AUROC = concordant_pairs / total_pairs
            auroc = concordant / (n_pos * n_neg)
            metrics["auroc"] = float(auroc)
    except Exception as e:
        metrics["auroc"] = None
    
    # Compute Precision (TP / (TP + FP))
    try:
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int32)
        
        # Use mean score as threshold
        threshold = np.mean(scores)
        predictions = (scores > threshold).astype(int)
        
        # If all predictions are same, try percentile
        if len(np.unique(predictions)) < 2:
            threshold = np.percentile(scores, 50)
            predictions = (scores > threshold).astype(int)
        
        if len(np.unique(predictions)) < 2:
            metrics["precision"] = None
        else:
            tp = np.sum((predictions == 1) & (labels == 1))
            fp = np.sum((predictions == 1) & (labels == 0))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            metrics["precision"] = float(precision)
    except Exception as e:
        metrics["precision"] = None
    
    # Compute Recall (TP / (TP + FN))
    try:
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int32)
        
        threshold = np.mean(scores)
        predictions = (scores > threshold).astype(int)
        
        if len(np.unique(predictions)) < 2:
            threshold = np.percentile(scores, 50)
            predictions = (scores > threshold).astype(int)
        
        if len(np.unique(predictions)) < 2:
            metrics["recall"] = None
        else:
            tp = np.sum((predictions == 1) & (labels == 1))
            fn = np.sum((predictions == 0) & (labels == 1))
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics["recall"] = float(recall)
    except Exception as e:
        metrics["recall"] = None
    
    return metrics


def analyze_single_experiment(function, experiment_settings, snapshot_root="snapshots", num_samples=30000, seed=42):
    """
    Analyze a single function/experiment combination.
    
    Args:
        function: Function from synth.py
        experiment_settings: Dict with keys: name, noise, optimizer, dropout, weight_decay
        snapshot_root: Root directory for snapshots
        num_samples: Number of samples for generating GT data
        seed: Random seed
    
    Returns:
        dict: {
            'function_name': str,
            'experiment_name': str,
            'noise': float,
            'optimizer': str,
            'dropout': float,
            'weight_decay': bool,
            'auroc': float or None,
            'precision': float or None,
            'recall': float or None,
            'num_gt': int,
            'num_detected': int,
            'num_exact_matched': int,
            'num_superset_matched_unique': int,
            'num_matched': int,
            'matched_pairs': list,
            'best_epoch': int,
            'val_loss': float,
            'success': bool,
        }
    """
    function_name = function.__name__
    experiment_name = experiment_settings.get("name", "unknown")
    noise = experiment_settings.get("noise", 0.0)
    
    result = {
        "function_name": function_name,
        "experiment_name": experiment_name,
        "noise": noise,
        "optimizer": experiment_settings.get("optimizer", "unknown"),
        "dropout": experiment_settings.get("dropout", 0.0),
        "weight_decay": experiment_settings.get("weight_decay", False),
        "auroc": None,
        "precision": None,
        "recall": None,
        "num_gt": 0,
        "num_detected": 0,
        "num_exact_matched": 0,
        "num_superset_matched_unique": 0,
        "num_matched": 0,
        "matched_pairs": [],
        "best_epoch": -1,
        "val_loss": float("inf"),
        "success": False,
    }
    
    try:
        # Find best epoch checkpoint
        snapshot_dir = Path(snapshot_root) / function_name / experiment_name
        best_epoch_files = list(snapshot_dir.glob("best_epoch_*.pt"))
        
        if not best_epoch_files:
            return result
        
        best_epoch_file = best_epoch_files[0]
        
        # Load model and NID interactions with correct dropout setting
        dropout_value = experiment_settings.get("dropout", 0.0)
        model, nid_interactions, val_loss, epoch = load_model_and_interactions(
            best_epoch_file, 
            dropout=dropout_value
        )
        
        if model is None or nid_interactions is None:
            return result
        
        # Get ground truth interactions
        gt_interactions = get_ground_truth_interactions(function, num_samples, noise, seed)

        # Compute one-to-one matching metrics regardless of whether AUROC can be formed.
        matching = match_interactions_one_to_one(gt_interactions, nid_interactions)
        
        # Compute AUROC data
        scores, labels = compute_auroc_data(gt_interactions, nid_interactions)
        
        if scores is None:
            # Not enough samples
            result["num_exact_matched"] = matching["num_exact_matched"]
            result["num_superset_matched_unique"] = matching["num_superset_matched_unique"]
            result["num_matched"] = matching["num_matched"]
            result["matched_pairs"] = matching["matched_pairs"]
            result["num_gt"] = len(gt_interactions)
            result["num_detected"] = len(nid_interactions)
            result["best_epoch"] = epoch
            result["val_loss"] = val_loss
            result["success"] = True
            return result
        
        # Compute metrics
        metrics = compute_metrics(scores, labels)
        
        result.update({
            "auroc": metrics["auroc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "num_gt": len(gt_interactions),
            "num_detected": len(nid_interactions),
            "num_exact_matched": matching["num_exact_matched"],
            "num_superset_matched_unique": matching["num_superset_matched_unique"],
            "num_matched": matching["num_matched"],
            "matched_pairs": matching["matched_pairs"],
            "best_epoch": epoch,
            "val_loss": val_loss,
            "success": True,
        })
    
    except Exception as e:
        print(f"Error analyzing {function_name}/{experiment_name}: {e}")
        result["error"] = str(e)
    
    return result


def analyze_all_experiments(snapshot_root="snapshots", num_samples=30000, seed=42, output_dir="results"):
    """
    Analyze all function/experiment combinations.
    
    Args:
        snapshot_root: Root directory for snapshots
        num_samples: Number of samples for generating GT data
        seed: Random seed
        output_dir: Directory to save results CSVs
    
    Returns:
        dict: Mapping of function_name to list of result dicts
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    print(f"Analyzing {len(functions)} functions × {len(EXPERIMENTS)} experiments...")
    print(f"Results will be saved to {output_dir}/")
    print()
    
    for function in functions:
        function_name = function.__name__
        function_results = []
        
        print(f"Analyzing {function_name}...")
        
        for experiment in EXPERIMENTS:
            result = analyze_single_experiment(
                function=function,
                experiment_settings=experiment,
                snapshot_root=snapshot_root,
                num_samples=num_samples,
                seed=seed,
            )
            function_results.append(result)
            
            # Print progress
            status = "✓" if result["success"] else "✗"
            auroc_str = f"{result['auroc']:.4f}" if result["auroc"] is not None else "N/A"
            print(f"  [{status}] {experiment['name']:20s} AUROC={auroc_str}")
        
        all_results[function_name] = function_results
        
        # Save results for this function
        df = pd.DataFrame(function_results)
        output_file = output_path / f"{function_name}_analysis.csv"
        df.to_csv(output_file, index=False)
        print(f"  → Saved to {output_file}")
        print()
    
    return all_results


def print_summary(all_results):
    """
    Print a summary of results across all functions and experiments.
    """
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    total_experiments = sum(len(results) for results in all_results.values())
    successful = sum(1 for results in all_results.values() for r in results if r["success"])
    
    print(f"Total experiments: {total_experiments}")
    print(f"Successful analyses: {successful} ({100*successful/total_experiments:.1f}%)")
    print()
    
    # Compute aggregate statistics
    all_aurocs = []
    for results in all_results.values():
        for r in results:
            if r["auroc"] is not None:
                all_aurocs.append(r["auroc"])
    
    if all_aurocs:
        print(f"AUROC Statistics:")
        print(f"  Mean:     {np.mean(all_aurocs):.4f}")
        print(f"  Median:   {np.median(all_aurocs):.4f}")
        print(f"  Std:      {np.std(all_aurocs):.4f}")
        print(f"  Min:      {np.min(all_aurocs):.4f}")
        print(f"  Max:      {np.max(all_aurocs):.4f}")
        print()
    
    # Print top performers
    print("Top 10 AUROC performances:")
    top_results = []
    for func_name, results in all_results.items():
        for r in results:
            if r["auroc"] is not None:
                top_results.append((func_name, r["experiment_name"], r["auroc"]))
    
    top_results.sort(key=lambda x: x[2], reverse=True)
    for i, (func, exp, auroc) in enumerate(top_results[:10], 1):
        print(f"  {i:2d}. {func}/{exp:20s} → AUROC = {auroc:.4f}")


if __name__ == "__main__":
    all_results = analyze_all_experiments(
        snapshot_root="snapshots",
        num_samples=30000,
        seed=42,
        output_dir="results",
    )
    
    print_summary(all_results)
