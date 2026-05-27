#!/usr/bin/env python3
"""
migrate_snapshots.py

Copies snapshots/ into snapshots_clean/{model_size}/{function_name}/{experiment_name}/
without modifying the original snapshots/ directory.

Source-directory selection priority (per experiment):
  1. --run-id subdir, if passed and contains best_epoch_*.pt files
  2. Most recently modified run_id subdir that contains best_epoch_*.pt files
  3. Loose files directly in the experiment dir — ONLY when no run_id subdirs exist at all

model_size is determined from the top-level "hidden_units" key in run_info.json:
  [64, 64]            -> small
  [140, 100, 60, 20]  -> big
"""

import argparse
import json
import re
import shutil
from pathlib import Path


_SMALL = [64, 64]
_BIG = [140, 100, 60, 20]

_FILES_TO_COPY = ("best_epoch_*.pt", "epoch_*.pt", "run_info.json", "metrics.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_size(hidden_units) -> str | None:
    if hidden_units == _SMALL:
        return "small"
    if hidden_units == _BIG:
        return "big"
    return None


def _has_best_epoch(d: Path) -> bool:
    """True if d contains at least one best_epoch_*.pt file (non-recursive)."""
    return any(d.glob("best_epoch_*.pt"))


def _most_recent_mtime(d: Path) -> float:
    """Max mtime across best_epoch_*.pt files in d."""
    return max(f.stat().st_mtime for f in d.glob("best_epoch_*.pt"))


def _read_run_info(d: Path) -> dict | None:
    ri = d / "run_info.json"
    if ri.exists():
        try:
            with open(ri) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _collect_files(d: Path) -> list[Path]:
    """Collect all files matching _FILES_TO_COPY from d (deduplicated, sorted)."""
    seen: set[Path] = set()
    for pattern in _FILES_TO_COPY:
        for f in d.glob(pattern):
            seen.add(f)
    return sorted(seen)


def _experiment_has_weight_decay(exp_dir: Path, experiment_name: str) -> bool:
    """Return True if '_wd' appears in the experiment directory name."""
    return "_wd" in experiment_name


# ---------------------------------------------------------------------------
# Source-directory selection
# ---------------------------------------------------------------------------

def _run_id_subdirs_with_best_epoch(exp_dir: Path) -> list[tuple[float, Path]]:
    """
    Return [(mtime, subdir)] for every immediate subdir of exp_dir that
    contains at least one best_epoch_*.pt file.  Sorted most-recent first.
    """
    candidates = []
    for p in exp_dir.iterdir():
        if p.is_dir() and _has_best_epoch(p):
            candidates.append((_most_recent_mtime(p), p))
    candidates.sort(reverse=True)
    return candidates


class SelectionResult:
    """Outcome of find_source_dir."""
    __slots__ = ("source", "reason", "ambiguous", "run_id_missing")

    def __init__(
        self,
        source: Path | None,
        reason: str,
        ambiguous: bool = False,
        run_id_missing: bool = False,
    ):
        self.source = source
        self.reason = reason
        self.ambiguous = ambiguous    # True: multiple valid run_ids, picked most recent
        self.run_id_missing = run_id_missing  # True: --run-id not present here


def find_source_dir(exp_dir: Path, run_id: str | None, loose_only: bool = False) -> SelectionResult:
    # --loose-only: skip all run_id subdir discovery and use only loose files in exp_dir
    if loose_only:
        if _has_best_epoch(exp_dir):
            return SelectionResult(
                source=exp_dir,
                reason="loose files (--loose-only)",
            )
        return SelectionResult(source=None, reason="no best_epoch_*.pt in exp dir (--loose-only)")

    valid = _run_id_subdirs_with_best_epoch(exp_dir)
    has_any_run_id_subdir = bool(valid)

    # 1. Explicit --run-id
    run_id_missing = False
    if run_id is not None:
        explicit = exp_dir / run_id
        if explicit.is_dir() and _has_best_epoch(explicit):
            return SelectionResult(
                source=explicit,
                reason=f"--run-id {run_id}",
            )
        run_id_missing = True  # specified but absent here

    # 2. Most recently modified run_id subdir with best_epoch files
    if valid:
        chosen = valid[0][1]
        ambiguous = len(valid) > 1
        if ambiguous:
            names = ", ".join(p.name for _, p in valid)
            reason = f"most-recent of {len(valid)}: {names}"
        else:
            reason = f"only run_id: {chosen.name}"
        return SelectionResult(
            source=chosen,
            reason=reason,
            ambiguous=ambiguous,
            run_id_missing=run_id_missing,
        )

    # 3. Loose files — only when no run_id subdirs exist at all
    if not has_any_run_id_subdir and _has_best_epoch(exp_dir):
        return SelectionResult(
            source=exp_dir,
            reason="loose files (no run_id subdirs)",
            run_id_missing=run_id_missing,
        )

    msg = "no best_epoch_*.pt found"
    if run_id_missing:
        msg = f"--run-id '{run_id}' absent; " + msg
    return SelectionResult(source=None, reason=msg, run_id_missing=run_id_missing)


# ---------------------------------------------------------------------------
# Main migration
# ---------------------------------------------------------------------------

def migrate(snapshot_root: Path, output_root: Path, run_id: str | None, loose_only: bool = False, no_weight_decay_only: bool = False, force_model_size: str | None = None) -> None:
    if not snapshot_root.exists():
        print(f"ERROR: snapshot root '{snapshot_root}' does not exist.")
        return

    # Discover experiments
    experiments: list[tuple[str, str, Path]] = []
    for fn_dir in sorted(snapshot_root.iterdir()):
        if not fn_dir.is_dir():
            continue
        for exp_dir in sorted(fn_dir.iterdir()):
            if exp_dir.is_dir():
                experiments.append((fn_dir.name, exp_dir.name, exp_dir))

    # Filter to non-weight-decay experiments only when requested
    filtered_out: list[str] = []
    if no_weight_decay_only:
        kept = []
        for function_name, experiment_name, exp_dir in experiments:
            if _experiment_has_weight_decay(exp_dir, experiment_name):
                filtered_out.append(f"{function_name}/{experiment_name}")
            else:
                kept.append((function_name, experiment_name, exp_dir))
        experiments = kept

    print(f"Snapshot root : {snapshot_root}")
    print(f"Output root   : {output_root}")
    if run_id:
        print(f"Preferred run-id : {run_id}")
    active_flags = []
    if no_weight_decay_only:
        active_flags.append("--no-weight-decay-only  (filter: _wd not in experiment name; implies loose-only)")
    if loose_only:
        active_flags.append("--loose-only  (skip run_id subdirs)")
    if force_model_size is not None:
        active_flags.append(f"--force-model-size={force_model_size}  (skip run_info.json arch detection)")
    if active_flags:
        print("Active flags  :")
        for flag in active_flags:
            print(f"  {flag}")
    if no_weight_decay_only and filtered_out:
        print(f"  Filtered out ({len(filtered_out)} weight-decay experiments):")
        for label in filtered_out:
            print(f"    {label}")
    print(f"Experiments found: {len(experiments)}")
    print()

    # Table header
    W_EXP, W_SRC, W_SZ, W_N = 36, 52, 7, 6
    sep = "-" * (W_EXP + W_SRC + W_SZ + W_N + 10)

    def row(exp, src, sz, n, flag=""):
        return f"  {exp:<{W_EXP}} {src:<{W_SRC}} {sz:<{W_SZ}} {n:>{W_N}}  {flag}"

    print(row("Experiment", "Source chosen", "Size", "Files"))
    print("  " + sep)

    # Tracking
    processed = 0
    total_files = 0
    size_counts: dict[str, int] = {"small": 0, "big": 0}
    skipped: list[tuple[str, str]] = []
    ambiguous_exps: list[str] = []
    run_id_missing_exps: list[str] = []

    for function_name, experiment_name, exp_dir in experiments:
        label = f"{function_name}/{experiment_name}"

        sel = find_source_dir(exp_dir, run_id, loose_only=loose_only or no_weight_decay_only)

        if sel.run_id_missing:
            run_id_missing_exps.append(label)

        if sel.source is None:
            skipped.append((label, sel.reason))
            print(row(label, sel.reason, "—", "SKIP", "[SKIP]"))
            continue

        # Determine model_size: use forced value when provided, otherwise detect from run_info.json
        if force_model_size is not None:
            model_size = force_model_size
        else:
            run_info = _read_run_info(sel.source)
            if run_info is None:
                reason = f"no run_info.json in {sel.source.name}"
                skipped.append((label, reason))
                print(row(label, sel.source.name, "—", "SKIP", "[no run_info]"))
                continue

            # Use top-level hidden_units (not experiment_settings.hidden_units,
            # which can be wrong when --hidden_units was passed to train.py)
            hidden_units = run_info.get("hidden_units")
            model_size = _model_size(hidden_units)
            if model_size is None:
                reason = f"unknown hidden_units={hidden_units}"
                skipped.append((label, reason))
                print(row(label, sel.source.name, "?", "SKIP", f"[hu={hidden_units}]"))
                continue

        # Copy
        dest = output_root / model_size / function_name / experiment_name
        dest.mkdir(parents=True, exist_ok=True)

        files = _collect_files(sel.source)
        for src_file in files:
            shutil.copy2(src_file, dest / src_file.name)

        n = len(files)
        total_files += n
        processed += 1
        size_counts[model_size] += 1
        if sel.ambiguous:
            ambiguous_exps.append(label)

        # Display: shorten source to just the run_id name (or "(loose)")
        if sel.source == exp_dir:
            src_display = "(loose)"
        else:
            src_display = sel.source.name

        flag = "[AMBIG]" if sel.ambiguous else ""
        if sel.run_id_missing:
            flag = (flag + " [run-id absent]").strip()

        print(row(label, src_display, model_size, str(n), flag))

    # Footer
    print("  " + sep)
    print()
    print("SUMMARY")
    print(f"  Processed : {processed} / {len(experiments)} experiments")
    print(f"  Files copied : {total_files}")
    print(f"  small : {size_counts['small']}   big : {size_counts['big']}")

    if ambiguous_exps:
        print(
            f"\n  [AMBIG] Multiple run_ids existed; used most-recent"
            f" ({len(ambiguous_exps)} experiments):"
        )
        for label in ambiguous_exps:
            print(f"    {label}")

    if run_id and run_id_missing_exps:
        print(
            f"\n  [run-id absent] --run-id '{run_id}' not found, fell back to"
            f" most-recent ({len(run_id_missing_exps)} experiments):"
        )
        for label in run_id_missing_exps:
            print(f"    {label}")

    if skipped:
        print(f"\n  [SKIP] Skipped ({len(skipped)}):")
        for label, reason in skipped:
            print(f"    {label}: {reason}")
    else:
        print("\n  No experiments skipped.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy snapshots/ into snapshots_clean/{model_size}/{function_name}/{experiment_name}/. "
            "Original files are never modified or deleted."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--snapshot-root",
        default="snapshots",
        help="Root of the existing snapshot tree",
    )
    parser.add_argument(
        "--output-root",
        default="snapshots_clean",
        help="Destination root directory",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Prefer this run_id subdirectory when selecting source files",
    )
    parser.add_argument(
        "--loose-only",
        action="store_true",
        default=False,
        help="Only copy loose files directly in each experiment directory; ignore all run_id subdirectories",
    )
    parser.add_argument(
        "--no-weight-decay-only",
        action="store_true",
        default=False,
        help=(
            "Only migrate experiments where '_wd' does not appear in the experiment directory name. "
            "Implies loose-only source selection for the included experiments."
        ),
    )
    parser.add_argument(
        "--force-model-size",
        choices=["small", "big"],
        default=None,
        help=(
            "Force model_size to this value for all experiments. "
            "Skips run_info.json architecture detection entirely; "
            "experiments without run_info.json are not skipped."
        ),
    )
    args = parser.parse_args()

    migrate(
        snapshot_root=Path(args.snapshot_root),
        output_root=Path(args.output_root),
        run_id=args.run_id,
        loose_only=args.loose_only,
        no_weight_decay_only=args.no_weight_decay_only,
        force_model_size=args.force_model_size,
    )


if __name__ == "__main__":
    main()
