#!/usr/bin/env python3
"""List available run_id subdirectories for a given function/experiment.

Usage:
  python3 list_runs.py --function f1 --experiment 0.1_adam --snapshot-root snapshots
"""
import argparse
from pathlib import Path
import sys

from analysis import list_available_run_ids


def main():
    parser = argparse.ArgumentParser(description="List run_id directories for an experiment")
    parser.add_argument("--snapshot-root", default="snapshots", help="Root snapshots folder")
    parser.add_argument("--function", required=True, help="Function name (e.g. f1)")
    parser.add_argument("--experiment", required=True, help="Experiment name (e.g. 0.1_adam)")
    args = parser.parse_args()

    runs = list_available_run_ids(args.snapshot_root, args.function, args.experiment)
    if not runs:
        print(f"No run_id subdirectories found under {args.snapshot_root}/{args.function}/{args.experiment}")
        sys.exit(0)

    print("Available run_id directories:")
    for r in runs:
        print(r)


if __name__ == "__main__":
    main()
