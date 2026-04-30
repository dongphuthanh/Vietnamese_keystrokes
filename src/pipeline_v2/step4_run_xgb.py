"""
Step 4: Run XGB.py on all scenarios × folds for both experiment types.

Usage:
  python step4_run_xgb.py                  # runs both context + user
  python step4_run_xgb.py --type context   # only context-independent
  python step4_run_xgb.py --type user      # only user-independent
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from config import (
    XGB_SCRIPT,
    CONTEXT_DIR, RUNS_CONTEXT,
    USER_DIR, RUNS_USER,
    SCENARIOS, FEATURE_PERCENTAGE, SEED, CV_SPLITS,
    POPULATION, GENERATIONS, CM_LABELS,
)

FOLDS = [1, 2, 3]


def run_one(train_csv: Path, test_csv: Path, outdir: Path, use_gpu: bool = False) -> bool:
    if not train_csv.exists() or not test_csv.exists():
        print(f"  [skip] missing CSV: {train_csv.name} or {test_csv.name}")
        return False

    # Skip if already completed
    if (outdir / "results.json").exists():
        print(f"  [done] skipping {outdir.name} (results.json exists)")
        return True

    # Skip empty train sets (can happen when cognitive-level filter removes all rows)
    if pd.read_csv(train_csv).shape[0] == 0:
        print(f"  [skip] empty train CSV: {train_csv.name}")
        return False

    cmd = [
        sys.executable, str(XGB_SCRIPT),
        "--train",              str(train_csv),
        "--test",               str(test_csv),
        "--outdir",             str(outdir),
        "--feature-percentage", str(FEATURE_PERCENTAGE),
        "--seed",               str(SEED),
        "--cv-splits",          str(CV_SPLITS),
        "--population",         str(POPULATION),
        "--generations",        str(GENERATIONS),
        "--cm-labels",
    ] + [str(l) for l in CM_LABELS]
    if use_gpu:
        cmd.append("--gpu")

    print(f"  Running: {outdir.name}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [error] {e}")
        return False


def run_experiment(data_dir: Path, runs_dir: Path, label: str, use_gpu: bool = False):
    runs_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Data:   {data_dir}")
    print(f"  Output: {runs_dir}")
    print(f"{'='*60}")

    total, done = 0, 0
    for scenario in SCENARIOS:
        for fold in FOLDS:
            total += 1
            train_csv = data_dir / f"train_{scenario}_fold{fold}.csv"
            test_csv  = data_dir / f"test_{scenario}_fold{fold}.csv"
            outdir    = runs_dir  / f"xgb_{scenario}_fold{fold}"
            if run_one(train_csv, test_csv, outdir, use_gpu):
                done += 1

    print(f"\n  Completed {done}/{total} runs for {label}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--type", choices=["context", "user", "both"], default="both")
    p.add_argument("--gpu", action="store_true", default=False, help="Use GPU for XGBoost.")
    args = p.parse_args()

    if not XGB_SCRIPT.exists():
        raise FileNotFoundError(f"XGB.py not found at {XGB_SCRIPT}")

    if args.type in ("context", "both"):
        run_experiment(CONTEXT_DIR, RUNS_CONTEXT, "CONTEXT-INDEPENDENT", args.gpu)

    if args.type in ("user", "both"):
        run_experiment(USER_DIR, RUNS_USER, "USER-INDEPENDENT", args.gpu)

    print("\n[Step 4] Done.")


if __name__ == "__main__":
    main()
