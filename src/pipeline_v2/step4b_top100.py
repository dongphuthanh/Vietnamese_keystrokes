"""
Step 4b: Re-run XGBoost using only the top-100 most important features.

Feature importance is pooled (mean Gain) from the existing step-4 runs.
Results are saved to runs_top100_context_indep/ — existing runs are untouched.

Usage:
  python step4b_top100.py                  # all scenarios
  python step4b_top100.py --scenario M5   # single scenario
  python step4b_top100.py --top-n 50      # different feature count
"""

import argparse
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd

from config import (
    XGB_SCRIPT,
    CONTEXT_DIR, RUNS_CONTEXT,
    SCENARIOS, SEED, CV_SPLITS, POPULATION, GENERATIONS, CM_LABELS,
)

FOLDS = [1, 2, 3]
BASE_DIR      = Path(__file__).parent
RUNS_TOP100   = BASE_DIR / "runs_top100_context_indep"
DATA_TOP100   = BASE_DIR / "top100_datasets"


# ============================================================
# POOL GAIN IMPORTANCE FROM EXISTING RUNS → TOP-N FEATURES
# ============================================================
def get_top_features(runs_dir: Path, scenario: str, n: int) -> list | None:
    all_imp = []
    for fold in FOLDS:
        model_path = runs_dir / f"xgb_{scenario}_fold{fold}" / "model.joblib"
        feat_path  = runs_dir / f"xgb_{scenario}_fold{fold}" / "selected_features.txt"
        if not model_path.exists() or not feat_path.exists():
            continue
        pipeline = joblib.load(model_path)
        imp      = pipeline.named_steps["classifier"].feature_importances_
        names    = [l.strip() for l in feat_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        all_imp.append(pd.DataFrame({"feature": names, "importance": imp}))

    if not all_imp:
        return None

    df = (
        pd.concat(all_imp, ignore_index=True)
        .groupby("feature")["importance"]
        .mean()
        .reset_index()
        .sort_values("importance", ascending=False)
    )
    return df["feature"].head(n).tolist()


# ============================================================
# FILTER CSV TO SELECTED FEATURES
# ============================================================
def filter_csv(src: Path, dst: Path, features: list) -> int:
    df = pd.read_csv(src)
    meta_cols = [c for c in ["label", "user_id", "session", "section"] if c in df.columns]
    available = [f for f in features if f in df.columns]
    df[meta_cols + available].to_csv(dst, index=False)
    return len(available)


# ============================================================
# RUN ONE FOLD
# ============================================================
def run_one(train_csv: Path, test_csv: Path, outdir: Path, population: int, generations: int) -> bool:
    if not train_csv.exists() or not test_csv.exists():
        print(f"  [skip] missing CSV")
        return False
    if (outdir / "results.json").exists():
        print(f"  [done] {outdir.name}")
        return True
    if pd.read_csv(train_csv).shape[0] == 0:
        print(f"  [skip] empty train CSV")
        return False

    cmd = [
        sys.executable, str(XGB_SCRIPT),
        "--train",              str(train_csv),
        "--test",               str(test_csv),
        "--outdir",             str(outdir),
        "--feature-percentage", "100",   # already filtered — keep all
        "--seed",               str(SEED),
        "--cv-splits",          str(CV_SPLITS),
        "--population",         str(population),
        "--generations",        str(generations),
        "--cm-labels",
    ] + [str(l) for l in CM_LABELS]

    print(f"  Running: {outdir.name}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [error] {e}")
        return False


# ============================================================
# MAIN
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="all",
                   help="Scenario (M2/M3/M4/M5) or 'all'")
    p.add_argument("--top-n", type=int, default=100,
                   help="Number of top features to use (default: 100)")
    p.add_argument("--population", type=int, default=POPULATION,
                   help="GA population size (default: from config)")
    p.add_argument("--generations", type=int, default=GENERATIONS,
                   help="GA generations (default: from config)")
    args = p.parse_args()

    scenarios = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]
    top_n = args.top_n

    DATA_TOP100.mkdir(parents=True, exist_ok=True)
    RUNS_TOP100.mkdir(parents=True, exist_ok=True)

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"  Scenario: {scenario}  |  Top-{top_n} features")
        print(f"{'='*60}")

        # Get top-N features from existing step-4 runs
        top_features = get_top_features(RUNS_CONTEXT, scenario, top_n)
        if top_features is None:
            print(f"  [skip] no existing model runs found for {scenario}")
            continue
        print(f"  Identified {len(top_features)} features from existing runs")

        # Save feature list for reference
        feat_file = RUNS_TOP100 / f"top{top_n}_features_{scenario}.txt"
        feat_file.write_text("\n".join(top_features), encoding="utf-8")
        print(f"  Feature list saved: {feat_file.name}")

        total, done = 0, 0
        for fold in FOLDS:
            total += 1
            src_train = CONTEXT_DIR / f"train_{scenario}_fold{fold}.csv"
            src_test  = CONTEXT_DIR / f"test_{scenario}_fold{fold}.csv"
            dst_train = DATA_TOP100  / f"train_{scenario}_fold{fold}.csv"
            dst_test  = DATA_TOP100  / f"test_{scenario}_fold{fold}.csv"
            outdir    = RUNS_TOP100  / f"xgb_{scenario}_fold{fold}"

            if not src_train.exists():
                print(f"  [skip] fold{fold}: source CSV missing")
                continue

            n_avail = filter_csv(src_train, dst_train, top_features)
            filter_csv(src_test, dst_test, top_features)
            print(f"  fold{fold}: {n_avail}/{top_n} features present in CSV")

            outdir.mkdir(parents=True, exist_ok=True)
            if run_one(dst_train, dst_test, outdir, args.population, args.generations):
                done += 1

        print(f"\n  Completed {done}/{total} folds for {scenario}")

    print("\n[Step 4b] Done.")


if __name__ == "__main__":
    main()
