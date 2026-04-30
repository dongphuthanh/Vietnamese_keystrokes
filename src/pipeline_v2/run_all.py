"""
Master pipeline runner.

Usage:
  python run_all.py                    # full pipeline (both experiment types)
  python run_all.py --type context     # only context-independent
  python run_all.py --type user        # only user-independent
  python run_all.py --start 2          # skip step 1 (PKL already exists)

Steps:
  1  extract        JSON → PKL
  2  context        PKL → context-independent CSVs
  3  user           PKL → user-independent CSVs
  4  run_xgb        train/evaluate XGB on all scenarios × folds
  5  collect        aggregate results
"""

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import step1_extract
import step2_context_indep
import step3_user_indep
import step4_run_xgb
import step5_collect


def run_step(name: str, fn):
    print(f"\n{'='*60}")
    print(f">>> {name}")
    print(f"{'='*60}")
    try:
        fn()
        print(f"[OK] {name} completed.")
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] User stopped at: {name}")
        sys.exit(1)
    except Exception:
        print(f"\n[ERROR] {name} failed:")
        traceback.print_exc()
        print(f"\nPipeline stopped at: {name}")
        print("Fix the error above, then re-run with --start <step_number> to resume.")
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description="Full Vietnamese keystroke pipeline.")
    p.add_argument("--type",  choices=["context", "user", "both"], default="both")
    p.add_argument("--start", type=int, default=1, choices=[1, 2, 3, 4, 5],
                   help="Start from this step number (skip earlier steps).")
    p.add_argument("--gpu", action="store_true", default=False,
                   help="Use GPU for XGBoost (step 4).")
    args = p.parse_args()

    print("=" * 60)
    print("  VIETNAMESE KEYSTROKE PIPELINE")
    print(f"  Experiment type : {args.type}")
    print(f"  Starting at step: {args.start}")
    print("=" * 60)

    if args.start <= 1:
        run_step("STEP 1: Feature Extraction (JSON → PKL)", step1_extract.main)

    if args.start <= 2 and args.type in ("context", "both"):
        run_step("STEP 2: Context-Independent Dataset", step2_context_indep.main)

    if args.start <= 3 and args.type in ("user", "both"):
        run_step("STEP 3: User-Independent Dataset", step3_user_indep.main)

    if args.start <= 4:
        step4_argv = ["step4_run_xgb.py", "--type", args.type]
        if args.gpu:
            step4_argv.append("--gpu")
        sys.argv = step4_argv
        run_step("STEP 4: Run XGB (all scenarios × folds)", step4_run_xgb.main)

    if args.start <= 5:
        sys.argv = ["step5_collect.py", "--type", args.type]
        run_step("STEP 5: Collect Results", step5_collect.main)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
