"""
Step 2: Context-Independent dataset split.

Split by cognitive level (question_index = session.level).
All users appear in both train and test, but different question types.

Fold splits:
  Fold 1: train levels [2,3,5,6]  test levels [1,4]
  Fold 2: train levels [1,3,4,6]  test levels [2,5]
  Fold 3: train levels [1,2,4,5]  test levels [3,6]

Output: context_indep_datasets/train_M2_fold1.csv ... (24 files total)
"""

import pandas as pd
from config import (
    NORMAL_PKL, ATTACK_PKL,
    CONTEXT_DIR, SCENARIOS, CONTEXT_FOLDS,
)
from utils import load_pkl_as_df, drop_meta_cols


def main():
    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    print("Loading PKL files...")
    df_normal = load_pkl_as_df(NORMAL_PKL, "normal")
    df_attack  = load_pkl_as_df(ATTACK_PKL, "attack")

    df_all = pd.concat([df_normal, df_attack], ignore_index=True)

    # Drop rows with invalid cognitive level or label
    n_before = len(df_all)
    df_all = df_all[
        df_all["cognitive_level"].notna() & (df_all["label"] != -1)
    ].copy()
    df_all["cognitive_level"] = df_all["cognitive_level"].astype(int)
    df_all["session"]         = df_all["session"].astype(int)
    print(f"Valid rows: {len(df_all)} / {n_before}")
    print(f"  Labels: {sorted(df_all['label'].unique())}")
    print(f"  Cognitive levels: {sorted(df_all['cognitive_level'].unique())}")
    print(f"  Users: {df_all['user_id'].nunique()}")

    # ---- Generate 3 folds × 4 scenarios ----
    print("\nGenerating context-independent datasets...")
    for fold_cfg in CONTEXT_FOLDS:
        fold        = fold_cfg["fold"]
        train_lvls  = fold_cfg["train_levels"]
        test_lvls   = fold_cfg["test_levels"]

        print(f"\n  Fold {fold} | train levels {train_lvls} | test levels {test_lvls}")

        # Test set: ALL users, test cognitive levels, all 5 labels
        df_test = df_all[df_all["cognitive_level"].isin(test_lvls)].copy()
        df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)
        df_test_out = drop_meta_cols(df_test)

        for scenario, conditions in SCENARIOS.items():
            train_frames = []
            for source, sessions in conditions:
                mask = (
                    (df_all["source"]          == source) &
                    (df_all["session"].isin(sessions)) &
                    (df_all["cognitive_level"].isin(train_lvls))
                )
                train_frames.append(df_all[mask])

            df_train = pd.concat(train_frames, ignore_index=True)
            df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
            df_train_out = drop_meta_cols(df_train)

            train_csv = CONTEXT_DIR / f"train_{scenario}_fold{fold}.csv"
            test_csv  = CONTEXT_DIR / f"test_{scenario}_fold{fold}.csv"
            df_train_out.to_csv(train_csv, index=False)
            df_test_out.to_csv(test_csv, index=False)

            print(f"    {scenario}: train {len(df_train):5d} rows | test {len(df_test):5d} rows")

    print(f"\n[Step 2] Done. 24 CSVs saved to {CONTEXT_DIR}")


if __name__ == "__main__":
    main()
