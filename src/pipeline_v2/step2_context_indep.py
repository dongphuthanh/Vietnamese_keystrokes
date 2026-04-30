"""
Step 2: Context-Independent dataset split.

Split by level group (group 1={1,4}, group 2={2,5}, group 3={3,6}).
All users appear in both train and test, but different question types.

Fold splits (leave-one-group-out):
  Fold 1: train groups [2,3]  test groups [1]
  Fold 2: train groups [1,3]  test groups [2]
  Fold 3: train groups [1,2]  test groups [3]

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

    print("Loading PKL files...")
    df_normal = load_pkl_as_df(NORMAL_PKL, "normal")
    df_attack  = load_pkl_as_df(ATTACK_PKL, "attack")

    df_all = pd.concat([df_normal, df_attack], ignore_index=True)

    n_before = len(df_all)
    df_all = df_all[
        df_all["group_id"].notna() & (df_all["label"] != -1)
    ].copy()
    df_all["group_id"] = df_all["group_id"].astype(int)
    df_all["session"]  = df_all["session"].astype(int)
    print(f"Valid rows: {len(df_all)} / {n_before}")
    print(f"  Labels: {sorted(df_all['label'].unique())}")
    print(f"  Groups: {sorted(df_all['group_id'].unique())}")
    print(f"  Users:  {df_all['user_id'].nunique()}")

    print("\nGenerating context-independent datasets...")
    for fold_cfg in CONTEXT_FOLDS:
        fold         = fold_cfg["fold"]
        train_groups = fold_cfg["train_groups"]
        test_groups  = fold_cfg["test_groups"]

        print(f"\n  Fold {fold} | train groups {train_groups} | test groups {test_groups}")

        # Test: ALL users, test groups, all labels
        df_test = df_all[df_all["group_id"].isin(test_groups)].copy()
        df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)
        df_test_out = drop_meta_cols(df_test)

        for scenario, conditions in SCENARIOS.items():
            train_frames = []
            for source, sessions in conditions:
                mask = (
                    (df_all["source"]        == source) &
                    df_all["session"].isin(sessions) &
                    df_all["group_id"].isin(train_groups)
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
