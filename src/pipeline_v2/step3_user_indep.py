"""
Step 3: User-Independent dataset split.

Train and test come from DIFFERENT users.
The user split is fixed (30 train / 15 test), then cognitive level
varies across the 3 folds (same pairs as context-independent).

Test  = test_users  AND test_cognitive_levels  (all 5 labels)
Train = train_users AND train_cognitive_levels  (filtered by scenario)

Output: user_indep_datasets/train_M2_fold1.csv ... (24 files)
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    NORMAL_PKL, ATTACK_PKL,
    USER_DIR, SCENARIOS, USER_FOLDS, N_TEST_USERS,
)
from utils import load_pkl_as_df, drop_meta_cols


def main():
    USER_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    print("Loading PKL files...")
    df_normal = load_pkl_as_df(NORMAL_PKL, "normal")
    df_attack  = load_pkl_as_df(ATTACK_PKL, "attack")

    df_all = pd.concat([df_normal, df_attack], ignore_index=True)

    n_before = len(df_all)
    df_all = df_all[
        df_all["cognitive_level"].notna() & (df_all["label"] != -1)
    ].copy()
    df_all["cognitive_level"] = df_all["cognitive_level"].astype(int)
    df_all["session"]         = df_all["session"].astype(int)
    print(f"Valid rows: {len(df_all)} / {n_before}")

    # ---- Fixed user split (based on normal users only) ----
    all_users  = df_normal["user_id"].unique()
    test_ratio = N_TEST_USERS / len(all_users)
    train_users, test_users = train_test_split(
        all_users, test_size=test_ratio, random_state=42
    )
    print(f"Users: {len(all_users)} total | {len(train_users)} train | {len(test_users)} test")

    # ---- Generate 3 folds × 4 scenarios ----
    print("\nGenerating user-independent datasets...")
    for fold_cfg in USER_FOLDS:
        fold        = fold_cfg["fold"]
        train_lvls  = fold_cfg["train_levels"]
        test_lvls   = fold_cfg["test_levels"]

        print(f"\n  Fold {fold} | train levels {train_lvls} | test levels {test_lvls}")

        # Test: test_users × test_levels, all 5 labels
        df_test = df_all[
            df_all["user_id"].isin(test_users) &
            df_all["cognitive_level"].isin(test_lvls)
        ].copy()
        df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)
        df_test_out = drop_meta_cols(df_test)

        for scenario, conditions in SCENARIOS.items():
            train_frames = []
            for source, sessions in conditions:
                mask = (
                    df_all["user_id"].isin(train_users) &
                    (df_all["source"]         == source) &
                    df_all["session"].isin(sessions) &
                    df_all["cognitive_level"].isin(train_lvls)
                )
                train_frames.append(df_all[mask])

            df_train = pd.concat(train_frames, ignore_index=True)
            df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
            df_train_out = drop_meta_cols(df_train)

            train_csv = USER_DIR / f"train_{scenario}_fold{fold}.csv"
            test_csv  = USER_DIR / f"test_{scenario}_fold{fold}.csv"
            df_train_out.to_csv(train_csv, index=False)
            df_test_out.to_csv(test_csv, index=False)

            print(f"    {scenario}: train {len(df_train):5d} rows | test {len(df_test):5d} rows")

    print(f"\n[Step 3] Done. 24 CSVs saved to {USER_DIR}")


if __name__ == "__main__":
    main()
