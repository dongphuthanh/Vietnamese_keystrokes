"""
Step 3: User-Independent dataset split.

Train and test come from DIFFERENT users.
Uses KFold(n_splits=3, shuffle=True, random_state=42) on unique users
— identical to the collaborator's split:

    n_splits = 3
    unique_users = np.unique(ID)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

All level groups are used in both train and test (no group filtering).

Output: user_indep_datasets/train_M2_fold1.csv ... (24 files)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from config import (
    NORMAL_PKL, ATTACK_PKL,
    USER_DIR, SCENARIOS,
)
from utils import load_pkl_as_df, drop_meta_cols


def main():
    USER_DIR.mkdir(parents=True, exist_ok=True)

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

    # KFold on unique normal users — matches collaborator's split exactly
    all_users = np.unique(df_normal["user_id"].values)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    print(f"Users: {len(all_users)} total | 3-fold split")

    print("\nGenerating user-independent datasets...")
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(all_users)):
        fold        = fold_idx + 1
        train_users = all_users[train_idx]
        test_users  = all_users[test_idx]

        print(f"\n  Fold {fold} | train {len(train_users)} users | test {len(test_users)} users")

        # Test: test_users, all groups, all labels
        df_test = df_all[df_all["user_id"].isin(test_users)].copy()
        df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)
        df_test_out = drop_meta_cols(df_test)

        for scenario, conditions in SCENARIOS.items():
            train_frames = []
            for source, sessions in conditions:
                mask = (
                    df_all["user_id"].isin(train_users) &
                    (df_all["source"]      == source) &
                    df_all["session"].isin(sessions)
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
