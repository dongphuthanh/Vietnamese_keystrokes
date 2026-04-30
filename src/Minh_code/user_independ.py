import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

# ==========================================
# CẤU HÌNH
# ==========================================
NORMAL_PKL = "full.pkl"
ATTACK_PKL = "attack.pkl"
OUT_DIR = Path("user_indep_cognitive_datasets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = {
    "M2": [("normal", [1, 3])],
    "M3": [("normal", [1, 2, 3])],
    "M4": [("normal", [1, 2, 3]), ("attack", [3])],
    "M5": [("normal", [1, 2, 3]), ("attack", [2, 3])],
}

FOLD_SPLITS = [
    (1, [2, 3, 5, 6], [1, 4]),
    (2, [1, 3, 4, 6], [2, 5]),
    (3, [1, 2, 4, 5], [3, 6]),
]

N_TEST_USERS = 15


# ==========================================
# HELPERS
# ==========================================
def assign_label(row):
    sess = int(row["session"])
    if row["source"] == "normal":
        if sess == 1: return 0
        if sess == 2: return 1
        if sess == 3: return 2
    elif row["source"] == "attack":
        if sess == 2: return 3
        if sess == 3: return 4
    return -1


def extract_cognitive_level(qi):
    try:
        s = str(qi).strip()
        if "." not in s:
            return None
        level = int(s.split(".")[-1])
        return level if 1 <= level <= 6 else None
    except Exception:
        return None


# ==========================================
# MAIN
# ==========================================
def main():
    print("=" * 70)
    print("    TAO DU LIEU: USER-INDEPENDENT + COGNITIVE LEVEL SPLIT")
    print("=" * 70)

    # Load
    df_normal = pd.DataFrame(pd.read_pickle(NORMAL_PKL))
    df_attack = pd.DataFrame(pd.read_pickle(ATTACK_PKL))
    df_normal["source"] = "normal"
    df_attack["source"] = "attack"

    df_all = pd.concat([df_normal, df_attack], ignore_index=True)
    num_cols = df_all.select_dtypes(include="number").columns
    df_all[num_cols] = df_all[num_cols].fillna(0)

    # Label + cognitive level
    df_all["label"] = df_all.apply(assign_label, axis=1)
    df_all["cognitive_level"] = df_all["question_index"].apply(extract_cognitive_level)

    # Filter invalid
    df_all = df_all[
        df_all["cognitive_level"].notna() & (df_all["label"] != -1)
    ].copy()
    df_all["cognitive_level"] = df_all["cognitive_level"].astype(int)

    print(f"Tong du lieu hop le: {len(df_all)}")
    print(f"  Labels    : {sorted(df_all['label'].unique())}")
    print(f"  Cog levels: {sorted(df_all['cognitive_level'].unique())}")

    # User split (based on normal users only — attack users are same people)
    users = df_normal["user_id"].unique()
    print(f"\nTong users: {len(users)}")

    test_ratio = N_TEST_USERS / len(users)
    train_users, test_users = train_test_split(
        users, test_size=test_ratio, random_state=42
    )
    print(f"  Train users: {len(train_users)}")
    print(f"  Test  users: {len(test_users)}")

    # Cols to drop before saving (XGB.py sẽ drop thêm user_id, session, file, label)
    cols_to_drop = ["source", "cognitive_level", "question_index"]

    print("\n" + "=" * 70)

    for fold_idx, train_levels, test_levels in FOLD_SPLITS:
        print(f"\nFOLD {fold_idx}  |  Train levels: {train_levels}  |  Test levels: {test_levels}")

        # Test: test users + test cognitive levels + all 5 labels
        test_mask = (
            df_all["user_id"].isin(test_users) &
            df_all["cognitive_level"].isin(test_levels)
        )
        df_test = df_all[test_mask].copy()
        df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)
        df_test_out = df_test.drop(columns=[c for c in cols_to_drop if c in df_test.columns])

        print(f"  Test: {len(df_test)} dong | labels={sorted(df_test['label'].unique())}")

        for scenario_name, conditions in SCENARIOS.items():
            train_frames = []
            for source_type, sessions in conditions:
                mask = (
                    df_all["user_id"].isin(train_users) &
                    (df_all["source"] == source_type) &
                    df_all["session"].isin(sessions) &
                    df_all["cognitive_level"].isin(train_levels)
                )
                train_frames.append(df_all[mask])

            df_train = pd.concat(train_frames, ignore_index=True)
            df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
            df_train_out = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns])

            train_csv = OUT_DIR / f"train_{scenario_name}_fold{fold_idx}.csv"
            test_csv  = OUT_DIR / f"test_{scenario_name}_fold{fold_idx}.csv"

            df_train_out.to_csv(train_csv, index=False)
            df_test_out.to_csv(test_csv, index=False)

            print(f"    {scenario_name}: Train {len(df_train):5d} labels={sorted(df_train['label'].unique())} | Test {len(df_test):5d}")

    print("\n" + "=" * 70)
    print(f"HOAN TAT! 24 file CSV da luu vao: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
