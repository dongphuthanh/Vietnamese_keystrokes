#!/usr/bin/env python3
"""
Chia dữ liệu theo COGNITIVE LEVEL thay vì theo USER
- question_index format: '1.1', '2.3', '3.6' (session.cognitive_level)
  Phần SAU dấu chấm = cognitive level (1-6)
  Chú ý: pkl có thể lưu dưới dạng float 1.1, 2.3 (OK)
         hoặc float 1.0, 2.0 (không có cognitive level → loại bỏ)

- 3-Fold CV theo cặp 14 / 25 / 36:
  * Fold 1: Test levels {1,4}  → Train levels {2,3,5,6}
  * Fold 2: Test levels {2,5}  → Train levels {1,3,4,6}
  * Fold 3: Test levels {3,6}  → Train levels {1,2,4,5}

- Train set: lọc theo (source, sessions) của kịch bản + train_levels
- Test set:  toàn bộ df_all (5 nhãn) lọc theo test_levels
"""

import pickle
import pandas as pd
import numpy as np
import os
from pathlib import Path

# ==========================================
# CẤU HÌNH
# ==========================================
NORMAL_PKL = "full.pkl"
ATTACK_PKL = "attack.pkl"
OUT_DIR = Path("cognitive_level_datasets")

FOLD_SPLITS = [
    {"train": [2, 3, 5, 6], "test": [1, 4]},  # Fold 1
    {"train": [1, 3, 4, 6], "test": [2, 5]},  # Fold 2
    {"train": [1, 2, 4, 5], "test": [3, 6]},  # Fold 3
]

SCENARIOS = {
    "M2": [("normal", [1, 3])],
    "M3": [("normal", [1, 2, 3])],
    "M4": [("normal", [1, 2, 3]), ("attack", [3])],
    "M5": [("normal", [1, 2, 3]), ("attack", [2, 3])],
}

os.makedirs(OUT_DIR, exist_ok=True)

# ==========================================
# HÀM TRÍCH XUẤT COGNITIVE LEVEL
# ==========================================
def extract_cognitive_level(question_index):
    """
    question_index dạng '1.1', '2.3', 3.6 (float), v.v.
    Cognitive level = phần sau dấu chấm (1-6).
    
    Loại bỏ các trường hợp:
    - float x.0 (vd 1.0, 2.0) → level = 0 → không hợp lệ
    - không có dấu chấm (vd integer 1, 2) → không hợp lệ
    - giá trị null/nan
    """
    try:
        if question_index is None:
            return None
        s = str(question_index).strip()
        if '.' not in s:
            return None  # integer thuần, không có cognitive level
        after_dot = s.split('.')[-1]
        level = int(after_dot)
        if level < 1 or level > 6:
            return None  # 0 hoặc ngoài range → loại
        return level
    except Exception:
        return None

# ==========================================
# HÀM GÁN NHÃN
# ==========================================
def assign_label(row):
    try:
        sess = int(row['session'])
    except Exception:
        return -1
    if row['source'] == 'normal':
        if sess == 1: return 0  # Bonafide
        if sess == 2: return 1  # Paraphrase
        if sess == 3: return 2  # Transcribe
    elif row['source'] == 'attack':
        if sess == 2: return 3  # Fake Paraphrase
        if sess == 3: return 4  # Fake Transcribe
    return -1

# ==========================================
# LOAD DỮ LIỆU
# ==========================================
def load_and_prep(pkl_path, source_name):
    print(f"📂 Đang đọc: {pkl_path}")
    if not os.path.exists(pkl_path):
        print(f"❌ Không tìm thấy file {pkl_path}!")
        return pd.DataFrame()

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    df = pd.DataFrame(data)
    df.fillna(0, inplace=True)
    df['source'] = source_name
    df['label'] = df.apply(assign_label, axis=1)
    df['cognitive_level'] = df['question_index'].apply(extract_cognitive_level)

    total = len(df)
    # Loại bỏ rows không có cognitive_level hợp lệ
    invalid = df[df['cognitive_level'].isna()]
    if len(invalid) > 0:
        print(f"   ⚠️  Loại bỏ {len(invalid)} rows có question_index không hợp lệ:")
        print(f"      question_index mẫu: {invalid['question_index'].unique()[:10].tolist()}")

    df = df[df['cognitive_level'].notna()].copy()
    df['cognitive_level'] = df['cognitive_level'].astype(int)

    print(f"✅ Đọc thành công {len(df)}/{total} dòng hợp lệ")
    print(f"   Cognitive levels: {sorted(df['cognitive_level'].unique())}")
    print(f"   Sessions: {sorted(df['session'].unique())}")
    print(f"   Labels: {sorted(df['label'].unique())}")
    return df

# ==========================================
# MAIN
# ==========================================
print("="*60)
print("    CHIA DỮ LIỆU THEO COGNITIVE LEVEL (cặp 14/25/36)")
print("="*60)

df_normal = load_and_prep(NORMAL_PKL, "normal")
df_attack = load_and_prep(ATTACK_PKL, "attack")

if df_normal.empty or df_attack.empty:
    print("❌ Thiếu dữ liệu. Dừng chương trình.")
    exit()

df_all = pd.concat([df_normal, df_attack], ignore_index=True)
print(f"\n📊 Tổng cộng: {len(df_all)} dòng hợp lệ")

# ==========================================
# TẠO DATASETS CHO 3 FOLDS × 4 KỊCH BẢN
# ==========================================
print("\n" + "="*60)
print("    TẠO DATASETS CHO 3 FOLDS")
print("="*60)

for fold_idx, fold_config in enumerate(FOLD_SPLITS, 1):
    train_levels = fold_config["train"]
    test_levels  = fold_config["test"]

    print(f"\n🔄 FOLD {fold_idx}  |  Train levels: {train_levels}  |  Test levels: {test_levels}")

    # TEST: toàn bộ df_all, chỉ lọc cognitive level
    test_df_master = df_all[df_all['cognitive_level'].isin(test_levels)].copy()
    test_df_master = test_df_master.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"   📋 Test (tất cả nhãn): {len(test_df_master)} dòng | Labels: {sorted(test_df_master['label'].unique())}")

    for scenario_name, conditions in SCENARIOS.items():
        # TRAIN: lọc theo (source, sessions) kịch bản + train_levels
        train_frames = []
        for source, sessions in conditions:
            mask = (
                (df_all['source'] == source) &
                (df_all['session'].isin(sessions)) &
                (df_all['cognitive_level'].isin(train_levels))
            )
            train_frames.append(df_all[mask])

        train_df = pd.concat(train_frames, ignore_index=True)
        if len(train_df) > 0:
            train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

        train_csv = OUT_DIR / f"train_{scenario_name}_fold{fold_idx}.csv"
        test_csv  = OUT_DIR / f"test_{scenario_name}_fold{fold_idx}.csv"

        train_df.to_csv(train_csv, index=False)
        test_df_master.to_csv(test_csv, index=False)

        print(f"      {scenario_name}: Train {len(train_df):5d} labels={sorted(train_df['label'].unique())} | Test {len(test_df_master):5d}")

print("\n" + "="*60)
print(f"✅ HOÀN TẤT! 24 file CSV đã được tạo trong: {OUT_DIR}")
print("="*60)

print("\n📊 THỐNG KÊ COGNITIVE LEVEL:")
for level in sorted(df_all['cognitive_level'].unique()):
    count = len(df_all[df_all['cognitive_level'] == level])
    print(f"   Level {level}: {count:5d} samples")

print("\n📊 THỐNG KÊ LABEL:")
label_names = {
    0: "Bonafide (B)", 1: "Paraphrase (P)", 2: "Transcribe (T)",
    3: "Fake Paraphrase (F_P)", 4: "Fake Transcribe (F_T)",
}
for label in sorted(df_all['label'].unique()):
    count = len(df_all[df_all['label'] == label])
    print(f"   {label_names.get(label, f'Unknown({label})')}: {count:5d} samples")