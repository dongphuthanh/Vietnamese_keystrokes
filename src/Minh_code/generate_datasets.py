import pickle
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold

# 1. TÊN FILE CHUẨN
NORMAL_PKL = "full.pkl"
ATTACK_PKL = "attack.pkl"
OUT_DIR = "prepared_datasets"

os.makedirs(OUT_DIR, exist_ok=True)

# 2. HÀM GÁN NHÃN
def assign_label(row):
    try:
        sess = int(row['session'])
    except:
        return -1

    if row['source'] == 'normal':
        if sess == 1: return 0
        if sess == 2: return 1
        if sess == 3: return 2
    elif row['source'] == 'attack':
        if sess == 2: return 3
        if sess == 3: return 4
    return -1

# 3. HÀM ĐỌC DỮ LIỆU
def load_and_prep(pkl_path, source_name):
    print(f"Đang đọc file: {pkl_path} ...")
    if not os.path.exists(pkl_path):
        print(f"❌ LỖI: Không tìm thấy file {pkl_path}!")
        return pd.DataFrame()
        
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    df = pd.DataFrame(data)
    df.fillna(0, inplace=True)
    df['source'] = source_name
    df['label'] = df.apply(assign_label, axis=1)
    print(f"✅ Đã đọc thành công {len(df)} dòng từ {pkl_path}")
    return df

df_normal = load_and_prep(NORMAL_PKL, "normal")
df_attack = load_and_prep(ATTACK_PKL, "attack")

if df_normal.empty or df_attack.empty:
    print("❌ Dừng chạy vì thiếu file.")
    exit()

df_all = pd.concat([df_normal, df_attack], ignore_index=True)

# 4. CHIA USER THÀNH 3 FOLDS BẰNG KFOLD
unique_users = df_normal['user_id'].unique()
print(f"\nTổng số users tìm thấy: {len(unique_users)} người.")

# Khởi tạo KFold chia 3 phần
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# 5. KỊCH BẢN M2, M3, M4, M5
scenarios = {
    "M2": [("normal", [1, 3])], 
    "M3": [("normal", [1, 2, 3])], 
    "M4": [("normal", [1, 2, 3]), ("attack", [3])],
    "M5": [("normal", [1, 2, 3]), ("attack", [2, 3])]
}

print("\n--- BẮT ĐẦU TẠO CSV CHO 3 FOLDS ---")

for fold, (train_idx, test_idx) in enumerate(kf.split(unique_users), 1):
    print(f"\n=====================================")
    print(f"🚀 ĐANG TẠO DỮ LIỆU CHO FOLD {fold}...")
    print(f"=====================================")
    
    train_users = unique_users[train_idx]
    test_users = unique_users[test_idx]
    
    # Tập Test chung cho Fold này
    test_df_master = df_all[df_all['user_id'].isin(test_users)].copy()
    test_df_master = test_df_master.sample(frac=1, random_state=42).reset_index(drop=True)

    for scenario_name, conditions in scenarios.items():
        train_frames = []
        for source, sessions in conditions:
            temp_df = df_all[
                (df_all['user_id'].isin(train_users)) & 
                (df_all['source'] == source) & 
                (df_all['session'].astype(int).isin(sessions))
            ]
            train_frames.append(temp_df)
            
        if not train_frames:
            continue
            
        train_df = pd.concat(train_frames, ignore_index=True)
        if len(train_df) > 0:
            train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        train_csv = os.path.join(OUT_DIR, f"train_{scenario_name}_fold{fold}.csv")
        test_csv = os.path.join(OUT_DIR, f"test_{scenario_name}_fold{fold}.csv")
        
        train_df.to_csv(train_csv, index=False)
        test_df_master.to_csv(test_csv, index=False)
        
        print(f"👉 {scenario_name}: Train {len(train_df)} dòng | Test {len(test_df_master)} dòng")

print(f"\n🎉 XONG! 24 file CSV (4 kịch bản x 3 folds x 2 train/test) đã nằm trong folder: {OUT_DIR}")
