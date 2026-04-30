import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

print("Đang tải dữ liệu...")
df_normal = pd.DataFrame(pd.read_pickle("full.pkl"))
df_attack = pd.DataFrame(pd.read_pickle("attack.pkl"))

df_normal['target_bin'] = 0
df_attack['target_bin'] = 1

df = pd.concat([df_normal, df_attack], ignore_index=True)
df = df[df['question_index'] != 0].copy()

# Gộp level (1.1 -> 1)
df['q_main'] = df['question_index'].astype(str).str.split('.').str[0]

# Gán nhãn 5 class: B(0), P(1), T(2), F_P(3), F_T(4)
conditions = [
    (df['target_bin'] == 0) & (df['q_main'] == '1'),
    (df['target_bin'] == 0) & (df['q_main'] == '2'),
    (df['target_bin'] == 0) & (df['q_main'] == '3'),
    (df['target_bin'] == 1) & (df['q_main'] == '2'),
    (df['target_bin'] == 1) & (df['q_main'] == '3')
]
choices = [0, 1, 2, 3, 4]
# Đặt tên cột là 'label' để khớp hoàn toàn với yêu cầu của file XGBoost
df['label'] = np.select(conditions, choices, default=-1)

# Xóa các cột rác để tránh việc XGBoost nhận nhầm thành Feature
cols_to_drop = ['question_index', 'target_bin', 'q_main']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

print("Đang chia dữ liệu 2/3 Train, 1/3 Test cho toàn bộ hệ thống...")
train_list = []
test_list = []
users = df['user_id'].unique()

for user in users:
    df_user = df[df['user_id'] == user]
    
    # Chia 2/3 Train, 1/3 Test (test_size ≈ 0.33)
    try:
        user_train, user_test = train_test_split(
            df_user, 
            test_size=0.33, 
            random_state=42, 
            stratify=df_user['label']
        )
        train_list.append(user_train)
        test_list.append(user_test)
    except ValueError:
        user_train, user_test = train_test_split(df_user, test_size=0.33, random_state=42)
        train_list.append(user_train)
        test_list.append(user_test)

df_train_global = pd.concat(train_list, ignore_index=True)
df_test_global = pd.concat(test_list, ignore_index=True)

# Lưu ra 2 file CSV
df_train_global.to_csv("train_m5.csv", index=False)
df_test_global.to_csv("test_m5.csv", index=False)

print(f"Xong! Đã tạo 'train_m5.csv' ({len(df_train_global)} dòng) và 'test_m5.csv' ({len(df_test_global)} dòng).")