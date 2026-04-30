import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

# Load data
with open('full.pkl', 'rb') as f:
    normal_data = pickle.load(f)

with open('attack.pkl', 'rb') as f:
    attack_data = pickle.load(f)

df_normal = pd.DataFrame(normal_data)
df_attack = pd.DataFrame(attack_data)

df_normal.fillna(0, inplace=True)
df_attack.fillna(0, inplace=True)

# Label encoding
le = LabelEncoder()
le.fit(['bonafide', 'paraphrase', 'transcribe'])
df_normal['label'] = le.transform(df_normal['label'])
df_attack['label'] = le.transform(df_attack['label'])
print("Label mapping: bonafide=0, paraphrase=1, transcribe=2")

# Thêm cột source để phân biệt normal vs attack
df_normal['source'] = 'normal'
df_attack['source'] = 'attack'

# 5-fold split theo user
all_users = sorted(df_normal['user_id'].unique())
kf = KFold(n_splits=3, shuffle=True, random_state=42)

drop_cols = ['user_id', 'session', 'section', 'file']
# KHÔNG drop 'source' — giữ lại để phân biệt F_P, F_T vs P, T

def make_csv(train_df, test_df, name, fold):
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
    train_df.fillna(0, inplace=True)
    test_df.fillna(0, inplace=True)
<<<<<<< HEAD:src/Minh code/code_make_datasets_temporal.py
    out = './'
=======
    out = '.'
>>>>>>> 8eea6159e8ceb28174a054173b7fe5040bc6eb00:src/Minh_code/code_make_datasets_temporal.py
    train_df.to_csv(f'{out}/train_{name}_fold{fold}.csv', index=False)
    test_df.to_csv(f'{out}/test_{name}_fold{fold}.csv', index=False)
    print(f'  {name} fold{fold} - Train: {len(train_df)} rows {train_df["label"].value_counts().to_dict()}, Test: {len(test_df)} rows {test_df["label"].value_counts().to_dict()}')

for fold, (train_idx, test_idx) in enumerate(kf.split(all_users), 1):
    print(f'\n=== Fold {fold} ===')
    train_users = [all_users[i] for i in train_idx]
    test_users = [all_users[i] for i in test_idx]

    df_train_users = df_normal[df_normal['user_id'].isin(train_users)]
    df_test_users = df_normal[df_normal['user_id'].isin(test_users)]
    df_attack_train = df_attack[df_attack['user_id'].isin(train_users)]
    df_attack_test = df_attack[df_attack['user_id'].isin(test_users)]

    # Test set chung: normal + attack của test users
    test_all = pd.concat([df_test_users, df_attack_test])

    # M2: Train = B + T
    train_m2 = df_train_users[df_train_users['label'].isin([0, 2])].copy()
    make_csv(train_m2, test_all, 'm2', fold)

    # M3: Train = B + T + P
    make_csv(df_train_users, test_all, 'm3', fold)

    # M4: Train = B + T + P + F_T
    attack_transcribe_train = df_attack_train[df_attack_train['label'] == 2]
    train_m4 = pd.concat([df_train_users, attack_transcribe_train])
    make_csv(train_m4, test_all, 'm4', fold)

    # M5: Train = tất cả
    train_m5 = pd.concat([df_train_users, df_attack_train])
    make_csv(train_m5, test_all, 'm5', fold)