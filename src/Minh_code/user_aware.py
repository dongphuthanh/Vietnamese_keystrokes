import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# ==========================================
# 1. LOAD VÀ CHUẨN BỊ DỮ LIỆU ĐA LỚP
# ==========================================
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
df['multi_target'] = np.select(conditions, choices, default=-1)

exclude_cols = ['user_id', 'label', 'file', 'session', 'question_index', 'target_bin', 'q_main', 'multi_target']
features = [c for c in df.columns if c not in exclude_cols]

# ==========================================
# 2. CHIA DATA 2/3 TRAIN - 1/3 TEST CHO TỪNG USER
# ==========================================
print("Đang chia dữ liệu 2/3 Train, 1/3 Test cho toàn bộ hệ thống...")
train_list = []
test_list = []

users = df['user_id'].unique()

for user in users:
    df_user = df[df['user_id'] == user]
    
    # Dùng train_test_split để chia ngẫu nhiên 2/3 Train, 1/3 Test (test_size ≈ 0.33)
    # stratify=df_user['multi_target'] để đảm bảo cả Train và Test đều có đủ các loại câu B, P, T, F_P, F_T
    try:
        user_train, user_test = train_test_split(
            df_user, 
            test_size=0.33, 
            random_state=42, 
            stratify=df_user['multi_target']
        )
        train_list.append(user_train)
        test_list.append(user_test)
    except ValueError:
        # Xử lý trường hợp user có quá ít dữ liệu ở 1 class không đủ để stratify
        user_train, user_test = train_test_split(df_user, test_size=0.33, random_state=42)
        train_list.append(user_train)
        test_list.append(user_test)

# Gom tất cả lại thành 1 tập Train lớn và 1 tập Test lớn
df_train_global = pd.concat(train_list, ignore_index=True)
df_test_global = pd.concat(test_list, ignore_index=True)

X_train, y_train = df_train_global[features], df_train_global['multi_target']
X_test, y_test = df_test_global[features], df_test_global['multi_target']

# ==========================================
# 3. HUẤN LUYỆN MODEL GLOBAL (M5)
# ==========================================
print(f"Bắt đầu Train mô hình XGBoost duy nhất trên {len(X_train)} mẫu...")
clf = XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=5, 
    random_state=42, 
    objective='multi:softprob', 
    num_class=5, 
    eval_metric='mlogloss'
)

clf.fit(X_train, y_train)

print(f"Đang dự đoán trên {len(X_test)} mẫu Test...")
y_pred = clf.predict(X_test)

# ==========================================
# 4. ĐÁNH GIÁ (MATRIX, FAR, FRR)
# ==========================================
class_names = ['B', 'P', 'T', 'F_P', 'F_T']
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4])
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

acc = accuracy_score(y_test, y_pred)

# FAR: Kẻ gian (3,4) lọt lưới thành Chủ nhân (0,1,2)
FA = np.sum(cm[3:5, 0:3])
Total_Impostors = np.sum(cm[3:5, :])
FAR = FA / Total_Impostors if Total_Impostors > 0 else 0

# FRR: Chủ nhân (0,1,2) bị đánh oan thành Kẻ gian (3,4)
FR = np.sum(cm[0:3, 3:5])
Total_Genuines = np.sum(cm[0:3, :])
FRR = FR / Total_Genuines if Total_Genuines > 0 else 0

print("\n" + "="*50)
print("CONFUSION MATRIX FOR M5 (GLOBAL MODEL - 2/3 TRAIN)")
print("="*50)
print(cm_df.to_string())
print("-" * 50)
print(f"Accuracy: {acc:.2%}")
print(f"FAR (False Acceptance Rate): {FAR:.2%}")
print(f"FRR (False Rejection Rate) : {FRR:.2%}")
print("="*50)