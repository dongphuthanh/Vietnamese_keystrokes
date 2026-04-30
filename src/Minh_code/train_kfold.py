import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1. LOAD DỮ LIỆU
print("Đang tải dữ liệu...")
df_normal = pd.DataFrame(pd.read_pickle("full.pkl"))
df_attack = pd.DataFrame(pd.read_pickle("attack.pkl"))

# Gán nhãn nhị phân: Normal = 0, Attack = 1
df_normal['target'] = 0
df_attack['target'] = 1

# Gộp dữ liệu
df = pd.concat([df_normal, df_attack], ignore_index=True)

# Lọc bỏ các dòng có question_index = 0 (dữ liệu rác)
df = df[df['question_index'] != 0]

# 2. CHUẨN BỊ ĐẶC TRƯNG (FEATURES)
# Loại bỏ các cột không phải là đặc trưng gõ phím
exclude_cols = ['user_id', 'label', 'file', 'session', 'question_index', 'target']
features = [c for c in df.columns if c not in exclude_cols]

X = df[features]
y = df['target']
q_idx = df['question_index']

# 3. THỰC HIỆN 3-FOLD THEO QUESTION_INDEX
questions = sorted(q_idx.unique()) # ['1.1', '2.1', '3.1']
fold_results = []

print(f"\nBắt đầu huấn luyện 3-Fold với các câu hỏi: {questions}")

for i, test_q in enumerate(questions):
    print(f"\n--- FOLD {i+1}: Test trên câu hỏi {test_q} ---")
    
    # Chia Train/Test dựa trên ID câu hỏi
    train_mask = (q_idx != test_q)
    test_mask = (q_idx == test_q)
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    # Khởi tạo và huấn luyện model
    # n_jobs=-1 để dùng hết nhân CPU chạy cho nhanh
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # Dự đoán
    y_pred = clf.predict(X_test)
    
    # Đánh giá
    acc = accuracy_score(y_test, y_pred)
    fold_results.append(acc)
    
    print(f"Accuracy cho Fold {i+1}: {acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# 4. TỔNG KẾT
print("\n" + "="*30)
print(f"KẾT QUẢ CUỐI CÙNG (Average Accuracy): {np.mean(fold_results):.4f}")
print("="*30)

# Lưu model của fold cuối cùng để test nhanh nếu cần
joblib.dump(clf, "final_rf_model.pkl")
print("Đã lưu model vào file final_rf_model.pkl")