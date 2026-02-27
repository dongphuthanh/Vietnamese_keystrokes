import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from cnn_gen_data import make_windows
from cnn_gen_data import TemporalCNN
import random

def set_seed(seed=42):
    """Sets the seed for reproducibility across runs."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Configuration
WIN_LENGTH = 200
STRIDE = 50
NUM_EPOCHS = 150
BATCH_SIZE = 64
SEED = 42

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 1. Dataset Splitting by User (30 Train, 15 Test)
# ---------------------------------------------------------
folder_path = "../dataset/Attack3"
all_users = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])

if len(all_users) != 45:
    print(f"Warning: Found {len(all_users)} users, but expected 45.")

train_users = all_users[0:15] + all_users[30:45]
test_users = all_users[15:30]
user2id = {user: i for i, user in enumerate(all_users)}

# ---------------------------------------------------------
# 2. Selective Window Generation Function
# ---------------------------------------------------------
def process_data(user_list, mode='train'):
    """
    mode 'train': original files for training users
    mode 'test_clean': original files for testing users
    mode 'test_attack': attack files for testing users
    """
    X_local, Y_local, ID_local = [], [], []
    
    for user_folder in user_list:
        user_id = user2id[user_folder]
        user_path = os.path.join(folder_path, user_folder)
        
        for root, _, files in os.walk(user_path):
            for filename in files:
                if not filename.endswith(".json"):
                    continue
                
                filepath = os.path.join(root, filename)
                is_attack = filename.lower().endswith("attack.json")
                
                # Filtering Logic
                if mode == 'train' and not is_attack:
                    make_windows(filepath, X_local, Y_local, ID_local, user_id, WIN_LENGTH, STRIDE)
                elif mode == 'test_clean' and not is_attack:
                    make_windows(filepath, X_local, Y_local, ID_local, user_id, WIN_LENGTH, STRIDE)
                elif mode == 'test_attack' and is_attack:
                    make_windows(filepath, X_local, Y_local, ID_local, user_id, WIN_LENGTH, STRIDE)
                    
    return np.array(X_local), np.array(Y_local), np.array(ID_local)

print("--- Generating Datasets ---")
X_train, Y_train, ID_train = process_data(train_users, mode='train')
X_test_clean, Y_test_clean, ID_test_clean = process_data(test_users, mode='test_clean')
X_test_attack, Y_test_attack, ID_test_attack = process_data(test_users, mode='test_attack')

print(f"Train Shape: {X_train.shape}")
print(f"Test (Clean) Shape: {X_test_clean.shape}")
print(f"Test (Attack) Shape: {X_test_attack.shape}")

# ---------------------------------------------------------
# 3. Scaling (Fitted ONLY on Train)
# ---------------------------------------------------------
# Assuming feature 0 is log(dt). Scale them separately to avoid leakage.
scaler = RobustScaler()

# Flatten for fitting
train_dt_col = X_train[:, :, 0].reshape(-1, 1)
scaler.fit(train_dt_col)

def apply_scaler(X_data, scaler_obj):
    col = X_data[:, :, 0].reshape(-1, 1)
    X_data[:, :, 0] = scaler_obj.transform(col).reshape(X_data.shape[0], X_data.shape[1])
    return X_data

X_train = apply_scaler(X_train, scaler)
X_test_clean = apply_scaler(X_test_clean, scaler)
X_test_attack = apply_scaler(X_test_attack, scaler)

# ---------------------------------------------------------
# 4. DataLoaders
# ---------------------------------------------------------
train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                        torch.tensor(Y_train, dtype=torch.long)), 
                          batch_size=BATCH_SIZE, shuffle=True)

clean_loader = DataLoader(TensorDataset(torch.tensor(X_test_clean, dtype=torch.float32), 
                                        torch.tensor(Y_test_clean, dtype=torch.long)), 
                          batch_size=BATCH_SIZE, shuffle=False)

attack_loader = DataLoader(TensorDataset(torch.tensor(X_test_attack, dtype=torch.float32), 
                                         torch.tensor(Y_test_attack, dtype=torch.long)), 
                           batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------------------------------------
# 5. Model Initialization and Training
# ---------------------------------------------------------
model = TemporalCNN(feature_dim=X_train.shape[2], num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

print("\n--- Starting Training ---")
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    scheduler.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss/len(train_loader):.4f}")

# ---------------------------------------------------------
# 6. Evaluation Function (Session-Level Majority Vote)
# ---------------------------------------------------------
def evaluate_model(loader, original_ids, title="Evaluation"):
    model.eval()
    session_results = {}
    current_idx = 0
    
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            preds = torch.argmax(F.softmax(out, dim=1), dim=1).cpu().numpy()
            labels = yb.numpy()
            
            # Match current batch to IDs
            batch_ids = original_ids[current_idx : current_idx + len(xb)]
            current_idx += len(xb)

            for j in range(len(preds)):
                uid = batch_ids[j]
                true_lab = labels[j]
                session_key = (uid, true_lab)
                
                if session_key not in session_results:
                    session_results[session_key] = []
                session_results[session_key].append(preds[j])

    # Aggregate Votes
    all_true, all_pred = [], []
    for (uid, true_label), window_preds in session_results.items():
        majority_vote = Counter(window_preds).most_common(1)[0][0]
        all_true.append(true_label)
        all_pred.append(majority_vote)
    
    acc = np.mean(np.array(all_true) == np.array(all_pred))
    print(f"\n[{title}] Session-Level Accuracy: {acc:.4f}")
    
    return all_true, all_pred, acc

# ---------------------------------------------------------
# 7. Comparison and Visualization
# ---------------------------------------------------------
# Evaluate Clean Test Set
y_true_c_full, y_pred_c_full, acc_c_full = evaluate_model(clean_loader, ID_test_clean, "CLEAN (ALL SESSIONS)")

# Evaluate Attack Test Set (Contains only S2 & S3)
y_true_a, y_pred_a, acc_a = evaluate_model(attack_loader, ID_test_attack, "ATTACK (S2 & S3)")

# --- Filter Clean Results for Fair Comparison ---
y_true_c_filtered = []
y_pred_c_filtered = []

for t, p in zip(y_true_c_full, y_pred_c_full):
    # Keep only Paraphrase (1) and Transcribe (2) sessions
    if t in [1, 2]:
        y_true_c_filtered.append(t)
        y_pred_c_filtered.append(p)

clean_s23_acc = np.mean(np.array(y_true_c_filtered) == np.array(y_pred_c_filtered))

print(f"\n--- Final Robustness Report (Sessions 2 & 3 Only) ---")
print(f"Clean Version Accuracy: {clean_s23_acc:.4f}")
print(f"Attack Version Accuracy: {acc_a:.4f}")
print(f"Vulnerability Gap: {clean_s23_acc - acc_a:.4f}")

# --- Side-by-Side Confusion Matrices ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
label_names = ["bonafide", "paraphrase", "transcribe"]
# Add this: the integer IDs of your classes
all_class_ids = [0, 1, 2] 

# Plot 1: Clean Versions (Filtered S2 & S3)
ConfusionMatrixDisplay.from_predictions(
    y_true_c_filtered, 
    y_pred_c_filtered, 
    labels=all_class_ids,         # <--- Add this line
    display_labels=label_names, 
    cmap=plt.cm.Blues, 
    ax=ax1, 
    colorbar=False,
    normalize='true'
)
ax1.set_title(f"Clean Versions (S2 & S3 Only)\nAccuracy: {clean_s23_acc:.2%}")

# Plot 2: Attack Versions (S2 & S3)
ConfusionMatrixDisplay.from_predictions(
    y_true_a, 
    y_pred_a, 
    labels=all_class_ids,         # <--- Add this line
    display_labels=label_names, 
    cmap=plt.cm.Reds, 
    ax=ax2, 
    colorbar=False,
    normalize='true'
)
ax2.set_title(f"Attack Versions (S2 & S3)\nAccuracy: {acc_a:.2%}")

plt.tight_layout()
plt.show()