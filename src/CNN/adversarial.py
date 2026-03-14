import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

# Assuming custom imports from your local files
from cnn_gen_data import make_windows, TemporalCNN

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Configuration
WIN_LENGTH, STRIDE = 200, 50
NUM_EPOCHS, BATCH_SIZE = 100, 64
SEED = 42
set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 1. Load Everything Separately
# ---------------------------------------------------------
folder_path = "../dataset/Attack3"
user_folders = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
user2id = {user: i for i, user in enumerate(user_folders)}

X_c, Y_c, ID_c = [], [], []
X_a, Y_a, ID_a = [], [], []

for user_folder in user_folders:
    u_id = user2id[user_folder]
    u_path = os.path.join(folder_path, user_folder)
    for root, _, files in os.walk(u_path):
        for f in files:
            if not f.endswith(".json"): continue
            path = os.path.join(root, f)
            if f.lower().endswith("attack.json"):
                make_windows(path, X_a, Y_a, ID_a, u_id, WIN_LENGTH, STRIDE)
            else:
                make_windows(path, X_c, Y_c, ID_c, u_id, WIN_LENGTH, STRIDE)

X_c, Y_c, ID_c = np.array(X_c), np.array(Y_c), np.array(ID_c)
X_a, Y_a, ID_a = np.array(X_a), np.array(Y_a), np.array(ID_a)

# ---------------------------------------------------------
# 2. Adversarial GroupKFold Loop
# ---------------------------------------------------------
gkf = GroupKFold(n_splits=5)
all_clean_true, all_clean_pred = [], []
all_attack_true, all_attack_pred = [], []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_c, Y_c, groups=ID_c)):
    print(f"\n--- Fold {fold+1}/5 ---")
    
    # Identify Test Users
    test_uids = np.unique(ID_c[test_idx])
    
    # BUILD ADVERSARIAL TRAIN SET
    # 1. Get Clean Train windows
    X_train_fold = X_c[train_idx].copy()
    Y_train_fold = Y_c[train_idx]
    
    # 2. Add Attack windows for the same training users
    train_uids = np.unique(ID_c[train_idx])
    atk_train_mask = np.isin(ID_a, train_uids)
    X_train_fold = np.concatenate([X_train_fold, X_a[atk_train_mask]], axis=0)
    Y_train_fold = np.concatenate([Y_train_fold, Y_a[atk_train_mask]], axis=0)

    # BUILD TEST SETS
    X_test_c_fold = X_c[test_idx].copy()
    Y_test_c_fold = Y_c[test_idx]
    
    atk_test_mask = np.isin(ID_a, test_uids)
    X_test_a_fold = X_a[atk_test_mask].copy()
    Y_test_a_fold = Y_a[atk_test_mask]

    # SCALING (Fit on Adv-Train)
    scaler = RobustScaler()
    X_train_fold[:,:,0] = scaler.fit_transform(X_train_fold[:,:,0].reshape(-1,1)).reshape(X_train_fold.shape[0], WIN_LENGTH)
    X_test_c_fold[:,:,0] = scaler.transform(X_test_c_fold[:,:,0].reshape(-1,1)).reshape(X_test_c_fold.shape[0], WIN_LENGTH)
    X_test_a_fold[:,:,0] = scaler.transform(X_test_a_fold[:,:,0].reshape(-1,1)).reshape(X_test_a_fold.shape[0], WIN_LENGTH)

    # DataLoaders
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train_fold, dtype=torch.float32), torch.tensor(Y_train_fold, dtype=torch.long)), batch_size=BATCH_SIZE, shuffle=True)
    c_loader = DataLoader(TensorDataset(torch.tensor(X_test_c_fold, dtype=torch.float32), torch.tensor(Y_test_c_fold, dtype=torch.long)), batch_size=BATCH_SIZE)
    a_loader = DataLoader(TensorDataset(torch.tensor(X_test_a_fold, dtype=torch.float32), torch.tensor(Y_test_a_fold, dtype=torch.long)), batch_size=BATCH_SIZE)

    # INIT MODEL & TRAIN
    model = TemporalCNN(feature_dim=X_c.shape[2], num_classes=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()

    # EVALUATION
    def get_preds(loader, ids):
        model.eval()
        res = {}
        ptr = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                p = torch.argmax(model(xb), dim=1).cpu().numpy()
                b_ids = ids[ptr:ptr+len(xb)]
                b_labs = yb.numpy()
                ptr += len(xb)
                for j in range(len(p)):
                    k = (b_ids[j], b_labs[j])
                    if k not in res: res[k] = []
                    res[k].append(p[j])
        y_t, y_p = [], []
        for (_, l), p_list in res.items():
            y_t.append(l); y_p.append(Counter(p_list).most_common(1)[0][0])
        return y_t, y_p

    # Store results
    ct, cp = get_preds(c_loader, ID_c[test_idx])
    at, ap = get_preds(a_loader, ID_a[atk_test_mask])
    all_clean_true.extend(ct); all_clean_pred.extend(cp)
    all_attack_true.extend(at); all_attack_pred.extend(ap)
    
    print(f"Fold {fold+1} Clean Acc: {np.mean(np.array(ct)==np.array(cp)):.4f}")
    print(f"Fold {fold+1} Attack Acc: {np.mean(np.array(at)==np.array(ap)):.4f}")

# ---------------------------------------------------------
# 3. Final Visualization
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
ConfusionMatrixDisplay.from_predictions(all_clean_true, all_clean_pred, display_labels=["B", "P", "T"], cmap='Blues', ax=ax1, normalize=None,values_format='d')
ax1.set_title("Aggregated Clean Accuracy")

ConfusionMatrixDisplay.from_predictions(
    all_attack_true, 
    all_attack_pred, 
    labels=[0, 1, 2],               # <--- Add this! This forces the 3x3 shape
    display_labels=["B", "P", "T"], 
    cmap='Reds', 
    ax=ax2, 
    normalize=None,
    values_format='d'
)
ax2.set_title("Aggregated Attack Accuracy")
plt.tight_layout()
plt.show()