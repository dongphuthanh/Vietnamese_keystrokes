import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import os
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler

WIN_LENGTH = 200
STRIDE = 50

# --- NEW PARAMETERS ---
VOTING_MODE = 'soft'  
SCENARIO = 'M5'       # CHANGE THIS TO 'M2', 'M3', 'M4', or 'M5'

# Define what classes the model is allowed to see during training
SCENARIO_TRAIN_CLASSES = {
    'M2': [0, 2],          # B, T
    'M3': [0, 1, 2],       # B, P, T
    'M4': [0, 1, 2, 4],    # B, P, T, F_T
    'M5': [0, 1, 2, 3, 4]  # B, P, T, F_P, F_T
}

# ---------------------------------------------------------
# 1. HYBRID CNN ARCHITECTURE
# ---------------------------------------------------------

class HybridCNN(nn.Module):
    def __init__(self, feature_dim, num_classes,stats_dim = 3, hidden_dim = 32):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=feature_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.bn1=nn.BatchNorm1d(hidden_dim)
        self.dropout1=nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=2, dilation=2)
        self.bn2=nn.BatchNorm1d(hidden_dim * 2)
        self.dropout2=nn.Dropout(0.2)
        self.conv3=nn.Conv1d(hidden_dim * 2, hidden_dim * 4,kernel_size=3,padding=4, dilation=4)
        self.bn3=nn.BatchNorm1d(hidden_dim * 4)
        self.dropout3=nn.Dropout(0.3)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.stat_proj = nn.Sequential(
            nn.Linear(stats_dim, 16),
            nn.BatchNorm1d(16), # Stabilizes the scaled stats
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fc = nn.Linear(hidden_dim * 4 + 16, num_classes)
        
    def forward(self, x, x_stats, return_features=False):
        # x: (batch, seq_len, feature_dim)
        x = x.permute(0, 2, 1)  # (batch, feature_dim, seq_len)
        x = self.conv1(x)
        x=F.relu(self.bn1(x))
        x=self.dropout1(x)
        x = self.conv2(x)
        x=F.relu(self.bn2(x))
        x=self.dropout2(x)
        x = self.conv3(x)
        x=F.relu(self.bn3(x))
        x=self.dropout3(x)
        features = self.pool(x).squeeze(-1)
        projected_stats = self.stat_proj(x_stats)
        fused_vector = torch.cat((features, projected_stats), dim=1)
        if return_features:
            return features
    
        x = self.fc(fused_vector)  # (batch, num_classes)
        return x
# ---------------------------------------------------------
# 2. MODIFIED WINDOW GENERATOR (WITH GLOBAL STATS)
# ---------------------------------------------------------
def make_windows(filepath, X_win, X_stats, Y, ID, user_id, win_length, stride):
    with open(filepath, "r", encoding="utf8") as f:
        data = json.load(f)
    _keystrokes = data.get("keystrokes", [])
    
    is_attack = filepath.endswith("attack.json")

    # 0: B, 1: P, 2: T, 3: F_P, 4: F_T
    sessions = {0: [], 1: [], 2: [], 3: [], 4: []}
    
    for k in _keystrokes:
        if k["event"] == "keydown":
            session_id = k["session"]
            if not is_attack:
                if session_id == 1: sessions[0].append(k)
                elif session_id == 2: sessions[1].append(k)
                elif session_id == 3: sessions[2].append(k)
            else:
                if session_id == 2: sessions[3].append(k)
                elif session_id == 3: sessions[4].append(k)

    for label, keys in sessions.items():
        if len(keys) < win_length: continue
        
        # --- CALCULATE SESSION-LEVEL GLOBAL STATS ---
        dts = []
        spaces = []
        backspaces = []
        for m in range(1, len(keys)):
            dt = np.clip(keys[m]["timestamp"] - keys[m - 1]["timestamp"], 1, 5000)
            dts.append(np.log(dt))
            spaces.append(1 if keys[m-1]["key"] == " " else 0)
            backspaces.append(1 if keys[m]["key"] == "Backspace" else 0)
            
        if len(dts) == 0: continue
        
        session_global_stats = [
            np.mean(dts),
            np.std(dts),
            np.median(dts)
        ]

        # --- SLIDING WINDOW ---
        for i in range(1, len(keys) - win_length, stride):
            window = []
            for j in range(i, i + win_length):
                dt = keys[j]["timestamp"] - keys[j - 1]["timestamp"]
                dt = np.clip(dt, 1, 5000)
                is_space = 1 if (keys[j - 1]["key"] == " ") else 0
                is_backspace = 1 if (keys[j]["key"] == "Backspace") else 0
                
                window.append([
                    np.log(dt),
                    is_space,
                    is_backspace
                ])

            X_win.append(window)
            X_stats.append(session_global_stats) # Pair the global stats with the local window
            Y.append(label)
            ID.append(user_id)

# ---------------------------------------------------------
# 3. DATA EXTRACTION
# ---------------------------------------------------------
X_win, X_stats, Y, ID = [], [], [], []

folder_path = "../../dataset/Attack4"
# Graceful fallback if testing without the folder structure locally
if not os.path.exists(folder_path):
    print(f"Warning: Data folder {folder_path} not found. Please verify the path.")

user_folders = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
user2id = {user: i for i, user in enumerate(user_folders)}

print("Generating windows and global statistics...")
for user_folder in user_folders:
    user_id = user2id[user_folder]
    user_path = os.path.join(folder_path, user_folder)
    
    for root, _, files in os.walk(user_path):
        for filename in sorted(files):
            if filename.endswith(".json"):
                filepath = os.path.join(root, filename)
                make_windows(filepath, X_win, X_stats, Y, ID, user_id, WIN_LENGTH, STRIDE)

X_win = np.array(X_win)
X_stats = np.array(X_stats)
Y = np.array(Y)
ID = np.array(ID)

print(f"Dataset Loaded -> X_win: {X_win.shape} | X_stats: {X_stats.shape} | Y: {Y.shape} | ID: {ID.shape}")

# ---------------------------------------------------------
# 4. TRAINING LOOP WITH SCENARIO FILTERING
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_splits = 3
gkf = GroupKFold(n_splits=n_splits)

fold_accuracies = []
all_true_labels = []
all_pred_labels = []

allowed_train_classes = SCENARIO_TRAIN_CLASSES[SCENARIO]
print(f"\nRunning Scenario {SCENARIO}: Training on classes {allowed_train_classes}, Testing on ALL 5 classes.")

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_win, Y, groups=ID)):
    print(f"\n--- Fold {fold + 1}/{n_splits} ---")
    
    X_win_train, X_win_test = X_win[train_idx].copy(), X_win[test_idx].copy()
    X_stats_train, X_stats_test = X_stats[train_idx].copy(), X_stats[test_idx].copy()
    Y_train_fold, Y_test_fold = Y[train_idx], Y[test_idx]
    test_ids_fold = ID[test_idx]

    # --- THE SCENARIO FILTER ---
    train_mask = np.isin(Y_train_fold, allowed_train_classes)
    X_win_train = X_win_train[train_mask]
    X_stats_train = X_stats_train[train_mask]
    Y_train_fold = Y_train_fold[train_mask]

    # --- Robust Scaling (Windows) ---
    scaler_win = RobustScaler()
    train_dt = X_win_train[:, :, 0].reshape(-1, 1)
    test_dt = X_win_test[:, :, 0].reshape(-1, 1)
    X_win_train[:, :, 0] = scaler_win.fit_transform(train_dt).reshape(X_win_train.shape[0], X_win_train.shape[1])
    X_win_test[:, :, 0] = scaler_win.transform(test_dt).reshape(X_win_test.shape[0], X_win_test.shape[1])

    # --- Robust Scaling (Global Stats) ---
    scaler_stats = RobustScaler()
    X_stats_train = scaler_stats.fit_transform(X_stats_train)
    X_stats_test = scaler_stats.transform(X_stats_test)

    # --- DataLoaders (Now accommodating 2 inputs) ---
    train_ds = TensorDataset(
        torch.tensor(X_win_train, dtype=torch.float32), 
        torch.tensor(X_stats_train, dtype=torch.float32), 
        torch.tensor(Y_train_fold, dtype=torch.long)
    )
    test_ds = TensorDataset(
        torch.tensor(X_win_test, dtype=torch.float32), 
        torch.tensor(X_stats_test, dtype=torch.float32), 
        torch.tensor(Y_test_fold, dtype=torch.long)
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # Initialize Hybrid Model
    model = HybridCNN(feature_dim=X_win.shape[2], stats_dim=X_stats.shape[1], num_classes=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Training
    num_epochs = 100 
    for epoch in range(num_epochs):
        model.train()
        for xb_win, xb_stats, yb in train_loader:
            xb_win, xb_stats, yb = xb_win.to(device), xb_stats.to(device), yb.to(device)
            optimizer.zero_grad()
            # Pass both window and stats to the HybridCNN
            loss = criterion(model(xb_win, xb_stats), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Evaluation
    model.eval()
    session_results = {}
    current_idx = 0
    with torch.no_grad():
        for xb_win, xb_stats, yb in test_loader:
            xb_win, xb_stats, yb = xb_win.to(device), xb_stats.to(device), yb.to(device)
            out = model(xb_win, xb_stats)
            
            if VOTING_MODE == 'soft':
                store_vals = F.softmax(out, dim=1).cpu().numpy()
            else:
                store_vals = torch.argmax(out, dim=1).cpu().numpy()
                
            labels = yb.cpu().numpy()
            batch_ids = test_ids_fold[current_idx : current_idx + len(xb_win)]
            current_idx += len(xb_win)

            for j in range(len(store_vals)):
                key = (batch_ids[j], labels[j])
                if key not in session_results: 
                    session_results[key] = []
                session_results[key].append(store_vals[j])

    # Aggregate session votes
    fold_preds = []
    fold_labels = []
    for (uid, true_lab), vals_list in session_results.items():
        if VOTING_MODE == 'soft':
            mean_probs = np.mean(vals_list, axis=0)
            final_prediction = np.argmax(mean_probs)
        else:
            final_prediction = Counter(vals_list).most_common(1)[0][0]
            
        fold_preds.append(final_prediction)
        fold_labels.append(true_lab)
        all_pred_labels.append(final_prediction)
        all_true_labels.append(true_lab)

    fold_acc = np.mean(np.array(fold_preds) == np.array(fold_labels))
    fold_accuracies.append(fold_acc)
    print(f"Fold {fold+1} Session Accuracy: {fold_acc:.4f}")

# --- FINAL SUMMARY ---
print("-" * 30)
print(f"Scenario {SCENARIO} Average CV Accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
print("-" * 30)

# Plot Final Integrated Confusion Matrix
cm = confusion_matrix(all_true_labels, all_pred_labels, labels=[0, 1, 2, 3, 4])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["B", "P", "T", "F_P", "F_T"])
disp.plot(cmap=plt.cm.Purples)
plt.title(f"{SCENARIO} Confusion Matrix ({VOTING_MODE.capitalize()} Voting)")
plt.show()