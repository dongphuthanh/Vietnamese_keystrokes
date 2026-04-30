import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import os
import copy
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import RobustScaler

# Make sure your cnn_gen_data.py has the TemporalCNN imported
from cnn_gen_data import TemporalCNN

WIN_LENGTH = 200
STRIDE = 50

# --- NEW PARAMETERS ---
VOTING_MODE = 'soft'  
SCENARIO = 'M5'       
PATIENCE = 15          # How many epochs to wait before early stopping

# Define what classes the model is allowed to see during training
SCENARIO_TRAIN_CLASSES = {
    'M2': [0, 2],          # B, T
    'M3': [0, 1, 2],       # B, P, T
    'M4': [0, 1, 2, 4],    # B, P, T, F_T
    'M5': [0, 1, 2, 3, 4], # B, P, T, F_P, F_T
    'M6': [0, 1]
}

# ---------------------------------------------------------
# 1. MODIFIED WINDOW GENERATOR
# ---------------------------------------------------------
def make_windows(filepath, X, Y, ID, user_id, win_length, stride):
    with open(filepath, "r", encoding="utf8") as f:
        data = json.load(f)
    _keystrokes = data.get("keystrokes", [])
    
    is_attack = filepath.endswith("attack.json")

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
        if len(keys) == 0: continue
        
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

            X.append(window)
            Y.append(label)
            ID.append(user_id)

# ---------------------------------------------------------
# 2. DATA EXTRACTION
# ---------------------------------------------------------
X, Y, ID = [], [], []

folder_path = "../../dataset/Attack4"
user_folders = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
user2id = {user: i for i, user in enumerate(user_folders)}

print("Generating windows...")
for user_folder in user_folders:
    user_id = user2id[user_folder]
    user_path = os.path.join(folder_path, user_folder)
    
    for root, _, files in os.walk(user_path):
        for filename in sorted(files):
            if filename.endswith(".json"):
                filepath = os.path.join(root, filename)
                make_windows(filepath, X, Y, ID, user_id, WIN_LENGTH, STRIDE)

X = np.array(X)
Y = np.array(Y)
ID = np.array(ID)

print(f"Dataset Loaded -> X: {X.shape} | Y: {Y.shape} | ID: {ID.shape}")

# ---------------------------------------------------------
# 3. TRAINING LOOP WITH SCENARIO FILTERING & EARLY STOPPING
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_splits = 3
gkf = GroupKFold(n_splits=n_splits)

fold_accuracies = []
all_true_labels = []
all_pred_labels = []

allowed_train_classes = SCENARIO_TRAIN_CLASSES[SCENARIO]
print(f"\nRunning Scenario {SCENARIO}: Training on classes {allowed_train_classes}, Testing on ALL 5 classes.")

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, Y, groups=ID)):
    print(f"\n--- Fold {fold + 1}/{n_splits} ---")
    
    X_train_fold, X_test_fold = X[train_idx].copy(), X[test_idx].copy()
    Y_train_fold, Y_test_fold = Y[train_idx], Y[test_idx]
    train_ids_fold = ID[train_idx].copy()
    test_ids_fold = ID[test_idx]

    # --- THE SCENARIO FILTER ---
    train_mask = np.isin(Y_train_fold, allowed_train_classes)
    X_train_fold = X_train_fold[train_mask]
    Y_train_fold = Y_train_fold[train_mask]
    train_ids_fold = train_ids_fold[train_mask]

    # --- VALIDATION SPLIT ---
    # Reserve 15% of the remaining training users for validation
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_sub_idx, val_idx = next(gss.split(X_train_fold, Y_train_fold, groups=train_ids_fold))

    X_train_sub, Y_train_sub = X_train_fold[train_sub_idx], Y_train_fold[train_sub_idx]
    X_val, Y_val = X_train_fold[val_idx], Y_train_fold[val_idx]

    # --- ROBUST SCALING ---
    scaler = RobustScaler()
    train_dt = X_train_sub[:, :, 0].reshape(-1, 1)
    val_dt = X_val[:, :, 0].reshape(-1, 1)
    test_dt = X_test_fold[:, :, 0].reshape(-1, 1)
    
    X_train_sub[:, :, 0] = scaler.fit_transform(train_dt).reshape(X_train_sub.shape[0], X_train_sub.shape[1])
    X_val[:, :, 0] = scaler.transform(val_dt).reshape(X_val.shape[0], X_val.shape[1])
    X_test_fold[:, :, 0] = scaler.transform(test_dt).reshape(X_test_fold.shape[0], X_test_fold.shape[1])

    # --- DATALOADERS ---
    train_ds = TensorDataset(torch.tensor(X_train_sub, dtype=torch.float32), torch.tensor(Y_train_sub, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test_fold, dtype=torch.float32), torch.tensor(Y_test_fold, dtype=torch.long))
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # Initialize Model
    model = TemporalCNN(feature_dim=X.shape[2], num_classes=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # --- EARLY STOPPING SETUP ---
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_weights = None
    num_epochs = 100

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()

        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Check Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}! Restoring best weights from epoch {epoch+1-PATIENCE}.")
            break

    # Restore the best model weights before evaluating the test set
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    # --- EVALUATION ---
    model.eval()
    session_results = {}
    current_idx = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            
            if VOTING_MODE == 'soft':
                store_vals = F.softmax(out, dim=1).cpu().numpy()
            else:
                store_vals = torch.argmax(out, dim=1).cpu().numpy()
                
            labels = yb.cpu().numpy()
            batch_ids = test_ids_fold[current_idx : current_idx + len(xb)]
            current_idx += len(xb)

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