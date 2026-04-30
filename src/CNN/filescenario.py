import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import os
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import RobustScaler

# Make sure your cnn_gen_data.py has the TemporalCNN imported
from cnn_gen_data import TemporalCNN

WIN_LENGTH = 200
STRIDE = 50

# --- NEW PARAMETERS ---
RUN_MODE = 'final'   # CHANGE THIS TO 'tuning' OR 'final'
VOTING_MODE = 'soft'  
SCENARIO = 'M5'       # CHANGE THIS TO 'M2', 'M3', 'M4', or 'M5'

# Define what classes the model is allowed to see during training
SCENARIO_TRAIN_CLASSES = {
    'M2': [0, 2],          # B, T
    'M3': [0, 1, 2],       # B, P, T
    'M4': [0, 1, 2, 4],    # B, P, T, F_T
    'M5': [0, 1, 2, 3, 4], # B, P, T, F_P, F_T
    'M6': [0, 1]
}

# ---------------------------------------------------------
# 1. MODIFIED WINDOW GENERATOR (WITH FILE ID)
# ---------------------------------------------------------
def make_windows(filepath, X, Y, ID, FILE_ID, user_id, file_id, win_length, stride):
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
            FILE_ID.append(file_id) 

# ---------------------------------------------------------
# 2. DATA EXTRACTION
# ---------------------------------------------------------
X, Y, ID, FILE_ID = [], [], [], []

folder_path = "../../dataset/Attack4"
user_folders = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
user2id = {user: i for i, user in enumerate(user_folders)}

print("Generating windows...")
file_counter = 0 

for user_folder in user_folders:
    user_id = user2id[user_folder]
    user_path = os.path.join(folder_path, user_folder)
    
    for root, _, files in os.walk(user_path):
        for filename in sorted(files):
            if filename.endswith(".json"):
                filepath = os.path.join(root, filename)
                make_windows(filepath, X, Y, ID, FILE_ID, user_id, file_counter, WIN_LENGTH, STRIDE)
                file_counter += 1

X = np.array(X)
Y = np.array(Y)
ID = np.array(ID)
FILE_ID = np.array(FILE_ID) 

print(f"Dataset Loaded -> X: {X.shape} | Y: {Y.shape} | ID: {ID.shape} | Files: {file_counter}")

# ---------------------------------------------------------
# 3. TRAINING LOOP WITH SCENARIO FILTERING & MODES
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_splits = 3
gkf = GroupKFold(n_splits=n_splits)

fold_accuracies = []
all_true_labels = []
all_pred_labels = []

allowed_train_classes = SCENARIO_TRAIN_CLASSES[SCENARIO]
print(f"\nRunning Scenario {SCENARIO}: Training on classes {allowed_train_classes}, Testing on ALL 5 classes.")
print(f"*** SCRIPT RUNNING IN [{RUN_MODE.upper()}] MODE ***\n")

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, Y, groups=ID)):
    print(f"\n--- Fold {fold + 1}/{n_splits} ---")
    
    # --- MODE ROUTING LOGIC ---
    if RUN_MODE == 'tuning':
        # Split the training fold further (80/20) while keeping users isolated
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        sub_train_idx, sub_val_idx = next(gss.split(X[train_idx], Y[train_idx], groups=ID[train_idx]))
        
        X_train_fold = X[train_idx][sub_train_idx].copy()
        Y_train_fold = Y[train_idx][sub_train_idx]
        
        X_eval = X[train_idx][sub_val_idx].copy()
        Y_eval = Y[train_idx][sub_val_idx]
        eval_fids = FILE_ID[train_idx][sub_val_idx]
        print(f"HYPERTUNING: Evaluating on Validation Set (Train: {len(X_train_fold)} w | Val: {len(X_eval)} w)")
        
    else:
        # Final mode: Train on the full fold, evaluate on the unseen Test fold
        X_train_fold = X[train_idx].copy()
        Y_train_fold = Y[train_idx]
        
        X_eval = X[test_idx].copy()
        Y_eval = Y[test_idx]
        eval_fids = FILE_ID[test_idx]
        print(f"FINAL: Evaluating on Unseen Test Set (Train: {len(X_train_fold)} w | Test: {len(X_eval)} w)")

    # --- THE SCENARIO FILTER ---
    # We only apply this to the Train set, so the eval set can still test false accepts
    train_mask = np.isin(Y_train_fold, allowed_train_classes)
    X_train_fold = X_train_fold[train_mask]
    Y_train_fold = Y_train_fold[train_mask]

    # --- Robust Scaling ---
    scaler = RobustScaler()
    train_dt = X_train_fold[:, :, 0].reshape(-1, 1)
    eval_dt = X_eval[:, :, 0].reshape(-1, 1)
    
    X_train_fold[:, :, 0] = scaler.fit_transform(train_dt).reshape(X_train_fold.shape[0], X_train_fold.shape[1])
    X_eval[:, :, 0] = scaler.transform(eval_dt).reshape(X_eval.shape[0], X_eval.shape[1])

    train_ds = TensorDataset(torch.tensor(X_train_fold, dtype=torch.float32), torch.tensor(Y_train_fold, dtype=torch.long))
    eval_ds = TensorDataset(torch.tensor(X_eval, dtype=torch.float32), torch.tensor(Y_eval, dtype=torch.long))
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=64, shuffle=False)

    # Initialize Model
    model = TemporalCNN(feature_dim=X.shape[2], num_classes=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.00)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Training
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Evaluation (Works seamlessly for both Validation and Test sets)
    model.eval()
    session_results = {}
    current_idx = 0
    with torch.no_grad():
        for xb, yb in eval_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            
            if VOTING_MODE == 'soft':
                store_vals = F.softmax(out, dim=1).cpu().numpy()
            else:
                store_vals = torch.argmax(out, dim=1).cpu().numpy()
                
            labels = yb.cpu().numpy()
            
            batch_fids = eval_fids[current_idx : current_idx + len(xb)]
            current_idx += len(xb)

            for j in range(len(store_vals)):
                # Group by (File ID, Session Label)
                key = (batch_fids[j], labels[j])
                if key not in session_results: 
                    session_results[key] = []
                session_results[key].append(store_vals[j])

    # Aggregate session votes
    fold_preds = []
    fold_labels = []
    
    for (fid, true_lab), vals_list in session_results.items():
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
    
    eval_type = "Validation" if RUN_MODE == 'tuning' else "Test"
    print(f"Fold {fold+1} Session-Level {eval_type} Accuracy: {fold_acc:.4f}")

# --- FINAL SUMMARY ---
print("-" * 30)
final_metric_name = "Validation" if RUN_MODE == 'tuning' else "Test"
print(f"Scenario {SCENARIO} Average Session-Level {final_metric_name} Accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
print("-" * 30)

# Plot Final Integrated Confusion Matrix
cm = confusion_matrix(all_true_labels, all_pred_labels, labels=[0, 1, 2, 3, 4])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["B", "P", "T", "F_P", "F_T"])
disp.plot(cmap=plt.cm.Purples)
plt.title(f"{SCENARIO} Confusion Matrix ({RUN_MODE.capitalize()} Mode)")
plt.show()