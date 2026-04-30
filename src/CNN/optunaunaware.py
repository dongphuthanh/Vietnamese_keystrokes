import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import os
import optuna
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler

# ---------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------
VOTING_MODE = 'soft'  
SCENARIO = 'M5'       
OPTUNA_TRIALS = 40    # Number of hyperparameter guesses per fold
OPTUNA_EPOCHS = 30    # Micro-epochs for fast tuning in the inner loop
FINAL_EPOCHS = 100
WIN_LENGTH = 200
STRIDE = 50

SCENARIO_TRAIN_CLASSES = {
    'M2': [0, 2],          
    'M3': [0, 1, 2],       
    'M4': [0, 1, 2, 4],    
    'M5': [0, 1, 2, 3, 4], 
    'M6': [0, 1]
}

# --- THE CUSTOM QUESTION GROUPS ---
QUESTION_GROUPS = [
    [1, 4],
    [2, 5],
    [3, 6]
]

# ---------------------------------------------------------
# 1. UPGRADED MODEL: DYNAMIC DIMENSIONS
# ---------------------------------------------------------
class TemporalCNN(nn.Module):
    def __init__(self, continuous_dim=1, num_classes=5, hidden_dim=64, 
                 proj_dim=8, embedding_dim=4, 
                 dropout1=0.2, dropout2=0.2, dropout3=0.3):
        super().__init__()
        self.projector = nn.Linear(continuous_dim, proj_dim)
        self.embedding = nn.Embedding(num_embeddings=5, embedding_dim=embedding_dim)
        conv_in_dim = proj_dim + embedding_dim
        
        self.conv1 = nn.Conv1d(in_channels=conv_in_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout1)
        
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.dropout2 = nn.Dropout(dropout2)
        
        self.conv3 = nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=4, dilation=4)
        self.bn3 = nn.BatchNorm1d(hidden_dim * 4)
        self.dropout3 = nn.Dropout(dropout3)
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden_dim * 4, num_classes)
        
    def forward(self, x, return_features=False):
        cont_feats = x[:, :, :-1]     
        cont_feats = F.relu(self.projector(cont_feats))
        key_ids = x[:, :, -1].long()    
        embedded_keys = self.embedding(key_ids) 
        
        x_combined = torch.cat([cont_feats, embedded_keys], dim=2) 
        x_combined = x_combined.permute(0, 2, 1)  
        
        x = self.conv1(x_combined)
        x = F.relu(self.bn1(x))
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.dropout3(x)
        
        features = self.pool(x).squeeze(-1)

        if return_features: return features
        return self.fc(features)

# ---------------------------------------------------------
# 2. DATA GENERATOR (TRACKING QUESTION & SESSION)
# ---------------------------------------------------------
def get_key_id(key_str):
    if len(key_str) == 1:
        if key_str.isalpha(): return 0
        elif key_str.isdigit(): return 1               
        elif key_str == ' ': return 2                               
    elif key_str == 'Backspace': return 3                                  
    return 4                                      

def make_windows(filepath, X, Y, ID, FILE_ID, Q_ID, user_id, file_id, win_length, stride):
    with open(filepath, "r", encoding="utf8") as f:
        data = json.load(f)
    _keystrokes = data.get("keystrokes", [])
    
    is_attack = filepath.endswith("attack.json")

    # Group dynamically by (label, q_id)
    grouped_keys = {}
    
    for k in _keystrokes:
        if k["event"] == "keydown":
            q_idx_str = k.get("question_index")
            if not q_idx_str: continue 
                
            parts = str(q_idx_str).split('.')
            if len(parts) != 2: continue
                
            q_id = int(parts[1])
            s_id = int(parts[0])
            
            label = None
            if not is_attack:
                if s_id == 1: label = 0     # B
                elif s_id == 2: label = 1   # P
                elif s_id == 3: label = 2   # T
            else:
                if s_id == 2: label = 3     # F_P
                elif s_id == 3: label = 4   # F_T

            if label is not None:
                group_key = (label, q_id)
                if group_key not in grouped_keys:
                    grouped_keys[group_key] = []
                grouped_keys[group_key].append(k)

    # Extract windows for each group
    for (label, q_id), keys in grouped_keys.items():
        if len(keys) == 0: continue
        for i in range(1, len(keys) - win_length, stride):
            window = []
            for j in range(i, i + win_length):
                dt = keys[j]["timestamp"] - keys[j - 1]["timestamp"]
                dt = np.clip(dt, 1, 5000)
                key_id = get_key_id(keys[j]["key"])
                window.append([np.log(dt), key_id])

            X.append(window)
            Y.append(label)
            ID.append(user_id)
            FILE_ID.append(file_id) 
            Q_ID.append(q_id) 

# ---------------------------------------------------------
# 3. DATA EXTRACTION
# ---------------------------------------------------------
X, Y, ID, FILE_ID, Q_ID = [], [], [], [], []

folder_path = "../../dataset/Attack4"
user_folders = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
user2id = {user: i for i, user in enumerate(user_folders)}

print("Generating windows with Categorical Key IDs and Question Tracking...")
file_counter = 0 

for user_folder in user_folders:
    user_id = user2id[user_folder]
    user_path = os.path.join(folder_path, user_folder)
    for root, _, files in os.walk(user_path):
        for filename in sorted(files):
            if filename.endswith(".json"):
                filepath = os.path.join(root, filename)
                make_windows(filepath, X, Y, ID, FILE_ID, Q_ID, user_id, file_counter, WIN_LENGTH, STRIDE)
                file_counter += 1

X = np.array(X)
Y = np.array(Y)
ID = np.array(ID)
FILE_ID = np.array(FILE_ID) 
Q_ID = np.array(Q_ID)

print(f"Dataset Loaded -> X: {X.shape} | Y: {Y.shape} | Files: {file_counter} | Questions: {np.unique(Q_ID)}")

# ---------------------------------------------------------
# 4. DOUBLE NESTED CROSS-CONTENT EVALUATION LOOP
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optuna.logging.set_verbosity(optuna.logging.WARNING) 
n_splits = len(QUESTION_GROUPS)

fold_accuracies = []
fold_window_accuracies = [] 
all_true_labels = []
all_pred_labels = []
all_pred_probs = [] 
fold_cm_data = []

allowed_train_classes = SCENARIO_TRAIN_CLASSES[SCENARIO]
print(f"\nRunning Scenario {SCENARIO}: Double Nested CV Tuning & Testing (Cross-Question)")

for fold in range(n_splits):
    test_questions = QUESTION_GROUPS[fold]
    train_questions = [q for i, group in enumerate(QUESTION_GROUPS) if i != fold for q in group]
    
    print(f"\n" + "="*50)
    print(f"OUTER FOLD {fold + 1}/{n_splits} - INITIALIZING OPTUNA EXPERIMENT")
    print(f"Train Questions: {train_questions} | Test Questions: {test_questions}")
    print("="*50)

    train_content_mask = np.isin(Q_ID, train_questions)
    test_content_mask = np.isin(Q_ID, test_questions)

    # ---------------------------------------------------------
    # INNER LOOP: OPTUNA TUNING (FULL INNER K-FOLD)
    # ---------------------------------------------------------
    X_train_full = X[train_content_mask].copy()
    Y_train_full = Y[train_content_mask]
    FID_train_full = FILE_ID[train_content_mask]
    QID_train_full = Q_ID[train_content_mask]
    
    def objective(trial):
        # Optuna guesses the parameters ONCE for this trial
        lr = trial.suggest_float('lr', 1e-4, 2e-3, log=True)
        hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64])
        proj_dim = trial.suggest_categorical('proj_dim', [2, 4, 8, 16])
        emb_dim = trial.suggest_categorical('embedding_dim', [2, 4, 8, 16])
        drop1 = trial.suggest_float('dropout1', 0.1, 0.5)
        drop2 = trial.suggest_float('dropout2', 0.1, 0.5)
        drop3 = trial.suggest_float('dropout3', 0.1, 0.5)
        ls = trial.suggest_categorical('label_smoothing', [0.0, 0.05, 0.1])

        
        # We have 4 training questions, so we create 4 inner folds
        inner_gkf = GroupKFold(n_splits=4)
        inner_fold_accs = []
        
        # Test this single guess across ALL 4 cognitive levels
        for sub_train_idx, val_idx in inner_gkf.split(X_train_full, Y_train_full, groups=QID_train_full):
            
            X_sub = X_train_full[sub_train_idx].copy()
            Y_sub = Y_train_full[sub_train_idx]
            
            X_val = X_train_full[val_idx].copy()
            Y_val = Y_train_full[val_idx]
            FID_val = FID_train_full[val_idx]
            
            # Scenario Filter (Only drop forbidden classes from Sub-Train!)
            train_mask = np.isin(Y_sub, allowed_train_classes)
            X_sub = X_sub[train_mask]
            Y_sub = Y_sub[train_mask]
            
            scaler = RobustScaler()
            X_sub[:, :, 0] = scaler.fit_transform(X_sub[:, :, 0].reshape(-1, 1)).reshape(X_sub.shape[0], X_sub.shape[1])
            X_val[:, :, 0] = scaler.transform(X_val[:, :, 0].reshape(-1, 1)).reshape(X_val.shape[0], X_val.shape[1])
            
            train_ds = TensorDataset(torch.tensor(X_sub, dtype=torch.float32), torch.tensor(Y_sub, dtype=torch.long))
            val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.long))
            t_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
            v_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
            
            model = TemporalCNN(
                continuous_dim=1, num_classes=5, hidden_dim=hidden_dim, 
                proj_dim=proj_dim, embedding_dim=emb_dim,
                dropout1=drop1, dropout2=drop2, dropout3=drop3
            ).to(device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            custom_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]).to(device)
            criterion = nn.CrossEntropyLoss(weight=custom_weights, label_smoothing=ls)
            
            # FAST MICRO-EPOCH TUNING
            for epoch in range(OPTUNA_EPOCHS):
                model.train()
                for xb, yb in t_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()
                    
            model.eval()
            session_results = {}
            current_idx = 0
            with torch.no_grad():
                for xb, yb in v_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    store_vals = F.softmax(model(xb), dim=1).cpu().numpy()
                    labels = yb.cpu().numpy()
                    batch_fids = FID_val[current_idx : current_idx + len(xb)]
                    current_idx += len(xb)
                    for j in range(len(store_vals)):
                        key = (batch_fids[j], labels[j]) 
                        if key not in session_results: session_results[key] = []
                        session_results[key].append(store_vals[j])
                        
            fold_preds, fold_labels = [], []
            for (fid, true_lab), vals_list in session_results.items():
                fold_preds.append(np.argmax(np.mean(vals_list, axis=0)))
                fold_labels.append(true_lab)
                
            inner_val_acc = np.mean(np.array(fold_preds) == np.array(fold_labels))
            inner_fold_accs.append(inner_val_acc)

        # Return the true average across all 4 cognitive levels
        return np.mean(inner_fold_accs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    best_params = study.best_params
    print(f"Optuna Winning Params for Outer Fold {fold + 1}: {best_params} (Avg Inner Val Acc: {study.best_value:.4f})")
    
    # ---------------------------------------------------------
    # OUTER LOOP: FINAL TESTING WITH WINNING PARAMS
    # ---------------------------------------------------------
    print(f"--- OUTER FOLD {fold + 1} FINAL TEST RUN ---")
    
    X_train_final = X[train_content_mask].copy()
    Y_train_final = Y[train_content_mask]
    X_test_final = X[test_content_mask].copy()
    Y_test_final = Y[test_content_mask]
    FID_test = FILE_ID[test_content_mask]

    # Scenario Filter applied to outer train set
    scenario_train_mask = np.isin(Y_train_final, allowed_train_classes)
    X_train_final = X_train_final[scenario_train_mask]
    Y_train_final = Y_train_final[scenario_train_mask]

    scaler = RobustScaler()
    X_train_final[:, :, 0] = scaler.fit_transform(X_train_final[:, :, 0].reshape(-1, 1)).reshape(X_train_final.shape[0], X_train_final.shape[1])
    X_test_final[:, :, 0] = scaler.transform(X_test_final[:, :, 0].reshape(-1, 1)).reshape(X_test_final.shape[0], X_test_final.shape[1])

    train_ds = TensorDataset(torch.tensor(X_train_final, dtype=torch.float32), torch.tensor(Y_train_final, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test_final, dtype=torch.float32), torch.tensor(Y_test_final, dtype=torch.long))
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = TemporalCNN(
        continuous_dim=1, num_classes=5, hidden_dim=best_params['hidden_dim'], 
        proj_dim=best_params['proj_dim'], embedding_dim=best_params['embedding_dim'],
        dropout1=best_params['dropout1'], dropout2=best_params['dropout2'], dropout3=best_params['dropout3']
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=1e-4)
    custom_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=custom_weights, label_smoothing=best_params['label_smoothing'])
    
    final_run_epochs = FINAL_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=final_run_epochs)

    for epoch in range(final_run_epochs):
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
        
        if (epoch + 1) % 20 == 0 or (epoch + 1) == final_run_epochs:
            print(f"Final Train Epoch [{epoch + 1}/{final_run_epochs}], Loss: {epoch_loss:.4f}")

    # Final Evaluation
    model.eval()
    session_results = {}
    current_idx = 0
    fold_win_preds, fold_win_labels = [], []  
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            store_vals = F.softmax(out, dim=1).cpu().numpy()
            labels = yb.cpu().numpy()
            
            fold_win_preds.extend(np.argmax(store_vals, axis=1))
            fold_win_labels.extend(labels)
            
            batch_fids = FID_test[current_idx : current_idx + len(xb)]
            current_idx += len(xb)

            for j in range(len(store_vals)):
                key = (batch_fids[j], labels[j]) 
                if key not in session_results: session_results[key] = []
                session_results[key].append(store_vals[j])

    fold_win_acc = np.mean(np.array(fold_win_preds) == np.array(fold_win_labels))
    fold_window_accuracies.append(fold_win_acc)

    fold_preds, fold_labels = [], []
    for (fid, true_lab), vals_list in session_results.items():
        mean_probs = np.mean(vals_list, axis=0)
        final_prediction = np.argmax(mean_probs)
            
        fold_preds.append(final_prediction)
        fold_labels.append(true_lab)
        
        all_pred_labels.append(final_prediction)
        all_true_labels.append(true_lab)
        all_pred_probs.append(mean_probs)

    fold_acc = np.mean(np.array(fold_preds) == np.array(fold_labels))
    fold_accuracies.append(fold_acc)
    fold_cm_data.append((fold_labels, fold_preds))
    
    print(f"-> Fold {fold+1} True Test Window Accuracy: {fold_win_acc:.4f}")
    print(f"-> Fold {fold+1} True Test Session Accuracy: {fold_acc:.4f}")

# ---------------------------------------------------------
# 5. FINAL SUMMARY & EER
# ---------------------------------------------------------
print("-" * 40)
print(f"Scenario {SCENARIO} Average True Test Window Acc: {np.mean(fold_window_accuracies):.4f} (+/- {np.std(fold_window_accuracies):.4f})")
print(f"Scenario {SCENARIO} Average True Test Session Acc: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
print("-" * 40)

print("\n" + "=" * 40)
print("BIOMETRIC AUTHENTICATION EER (PER-SESSION)")
print("Target: Bonafide (Class 0)")
print("=" * 40)

y_true_array = np.array(all_true_labels)
y_probs_array = np.array(all_pred_probs)
y_true_binary = (y_true_array == 0).astype(int)
y_probs_bonafide = y_probs_array[:, 0]

fpr, tpr, thresholds = roc_curve(y_true_binary, y_probs_bonafide)
fnr = 1 - tpr
eer_index = np.nanargmin(np.absolute((fnr - fpr)))
global_eer = (fpr[eer_index] + fnr[eer_index]) / 2.0
optimal_threshold = thresholds[eer_index]

print(f"Global System EER (Bonafide vs All) : {global_eer:.4f} ({global_eer * 100:.2f}%)")
print(f"Optimal Confidence Threshold        : {optimal_threshold:.4f}\n")

print("-" * 40)
print("Attack-Specific False Acceptance Rates (FAR) at EER Threshold:")
print("-" * 40)

class_names = {1: "Paraphrase (P)", 2: "Transcribe (T)", 3: "Fake Paraphrase (F_P)", 4: "Fake Transcribe (F_T)"}

for class_idx, name in class_names.items():
    attack_mask = (y_true_array == class_idx)
    if np.sum(attack_mask) == 0: continue 
        
    attack_probs = y_probs_bonafide[attack_mask]
    false_accepts = np.sum(attack_probs >= optimal_threshold)
    specific_far = false_accepts / len(attack_probs)
    print(f"{name:<25}: {specific_far:.4f} ({specific_far * 100:.2f}%)")

print("=" * 40 + "\n")

# ---------------------------------------------------------
# 6. PLOT THREE SEPARATE CONFUSION MATRICES
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (true_labs, pred_labs) in enumerate(fold_cm_data):
    cm = confusion_matrix(true_labs, pred_labs, labels=[0, 1, 2, 3, 4])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["B", "P", "T", "F_P", "F_T"])
    disp.plot(cmap=plt.cm.Purples, ax=axes[i], colorbar=False)
    axes[i].set_title(f"Outer Fold {i+1} True Test Matrix")

plt.tight_layout()
plt.show()