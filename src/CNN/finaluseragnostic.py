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
OPTUNA_TRIALS = 10   
OPTUNA_EPOCHS = 50    
HIDDEN_DIM = 128
FINAL_EPOCHS = 50     

WIN_LENGTH = 100
STRIDE = 50

SCENARIO_TRAIN_CLASSES = {
    'M2': [0, 2],          
    'M3': [0, 1, 2],       
    'M4': [0, 1, 2, 4],    
    'M5': [0, 1, 2, 3, 4], 
    'M6': [0, 1]
}

# ---------------------------------------------------------
# DEVICE SETUP & GPU LEASH
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





class PeriodicLoggingCallback:
    def __init__(self, print_every=5, total_trials=100):
        self.print_every = print_every
        self.total_trials = total_trials

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        current_trial = trial.number + 1 
        if current_trial % self.print_every == 0:
            print(f"  -> [Optuna Progress] Trial {current_trial:03d}/{self.total_trials} "
                  f"| Current Weighted Score: {trial.value:.4f} "
                  f"| Best Score So Far: {study.best_value:.4f}")
            
class EarlyStoppingCallback:
    def __init__(self, patience: int):
        self.patience = patience

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        try:
            best_trial = study.best_trial
        except ValueError:
            return  
        if (trial.number - best_trial.number) >= self.patience:
            print(f"\n[EARLY STOPPING TRIGGERED] No improvement in {self.patience} trials!")
            print(f"Halting the inner experiment to save compute.")
            study.stop()
            
    __call__.experimental_disable_run_after_update = False


class TemporalCNN(nn.Module):
    def __init__(self, continuous_dim=1, num_classes=5, hidden_dim=64, dropout_fc=0.3):
        super().__init__()
        
        # Continuous Features + 6 One-Hot Key Classes
        unified_in_dim = 8
        
        # Single Unified Projector
        self.time_proj = nn.Linear(continuous_dim, 4)
        self.unified_projector = nn.Linear(unified_in_dim, hidden_dim)
        self.embedding = nn.Embedding(5, 4)
        
        self.conv1 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout1d(0.1)
        
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding="same", dilation=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout1d(0.1)

        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding="same", dilation=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout3 = nn.Dropout1d(0.1)

        self.pool = nn.AdaptiveMaxPool1d(1) 
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_fc = nn.Dropout(dropout_fc)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x, return_features=False):
        cont_feats = x[:, :, :-1]     
        key_ids = x[:, :, -1].long() 
        cont_feats = F.relu(self.time_proj(cont_feats))
        key_ids = self.embedding(key_ids)
        unified_input = torch.cat([cont_feats, key_ids], dim=2) 
        x_proj = F.relu(self.unified_projector(unified_input))
        
        # Permute for CNN [Batch, Channels, Seq_Length]
        x_cnn = x_proj.permute(0, 2, 1)  
        
        x_cnn = self.conv1(x_cnn)
        x_cnn = F.relu(self.bn1(x_cnn))
        x_cnn = self.dropout1(x_cnn)
        
        x_cnn = self.conv2(x_cnn)
        x_cnn = F.relu(self.bn2(x_cnn))
        x_cnn = self.dropout2(x_cnn)
        
        x_cnn = self.conv3(x_cnn)
        x_cnn = F.relu(self.bn3(x_cnn))
        x_cnn = self.dropout3(x_cnn)


        features = self.pool(x_cnn).squeeze(-1)
        features = F.relu(self.fc1(features))
        features = self.dropout_fc(features)
        
        if return_features: return features
        return self.fc2(features)

# ---------------------------------------------------------
# 2. DATA GENERATORS
# ---------------------------------------------------------
def get_key_id(key_str):
    if len(key_str) == 1:
        if key_str.isalpha(): return 0
        elif key_str.isdigit(): return 1               
        elif key_str == ' ': return 2                               
    elif key_str == 'Backspace': return 3                                  
    return 4                                          

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
                key_id = get_key_id(keys[j]["key"])
                window.append([np.log(dt), key_id])
            X.append(window)
            Y.append(label)
            ID.append(user_id)
            FILE_ID.append(file_id) 

# ---------------------------------------------------------
# 3. DATA EXTRACTION
# ---------------------------------------------------------
X, Y, ID, FILE_ID = [], [], [], []

folder_path = "../../dataset/Attack4"
user_folders = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
user2id = {user: i for i, user in enumerate(user_folders)}

print(f"Generating User-Agnostic windows...")
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
# 4. NESTED CROSS-VALIDATION WITH OPTUNA
# ---------------------------------------------------------
optuna.logging.set_verbosity(optuna.logging.WARNING) 

n_splits = 3
gkf = GroupKFold(n_splits=n_splits)

fold_accuracies = []
fold_window_accuracies = [] 
all_true_labels = []
all_pred_labels = []
all_pred_probs = [] 
fold_cm_data = []

allowed_train_classes = SCENARIO_TRAIN_CLASSES[SCENARIO]
print(f"\nRunning Scenario {SCENARIO}: Automated 3x3 Nested CV Tuning & Testing (User-Agnostic)")

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, Y, groups=ID)):
    print(f"\n" + "="*50)
    print(f"FOLD {fold + 1}/{n_splits} - INITIALIZING OPTUNA EXPERIMENT")
    print("="*50)
    
    X_train_full = X[train_idx]
    Y_train_full = Y[train_idx]
    ID_train_full = ID[train_idx]
    FID_train_full = FILE_ID[train_idx]
    
    def objective(trial):
        lr = trial.suggest_float('lr', 1e-4, 2e-3, log=True)

        
        inner_gkf = GroupKFold(n_splits=3)
        inner_fold_scores = []
        inner_fold_thresholds = [] # <--- Zero Leakage tracking
        
        for inner_train_idx, inner_val_idx in inner_gkf.split(X_train_full, Y_train_full, groups=ID_train_full):
            X_sub = X_train_full[inner_train_idx].copy()
            Y_sub = Y_train_full[inner_train_idx]
            
            X_val = X_train_full[inner_val_idx].copy()
            Y_val = Y_train_full[inner_val_idx]
            
            train_mask = np.isin(Y_sub, allowed_train_classes)
            X_sub = X_sub[train_mask]
            Y_sub = Y_sub[train_mask]
            
            scaler = RobustScaler()
            X_sub[:, :, 0] = scaler.fit_transform(X_sub[:, :, 0].reshape(-1, 1)).reshape(X_sub.shape[0], X_sub.shape[1])
            X_val[:, :, 0] = scaler.transform(X_val[:, :, 0].reshape(-1, 1)).reshape(X_val.shape[0], X_val.shape[1])
            
            train_ds = TensorDataset(torch.tensor(X_sub, dtype=torch.float32), torch.tensor(Y_sub, dtype=torch.long))
            val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.long))
            t_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
            v_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
            
            model = TemporalCNN(
            continuous_dim=1, num_classes=5, hidden_dim=HIDDEN_DIM
        ).to(device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss() 
            
            for epoch in range(OPTUNA_EPOCHS):
                model.train()
                for xb, yb in t_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                        
                    loss.backward()
                    optimizer.step()
                    
            # --- EVALUATE VALIDATION FOLD (WEIGHTED RULEBOOK) ---
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
                
            inner_fold_scores.append(np.mean(np.array(fold_preds) == np.array(fold_labels)))
            
        return np.mean(inner_fold_scores)

    study_name = f"user_agnostic_scenario_{SCENARIO}_fold_{fold + 1}"
    study = optuna.create_study(
        study_name=study_name,
        storage="sqlite:///keystroke_optuna_final_eer.db",
        load_if_exists=True,
        direction="minimize" # <--- Must Minimize the EER score
    )
    study.optimize(
        objective, 
        n_trials=OPTUNA_TRIALS,
        callbacks=[
            EarlyStoppingCallback(patience=15),
            PeriodicLoggingCallback(print_every=5, total_trials=OPTUNA_TRIALS)
        ]
    )
    
    best_params = study.best_params
    tuned_threshold = study.best_trial.user_attrs["optimal_threshold"] # <--- RETRIEVE BLIND THRESHOLD
    
    print(f"Optuna Winning Params for Fold {fold + 1}:")
    print(json.dumps(best_params, indent=4))
    print(f"(Inner 3-Fold Val Weighted Score: {study.best_value:.4f})")
    print(f"(Blind Threshold Locked At: {tuned_threshold:.4f})")
    
    # ---------------------------------------------------------
    # OUTER LOOP: FINAL TESTING WITH WINNING PARAMS
    # ---------------------------------------------------------
    print(f"--- FOLD {fold + 1} ZERO-LEAKAGE TEST RUN ---")
    
    X_train_final = X[train_idx].copy()
    Y_train_final = Y[train_idx]
    X_test_final = X[test_idx].copy()
    Y_test_final = Y[test_idx]
    FID_test = FILE_ID[test_idx]

    train_mask = np.isin(Y_train_final, allowed_train_classes)
    X_train_final = X_train_final[train_mask]
    Y_train_final = Y_train_final[train_mask]

    scaler = RobustScaler()
    X_train_final[:, :, 0] = scaler.fit_transform(X_train_final[:, :, 0].reshape(-1, 1)).reshape(X_train_final.shape[0], X_train_final.shape[1])
    X_test_final[:, :, 0] = scaler.transform(X_test_final[:, :, 0].reshape(-1, 1)).reshape(X_test_final.shape[0], X_test_final.shape[1])

    train_ds = TensorDataset(torch.tensor(X_train_final, dtype=torch.float32), torch.tensor(Y_train_final, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test_final, dtype=torch.float32), torch.tensor(Y_test_final, dtype=torch.long))
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    best_pool = best_params.get('pool_type', 'max')
    
    model = TemporalCNN(
        continuous_dim=1, num_classes=5, hidden_dim=best_params['hidden_dim'], 
        proj_dim=best_params['proj_dim'], embedding_dim=best_params['embedding_dim'],
        dropout1=best_params['dropout1'], dropout2=best_params['dropout2'], dropout3=best_params['dropout3'],
        pool_type=best_pool
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=1e-4)
    
    # --- FINAL RUN CRITERION UPDATED TO STANDARD CROSS ENTROPY ---
    best_ls = best_params.get('label_smoothing', 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=best_ls)
    
    best_jitter = best_params.get('jitter_std', 0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINAL_EPOCHS)

    for epoch in range(FINAL_EPOCHS):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            
            if best_jitter > 0.0:
                noise_adder = torch.normal(mean=0.0, std=best_jitter, size=xb[:, :, :-1].shape).to(device)
                noise_adder = torch.clamp(noise_adder, min=-(3 * best_jitter), max=(3 * best_jitter))
                xb_augmented = xb.clone()
                xb_augmented[:, :, :-1] = xb[:, :, :-1] + noise_adder
                loss = criterion(model(xb_augmented), yb)
            else:
                loss = criterion(model(xb), yb)
                
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        
        if (epoch + 1) % 10 == 0 or (epoch + 1) == FINAL_EPOCHS:
            print(f"Final Train Epoch [{epoch + 1}/{FINAL_EPOCHS}], Loss: {epoch_loss:.4f}")

    # --- FINAL ZERO-LEAKAGE EVALUATION ---
    model.eval()
    file_results = {}
    current_idx = 0
    fold_win_preds, fold_win_labels = [], []  
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            store_vals = F.softmax(model(xb), dim=1).cpu().numpy()
            labels = yb.cpu().numpy()
            
            fold_win_preds.extend(np.argmax(store_vals, axis=1))
            fold_win_labels.extend(labels)
            
            batch_fids = FID_test[current_idx : current_idx + len(xb)]
            current_idx += len(xb)

            for j in range(len(store_vals)):
                key = (batch_fids[j], labels[j]) 
                if key not in file_results: file_results[key] = []
                file_results[key].append(store_vals[j])

    fold_win_acc = np.mean(np.array(fold_win_preds) == np.array(fold_win_labels))
    fold_window_accuracies.append(fold_win_acc)

    fold_preds, fold_labels = [], []
    for (fid, true_lab), vals_list in file_results.items():
        mean_probs = np.mean(vals_list, axis=0)
        probs_copy = mean_probs.copy() 
        
        # ---------------------------------------------------------
        # THE ZERO-LEAKAGE BOUNCER RULE
        # ---------------------------------------------------------
        if probs_copy[0] >= tuned_threshold:
            final_prediction = 0  # Accept: It's Bonafide
        else:
            probs_copy[0] = -1.0  # Reject: Force model to pick the highest attack class
            final_prediction = np.argmax(probs_copy)
            
        fold_preds.append(final_prediction)
        fold_labels.append(true_lab)
        
        all_pred_labels.append(final_prediction)
        all_true_labels.append(true_lab)
        all_pred_probs.append(mean_probs)

    fold_acc = np.mean(np.array(fold_preds) == np.array(fold_labels))
    fold_accuracies.append(fold_acc)
    fold_cm_data.append((fold_labels, fold_preds))
    
    f_labels = np.array(fold_labels)
    f_preds = np.array(fold_preds)
    
    bonafide_mask = (f_labels == 0)
    frr = np.sum(f_preds[bonafide_mask] != 0) / np.sum(bonafide_mask) if np.sum(bonafide_mask) > 0 else 0.0
        
    attack_mask = (f_labels != 0)
    far_all = np.sum(f_preds[attack_mask] == 0) / np.sum(attack_mask) if np.sum(attack_mask) > 0 else 0.0
    
    para_mask = (f_labels == 1)
    far_para = np.sum(f_preds[para_mask] == 0) / np.sum(para_mask) if np.sum(para_mask) > 0 else 0.0
    
    hter = (frr + far_all) / 2.0

    print(f"-> Fold {fold+1} Blind Threshold Applied : {tuned_threshold:.4f}")
    print(f"-> Fold {fold+1} HTER (Half Total Error) : {hter:.4f} ({hter * 100:.2f}%)")
    print(f"-> Fold {fold+1} FRR (Bonafide Rejected) : {frr:.4f} ({frr * 100:.2f}%)")
    print(f"-> Fold {fold+1} FAR (Paraphrase Accpt)  : {far_para:.4f} ({far_para * 100:.2f}%)")

# ---------------------------------------------------------
# 5. FINAL SUMMARY & HTER
# ---------------------------------------------------------
print("-" * 40)
print(f"Scenario {SCENARIO} Average True Test Window Acc: {np.mean(fold_window_accuracies):.4f} (+/- {np.std(fold_window_accuracies):.4f})")
print(f"Scenario {SCENARIO} Average True Test Session Acc: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
print("-" * 40)

print("\n" + "=" * 40)
print("ZERO-LEAKAGE BIOMETRIC EVALUATION (HTER)")
print("Target: Bonafide (Class 0)")
print("=" * 40)

y_true_array = np.array(all_true_labels)
y_preds_array = np.array(all_pred_labels)

global_bonafide_mask = (y_true_array == 0)
global_frr = np.sum(y_preds_array[global_bonafide_mask] != 0) / np.sum(global_bonafide_mask)

global_attack_mask = (y_true_array != 0)
global_far = np.sum(y_preds_array[global_attack_mask] == 0) / np.sum(global_attack_mask)

global_hter = (global_frr + global_far) / 2.0

print(f"Global System HTER (Half Total Error) : {global_hter:.4f} ({global_hter * 100:.2f}%)")
print(f"Global FRR (Bonafide Rejected)        : {global_frr:.4f} ({global_frr * 100:.2f}%)")
print(f"Global FAR (All Attacks Accepted)     : {global_far:.4f} ({global_far * 100:.2f}%)\n")

print("-" * 40)
print("Attack-Specific False Acceptance Rates (FAR):")
print("-" * 40)

class_names = {1: "Paraphrase (P)", 2: "Transcribe (T)", 3: "Fake Paraphrase (F_P)", 4: "Fake Transcribe (F_T)"}

for class_idx, name in class_names.items():
    specific_attack_mask = (y_true_array == class_idx)
    if np.sum(specific_attack_mask) == 0: continue 
        
    false_accepts = np.sum(y_preds_array[specific_attack_mask] == 0)
    specific_far = false_accepts / np.sum(specific_attack_mask)
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
    axes[i].set_title(f"Fold {i+1} True Test Matrix")

plt.tight_layout()

# Headless Saving
plt.savefig(f"Scenario_{SCENARIO}_User_Agnostic_CM_EER.png", dpi=300, bbox_inches='tight')
plt.close()