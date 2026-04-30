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
OPTUNA_TRIALS = 100    
OPTUNA_EPOCHS = 30    
FINAL_EPOCHS = 50

WIN_LENGTH = 200
STRIDE = 50

SCENARIO_TRAIN_CLASSES = {
    'M2': [0, 2],          
    'M3': [0, 1, 2],       
    'M4': [0, 1, 2, 4],    
    'M5': [0, 1, 2, 3, 4], 
    'M6': [0, 1]
}

# --- THE CUSTOM QUESTION GROUPS (Outer Cross-Validation) ---
QUESTION_GROUPS = [
    [1, 4],
    [2, 5],
    [3, 6]
]

# OPTIMIZATION TWEAKS
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.75, device=0)

# ---------------------------------------------------------
# 0. CUSTOM UTILITIES: EARLY STOPPING & FOCAL LOSS
# ---------------------------------------------------------
class PeriodicLoggingCallback:
    def __init__(self, print_every=5, total_trials=100):
        self.print_every = print_every
        self.total_trials = total_trials

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        # Optuna trial numbers are 0-indexed, so we add 1 for display
        current_trial = trial.number + 1 
        
        # Print an update every N trials
        if current_trial % self.print_every == 0:
            print(f"  -> [Optuna Progress] Trial {current_trial:03d}/{self.total_trials} "
                  f"| Current Score: {trial.value:.4f} "
                  f"| Best Score So Far: {study.best_value:.4f}")
            
class EarlyStoppingCallback:
    def __init__(self, patience: int):
        self.patience = patience

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        try:
            best_trial = study.best_trial
        except ValueError:
            return  

        trials_since_best = trial.number - best_trial.number
        if trials_since_best >= self.patience:
            print(f"\n[EARLY STOPPING TRIGGERED] No improvement in {self.patience} trials!")
            print(f"Halting the inner experiment to save compute.")
            study.stop()
            
    __call__.experimental_disable_run_after_update = False

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss) 
        pt = torch.clamp(pt, min=1e-7, max=1.0 - 1e-7)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if torch.isnan(focal_loss).any():
            focal_loss = torch.nan_to_num(focal_loss, nan=0.0)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# ---------------------------------------------------------
# 1. UPGRADED MODEL: EXCLUSIVELY MAX POOLING
# ---------------------------------------------------------
class TemporalCNN(nn.Module):
    def __init__(self, continuous_dim=1, num_classes=5, hidden_dim=32, 
                 proj_dim=8, embedding_dim=8, 
                 dropout1=0.2, dropout2=0.2, dropout3=0.3):
        super().__init__()
        
        self.projector = nn.Linear(continuous_dim, proj_dim)
        self.embedding = nn.Embedding(num_embeddings=3, embedding_dim=embedding_dim) 
        conv_in_dim = proj_dim + embedding_dim
        
        self.conv1 = nn.Conv1d(in_channels=conv_in_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout1)
        
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout2)
        
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout3 = nn.Dropout(dropout3)
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
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
# 2. DATA GENERATORS (STANDARD)
# ---------------------------------------------------------
def get_key_id(key_str):
    if len(key_str) == 1:
        if key_str.isalpha(): return 0
        elif key_str.isdigit(): return 1               
        elif key_str == ' ': return 2                               
    elif key_str == 'Backspace': return 3                                  
    return 4  
def get_key_id3(key_str):
            
    if key_str == ' ': return 0                               
    elif key_str == 'Backspace': return 1                               
    return 2  

def make_windows(filepath, X, Y, ID, FILE_ID, Q_ID, user_id, file_id, win_length, stride):
    with open(filepath, "r", encoding="utf8") as f:
        data = json.load(f)
    _keystrokes = data.get("keystrokes", [])
    is_attack = filepath.endswith("attack.json")

    grouped_keys = {}
    for k in _keystrokes:
        if k["event"] == "keydown":
            q_idx_str = k.get("question_index")
            if not q_idx_str: continue 
            parts = str(q_idx_str).split('.')
            if len(parts) != 2: continue
            
            q_id, s_id = int(parts[1]), int(parts[0])
            label = None
            if not is_attack:
                if s_id == 1: label = 0    
                elif s_id == 2: label = 1  
                elif s_id == 3: label = 2  
            else:
                if s_id == 2: label = 3    
                elif s_id == 3: label = 4  

            if label is not None:
                group_key = (label, q_id)
                if group_key not in grouped_keys: grouped_keys[group_key] = []
                grouped_keys[group_key].append(k)

    for (label, q_id), keys in grouped_keys.items():
        if len(keys) == 0: continue
        for i in range(1, len(keys) - win_length, stride):
            window = []
            for j in range(i, i + win_length):
                dt = np.clip(keys[j]["timestamp"] - keys[j - 1]["timestamp"], 1, 5000)
                key_id = get_key_id3(keys[j]["key"])
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

print("Generating standard context-agnostic windows...")
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

NUM_CONTINUOUS_FEATURES = X.shape[2] - 1 

print(f"Dataset Loaded -> X: {X.shape} | Y: {Y.shape} | Continuous Features Detected: {NUM_CONTINUOUS_FEATURES}")

# ---------------------------------------------------------
# 4. NESTED CROSS-CONTENT OPTUNA EVALUATION LOOP
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
print(f"\nRunning Scenario {SCENARIO}: Automated Nested CV Tuning & Testing (Cross-Question)")

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
    # INNER LOOP: OPTUNA TUNING (4-FOLD CROSS-VALIDATION)
    # ---------------------------------------------------------
    X_train_full = X[train_content_mask].copy()
    Y_train_full = Y[train_content_mask]
    FID_train_full = FILE_ID[train_content_mask]
    QID_train_full = Q_ID[train_content_mask]
    
    def objective(trial):
        lr = trial.suggest_float('lr', 1e-4, 2e-3, log=True)
        hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128])
        proj_dim = trial.suggest_categorical('proj_dim', [2, 4, 8, 16])
        emb_dim = trial.suggest_categorical('embedding_dim', [2, 4, 8, 16])
        drop1 = trial.suggest_float('dropout1', 0.1, 0.5)
        drop2 = trial.suggest_float('dropout2', 0.1, 0.5)
        drop3 = trial.suggest_float('dropout3', 0.1, 0.5)
        jitter_std = trial.suggest_float('jitter_std', 0.0, 0.15) 
        

        focal_gamma = trial.suggest_float('gamma', 1.0, 5.0)
        
        inner_gkf = GroupKFold(n_splits=4)
        inner_fold_scores = []
        inner_fold_thresholds = [] # <--- Added for ZERO LEAKAGE
        
        for inner_train_idx, inner_val_idx in inner_gkf.split(X_train_full, Y_train_full, groups=QID_train_full):
            
            X_sub = X_train_full[inner_train_idx].copy()
            Y_sub = Y_train_full[inner_train_idx]
            X_val = X_train_full[inner_val_idx].copy()
            Y_val = Y_train_full[inner_val_idx]
            
            train_mask = np.isin(Y_sub, allowed_train_classes)
            X_sub = X_sub[train_mask]
            Y_sub = Y_sub[train_mask]
            
            scaler = RobustScaler()
            train_cont = X_sub[:, :, :NUM_CONTINUOUS_FEATURES].reshape(-1, NUM_CONTINUOUS_FEATURES)
            val_cont = X_val[:, :, :NUM_CONTINUOUS_FEATURES].reshape(-1, NUM_CONTINUOUS_FEATURES)
            
            X_sub[:, :, :NUM_CONTINUOUS_FEATURES] = scaler.fit_transform(train_cont).reshape(X_sub.shape[0], X_sub.shape[1], NUM_CONTINUOUS_FEATURES)
            X_val[:, :, :NUM_CONTINUOUS_FEATURES] = scaler.transform(val_cont).reshape(X_val.shape[0], X_val.shape[1], NUM_CONTINUOUS_FEATURES)
            
            train_ds = TensorDataset(torch.tensor(X_sub, dtype=torch.float32), torch.tensor(Y_sub, dtype=torch.long))
            val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.long))
            
            t_loader = DataLoader(train_ds, batch_size=128, shuffle=True, pin_memory=True)
            v_loader = DataLoader(val_ds, batch_size=128, shuffle=False, pin_memory=True)
            
            model = TemporalCNN(
                continuous_dim=NUM_CONTINUOUS_FEATURES, num_classes=5, hidden_dim=hidden_dim, 
                proj_dim=proj_dim, embedding_dim=emb_dim,
                dropout1=drop1, dropout2=drop2, dropout3=drop3
            ).to(device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            
            criterion = WeightedFocalLoss(alpha=None, gamma=focal_gamma)
            
            for epoch in range(OPTUNA_EPOCHS):
                model.train()
                for xb, yb in t_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    
                    if jitter_std > 0.0:
                        noise_adder = torch.normal(mean=0.0, std=jitter_std, size=xb[:, :, :NUM_CONTINUOUS_FEATURES].shape).to(device)
                        noise_adder = torch.clamp(noise_adder, min=-(3 * jitter_std), max=(3 * jitter_std))
                        xb[:, :, :NUM_CONTINUOUS_FEATURES] += noise_adder
                        
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()
                    
            # --- EVALUATE VALIDATION FOLD ---
            # --- EVALUATE VALIDATION FOLD (WEIGHTED COMPROMISE STRATEGY) ---
            model.eval()
            val_probs = []
            val_labels = []
            
            with torch.no_grad():
                for xb, yb in v_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    val_probs.extend(F.softmax(out, dim=1)[:, 0].cpu().numpy())
                    val_labels.extend(yb.cpu().numpy())
            
            val_probs = np.array(val_probs)
            val_labels = np.array(val_labels)
            
            # ---------------------------------------------------------
            # 1. CALCULATE GLOBAL EER (Bonafide vs. ALL Attacks)
            # ---------------------------------------------------------
            global_binary_labels = (val_labels == 0).astype(int)
            g_fpr, g_tpr, g_thresholds = roc_curve(global_binary_labels, val_probs)
            g_fnr = 1 - g_tpr
            g_eer_index = np.nanargmin(np.absolute((g_fnr - g_fpr)))
            
            global_eer = (g_fpr[g_eer_index] + g_fnr[g_eer_index]) / 2.0
            global_threshold = g_thresholds[g_eer_index]
            
            # ---------------------------------------------------------
            # 2. CALCULATE WORST-CASE EER (The biggest vulnerability)
            # ---------------------------------------------------------
            worst_eer = 0.0
            worst_threshold = 0.5 # Default fallback
            
            for attack_class in [1, 2, 3, 4]:
                mask = (val_labels == 0) | (val_labels == attack_class)
                if np.sum(val_labels == attack_class) == 0: continue
                
                specific_binary = (val_labels[mask] == 0).astype(int)
                specific_probs = val_probs[mask]
                
                fpr, tpr, atk_thresholds = roc_curve(specific_binary, specific_probs)
                fnr = 1 - tpr
                eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
                specific_eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
                
                if specific_eer > worst_eer:
                    worst_eer = specific_eer
                    worst_threshold = atk_thresholds[eer_idx] # <--- SAVE THE STRICT THRESHOLD
            
            # 3. APPLY THE WEIGHTED PENALTY 
            fold_weighted_score = (0.0 * global_eer) + (1.0 * worst_eer)
            inner_fold_scores.append(fold_weighted_score)
            
            # --- CRITICAL FIX ---
            # If you are optimizing 70% for the worst case, pass the worst-case threshold!
            inner_fold_thresholds.append(worst_threshold)
            
        # --- ATTACH MEAN THRESHOLD TO OPTUNA TRIAL METADATA ---
        trial.set_user_attr("optimal_threshold", float(np.mean(inner_fold_thresholds)))
        return np.mean(inner_fold_scores)

    study_name = f"context_scenario_{SCENARIO}_fold_{fold + 1}"
    study = optuna.create_study(
        study_name=study_name,
        storage="sqlite:///keystroke_optuna_context_agnostic.db",
        load_if_exists=True,
        direction="minimize" 
    )
    
    study.optimize(
        objective, 
        n_trials=OPTUNA_TRIALS,
        callbacks=[
            EarlyStoppingCallback(patience=35),
            PeriodicLoggingCallback(print_every=5, total_trials=OPTUNA_TRIALS)
        ]
    )
    
    best_params = study.best_params
    tuned_threshold = study.best_trial.user_attrs["optimal_threshold"] # <--- RETRIEVE BLIND THRESHOLD
    
    print(f"Optuna Winning Params for Outer Fold {fold + 1}:")
    print(json.dumps(best_params, indent=4))
    print(f"(Inner 4-Fold Val EER: {study.best_value:.4f})")
    print(f"(Blind Threshold Locked At: {tuned_threshold:.4f})")
    
    # ---------------------------------------------------------
    # OUTER LOOP: FINAL TESTING WITH WINNING PARAMS
    # ---------------------------------------------------------
    print(f"--- OUTER FOLD {fold + 1} ZERO-LEAKAGE TEST RUN ---")
    
    X_train_final = X[train_content_mask].copy()
    Y_train_final = Y[train_content_mask]
    X_test_final = X[test_content_mask].copy()
    Y_test_final = Y[test_content_mask]
    FID_test = FILE_ID[test_content_mask]

    scenario_train_mask = np.isin(Y_train_final, allowed_train_classes)
    X_train_final = X_train_final[scenario_train_mask]
    Y_train_final = Y_train_final[scenario_train_mask]

    scaler = RobustScaler()
    train_cont = X_train_final[:, :, :NUM_CONTINUOUS_FEATURES].reshape(-1, NUM_CONTINUOUS_FEATURES)
    test_cont = X_test_final[:, :, :NUM_CONTINUOUS_FEATURES].reshape(-1, NUM_CONTINUOUS_FEATURES)
    
    X_train_final[:, :, :NUM_CONTINUOUS_FEATURES] = scaler.fit_transform(train_cont).reshape(X_train_final.shape[0], X_train_final.shape[1], NUM_CONTINUOUS_FEATURES)
    X_test_final[:, :, :NUM_CONTINUOUS_FEATURES] = scaler.transform(test_cont).reshape(X_test_final.shape[0], X_test_final.shape[1], NUM_CONTINUOUS_FEATURES)

    train_ds = TensorDataset(torch.tensor(X_train_final, dtype=torch.float32), torch.tensor(Y_train_final, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test_final, dtype=torch.float32), torch.tensor(Y_test_final, dtype=torch.long))
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, pin_memory=True)

    model = TemporalCNN(
        continuous_dim=NUM_CONTINUOUS_FEATURES, num_classes=5, hidden_dim=best_params['hidden_dim'], 
        proj_dim=best_params['proj_dim'], embedding_dim=best_params['embedding_dim'],
        dropout1=best_params['dropout1'], dropout2=best_params['dropout2'], dropout3=best_params['dropout3']
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=1e-4)
    
    best_gamma = best_params.get('gamma', 2.0)
    
    criterion = WeightedFocalLoss(alpha=None, gamma=best_gamma)
    
    final_run_epochs = 50
    best_jitter = best_params.get('jitter_std', 0.0)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=final_run_epochs)

    for epoch in range(final_run_epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            if best_jitter > 0.0:
                noise_adder = torch.normal(mean=0.0, std=best_jitter, size=xb[:, :, :NUM_CONTINUOUS_FEATURES].shape).to(device)
                noise_adder = torch.clamp(noise_adder, min=-(3 * best_jitter), max=(3 * best_jitter))
                xb[:, :, :NUM_CONTINUOUS_FEATURES] += noise_adder
                
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        #scheduler.step()
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
    
    # ---------------------------------------------------------
    # CALCULATE ZERO-LEAKAGE METRICS FOR THIS FOLD
    # ---------------------------------------------------------
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
# 5. FINAL SUMMARY (HTER REPLACES EER)
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
    axes[i].set_title(f"Outer Fold {i+1} True Test Matrix")

plt.tight_layout()

# Headless Saving for Cloud Environments
plt.savefig(f"Scenario_{SCENARIO}_Context_Agnostic_CM.png", dpi=300, bbox_inches='tight')
plt.close()