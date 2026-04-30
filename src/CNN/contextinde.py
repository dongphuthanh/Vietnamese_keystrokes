import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import os
import optuna
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler

# ---------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------
VOTING_MODE = 'soft'  
SCENARIO = 'M5'       
OPTUNA_TRIALS = 30    
OPTUNA_EPOCHS = 30    
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

# --- LEAVE-ONE-QUESTION-OUT (LOQO) GROUPS ---
QUESTION_GROUPS = [
    [1,4],
    [2,5],
    [3,6]
]

# OPTIMIZATION TWEAKS
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.75, device=0)

# ---------------------------------------------------------
# 0. CUSTOM UTILITIES: EARLY STOPPING
# ---------------------------------------------------------
class PeriodicLoggingCallback:
    def __init__(self, print_every=5, total_trials=100):
        self.print_every = print_every
        self.total_trials = total_trials

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        current_trial = trial.number + 1 
        if current_trial % self.print_every == 0:
            print(f"  -> [Optuna Progress] Trial {current_trial:03d}/{self.total_trials} "
                  f"| Current Acc: {trial.value:.4f} "
                  f"| Best Acc So Far: {study.best_value:.4f}")
            
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
class TemporalAttention(nn.Module):
    def __init__(self, channel_dim):
        super().__init__()
        self.attn_weights = nn.Sequential(
            nn.Conv1d(channel_dim, channel_dim // 2, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(channel_dim // 2, 1, kernel_size=1)
        )

    def forward(self, x):
        weights = F.softmax(self.attn_weights(x), dim=2) 
        context = torch.sum(x * weights, dim=2) 
        return context
# ---------------------------------------------------------
# 1. MODEL: EXCLUSIVELY MAX POOLING
# ---------------------------------------------------------
class TemporalCNN(nn.Module):
    def __init__(self, continuous_dim=1, num_classes=5, hidden_dim=32, 
                 proj_dim=8, embedding_dim=8, 
                 dropout1=0.2, dropout2=0.2, dropout3=0.3,
                 pool_type='attention'):
        super().__init__()
        
        self.projector = nn.Linear(continuous_dim, proj_dim)
        self.embedding = nn.Embedding(num_embeddings=5, embedding_dim=embedding_dim) 
        conv_in_dim = proj_dim + embedding_dim
        
        self.conv1 = nn.Conv1d(in_channels=conv_in_dim, out_channels=hidden_dim, kernel_size=2, padding="same")
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout1)
        
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2, padding="same")
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout2)
        

        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2, padding="same")
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout3 = nn.Dropout(dropout3)


        #self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
            
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

        max_feats = self.max_pool(x).squeeze(-1)
        
        # 2. Extract the statistical baselines (Avg)
        #avg_feats = self.avg_pool(x).squeeze(-1)
        
        # 3. Stitch them together side-by-side
        #features = torch.cat([max_feats, avg_feats], dim=1)
        features = max_feats

        if return_features: return features
        return self.fc(features)

# ---------------------------------------------------------
# 2. DATA GENERATORS
# ---------------------------------------------------------
def get_key_id3(key_str):
    if key_str == ' ': return 0                               
    elif key_str == 'Backspace': return 1                               
    return 2  
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

X, Y, ID, FILE_ID, Q_ID = np.array(X), np.array(Y), np.array(ID), np.array(FILE_ID), np.array(Q_ID)
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
all_true_labels, all_pred_labels = [], []
fold_cm_data = []

allowed_train_classes = SCENARIO_TRAIN_CLASSES[SCENARIO]
print(f"\nRunning Scenario {SCENARIO}: LOQO ACCURACY-OPTIMIZED Pipeline")

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
    # INNER LOOP: OPTUNA TUNING (ACCURACY MAXIMIZATION)
    # ---------------------------------------------------------
    X_train_full = X[train_content_mask].copy()
    Y_train_full = Y[train_content_mask]
    FID_train_full = FILE_ID[train_content_mask]
    QID_train_full = Q_ID[train_content_mask]
    
    def objective(trial):
        lr = trial.suggest_float('lr', 1e-4, 2e-3, log=True)
        hidden_dim = trial.suggest_categorical('hidden_dim', [32,64, 128, 256])
        proj_dim = trial.suggest_categorical('proj_dim', [4, 8, 16, 32])
        emb_dim = trial.suggest_categorical('embedding_dim', [2, 4, 8, 16])
        drop1 = trial.suggest_float('dropout1', 0.0, 0.5)
        drop2 = trial.suggest_float('dropout2', 0.0, 0.5)
        drop3 = trial.suggest_float('dropout3', 0.0, 0.5)
        jitter_std = trial.suggest_float('jitter_std', 0.0, 0.15) 
        ls = trial.suggest_float('label_smoothing', 0.0, 0.15)
        
        # Inner fold shifted to 5 to match LOQO training logic
        inner_gkf = GroupKFold(n_splits=4)
        inner_fold_scores = []
        
        for inner_train_idx, inner_val_idx in inner_gkf.split(X_train_full, Y_train_full, groups=QID_train_full):
            
            X_sub, Y_sub = X_train_full[inner_train_idx].copy(), Y_train_full[inner_train_idx]
            X_val, Y_val = X_train_full[inner_val_idx].copy(), Y_train_full[inner_val_idx]
            FID_val = FID_train_full[inner_val_idx]
            
            train_mask = np.isin(Y_sub, allowed_train_classes)
            X_sub, Y_sub = X_sub[train_mask], Y_sub[train_mask]
            
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
            criterion = nn.CrossEntropyLoss(label_smoothing=ls)
            
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
                    
            # --- EVALUATE INNER FOLD (ACCURACY AGGREGATION) ---
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
            
            # Simple Softmax Averaging
            fold_preds, fold_labels = [], []
            for (fid, true_lab), vals_list in session_results.items():
                mean_probs = np.mean(vals_list, axis=0)
                fold_preds.append(np.argmax(mean_probs))
                fold_labels.append(true_lab)
                
            inner_fold_scores.append(accuracy_score(fold_labels, fold_preds))
            
        return np.mean(inner_fold_scores)

    study_name = f"context_scenario_{SCENARIO}_fold_{fold + 1}"
    study = optuna.create_study(
        study_name=study_name,
        storage="sqlite:///keystroke_optuna_baseline.db",
        load_if_exists=True,
        direction="maximize" # MAXIMIZE ACCURACY
    )
    
    study.optimize(
        objective, 
        n_trials=OPTUNA_TRIALS,
        callbacks=[
            EarlyStoppingCallback(patience=25),
            PeriodicLoggingCallback(print_every=5, total_trials=OPTUNA_TRIALS)
        ]
    )
    
    best_params = study.best_params
    
    print(f"Optuna Winning Params for Outer Fold {fold + 1}:")
    print(json.dumps(best_params, indent=4))
    print(f"(Inner 5-Fold Val Session Accuracy: {study.best_value:.4f})")
    
    # ---------------------------------------------------------
    # OUTER LOOP: FINAL TESTING WITH WINNING PARAMS
    # ---------------------------------------------------------
    print(f"--- OUTER FOLD {fold + 1} FINAL TEST RUN ---")
    
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
    best_ls = best_params.get('label_smoothing', 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=best_ls)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINAL_EPOCHS)
    
    best_jitter = best_params.get('jitter_std', 0.0)

    for epoch in range(FINAL_EPOCHS):
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
            scheduler.step()
            running_loss += loss.item() * xb.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        if (epoch + 1) % 10 == 0 or (epoch + 1) == FINAL_EPOCHS:
            print(f"Final Train Epoch [{epoch + 1}/{FINAL_EPOCHS}], Loss: {epoch_loss:.4f}")

    # Final Evaluation (Simple Argmax Voting)
    model.eval()
    session_results = {}
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
                if key not in session_results: session_results[key] = []
                session_results[key].append(store_vals[j])

    fold_win_acc = np.mean(np.array(fold_win_preds) == np.array(fold_win_labels))
    fold_window_accuracies.append(fold_win_acc)

    fold_preds, fold_labels = [], []
    for (fid, true_lab), vals_list in session_results.items():
        mean_probs = np.mean(vals_list, axis=0)
        
        # STANDARD ARGMAX VOTING
        final_prediction = np.argmax(mean_probs)
            
        fold_preds.append(final_prediction)
        fold_labels.append(true_lab)
        
        all_pred_labels.append(final_prediction)
        all_true_labels.append(true_lab)

    fold_acc = accuracy_score(fold_labels, fold_preds)
    fold_accuracies.append(fold_acc)
    fold_cm_data.append((fold_labels, fold_preds))
    
    # ---------------------------------------------------------
    # CALCULATE RAW DEFAULT BIOMETRIC METRICS (FROM ARGMAX)
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

    print(f"-> Fold {fold+1} Session Accuracy : {fold_acc * 100:.2f}%")
    print(f"-> Fold {fold+1} Uncalibrated HTER: {hter:.4f} ({hter * 100:.2f}%)")
    print(f"-> Fold {fold+1} FRR (Bonafide)   : {frr:.4f} ({frr * 100:.2f}%)")
    print(f"-> Fold {fold+1} FAR (Paraphrase) : {far_para:.4f} ({far_para * 100:.2f}%)")

# ---------------------------------------------------------
# 5. FINAL SUMMARY 
# ---------------------------------------------------------
print("-" * 40)
print(f"Scenario {SCENARIO} Average Test Window Acc: {np.mean(fold_window_accuracies):.4f} (+/- {np.std(fold_window_accuracies):.4f})")
print(f"Scenario {SCENARIO} Average Test Session Acc: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
print("-" * 40)

global_acc = accuracy_score(all_true_labels, all_pred_labels)
print(f"\nGlobal System Accuracy: {global_acc * 100:.2f}%\n")

print("=" * 40)
print("UNCALIBRATED BIOMETRIC EVALUATION (ARGMAX)")
print("Target: Bonafide (Class 0)")
print("=" * 40)

y_true_array = np.array(all_true_labels)
y_preds_array = np.array(all_pred_labels)

global_bonafide_mask = (y_true_array == 0)
global_frr = np.sum(y_preds_array[global_bonafide_mask] != 0) / np.sum(global_bonafide_mask)

global_attack_mask = (y_true_array != 0)
global_far = np.sum(y_preds_array[global_attack_mask] == 0) / np.sum(global_attack_mask)

global_hter = (global_frr + global_far) / 2.0

print(f"Global Default HTER (Half Total Error) : {global_hter:.4f} ({global_hter * 100:.2f}%)")
print(f"Global FRR (Bonafide Rejected)         : {global_frr:.4f} ({global_frr * 100:.2f}%)")
print(f"Global FAR (All Attacks Accepted)      : {global_far:.4f} ({global_far * 100:.2f}%)\n")

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
# 6. PLOT SIX LOQO CONFUSION MATRICES (2x3 Grid)
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 10))
axes = axes.flatten() 

for i, (true_labs, pred_labs) in enumerate(fold_cm_data):
    cm = confusion_matrix(true_labs, pred_labs, labels=[0, 1, 2, 3, 4])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["B", "P", "T", "F_P", "F_T"])
    disp.plot(cmap=plt.cm.Purples, ax=axes[i], colorbar=False)
    
    test_q = QUESTION_GROUPS[i][0]
    axes[i].set_title(f"Fold {i+1} (Tested blindly on Q{test_q})")

plt.tight_layout()

# Headless Saving for Cloud Environments
plt.savefig(f"Scenario_{SCENARIO}_LOQO_Baseline_CM2.png", dpi=300, bbox_inches='tight')
plt.close()