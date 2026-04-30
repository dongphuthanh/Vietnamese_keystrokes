import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import os
import optuna
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import RobustScaler
from optuna.samplers import TPESampler
import random

# ---------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------
USE_DWELL_TIME = False 
POOLING_MODE = 'mean' # TOGGLE: 'mean' or 'attention'
SCENARIO = 'M5'       
OPTUNA_TRIALS = 10
OPTUNA_EPOCHS = 30    
FINAL_EPOCHS = 30
HIDDEN_DIM = 64 # Kept at 64 to prevent OOM/Overfitting on session grouping
WIN_LENGTH = 100
STRIDE = 50
BATCH_SIZE = 8 # Lowered because 1 batch item = 1 full session of windows!

SCENARIO_TRAIN_CLASSES = {
    'M2': [0, 2],          
    'M3': [0, 1, 2],       
    'M4': [0, 1, 2, 4],    
    'M5': [0, 1, 2, 3, 4], 
    'M6': [0, 1]
}

QUESTION_GROUPS = [
    [1,4], [2,5], [3,6],
]

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

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
            study.stop()
            
    __call__.experimental_disable_run_after_update = False


# ---------------------------------------------------------
# 1. UPGRADED SESSION MODEL
# ---------------------------------------------------------
class SessionTemporalCNN(nn.Module):
    def __init__(self, continuous_dim=1, num_classes=5, hidden_dim=64, dropout_fc=0.3, pool_mode='mean'):
        super().__init__()
        self.pool_mode = pool_mode
        
        # --- Window-Level Feature Extractor ---
        unified_in_dim = 8
        self.time_proj = nn.Linear(continuous_dim, 4)
        self.unified_projector = nn.Linear(unified_in_dim, hidden_dim)
        self.embedding = nn.Embedding(5, 4)
        
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout1d(0.1)
        
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout1d(0.1)

        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=4)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout3 = nn.Dropout1d(0.1)

        self.pool = nn.AdaptiveMaxPool1d(1) 
        
        # --- Session-Level Pooling Mechanism ---
        if self.pool_mode == 'attention':
            self.attn_weights = nn.Linear(hidden_dim, 1)
        
        # --- Final Classifier ---
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_fc = nn.Dropout(dropout_fc)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def extract_window_features(self, x):
        cont_feats = x[:, :, :-1]     
        key_ids = x[:, :, -1].long() 
        cont_feats = F.relu(self.time_proj(cont_feats))
        key_ids = self.embedding(key_ids)
        unified_input = torch.cat([cont_feats, key_ids], dim=2) 
        x_proj = F.relu(self.unified_projector(unified_input))
        
        x_cnn = x_proj.permute(0, 2, 1)  
        x_cnn = self.dropout1(F.relu(self.bn1(self.conv1(x_cnn))))
        x_cnn = self.dropout2(F.relu(self.bn2(self.conv2(x_cnn))))
        x_cnn = self.dropout3(F.relu(self.bn3(self.conv3(x_cnn))))

        features = self.pool(x_cnn).squeeze(-1)
        return features

    def forward(self, x, mask):
        # x shape: [Batch, Num_Windows, Seq_Len, Features]
        B, W, S, F_dim = x.shape
        
        # 1. Flatten to extract features for every window at once
        x_flat = x.view(B * W, S, F_dim)
        feat_flat = self.extract_window_features(x_flat)
        
        # 2. Reshape back into distinct sessions
        features = feat_flat.view(B, W, -1) # [Batch, Num_Windows, Hidden_Dim]
        
        # 3. Apply Session Pooling (Ignoring padded zeros using mask)
        if self.pool_mode == 'mean':
            mask_expanded = mask.unsqueeze(-1).float() # [B, W, 1]
            masked_features = features * mask_expanded
            sum_features = masked_features.sum(dim=1)
            valid_counts = mask_expanded.sum(dim=1).clamp(min=1.0)
            session_emb = sum_features / valid_counts
            
        elif self.pool_mode == 'attention':
            attn_scores = self.attn_weights(features) # [B, W, 1]
            # Force padded windows to -infinity so Softmax ignores them
            attn_scores = attn_scores.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            attn_probs = F.softmax(attn_scores, dim=1) 
            session_emb = (features * attn_probs).sum(dim=1) # [B, Hidden_Dim]
            
        # 4. Final Classification
        out = F.relu(self.fc1(session_emb))
        out = self.dropout_fc(out)
        return self.fc2(out)

# ---------------------------------------------------------
# 2. SESSION DATA LOADERS & DATA GENERATION
# ---------------------------------------------------------
class SessionDataset(Dataset):
    def __init__(self, X_list, Y_list, FID_list):
        self.X_list = X_list
        self.Y_list = Y_list
        self.FID_list = FID_list
        
    def __len__(self):
        return len(self.X_list)
        
    def __getitem__(self, idx):
        return torch.tensor(self.X_list[idx], dtype=torch.float32), self.Y_list[idx], self.FID_list[idx]

def session_collate_fn(batch):
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    fids = [item[2] for item in batch]
    
    lengths = [x.shape[0] for x in xs]
    max_len = max(lengths)
    
    # Pad sequences with zeros
    batch_x = torch.zeros(len(batch), max_len, xs[0].shape[1], xs[0].shape[2])
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    
    for i, x in enumerate(xs):
        batch_x[i, :lengths[i], :, :] = x
        mask[i, :lengths[i]] = True
        
    return batch_x, torch.tensor(ys, dtype=torch.long), mask, fids

def group_by_session(X, Y, FID):
    # Create a composite key to ensure we don't mix different classes from the same file!
    # Example key: "User5_Class0" vs "User5_Class3"
    composite_keys = np.array([f"{fid}_{y}" for fid, y in zip(FID, Y)])
    unique_keys = np.unique(composite_keys)
    
    X_sess, Y_sess, FID_sess = [], [], []
    for key in unique_keys:
        idx = np.where(composite_keys == key)[0]
        
        # Now these are purely one class!
        X_sess.append(X[idx])
        Y_sess.append(Y[idx[0]]) 
        
        # Extract the original FID back out of the string
        original_fid = int(key.split('_')[0])
        FID_sess.append(original_fid)
        
    return X_sess, np.array(Y_sess), np.array(FID_sess)

# ... [KEEP YOUR EXISTING get_key_id AND make_windows FUNCTIONS HERE] ...
def get_key_id(key_str):
    if len(key_str) == 1:
        if key_str.isalpha(): return 0
        elif key_str.isdigit(): return 3            
        elif key_str == ' ': return 1                            
    elif key_str == 'Backspace': return 2                              
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

print(f"Generating standard windows...")
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
    # INNER LOOP: OPTUNA TUNING (FAST 75/25 SPLIT)
    # ---------------------------------------------------------
    X_train_full = X[train_content_mask].copy()
    Y_train_full = Y[train_content_mask]
    FID_train_full = FILE_ID[train_content_mask]
    QID_train_full = Q_ID[train_content_mask]
    
    def objective(trial):
        lr = trial.suggest_float('lr', 1e-4, 2e-3, log=True)
        
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        sub_train_idx, val_idx = next(gss.split(X_train_full, Y_train_full, groups=QID_train_full))
        
        # 1. Grab splits (Including FID for later grouping!)
        X_sub = X_train_full[sub_train_idx].copy()
        Y_sub = Y_train_full[sub_train_idx]
        FID_sub = FID_train_full[sub_train_idx]
        
        X_val = X_train_full[val_idx].copy()
        Y_val = Y_train_full[val_idx]
        FID_val = FID_train_full[val_idx]
        
        train_mask = np.isin(Y_sub, allowed_train_classes)
        X_sub = X_sub[train_mask]
        Y_sub = Y_sub[train_mask]
        FID_sub = FID_sub[train_mask] # Mask FID too
        
        # 2. Scale Flat Data
        scaler = RobustScaler()
        train_cont = X_sub[:, :, :NUM_CONTINUOUS_FEATURES].reshape(-1, NUM_CONTINUOUS_FEATURES)
        val_cont = X_val[:, :, :NUM_CONTINUOUS_FEATURES].reshape(-1, NUM_CONTINUOUS_FEATURES)
        
        X_sub[:, :, :NUM_CONTINUOUS_FEATURES] = scaler.fit_transform(train_cont).reshape(X_sub.shape[0], X_sub.shape[1], NUM_CONTINUOUS_FEATURES)
        X_val[:, :, :NUM_CONTINUOUS_FEATURES] = scaler.transform(val_cont).reshape(X_val.shape[0], X_val.shape[1], NUM_CONTINUOUS_FEATURES)
        
        # 3. Group Scaled Data into Sessions
        X_sub_sess, Y_sub_sess, FID_sub_sess = group_by_session(X_sub, Y_sub, FID_sub)
        X_val_sess, Y_val_sess, FID_val_sess = group_by_session(X_val, Y_val, FID_val)

        train_ds = SessionDataset(X_sub_sess, Y_sub_sess, FID_sub_sess)
        val_ds = SessionDataset(X_val_sess, Y_val_sess, FID_val_sess)
        
        g = torch.Generator()
        g.manual_seed(42)
        t_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=session_collate_fn, num_workers=0, pin_memory=True, generator=g)
        v_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=session_collate_fn, num_workers=0, pin_memory=True)
        
        model = SessionTemporalCNN(
            continuous_dim=NUM_CONTINUOUS_FEATURES, num_classes=5, hidden_dim=HIDDEN_DIM, pool_mode=POOLING_MODE
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=OPTUNA_EPOCHS)
        
        for epoch in range(OPTUNA_EPOCHS):
            model.train()
            for xb, yb, mask, _ in t_loader:
                xb, yb, mask = xb.to(device), yb.to(device), mask.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb, mask), yb)
                loss.backward()
                optimizer.step()  
            scheduler.step()
                
        model.eval()
        fold_preds, fold_labels = [], []
        with torch.no_grad():
            for xb, yb, mask, _ in v_loader:
                xb, mask = xb.to(device), mask.to(device)
                out = model(xb, mask)
                preds = torch.argmax(out, dim=1).cpu().numpy()
                fold_preds.extend(preds)
                fold_labels.extend(yb.numpy())
                
        inner_val_acc = np.mean(np.array(fold_preds) == np.array(fold_labels))
        return inner_val_acc

    sampler = TPESampler(seed=42)
    study_name = f"context_scenario_M5_fold_{fold + 1}"
    study = optuna.create_study(direction="maximize",study_name=study_name, sampler=sampler, storage="sqlite:///sessionfusion.db", load_if_exists=True)
    study.optimize(objective, n_trials=OPTUNA_TRIALS, callbacks=[
            EarlyStoppingCallback(patience=35),
            PeriodicLoggingCallback(print_every=5, total_trials=OPTUNA_TRIALS)
        ])
    best_params = study.best_params
    
    print(f"Optuna Winning Params for Outer Fold {fold + 1}:")
    print(json.dumps(best_params, indent=4))
    print(f"(Inner Val Acc: {study.best_value:.4f})")
    
    # ---------------------------------------------------------
    # OUTER LOOP: FINAL TESTING WITH WINNING PARAMS
    # ---------------------------------------------------------
    print(f"--- OUTER FOLD {fold + 1} FINAL TEST RUN ---")
    
    X_train_final = X[train_content_mask].copy()
    Y_train_final = Y[train_content_mask]
    FID_train_final = FILE_ID[train_content_mask]
    
    X_test_final = X[test_content_mask].copy()
    Y_test_final = Y[test_content_mask]
    FID_test_final = FILE_ID[test_content_mask]

    scenario_train_mask = np.isin(Y_train_final, allowed_train_classes)
    X_train_final = X_train_final[scenario_train_mask]
    Y_train_final = Y_train_final[scenario_train_mask]
    FID_train_final = FID_train_final[scenario_train_mask]

    scaler = RobustScaler()
    train_cont = X_train_final[:, :, :NUM_CONTINUOUS_FEATURES].reshape(-1, NUM_CONTINUOUS_FEATURES)
    test_cont = X_test_final[:, :, :NUM_CONTINUOUS_FEATURES].reshape(-1, NUM_CONTINUOUS_FEATURES)
    
    X_train_final[:, :, :NUM_CONTINUOUS_FEATURES] = scaler.fit_transform(train_cont).reshape(X_train_final.shape[0], X_train_final.shape[1], NUM_CONTINUOUS_FEATURES)
    X_test_final[:, :, :NUM_CONTINUOUS_FEATURES] = scaler.transform(test_cont).reshape(X_test_final.shape[0], X_test_final.shape[1], NUM_CONTINUOUS_FEATURES)

    # Group into sessions!
    X_tr_sess, Y_tr_sess, FID_tr_sess = group_by_session(X_train_final, Y_train_final, FID_train_final)
    X_ts_sess, Y_ts_sess, FID_ts_sess = group_by_session(X_test_final, Y_test_final, FID_test_final)

    train_ds = SessionDataset(X_tr_sess, Y_tr_sess, FID_tr_sess)
    test_ds = SessionDataset(X_ts_sess, Y_ts_sess, FID_ts_sess)
    
    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=session_collate_fn, num_workers=0, pin_memory=True, generator=g)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=session_collate_fn, num_workers=0, pin_memory=True)

    model = SessionTemporalCNN(
        continuous_dim=NUM_CONTINUOUS_FEATURES, num_classes=5, hidden_dim=HIDDEN_DIM, pool_mode=POOLING_MODE
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINAL_EPOCHS)
    
    for epoch in range(FINAL_EPOCHS):
        model.train()
        running_loss = 0.0
        for xb, yb, mask, _ in train_loader:
            xb, yb, mask = xb.to(device), yb.to(device), mask.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb, mask), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        if (epoch + 1) % 10 == 0 or (epoch + 1) == FINAL_EPOCHS:
            print(f"Final Train Epoch [{epoch + 1}/{FINAL_EPOCHS}], Loss: {epoch_loss:.4f}")

    # Final Evaluation (Drastically simpler now!)
    model.eval()
    fold_preds, fold_labels = [], []
    
    with torch.no_grad():
        for xb, yb, mask, fids in test_loader:
            xb, mask = xb.to(device), mask.to(device)
            out = model(xb, mask)
            probs = F.softmax(out, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            fold_preds.extend(preds)
            fold_labels.extend(yb.numpy())
            all_pred_labels.extend(preds)
            all_true_labels.extend(yb.numpy())
            all_pred_probs.extend(probs)

    fold_acc = np.mean(np.array(fold_preds) == np.array(fold_labels))
    fold_accuracies.append(fold_acc)
    fold_cm_data.append((fold_labels, fold_preds))
    
    print(f"-> Fold {fold+1} True Test Session Accuracy: {fold_acc:.4f}")

# ---------------------------------------------------------
# 5. FINAL SUMMARY & EER
# ---------------------------------------------------------
print("-" * 40)
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