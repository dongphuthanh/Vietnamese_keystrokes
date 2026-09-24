import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import os
import optuna
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GroupShuffleSplit
from sklearn.preprocessing import RobustScaler
from optuna.samplers import TPESampler
import random
import sys

# ---------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------
VOTING_MODE = 'soft'  
SCENARIO = sys.argv[1] if len(sys.argv) > 1 else 'M2'
OPTUNA_TRIALS = 0   
OPTUNA_EPOCHS = 30
FINAL_EPOCHS = 100
HIDDEN_DIM = 128
WIN_LENGTH = 100
STRIDE = 50
NUM_CONT_FEATURES = 4

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


def set_seed(seed=42):
    # 1. Set pure Python random seed (used by some data augmentations)
    random.seed(seed)
    
    # 2. Set environment variable to strictly hash Python dictionaries/sets
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 3. Set NumPy random seed (used by Scikit-Learn and Pandas)
    np.random.seed(seed)
    
    # 4. Set PyTorch seed for CPU operations
    torch.manual_seed(seed)
    
    # 5. Set PyTorch seed for all GPU operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # If using multi-GPU
        
    # 6. Force cuDNN to be deterministic (CRITICAL for full reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call the function before you initialize any models or dataloaders
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
            print(f"Halting the inner experiment to save compute.")
            study.stop()
            
    __call__.experimental_disable_run_after_update = False

class TypeNet(nn.Module):
    def __init__(self, sequence_length, in_dim, hidden_dim_1, hidden_dim_2, output_dim, dropout):
        super(TypeNet, self).__init__()
        self.dropout = dropout
        in_dim = 8
        self.embedding = nn.Embedding(5, 4)
        self.lstm1 = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim_1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_dim_1, hidden_size=hidden_dim_2, batch_first=True)
        self.bn1 = nn.BatchNorm1d(num_features=in_dim)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_dim_1)
        self.bn3 = nn.BatchNorm1d(num_features=hidden_dim_2)
        self.act1 = nn.Tanh()
        self.act2 = nn.Tanh()
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(hidden_dim_2, output_dim)

        for name, param in self.lstm1.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.kaiming_uniform_(param.data)
        for name, param in self.lstm2.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.kaiming_uniform_(param.data)
        nn.init.xavier_uniform_(self.fc1.weight)


    def forward(self, x): 
        cont_feats = x[:, :, :-1]     
        key_ids = x[:, :, -1].long() 
        key_emb = self.embedding(key_ids)
        x = torch.cat((cont_feats, key_emb), dim=2)
        x = torch.movedim(x, 2, 1)
        out = self.bn1(x)
        out = torch.movedim(out, 2, 1)

        # 2. Feed directly into LSTM 1 (No packing needed!)
        out, _ = self.lstm1(out) 

        out = self.act1(out)
        out = self.dropout_1(out)

        out = torch.movedim(out, 2, 1)
        out = self.bn2(out)
        out = torch.movedim(out, 2, 1)

        # 3. Feed directly into LSTM 2 (No packing needed!)
        out_p, _ = self.lstm2(out) 

        out_p = self.act2(out_p)
        
        # 4. Safely flatten and classify
        out_p = out_p[:, -1, :]
        out = self.bn3(out_p)
        out = self.dropout_2(out)
        out = self.fc1(out)

        return out

# ---------------------------------------------------------
# 2. DATA GENERATORS (UPGRADED WITH HOLD TIME)
# ---------------------------------------------------------
def get_key_id(key_str):
    if len(key_str) == 1:
        if key_str.isalpha(): return 0
        elif key_str.isdigit(): return 3               
        elif key_str == ' ': return 1                               
    elif key_str == 'Backspace': return 2                                  
    return 4                                      

def make_windows(filepath, X, Y, ID, FILE_ID, user_id, file_id, win_length, stride):
    with open(filepath, "r", encoding="utf8") as f:
        data = json.load(f)
    _keystrokes = data.get("keystrokes", [])
    is_attack = filepath.endswith("attack.json")

    # Step 1: Group ALL events strictly by their Class Label (0-4), ignoring Question IDs
    grouped_events = {0: [], 1: [], 2: [], 3: [], 4: []}
    
    for k in _keystrokes:
        if k["event"] in ["keydown", "keyup"]:
            session_id = k.get("session")
            if session_id is None:
                continue
            
            label = None
            if not is_attack:
                if session_id == 1: label = 0     
                elif session_id == 2: label = 1   
                elif session_id == 3: label = 2   
            else:
                if session_id == 2: label = 3     
                elif session_id == 3: label = 4   

            if label is not None:
                grouped_events[label].append(k)

    # Step 2: Pair Press and Release times to create full Keystrokes
    for label, events in grouped_events.items():
        if len(events) == 0:
            continue
            
        pending_presses = {}
        paired_keys = []
        
        for ev in events:
            char = ev["key"]
            if ev["event"] == "keydown" and char not in pending_presses:
                pending_presses[char] = ev["timestamp"]
            elif ev["event"] == "keyup" and char in pending_presses:
                p_time = pending_presses.pop(char)
                r_time = ev["timestamp"]
                
                # Assumes your robust modulo get_key_id function is still in memory
                keycode = get_key_id(char) 
                
                paired_keys.append({
                    'key': keycode,
                    'press': p_time,
                    'release': r_time
                })
                
        # Must sort by press time to guarantee biological sequence order
        paired_keys.sort(key=lambda x: x['press'])
        
        if len(paired_keys) <= win_length:
            continue

        # Step 3: Extract the exact TypeNet 5 Features
        for i in range(1, len(paired_keys) - win_length + 1, stride):
            window = []
            
            for j in range(i, i + win_length):
                curr = paired_keys[j]
                prev = paired_keys[j - 1]
                
                # 1. Calculate raw milliseconds
                HL = curr['release'] - curr['press']
                IL = curr['press'] - prev['release']
                PL = curr['press'] - prev['press']
                RL = curr['release'] - prev['release']
                
                # 2. TypeNet Normalization (Convert to Seconds)
                # Clip limits extreme anomalies from web capture errors
                HL_sec = np.clip(HL, 0, 5000) / 1000.0
                IL_sec = np.clip(IL, -5000, 5000) / 1000.0
                PL_sec = np.clip(PL, 0, 5000) / 1000.0
                RL_sec = np.clip(RL, -5000, 5000) / 1000.0
                
                # 3. Form the 1x5 vector for this specific timestep
                # Note: We pass the raw 'key' integer so it can feed into PyTorch's nn.Embedding
                window.append([HL_sec, IL_sec, PL_sec, RL_sec, curr['key']])

            # Append the completed (win_length x 5) matrix to the dataset
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

print(f"Generating windows with Up-Events (Hold Time)...")
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
unique_users = np.unique(ID)
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_accuracies = []
fold_window_accuracies = [] 
all_true_labels = []
all_pred_labels = []
all_pred_probs = [] 
fold_cm_data = []

allowed_train_classes = SCENARIO_TRAIN_CLASSES[SCENARIO]
print(f"\nRunning Scenario {SCENARIO}: Automated 3x3 Nested CV Tuning & Testing (WITH HOLD TIME)")

for fold, (train_user_idx, test_user_idx) in enumerate(kf.split(unique_users)):
    print(f"\n" + "="*50)
    print(f"FOLD {fold + 1}/{n_splits} - INITIALIZING OPTUNA EXPERIMENT")
    print("="*50)
    train_users = unique_users[train_user_idx]
    test_users = unique_users[test_user_idx]
    
    # 2. Map those users back to the actual row indices in X
    train_idx = np.where(np.isin(ID, train_users))[0]
    test_idx = np.where(np.isin(ID, test_users))[0]

    X_train_full = X[train_idx]
    Y_train_full = Y[train_idx]
    ID_train_full = ID[train_idx]
    FID_train_full = FILE_ID[train_idx]
    
    def objective(trial):
        lr = trial.suggest_float('lr', 1e-4, 5e-4, log=True)

        # --- 75/25 USER-INDEPENDENT SPLIT ---
        # n_splits=1 means it only generates one train/val pair instead of looping
        gss = GroupShuffleSplit(n_splits=1, test_size=1/3, random_state=42)
        inner_train_idx, inner_val_idx = next(gss.split(X_train_full, Y_train_full, groups=ID_train_full))
        
        X_sub = X_train_full[inner_train_idx].copy()
        Y_sub = Y_train_full[inner_train_idx]
        X_val = X_train_full[inner_val_idx].copy()
        Y_val = Y_train_full[inner_val_idx]
        FID_val = FID_train_full[inner_val_idx]
        
        train_mask = np.isin(Y_sub, allowed_train_classes)
        X_sub = X_sub[train_mask]
        Y_sub = Y_sub[train_mask]
        
        # --- UPGRADED SCALER: Scales both log(dt) and log(hold) without leakage ---
        scaler = RobustScaler()
        X_sub_cont = X_sub[:, :, :NUM_CONT_FEATURES].reshape(-1, NUM_CONT_FEATURES)
        X_sub_cont_scaled = scaler.fit_transform(X_sub_cont)
        X_sub[:, :, :NUM_CONT_FEATURES] = X_sub_cont_scaled.reshape(X_sub.shape[0], X_sub.shape[1], NUM_CONT_FEATURES)
        
        X_val_cont = X_val[:, :, :NUM_CONT_FEATURES].reshape(-1, NUM_CONT_FEATURES)
        X_val_cont_scaled = scaler.transform(X_val_cont)
        X_val[:, :, :NUM_CONT_FEATURES] = X_val_cont_scaled.reshape(X_val.shape[0], X_val.shape[1], NUM_CONT_FEATURES)
        
        train_ds = TensorDataset(torch.tensor(X_sub, dtype=torch.float32), torch.tensor(Y_sub, dtype=torch.long))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.long))
        g = torch.Generator()
        g.manual_seed(42)

        t_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=True, generator=g)
        v_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
        
        model = TypeNet(sequence_length=100, in_dim=5, hidden_dim_1=128, hidden_dim_2=128, output_dim=5, dropout=0.5).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=OPTUNA_EPOCHS)
        
        for epoch in range(OPTUNA_EPOCHS):
            model.train()
            for xb, yb in t_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                    
                loss.backward()
                optimizer.step()
            scheduler.step()

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
            

        return np.mean(np.array(fold_preds) == np.array(fold_labels))

    # --- NEW DB NAME FOR WITH_UP RUNS ---
    sampler = TPESampler(seed=42)
    study_name = f"scenario_M5_fold_{fold + 1}"
    study = optuna.create_study(
        study_name=study_name,
        sampler=sampler,
        storage="sqlite:///optuna_studies/typenet_uie.db",
        load_if_exists=True,
        direction="maximize"
    )
    study.optimize(objective, n_trials=OPTUNA_TRIALS,
                   callbacks=[
            EarlyStoppingCallback(patience=35),
            PeriodicLoggingCallback(print_every=5, total_trials=OPTUNA_TRIALS)
        ])
    best_params = study.best_params
    print(f"Optuna Winning Params for Fold {fold + 1}: {best_params} (Val Acc: {study.best_value:.4f})")
    
    # ---------------------------------------------------------
    # OUTER LOOP: FINAL TESTING WITH WINNING PARAMS
    # ---------------------------------------------------------
    print(f"--- FOLD {fold + 1} FINAL TEST RUN ---")
    
    X_train_final = X[train_idx].copy()
    Y_train_final = Y[train_idx]
    X_test_final = X[test_idx].copy()
    Y_test_final = Y[test_idx]
    FID_test = FILE_ID[test_idx]

    train_mask = np.isin(Y_train_final, allowed_train_classes)
    X_train_final = X_train_final[train_mask]
    Y_train_final = Y_train_final[train_mask]

    # --- UPGRADED SCALER ---
    scaler = RobustScaler()
    X_train_final_cont = X_train_final[:, :, :NUM_CONT_FEATURES].reshape(-1, NUM_CONT_FEATURES)
    X_train_final_scaled = scaler.fit_transform(X_train_final_cont)
    X_train_final[:, :, :NUM_CONT_FEATURES] = X_train_final_scaled.reshape(X_train_final.shape[0], X_train_final.shape[1], NUM_CONT_FEATURES)
    
    X_test_final_cont = X_test_final[:, :, :NUM_CONT_FEATURES].reshape(-1, NUM_CONT_FEATURES)
    X_test_final_scaled = scaler.transform(X_test_final_cont)
    X_test_final[:, :, :NUM_CONT_FEATURES] = X_test_final_scaled.reshape(X_test_final.shape[0], X_test_final.shape[1], NUM_CONT_FEATURES)

    train_ds = TensorDataset(torch.tensor(X_train_final, dtype=torch.float32), torch.tensor(Y_train_final, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test_final, dtype=torch.float32), torch.tensor(Y_test_final, dtype=torch.long))
    
    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=True, generator=g)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)


    model = TypeNet(sequence_length=100, in_dim=5, hidden_dim_1=128, hidden_dim_2=128, output_dim=5, dropout=0.5).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINAL_EPOCHS)

    for epoch in range(FINAL_EPOCHS):
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
        
        if (epoch + 1) % 20 == 0 or (epoch + 1) == FINAL_EPOCHS:
            print(f"Final Train Epoch [{epoch + 1}/{FINAL_EPOCHS}], Loss: {epoch_loss:.4f}")

    # Final Evaluation
    model.eval()
    file_results = {}
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
                if key not in file_results: file_results[key] = []
                file_results[key].append(store_vals[j])

    fold_win_acc = np.mean(np.array(fold_win_preds) == np.array(fold_win_labels))
    fold_window_accuracies.append(fold_win_acc)

    fold_preds, fold_labels = [], []
    for (fid, true_lab), vals_list in file_results.items():
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

# HEADLESS SAVING: New plot name for the run with Hold Time!
plt.savefig(f"figures/typenet_uie_{SCENARIO}_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()