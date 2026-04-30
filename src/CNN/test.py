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
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import RobustScaler
from optuna.samplers import TPESampler
import random

# ---------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------
USE_DWELL_TIME = False  # TOGGLE: True to use make_windows_with_up, False for standard
VOTING_MODE = 'soft'  
SCENARIO = 'M5'       
OPTUNA_TRIALS = 0
OPTUNA_EPOCHS = 30    
FINAL_EPOCHS = 90
HIDDEN_DIM = 32
WIN_LENGTH = 100
STRIDE = 50
import math

SCENARIO_TRAIN_CLASSES = {
    'M2': [0, 2],          
    'M3': [0, 1, 2],       
    'M4': [0, 1, 2, 4],    
    'M5': [0, 1, 2, 3, 4], 
    'M6': [0, 1]
}

# --- THE CUSTOM QUESTION GROUPS ---
QUESTION_GROUPS = [
    [1,4],
    [2,5],
    [3,6],
]
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



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TemporalCNN(nn.Module):
    # CHANGED: Default num_continuous is now 1 (representing only Flight Time)
    def __init__(self, num_continuous=1, num_classes=5, hidden_dim=64):
        super().__init__()
        
        self.key_embedding = nn.Embedding(num_embeddings=5, embedding_dim=4)
        
        # input_channels is now 1 + 4 = 5 (instead of 8)
        input_channels = num_continuous + 4 
        
        self.resize_layer = nn.Linear(input_channels, hidden_dim)
        self.pos_encoder = PositionalEncoding(d_model=hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=4, 
            batch_first=True, 
            dim_feedforward=hidden_dim * 2
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        self.conv1 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        
        self.dropout = nn.Dropout(0.3)
        self.fc_out = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x arrives as shape: (Batch, Seq_Len, 2)
        
        # 1. The Slice
        # Grabs everything EXCEPT the last column. Shape becomes: (Batch, Seq_Len, 1)
        timings = x[:, :, :-1]              
        
        # Grabs ONLY the last column. Shape becomes: (Batch, Seq_Len)
        key_ids = x[:, :, -1].long()        
        
        # 2. Embed and Fuse
        # key_emb shape: (Batch, Seq_Len, 4)
        key_emb = self.key_embedding(key_ids)       
        
        # fused shape becomes: (Batch, Seq_Len, 5)
        fused = torch.cat([timings, key_emb], dim=-1) 
        
        # 3. Transformer Pipeline 
        # Instantly scales the 5 channels up to your hidden_dim (e.g., 64)
        x_trans = self.resize_layer(fused)
        x_trans = self.pos_encoder(x_trans)
        x_trans = self.transformer(x_trans)
        
        # 4. Shape-Shift for CNN
        x_cnn = x_trans.transpose(1, 2)
        
        # 5. CNN Pipeline 
        x_cnn = self.conv1(x_cnn)
        x_cnn = self.bn1(x_cnn)
        x_cnn = self.relu1(x_cnn)
        
        x_cnn = self.conv2(x_cnn)
        x_cnn = self.bn2(x_cnn)
        x_cnn = self.relu2(x_cnn)
        
        # 6. Global Max Pool & Classify
        x_pool = x_cnn.max(dim=2)[0] 
        out = self.dropout(x_pool)
        out = self.fc_out(out)
        
        return out


def get_key_id(key_str):
    if len(key_str) == 1:
        if key_str.isalpha(): return 0
        elif key_str.isdigit(): return 3            
        elif key_str == ' ': return 1                            
    elif key_str == 'Backspace': return 2                             
    return 4
# ---------------------------------------------------------
# 2. DATA GENERATORS (STANDARD & UPGRADED)
# ---------------------------------------------------------
def get_key_id_6class(current_key_str, prev_key_str=""):
    """
    Maps a keystroke into exactly 6 macro-categories:
    0: Normal Character (a-z)
    1: Digit (0-9)
    2: Space (' ')
    3: Backspace ('Backspace')
    4: Telex Diacritic / Tone
    5: Everything Else (Shift, Enter, Punctuation, etc.)
    """
    ck = str(current_key_str).lower()
    pk = str(prev_key_str).lower() if prev_key_str else ""
    
    is_diacritic = False
    
    if len(ck) == 1:
        if ck in {'a', 'e', 'o', 'd'} and ck == pk:
            is_diacritic = True
        elif ck == 'w' and pk in {'a', 'o', 'u'}:
            is_diacritic = True
        elif ck in {'s', 'f', 'r', 'x', 'j'} and pk.isalpha():
            if ck == 'r' and pk == 't':
                is_diacritic = False
            else:
                is_diacritic = True

    if is_diacritic: return 4                      # Class 4: Diacritic
    if len(ck) == 1:
        if ck.isalpha(): return 0                  # Class 0: Normal Character
        elif ck.isdigit(): return 1                # Class 1: Digit
        elif ck == ' ': return 2                   # Class 2: Space
    if ck == 'backspace': return 3                 # Class 3: Backspace
    return 5                                       # Class 5: Everything Else

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



def make_windows_with_up(filepath, X, Y, ID, FILE_ID, Q_ID, user_id, file_id, win_length, stride):
    with open(filepath, "r", encoding="utf8") as f:
        data = json.load(f)
    _keystrokes = data.get("keystrokes", [])
    is_attack = filepath.endswith("attack.json")

    for i in range(len(_keystrokes) - 1):
        if _keystrokes[i]["event"] == "keydown":
            dt_dwell = np.clip(_keystrokes[i+1]["timestamp"] - _keystrokes[i]["timestamp"], 1, 2000)
            _keystrokes[i]["dwell_time"] = np.log(dt_dwell)
            
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
                dwell_time = keys[j - 1].get("dwell_time", np.log(100))
                
                window.append([np.log(dt), dwell_time, key_id]) 

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

mode_str = "Upgraded Features (Dwell Time + Flags)" if USE_DWELL_TIME else "Standard Features"
print(f"Generating windows with {mode_str}...")
file_counter = 0 

for user_folder in user_folders:
    user_id = user2id[user_folder]
    user_path = os.path.join(folder_path, user_folder)
    for root, _, files in os.walk(user_path):
        for filename in sorted(files):
            if filename.endswith(".json"):
                filepath = os.path.join(root, filename)
                if USE_DWELL_TIME:
                    make_windows_with_up(filepath, X, Y, ID, FILE_ID, Q_ID, user_id, file_counter, WIN_LENGTH, STRIDE)
                else:
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
        
        X_sub = X_train_full[sub_train_idx].copy()
        Y_sub = Y_train_full[sub_train_idx]
        
        X_val = X_train_full[val_idx].copy()
        Y_val = Y_train_full[val_idx]
        FID_val = FID_train_full[val_idx]
        
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
        g = torch.Generator()
        g.manual_seed(42)
        t_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=True, generator=g)
        v_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
        
        model = TemporalCNN(
            num_continuous=NUM_CONTINUOUS_FEATURES, num_classes=5, hidden_dim=HIDDEN_DIM
        ).to(device)
        
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
            
        inner_val_acc = np.mean(np.array(fold_preds) == np.array(fold_labels))
        return inner_val_acc

    sampler = TPESampler(seed=42)
    study_name = f"context_scenario_M5_fold_{fold + 1}"
    study = optuna.create_study(direction="maximize",study_name=study_name, sampler=sampler, storage="sqlite:///test2.db", load_if_exists=True)
    study.optimize(objective, n_trials=OPTUNA_TRIALS,
                   callbacks=[
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
    
    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=True, generator=g)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    model = TemporalCNN(
        num_continuous=NUM_CONTINUOUS_FEATURES, num_classes=5, hidden_dim=HIDDEN_DIM
    ).to(device)
    
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

    targets_to_inspect = [0, 1] 

    for target_label in targets_to_inspect:
        target_fid = None
        
        for (fid, true_lab) in session_results.keys():
            if true_lab == target_label:
                target_fid = fid
                break

        if target_fid is not None:
            print(f"\n" + "="*75)
            class_name = "BONAFIDE" if target_label == 0 else "PARAPHRASE ATTACK"
            print(f"SOFTMAX TIMELINE | File ID: {target_fid} | True Label: {class_name} ({target_label})")
            print("="*75)
            
            window_probs = session_results[(target_fid, target_label)]
            
            print(f"{'Win #':<6} | {'Bonafide':<12} | {'Paraph':<12} | {'Transc':<10} | {'Fake P':<10} | {'Fake T':<10}")
            print("-" * 75)
            
            for i, p in enumerate(window_probs):
                b_star = "*" if np.argmax(p) == 0 else " "
                p_star = "*" if np.argmax(p) == 1 else " "
                
                print(f"W_{i:<4} | {p[0]:.4f} {b_star}   | {p[1]:.4f} {p_star}   | {p[2]:.4f}     | {p[3]:.4f}     | {p[4]:.4f}")
            
            print("="*75 + "\n")

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