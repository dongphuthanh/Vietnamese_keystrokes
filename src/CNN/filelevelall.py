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
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import RobustScaler

# ---------------------------------------------------------
# PARAMETERS & MASTER TOGGLES
# ---------------------------------------------------------
VOTING_MODE = 'soft'  
SCENARIO = 'M5'       
OPTUNA_TRIALS = 100   
OPTUNA_EPOCHS = 30
WIN_LENGTH = 200
STRIDE = 50

# --- THE UPGRADE TOGGLES ---
USE_ATTENTION = False # True to use Temporal Attention, False to use AdaptiveMaxPool1d
USE_JITTER =   True  # True to inject Additive Gaussian noise to continuous features during training

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

if device.type == 'cuda':
    # Limits PyTorch to 25% of your GPU VRAM to prevent game crashes
    torch.cuda.set_per_process_memory_fraction(0.5, device=0)
    torch.backends.cudnn.benchmark = True

# ---------------------------------------------------------
# 1. UPGRADED LOSS & MODEL (ATTENTION + DYNAMIC PROJECTION)
# ---------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, label_smoothing=self.label_smoothing, reduction='none')
        probs = F.softmax(inputs, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

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

class TemporalCNN(nn.Module):
    def __init__(self, continuous_dim=1, num_classes=5, hidden_dim=32, 
                 proj_dim=8, embedding_dim=8, 
                 dropout1=0.2, dropout2=0.2, dropout3=0.3,
                 use_attention=False):
        super().__init__()
        
        self.projector = nn.Linear(continuous_dim, proj_dim)
        self.embedding = nn.Embedding(num_embeddings=5, embedding_dim=embedding_dim) 
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
        
        self.use_attention = use_attention
        if self.use_attention:
            self.attention = TemporalAttention(hidden_dim)
        else:
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
        
        if self.use_attention:
            features = self.attention(x)
        else:
            features = self.pool(x).squeeze(-1)

        if return_features: return features
        return self.fc(features)

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

print(f"Generating windows... (Attention: {USE_ATTENTION}, Jitter: {USE_JITTER})")
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
print(f"\nRunning Scenario {SCENARIO}: Automated Nested CV Tuning & Testing")

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, Y, groups=ID)):
    print(f"\n" + "="*50)
    print(f"FOLD {fold + 1}/{n_splits} - INITIALIZING OPTUNA EXPERIMENT")
    print("="*50)
    
    # ---------------------------------------------------------
    # INNER LOOP: OPTUNA TUNING
    # ---------------------------------------------------------
    X_train_full = X[train_idx]
    Y_train_full = Y[train_idx]
    ID_train_full = ID[train_idx]
    FID_train_full = FILE_ID[train_idx]
    
    def objective(trial):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        sub_train_idx, val_idx = next(gss.split(X_train_full, Y_train_full, groups=ID_train_full))
        
        X_sub = X_train_full[sub_train_idx].copy()
        Y_sub = Y_train_full[sub_train_idx]
        
        X_val = X_train_full[val_idx].copy()
        Y_val = Y_train_full[val_idx]
        FID_val = FID_train_full[val_idx]
        
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
        
        lr = trial.suggest_float('lr', 1e-4, 2e-3, log=True)
        hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128])
        proj_dim = trial.suggest_categorical('proj_dim', [2, 4, 8, 16])
        emb_dim = trial.suggest_categorical('embedding_dim', [2, 4, 8, 16])
        drop1 = trial.suggest_float('dropout1', 0.1, 0.5)
        drop2 = trial.suggest_float('dropout2', 0.1, 0.5)
        drop3 = trial.suggest_float('dropout3', 0.1, 0.5)
        
        # --- NEW OPTUNA DYNAMIC PARAMS ---
        ls = trial.suggest_float('label_smoothing', 0.0, 0.14, step=0.02)
        #gamma_val = trial.suggest_float('gamma', 0.0, 5.0) # Search from no focal loss (0.0) to extreme focus (5.0)
        jitter_std = trial.suggest_float('jitter_std', 0.01, 0.15) if USE_JITTER else 0.0
        
        model = TemporalCNN(
            continuous_dim=1, num_classes=5, hidden_dim=hidden_dim, 
            proj_dim=proj_dim, embedding_dim=emb_dim,
            dropout1=drop1, dropout2=drop2, dropout3=drop3,
            use_attention=USE_ATTENTION
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # --- PASS GAMMA TO CRITERION ---
        criterion = nn.CrossEntropyLoss(label_smoothing=ls)
        
        for epoch in range(OPTUNA_EPOCHS):
            model.train()
            for xb, yb in t_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                
                # --- ADDITIVE JITTER (INNER LOOP) ---
                if USE_JITTER:
                    noise_adder = torch.normal(mean=0.0, std=jitter_std, size=xb[:, :, :-1].shape).to(device)
                    noise_adder = torch.clamp(noise_adder, min=-(3 * jitter_std), max=(3 * jitter_std))
                    xb_augmented = xb.clone()
                    xb_augmented[:, :, :-1] = xb[:, :, :-1] + noise_adder
                    loss = criterion(model(xb_augmented), yb)
                else:
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
            
        return np.mean(np.array(fold_preds) == np.array(fold_labels))

    # --- SAVE STATE DATABASE CONNECTION ---
    study_name = f"scenario_{SCENARIO}_fold_{fold + 1}"
    study = optuna.create_study(
        study_name=study_name,
        storage="sqlite:///keystroke_optuna.db",
        load_if_exists=True,
        direction="maximize"
    )
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
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
        dropout1=best_params['dropout1'], dropout2=best_params['dropout2'], dropout3=best_params['dropout3'],
        use_attention=USE_ATTENTION
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=1e-4)
    
    # --- PULL THE BEST GAMMA FOR THE FINAL RUN ---
    best_gamma = best_params.get('gamma', 2.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=best_params['label_smoothing'])
    
    final_run_epochs = 100
    best_jitter = best_params.get('jitter_std', 0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=final_run_epochs)

    for epoch in range(final_run_epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            
            # --- ADDITIVE JITTER (OUTER LOOP) ---
            if USE_JITTER:
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
        
        if (epoch + 1) % 20 == 0 or (epoch + 1) == final_run_epochs:
            print(f"Final Train Epoch [{epoch + 1}/{final_run_epochs}], Loss: {epoch_loss:.4f}")

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
plt.show()