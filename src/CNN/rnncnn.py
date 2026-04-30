import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import os
import optuna
import copy
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler

# ---------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------
USE_DWELL_TIME = False  # Set to True to use Dwell Time + Space/Backspace flags
SCENARIO = 'M5'       
OPTUNA_TRIALS = 30    
OPTUNA_EPOCHS = 30    
FINAL_EPOCHS = 100

SCENARIO_TRAIN_CLASSES = {
    'M2': [0, 2],          
    'M3': [0, 1, 2],       
    'M4': [0, 1, 2, 4],    
    'M5': [0, 1, 2, 3, 4], 
    'M6': [0, 1]
}

QUESTION_GROUPS = [
    [1,4],
    [2,5],
    [3,6],
]

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ---------------------------------------------------------
# HELPER CLASSES & FUNCTIONS
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
            study.stop()
            
    __call__.experimental_disable_run_after_update = False

class HolisticKeystrokeDataset(Dataset):
    def __init__(self, X, Y):
        self.X = [torch.tensor(seq, dtype=torch.float32) for seq in X]
        self.Y = torch.tensor(Y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def collate_fn_pad(batch):
    sequences, labels = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels)
    return sequences_padded, labels

def scale_jagged_sequences(X_train_list, X_test_list, num_cont_features):
    scaler = RobustScaler()
    train_cont_flat = []
    
    for seq in X_train_list:
        for step in seq:
            train_cont_flat.append(step[:num_cont_features])
    scaler.fit(train_cont_flat)
    
    def transform_in_place(X_list):
        for i in range(len(X_list)):
            cont_vals = [step[:num_cont_features] for step in X_list[i]]
            scaled_vals = scaler.transform(cont_vals)
            for j in range(len(X_list[i])):
                for k in range(num_cont_features):
                    X_list[i][j][k] = scaled_vals[j][k]
                    
    transform_in_place(X_train_list)
    transform_in_place(X_test_list)

# ---------------------------------------------------------
# MODEL ARCHITECTURE (CRNN Hybrid)
# ---------------------------------------------------------
class TemporalCRNN(nn.Module):
    def __init__(self, continuous_dim=1, num_classes=5, cnn_hidden=64, rnn_hidden=64, 
                 proj_dim=8, embedding_dim=4, num_layers=2,
                 dropout_cnn=0.2, dropout_rnn=0.3, dropout_fc=0.4):
        super().__init__()
        self.projector = nn.Linear(continuous_dim, proj_dim)
        self.embedding = nn.Embedding(num_embeddings=40, embedding_dim=embedding_dim) 
        
        input_dim = proj_dim + embedding_dim
        
        # 1. The Time Compressor (CNN) - Cuts length by 4x
        self.compressor = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_hidden, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(cnn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_cnn),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # 2. The Sequence Learner (LSTM)
        self.lstm = nn.LSTM(
            input_size=cnn_hidden, 
            hidden_size=rnn_hidden, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout_rnn if num_layers > 1 else 0.0,
            bidirectional=True 
        )
        
        lstm_out_dim = rnn_hidden * 2 
        
        # 3. Final Aggregation
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(lstm_out_dim, rnn_hidden)
        self.dropout_fc = nn.Dropout(dropout_fc)
        self.fc2 = nn.Linear(rnn_hidden, num_classes)
        
    def forward(self, x):
        cont_feats = x[:, :, :-1]     
        cont_feats = F.relu(self.projector(cont_feats))
        key_ids = x[:, :, -1].long()    
        embedded_keys = self.embedding(key_ids) 
        
        x_combined = torch.cat([cont_feats, embedded_keys], dim=2) 
        
        # FLIP FOR CNN: (Batch, Features, Seq_Len)
        x_cnn_in = x_combined.permute(0, 2, 1)  
        
        # Compress Timeline
        x_cnn_out = self.compressor(x_cnn_in)
        
        # FLIP FOR LSTM: (Batch, Smaller_Seq_Len, CNN_Channels)
        x_rnn_in = x_cnn_out.permute(0, 2, 1)
        
        # Pass to LSTM
        rnn_out, _ = self.lstm(x_rnn_in) 
        
        # Pool & Classify
        x_pool = rnn_out.permute(0, 2, 1)
        features = self.pool(x_pool).squeeze(-1)
        
        features = F.relu(self.fc1(features))
        features = self.dropout_fc(features)
        
        return self.fc2(features)

# ---------------------------------------------------------
# DATA EXTRACTION
# ---------------------------------------------------------
def get_key_id2(key_str):
    if len(key_str) == 1:
        if key_str.isalpha(): return ord(key_str.lower()) - ord('a') 
        elif key_str.isdigit(): return 26 + int(key_str)               
        elif key_str == ' ': return 36                              
    elif key_str == 'Backspace': return 37                                 
    return 38   

def process_file_holistic(filepath, X, Y, ID, FILE_ID, Q_ID, user_id, file_id, use_dwell):
    with open(filepath, "r", encoding="utf8") as f:
        data = json.load(f)
    _keystrokes = data.get("keystrokes", [])
    is_attack = filepath.endswith("attack.json")

    if use_dwell:
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
        if len(keys) < 2: continue
        sequence = []
        for j in range(1, len(keys)):
            dt = np.clip(keys[j]["timestamp"] - keys[j - 1]["timestamp"], 1, 5000)
            key_id = get_key_id2(keys[j]["key"])
            
            if use_dwell:
                is_space = 1.0 if (keys[j - 1]["key"] == " ") else 0.0
                is_backspace = 1.0 if (keys[j]["key"] == "Backspace") else 0.0
                dwell_time = keys[j - 1].get("dwell_time", np.log(100))
                sequence.append([np.log(dt), dwell_time, is_space, is_backspace, key_id]) 
            else:
                sequence.append([np.log(dt), key_id])

        # Increased cap to 2000 because CNN cuts it to 500 for the LSTM
        sequence = sequence[:1200] 

        X.append(sequence)
        Y.append(label)
        ID.append(user_id)
        FILE_ID.append(file_id) 
        Q_ID.append(q_id)

X, Y, ID, FILE_ID, Q_ID = [], [], [], [], []

# !!!!!!!!!!!!!!!! CHANGE PATH HERE AS NEEDED !!!!!!!!!!!!!!!!
folder_path = "../../dataset/Attack4" 
if not os.path.exists(folder_path):
    print(f"WARNING: Directory {folder_path} not found. Ensure the path is correct.")

user_folders = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
user2id = {user: i for i, user in enumerate(user_folders)}

mode_str = "Upgraded Features (Dwell Time + Flags)" if USE_DWELL_TIME else "Standard Features"
print(f"Generating holistic CRNN sequences with {mode_str}...")
file_counter = 0 

for user_folder in user_folders:
    user_id = user2id[user_folder]
    user_path = os.path.join(folder_path, user_folder)
    for root, _, files in os.walk(user_path):
        for filename in sorted(files):
            if filename.endswith(".json"):
                filepath = os.path.join(root, filename)
                process_file_holistic(filepath, X, Y, ID, FILE_ID, Q_ID, user_id, file_counter, USE_DWELL_TIME)
                file_counter += 1

Y = np.array(Y)
ID = np.array(ID)
FILE_ID = np.array(FILE_ID) 
Q_ID = np.array(Q_ID)

if len(X) == 0:
    raise ValueError("No data loaded. Check your folder_path!")

NUM_CONTINUOUS_FEATURES = len(X[0][0]) - 1
print(f"Dataset Loaded -> Sessions: {len(X)} | Continuous Features Detected: {NUM_CONTINUOUS_FEATURES}")

# ---------------------------------------------------------
# NESTED CROSS-CONTENT OPTUNA EVALUATION LOOP
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

for fold in range(n_splits):
    test_questions = QUESTION_GROUPS[fold]
    train_questions = [q for i, group in enumerate(QUESTION_GROUPS) if i != fold for q in group]
    
    print(f"\n" + "="*50)
    print(f"OUTER FOLD {fold + 1}/{n_splits} - Optuna Tuning & Final Test (CRNN)")
    print(f"Train Qs: {train_questions} | Test Qs: {test_questions}")
    print("="*50)

    train_content_mask = np.isin(Q_ID, train_questions)
    test_content_mask = np.isin(Q_ID, test_questions)

    X_train_full = [X[i] for i, m in enumerate(train_content_mask) if m]
    Y_train_full = Y[train_content_mask]
    QID_train_full = Q_ID[train_content_mask]
    FID_train_full = FILE_ID[train_content_mask]
    
    def objective(trial):
        lr = trial.suggest_float('lr', 1e-4, 2e-3, log=True)
        cnn_hidden = trial.suggest_categorical('cnn_hidden', [32, 64])
        rnn_hidden = trial.suggest_categorical('rnn_hidden', [32, 64])
        proj_dim = trial.suggest_categorical('proj_dim', [4, 8, 16])
        emb_dim = trial.suggest_categorical('embedding_dim', [4, 8, 16])
        
        num_layers = trial.suggest_int('num_layers', 1, 2)
        drop_cnn = trial.suggest_float('dropout_cnn', 0.1, 0.5)
        drop_rnn = trial.suggest_float('dropout_rnn', 0.1, 0.5)
        drop_fc = trial.suggest_float('dropout_fc', 0.1, 0.5)
        ls = trial.suggest_float('label_smoothing', 0.0, 0.2)
        
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        sub_train_idx, val_idx = next(gss.split(X_train_full, Y_train_full, groups=QID_train_full))
        
        X_sub = copy.deepcopy([X_train_full[i] for i in sub_train_idx])
        Y_sub = Y_train_full[sub_train_idx]
        X_val = copy.deepcopy([X_train_full[i] for i in val_idx])
        Y_val = Y_train_full[val_idx]
        
        train_mask = np.isin(Y_sub, allowed_train_classes)
        X_sub = [X_sub[i] for i, m in enumerate(train_mask) if m]
        Y_sub = Y_sub[train_mask]
        
        scale_jagged_sequences(X_sub, X_val, NUM_CONTINUOUS_FEATURES)
        
        train_ds = HolisticKeystrokeDataset(X_sub, Y_sub)
        val_ds = HolisticKeystrokeDataset(X_val, Y_val)
        
        t_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn_pad)
        v_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn_pad)
        
        model = TemporalCRNN(
            continuous_dim=NUM_CONTINUOUS_FEATURES, num_classes=5, cnn_hidden=cnn_hidden, rnn_hidden=rnn_hidden, 
            proj_dim=proj_dim, embedding_dim=emb_dim,
            num_layers=num_layers, dropout_cnn=drop_cnn, dropout_rnn=drop_rnn, dropout_fc=drop_fc
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(label_smoothing=ls)
        
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
        
        FID_val = FID_train_full[val_idx] 
        with torch.no_grad():
            for xb, yb in v_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                store_vals = F.softmax(out, dim=1).cpu().numpy()
                labels = yb.cpu().numpy()
                
                batch_fids = FID_val[current_idx : current_idx + len(xb)]
                current_idx += len(xb)
                
                for j in range(len(store_vals)):
                    key = (batch_fids[j], labels[j]) 
                    if key not in session_results: session_results[key] = []
                    session_results[key].append(store_vals[j])
                    
        fold_preds, fold_labels = [], []
        # STANDARD SOFT VOTING
        for (fid, true_lab), vals_list in session_results.items():
            mean_probs = np.mean(vals_list, axis=0) 
            fold_preds.append(np.argmax(mean_probs))
            fold_labels.append(true_lab)
                
        return np.mean(np.array(fold_preds) == np.array(fold_labels))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS,
                   callbacks=[EarlyStoppingCallback(patience=15), PeriodicLoggingCallback(print_every=5, total_trials=OPTUNA_TRIALS)])
    best_params = study.best_params
    
    print(f"Optuna Winning Params for Outer Fold {fold + 1}:")
    print(json.dumps(best_params, indent=4))
    
    # ---------------------------------------------------------
    # FINAL TESTING
    # ---------------------------------------------------------
    X_train_final = copy.deepcopy([X[i] for i, m in enumerate(train_content_mask) if m])
    Y_train_final = Y[train_content_mask]
    X_test_final = copy.deepcopy([X[i] for i, m in enumerate(test_content_mask) if m])
    Y_test_final = Y[test_content_mask]
    FID_test = FILE_ID[test_content_mask]

    scenario_train_mask = np.isin(Y_train_final, allowed_train_classes)
    X_train_final = [X_train_final[i] for i, m in enumerate(scenario_train_mask) if m]
    Y_train_final = Y_train_final[scenario_train_mask]

    scale_jagged_sequences(X_train_final, X_test_final, NUM_CONTINUOUS_FEATURES)

    train_ds = HolisticKeystrokeDataset(X_train_final, Y_train_final)
    test_ds = HolisticKeystrokeDataset(X_test_final, Y_test_final)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn_pad)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn_pad)

    model = TemporalCRNN(
        continuous_dim=NUM_CONTINUOUS_FEATURES, num_classes=5, cnn_hidden=best_params['cnn_hidden'], rnn_hidden=best_params['rnn_hidden'], 
        proj_dim=best_params['proj_dim'], embedding_dim=best_params['embedding_dim'],
        num_layers=best_params['num_layers'], dropout_cnn=best_params['dropout_cnn'], dropout_rnn=best_params['dropout_rnn'], dropout_fc=best_params['dropout_fc']
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=best_params['label_smoothing'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINAL_EPOCHS)

    for epoch in range(FINAL_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Final Evaluation
    model.eval()
    session_results = {}
    current_idx = 0
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            store_vals = F.softmax(out, dim=1).cpu().numpy()
            labels = yb.cpu().numpy()
            
            batch_fids = FID_test[current_idx : current_idx + len(xb)]
            current_idx += len(xb)

            for j in range(len(store_vals)):
                key = (batch_fids[j], labels[j]) 
                if key not in session_results: session_results[key] = []
                session_results[key].append(store_vals[j])

    fold_preds, fold_labels = [], []
    
    # STANDARD SOFT VOTING
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
    
    print(f"-> Fold {fold+1} True Test Session Accuracy: {fold_acc:.4f}")

# ---------------------------------------------------------
# FINAL SUMMARY & EER
# ---------------------------------------------------------
print("\n" + "-" * 40)
print(f"Scenario {SCENARIO} Average Session Acc: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
print("-" * 40)

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

# Display CMs
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (true_labs, pred_labs) in enumerate(fold_cm_data):
    cm = confusion_matrix(true_labs, pred_labs, labels=[0, 1, 2, 3, 4])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["B", "P", "T", "F_P", "F_T"])
    disp.plot(cmap=plt.cm.Purples, ax=axes[i], colorbar=False)
    axes[i].set_title(f"Outer Fold {i+1} True Test Matrix")

plt.tight_layout()
plt.show()