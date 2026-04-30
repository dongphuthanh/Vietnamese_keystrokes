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
from sklearn.preprocessing import RobustScaler

# Make sure your cnn_gen_data.py has the TemporalCNN imported
from cnn_gen_data import TemporalCNN

WIN_LENGTH = 200
STRIDE = 50

# --- NEW PARAMETERS ---
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

# --- THE CUSTOM QUESTION GROUPS ---
QUESTION_GROUPS = [
    [1, 4],
    [2, 5],
    [3, 6]
]

# ---------------------------------------------------------
# 1. UPGRADED MODEL: DILATED CNN + EMBEDDINGS
# ---------------------------------------------------------
class TemporalCNN(nn.Module):
    def __init__(self, continuous_dim=1, num_classes=5, hidden_dim=64, embedding_dim=2):
        super().__init__()
        self.projector = nn.Linear(continuous_dim, 8)
        self.embedding = nn.Embedding(num_embeddings=5, embedding_dim=embedding_dim)
        conv_in_dim = 8 + embedding_dim
        
        self.conv1 = nn.Conv1d(in_channels=conv_in_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.dropout2 = nn.Dropout(0.2)
        
        self.conv3 = nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=4, dilation=4)
        self.bn3 = nn.BatchNorm1d(hidden_dim * 4)
        self.dropout3 = nn.Dropout(0.3)
        
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

        if return_features:
            return features
    
        return self.fc(features)

# ---------------------------------------------------------
# 2. DATA GENERATOR (NOW PARSING QUESTION & SESSION)
# ---------------------------------------------------------
def get_key_id(key_str):
    if len(key_str) == 1:
        if key_str.isalpha(): return 0
        elif key_str.isdigit(): return 1               
        elif key_str == ' ': return 2                               
    elif key_str == 'Backspace': return 3                                  
    return 4                                      
def get_key_id2(key_str):
    """Maps a key string to an integer ID from 0 to 38"""
    if len(key_str) == 1:
        if key_str.isalpha():
            return ord(key_str.lower()) - ord('a') # 0 to 25
        elif key_str.isdigit():
            return 26 + int(key_str)               # 26 to 35
        elif key_str == ' ':
            return 36                              # 36
    elif key_str == 'Backspace':
        return 37                                  # 37
    return 38     
def get_key_id3(key_str):
            
    if key_str == ' ': return 0                               
    elif key_str == 'Backspace': return 1                               
    return 2  

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
            if not q_idx_str: 
                continue 
                
            parts = str(q_idx_str).split('.')
            if len(parts) != 2: 
                continue
                
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
            Q_ID.append(q_id) # NEW: Tag this window with its Question ID

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
# 4. DETERMINISTIC CROSS-CONTENT EVALUATION LOOP
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_splits = len(QUESTION_GROUPS)

fold_accuracies = []
fold_window_accuracies = [] 
all_true_labels = []
all_pred_labels = []
all_pred_probs = [] 

allowed_train_classes = SCENARIO_TRAIN_CLASSES[SCENARIO]
print(f"\nRunning Scenario {SCENARIO}: Training on classes {allowed_train_classes}")

for fold in range(n_splits):
    test_questions = QUESTION_GROUPS[fold]
    train_questions = [q for i, group in enumerate(QUESTION_GROUPS) if i != fold for q in group]
    
    print(f"\n--- Fold {fold + 1}/{n_splits} ---")
    print(f"Train Questions: {train_questions} | Test Questions: {test_questions}")
    
    train_content_mask = np.isin(Q_ID, train_questions)
    test_content_mask = np.isin(Q_ID, test_questions)

    X_train_fold, X_test_fold = X[train_content_mask].copy(), X[test_content_mask].copy()
    Y_train_fold, Y_test_fold = Y[train_content_mask], Y[test_content_mask]
    test_fids_fold = FILE_ID[test_content_mask]

    # Scenario Filter (Only drop forbidden classes from Train!)
    scenario_train_mask = np.isin(Y_train_fold, allowed_train_classes)
    X_train_fold = X_train_fold[scenario_train_mask]
    Y_train_fold = Y_train_fold[scenario_train_mask]

    # Robust Scaling
    scaler = RobustScaler()
    train_dt = X_train_fold[:, :, 0].reshape(-1, 1)
    test_dt = X_test_fold[:, :, 0].reshape(-1, 1)
    
    X_train_fold[:, :, 0] = scaler.fit_transform(train_dt).reshape(X_train_fold.shape[0], X_train_fold.shape[1])
    X_test_fold[:, :, 0] = scaler.transform(test_dt).reshape(X_test_fold.shape[0], X_test_fold.shape[1])

    train_ds = TensorDataset(torch.tensor(X_train_fold, dtype=torch.float32), torch.tensor(Y_train_fold, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test_fold, dtype=torch.float32), torch.tensor(Y_test_fold, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = TemporalCNN(continuous_dim=1, num_classes=5, hidden_dim=64, embedding_dim=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    custom_weights = torch.tensor([1.0, 1.1, 1.0, 1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=custom_weights, label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

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

    # Evaluation
    model.eval()
    session_results = {}
    current_idx = 0
    
    fold_win_preds = []   
    fold_win_labels = []  
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            
            store_vals = F.softmax(out, dim=1).cpu().numpy()
            labels = yb.cpu().numpy()
            
            # Compute and store window-level predictions for this batch
            batch_preds = np.argmax(store_vals, axis=1)
            fold_win_preds.extend(batch_preds)
            fold_win_labels.extend(labels)
            
            batch_fids = test_fids_fold[current_idx : current_idx + len(xb)]
            current_idx += len(xb)

            for j in range(len(store_vals)):
                # Key strictly groups by (File ID, Session Label)
                key = (batch_fids[j], labels[j]) 
                if key not in session_results: 
                    session_results[key] = []
                session_results[key].append(store_vals[j])

    # Calculate and store fold window-level accuracy
    fold_win_acc = np.mean(np.array(fold_win_preds) == np.array(fold_win_labels))
    fold_window_accuracies.append(fold_win_acc)

    # Aggregate Session Votes
    fold_preds = []
    fold_labels = []
    
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
    
    # Print both window and file accuracies
    print(f"Fold {fold+1} Window-Level Accuracy: {fold_win_acc:.4f}")
    print(f"Fold {fold+1} Session-Level Accuracy: {fold_acc:.4f}")

# ---------------------------------------------------------
# 5. FINAL SUMMARY & EER
# ---------------------------------------------------------
print("-" * 40)
print(f"Scenario {SCENARIO} Average Window-Level CV Accuracy: {np.mean(fold_window_accuracies):.4f} (+/- {np.std(fold_window_accuracies):.4f})")
print(f"Scenario {SCENARIO} Average Session-Level CV Accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
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

# Final Confusion Matrix
cm = confusion_matrix(all_true_labels, all_pred_labels, labels=[0, 1, 2, 3, 4])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["B", "P", "T", "F_P", "F_T"])
disp.plot(cmap=plt.cm.Purples)
plt.title(f"{SCENARIO} Zero-Shot Content Confusion Matrix")
plt.show()