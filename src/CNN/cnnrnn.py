import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler

WIN_LENGTH = 200
STRIDE = 50

# --- NEW CHUNKING PARAMETERS ---
SEQ_LENGTH = 50     # How many windows the RNN sees in a single sample
SEQ_STRIDE = 5        # How many windows to slide forward for the next sub-session

# --- SCENARIO PARAMETERS ---
VOTING_MODE = 'soft'  
SCENARIO = 'M5'       

# Define what classes the model is allowed to see during training
SCENARIO_TRAIN_CLASSES = {
    'M2': [0, 2],          # B, T
    'M3': [0, 1, 2],       # B, P, T
    'M4': [0, 1, 2, 4],    # B, P, T, F_T
    'M5': [0, 1, 2, 3, 4], # B, P, T, F_P, F_T
    'M6': [0, 1]
}

# ---------------------------------------------------------
# NEW: HIERARCHICAL CNN-RNN ARCHITECTURE
# ---------------------------------------------------------
class HierarchicalModel(nn.Module):
    def __init__(self, feature_dim=3, cnn_hidden=64, rnn_hidden=64, num_classes=5):
        super(HierarchicalModel, self).__init__()
        
        # 1. The Window Encoder (Time-Distributed CNN)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=feature_dim, out_channels=cnn_hidden, kernel_size=3, padding=1),
            nn.Dropout(0.2),
            nn.Conv1d(cnn_hidden, cnn_hidden * 2, kernel_size=3, padding=2, dilation=2),
            nn.Dropout(0.2),
            nn.AdaptiveMaxPool1d(1)
        )
        
        # 2. The Session Reader (Bidirectional GRU)
        self.rnn = nn.GRU(
            input_size=cnn_hidden * 2, 
            hidden_size=rnn_hidden, 
            batch_first=True, 
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        
        # 3. Final Decision Head
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x, lengths):
        batch_size, max_windows, win_length, feat_dim = x.shape
        x = x.view(batch_size * max_windows, win_length, feat_dim)
        x = x.permute(0, 2, 1) 
        
        cnn_out = self.cnn(x)                 
        cnn_out = cnn_out.squeeze(-1)         
        
        rnn_in = cnn_out.view(batch_size, max_windows, -1) 
        packed_in = nn.utils.rnn.pack_padded_sequence(rnn_in, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        packed_out, hidden = self.rnn(packed_in)
        final_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        
        final_hidden = self.dropout(final_hidden)
        out = self.fc(final_hidden)
        
        return out

# ---------------------------------------------------------
# NEW: DYNAMIC SUBSESSION DATASET & COLLATOR
# ---------------------------------------------------------
class SubSessionDataset(Dataset):
    def __init__(self, X_list, Y_list, SessID_list):
        self.X = []
        for x in X_list:
            # The Nuclear Option: Destroy the object wrapper and rebuild as pure float32
            clean_x = np.array(x.tolist(), dtype=np.float32)
            self.X.append(torch.tensor(clean_x, dtype=torch.float32))
            
        self.Y = torch.tensor(Y_list, dtype=torch.long)
        self.SessID = SessID_list
        
    def __len__(self):
        return len(self.Y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.SessID[idx]
    
def subsession_collate_fn(batch):
    X = [item[0] for item in batch]
    Y = torch.tensor([item[1] for item in batch], dtype=torch.long)
    SessID = [item[2] for item in batch]
    lengths = torch.tensor([len(x) for x in X], dtype=torch.long)
    
    X_padded = nn.utils.rnn.pad_sequence(X, batch_first=True) 
    
    return X_padded, Y, lengths, SessID

# ---------------------------------------------------------
# 1. MODIFIED CHUNKING GENERATOR
# ---------------------------------------------------------
# We use this to track which sub-sessions belong to which parent session
global_session_counter = 0 

def make_subsessions(filepath, X, Y, ID, SESS_ID, user_id, win_length, stride, seq_length, seq_stride):
    global global_session_counter
    
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
        if len(keys) < win_length: continue 
        
        # Step 1: Generate all raw 200-key windows for this session
        session_windows = []
        for i in range(1, len(keys) - win_length, stride):
            window = []
            for j in range(i, i + win_length):
                dt = keys[j]["timestamp"] - keys[j - 1]["timestamp"]
                dt = np.clip(dt, 1, 5000)
                is_space = 1 if (keys[j - 1]["key"] == " ") else 0
                is_backspace = 1 if (keys[j]["key"] == "Backspace") else 0
                
                window.append([np.log(dt), is_space, is_backspace])
                
            session_windows.append(window)
            
        # Step 2: Chunk the windows into overlapping Sub-Sessions
        if len(session_windows) > 0:
            current_parent_session_id = global_session_counter
            global_session_counter += 1
            
            # If the session is shorter than SEQ_LENGTH, just append the whole thing
            if len(session_windows) <= seq_length:
                X.append(np.array(session_windows, dtype=np.float32)) # <-- FORCE FLOAT32
                Y.append(label)
                ID.append(user_id)
                SESS_ID.append(current_parent_session_id)
            else:
                # Slide over the windows and create chunks
                for w in range(0, len(session_windows) - seq_length + 1, seq_stride):
                    sub_session = session_windows[w : w + seq_length]
                    X.append(np.array(sub_session, dtype=np.float32))
                    Y.append(label)
                    ID.append(user_id)
                    SESS_ID.append(current_parent_session_id)

# ---------------------------------------------------------
# 2. DATA EXTRACTION
# ---------------------------------------------------------
X, Y, ID, SESS_ID = [], [], [], []

folder_path = "../../dataset/Attack4"
user_folders = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
user2id = {user: i for i, user in enumerate(user_folders)}

print("Generating sub-session chunks...")
for user_folder in user_folders:
    user_id = user2id[user_folder]
    user_path = os.path.join(folder_path, user_folder)
    
    for root, _, files in os.walk(user_path):
        for filename in sorted(files):
            if filename.endswith(".json"):
                filepath = os.path.join(root, filename)
                make_subsessions(filepath, X, Y, ID, SESS_ID, user_id, WIN_LENGTH, STRIDE, SEQ_LENGTH, SEQ_STRIDE)

X = np.array(X, dtype=object) 
Y = np.array(Y)
ID = np.array(ID)
SESS_ID = np.array(SESS_ID)

print(f"Dataset Loaded -> Total Sub-Session Samples: {len(X)}")

# ---------------------------------------------------------
# 3. TRAINING LOOP
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_splits = 3
gkf = GroupKFold(n_splits=n_splits)

fold_accuracies = []
all_true_labels = []
all_pred_labels = []

allowed_train_classes = SCENARIO_TRAIN_CLASSES[SCENARIO]
print(f"\nRunning Scenario {SCENARIO}: Training on classes {allowed_train_classes}, Testing on ALL 5 classes.")

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, Y, groups=ID)):
    print(f"\n--- Fold {fold + 1}/{n_splits} ---")
    
    X_train_fold = [X[i].copy() for i in train_idx]
    X_test_fold = [X[i].copy() for i in test_idx]
    
    Y_train_fold = Y[train_idx]
    Y_test_fold = Y[test_idx]
    SESS_ID_test_fold = SESS_ID[test_idx]

    # --- THE SCENARIO FILTER ---
    train_mask = np.isin(Y_train_fold, allowed_train_classes)
    X_train_fold = [X_train_fold[i] for i in range(len(X_train_fold)) if train_mask[i]]
    Y_train_fold = Y_train_fold[train_mask]

    # --- Robust Scaling ---
    scaler = RobustScaler()
    train_dts = np.concatenate([sess[:, :, 0].reshape(-1) for sess in X_train_fold]).reshape(-1, 1)
    scaler.fit(train_dts)
    
    for i in range(len(X_train_fold)):
        orig_shape = X_train_fold[i][:, :, 0].shape
        X_train_fold[i][:, :, 0] = scaler.transform(X_train_fold[i][:, :, 0].reshape(-1, 1)).reshape(orig_shape)
        
    for i in range(len(X_test_fold)):
        orig_shape = X_test_fold[i][:, :, 0].shape
        X_test_fold[i][:, :, 0] = scaler.transform(X_test_fold[i][:, :, 0].reshape(-1, 1)).reshape(orig_shape)

    # Dataloaders
    train_ds = SubSessionDataset(X_train_fold, Y_train_fold, [0]*len(Y_train_fold)) # We don't need SESS_ID for training
    test_ds = SubSessionDataset(X_test_fold, Y_test_fold, SESS_ID_test_fold)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=subsession_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=subsession_collate_fn)

    # Initialize Hierarchical Model
    model = HierarchicalModel(feature_dim=3, num_classes=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=70)

    # Training
    num_epochs = 70
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for xb, yb, lengths, _ in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            
            loss = criterion(model(xb, lengths), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            
        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Evaluation (Aggregating Sub-Sessions back to Parent Session)
    model.eval()
    parent_session_results = {}
    
    with torch.no_grad():
        for xb, yb, lengths, sess_ids in test_loader:
            xb = xb.to(device)
            out = model(xb, lengths)
            probs = F.softmax(out, dim=1).cpu().numpy()
            labels = yb.numpy()
            
            for j in range(len(probs)):
                sid = sess_ids[j]
                true_label = labels[j]
                
                if sid not in parent_session_results:
                    parent_session_results[sid] = {'probs': [], 'true_label': true_label}
                    
                parent_session_results[sid]['probs'].append(probs[j])

    # Recombine the chunks to calculate the final fold accuracy
    fold_preds = []
    fold_labels = []
    
    for sid, data in parent_session_results.items():
        if VOTING_MODE == 'soft':
            mean_probs = np.mean(data['probs'], axis=0)
            final_pred = np.argmax(mean_probs)
        else:
            discrete_preds = [np.argmax(p) for p in data['probs']]
            final_pred = Counter(discrete_preds).most_common(1)[0][0]
            
        fold_preds.append(final_pred)
        fold_labels.append(data['true_label'])
        
        all_pred_labels.append(final_pred)
        all_true_labels.append(data['true_label'])

    fold_acc = np.mean(np.array(fold_preds) == np.array(fold_labels))
    fold_accuracies.append(fold_acc)
    print(f"Fold {fold+1} Parent Session Accuracy: {fold_acc:.4f}")

# --- FINAL SUMMARY ---
print("-" * 30)
print(f"Scenario {SCENARIO} Average CV Accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
print("-" * 30)

# Plot Final Integrated Confusion Matrix
cm = confusion_matrix(all_true_labels, all_pred_labels, labels=[0, 1, 2, 3, 4])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["B", "P", "T", "F_P", "F_T"])
disp.plot(cmap=plt.cm.Purples)
plt.title(f"{SCENARIO} Confusion Matrix (Sub-Session Chunking)")
plt.show()