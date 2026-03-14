import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from gnn_gen_data import gin_make_windows
import torch.nn.functional as F
import math

########################################
# 1. DATASET (Added UID tracking)
########################################

class KeystrokeDataset(Dataset):
    def __init__(self, X, Y, ID):
        self.X = X
        self.Y = Y
        self.ID = ID

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        window = self.X[idx]
        keys = np.array([k for k,t in window])
        times = np.array([t for k,t in window])

        # inter-event times
        delta_t = np.diff(times, prepend=times[0])
        # log transform (helps stability)
        delta_t = np.log(delta_t + 1)

        return {
            "keys": torch.tensor(keys, dtype=torch.long),
            "delta_t": torch.tensor(delta_t, dtype=torch.float32),
            "label": torch.tensor(self.Y[idx], dtype=torch.long),
            "uid": self.ID[idx]
        }

########################################
# 2. COLLATE (Padding & UID pass-through)
########################################

def collate_fn(batch):
    keys = [b["keys"] for b in batch]
    delta_t = [b["delta_t"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])
    uids = [b["uid"] for b in batch]

    keys = pad_sequence(keys, batch_first=True)
    delta_t = pad_sequence(delta_t, batch_first=True)

    return keys, delta_t, labels, uids

########################################
# 3. MODEL (Unchanged)
########################################

class PointProcessClassifier(nn.Module):
    def __init__(self, num_keys, hidden_dim, num_classes):
        super().__init__()
        
        # 1. Input Embeddings
        self.key_embedding = nn.Embedding(num_keys, 16)
        self.time_projection = nn.Linear(1, 16)
        # 2. Bidirectional GRU
        # Setting bidirectional=True doubles the hidden output size automatically.
        # So if hidden_dim=64, the GRU outputs 128 features per time step.
        self.gru = nn.GRU(
            input_size=32,  # 32 key embedding + 1 delta_t
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.3,
            batch_first=True,
            bidirectional=True
        )
        
        # 3. Attention Mechanism
        # Learns which keystrokes/pauses in the sequence are the most important
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 4. Final Classifier
        # The input is (hidden_dim * 2 from Attention) + (hidden_dim * 2 from Max Pooling) = hidden_dim * 4
        classifier_in_dim = hidden_dim * 4
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, keys, delta_t):
        # --- A. Prepare Inputs ---
        key_emb = self.key_embedding(keys)
        delta_t = delta_t.unsqueeze(-1)
        time_emb = torch.relu(self.time_projection(delta_t))
        
        # Concatenate [32] and [16] -> [48]
        x = torch.cat([key_emb, time_emb], dim=-1)
        
        # --- B. Sequence Processing (BiGRU) ---
        # h shape: [Batch, Seq_Len, Hidden * 2]
        h, _ = self.gru(x)
        
        # --- C. Attention Layer ---
        # 1. Calculate raw attention scores for each time step
        # attn_scores shape: [Batch, Seq_Len, 1]
        attn_scores = self.attention(h)
        
        # 2. Convert scores to probabilities (weights) using Softmax
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # 3. Multiply the GRU outputs by the weights and sum them up (Context Vector)
        # h_attended shape: [Batch, Hidden * 2]
        h_attended = torch.sum(h * attn_weights, dim=1)
        
        # --- D. Global Max Pooling ---
        # Capture the strongest signals (e.g., the longest pause or fastest burst) anywhere in the sequence
        # h_max shape: [Batch, Hidden * 2]
        h_max, _ = torch.max(h, dim=1)
        
        # --- E. Fusion & Classification ---
        # Combine the "Attended" summary with the "Extreme" summary
        # h_final shape: [Batch, Hidden * 4]
        h_final = torch.cat([h_attended, h_max], dim=1)
        
        logits = self.classifier(h_final)
        
        return logits

########################################
# 4. CROSS-VALIDATION PIPELINE
########################################
class TruePointProcessClassifier(nn.Module):
    def __init__(self, num_keys, hidden_dim = 64, num_classes = 3):
        super().__init__()
        self.key_embedding = nn.Embedding(num_keys, 16)
        self.time_proj = nn.Linear(1, 32)
        
        self.gru = nn.GRU(
            input_size=48, 
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # --- NEW: The Point Process Intensity Predictor ---
        # Predicts the rate parameter (lambda) for the next inter-event time
        self.lambda_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus() # Softplus ensures lambda is always strictly positive (> 0)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, keys, delta_t):
        key_emb = self.key_embedding(keys)
        time_emb = torch.relu(self.time_proj(delta_t.unsqueeze(-1)))
        
        x = torch.cat([key_emb, time_emb], dim=-1)
        
        # h shape: [Batch, Seq_Len, Hidden]
        h, _ = self.gru(x)
        
        # --- 1. Point Process Output ---
        # Predict lambda for every step in the sequence
        # lambdas shape: [Batch, Seq_Len]
        lambdas = self.lambda_predictor(h).squeeze(-1)
        
        # --- 2. Classification Output ---
        # Still use the final state for the overall sequence classification
        h_last = h[:, -1]
        logits = self.classifier(h_last)
        
        return logits, lambdas
    


def run_cv_training_tpp(X, Y, ID, n_splits=5, epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running {n_splits}-Fold CV for True Point Process on {device}...")

    X_np = np.array(X, dtype=object)
    Y_np = np.array(Y)
    ID_np = np.array(ID)

    gkf = GroupKFold(n_splits=n_splits)
    
    num_keys = 3 # Adjust if you have more key categories
    num_classes = len(set(Y))
    
    all_voted_preds = []
    all_voted_labels = []
    fold_window_accs = []
    fold_session_accs = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_np, Y_np, groups=ID_np)):
        print(f"\n{'='*15} FOLD {fold + 1}/{n_splits} {'='*15}")
        
        train_dataset = KeystrokeDataset(X_np[train_idx], Y_np[train_idx], ID_np[train_idx])
        test_dataset = KeystrokeDataset(X_np[test_idx], Y_np[test_idx], ID_np[test_idx])

        # Note: drop_last=True is added here to prevent the BatchNorm crash!
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

        model = TruePointProcessClassifier(num_keys=num_keys).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4) # Added weight decay to fight overfitting
        criterion_class = nn.CrossEntropyLoss()

        # --- Train Loop ---
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            
            for keys, delta_t, labels, _ in train_loader:
                keys, delta_t, labels = keys.to(device), delta_t.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                # Model now returns two outputs
                logits, lambdas = model(keys, delta_t)
                
                # 1. Classification Loss
                loss_class = criterion_class(logits, labels)
                
                # 2. Point Process Setup
                # Shift delta_t left to get the "target" time for the next key
                next_delta_t = torch.roll(delta_t, shifts=-1, dims=1)
                next_delta_t[:, -1] = 0.0 # Clear the last column
                
                # Create a mask so padded zeros don't affect the likelihood loss
                # Assuming pad_sequence uses 0.0 for padding delta_t
                mask = (delta_t != 0.0).float()
                mask[:, -1] = 0.0 # We can't predict the next key for the final step
                
                # 3. Negative Log-Likelihood (NLL) for Exponential Distribution
                # Loss = -log(lambda) + lambda * actual_time
                nll_loss = -torch.log(lambdas + 1e-6) + (lambdas * next_delta_t)
                nll_loss = (nll_loss * mask).sum() / (mask.sum() + 1e-6) # Average over valid steps
                
                # 4. Total Multi-Task Loss (alpha parameter balances the two tasks)
                alpha = 0.5
                total_loss = loss_class + (alpha * nll_loss)
                
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
                
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:02d} | Total Loss: {train_loss/len(train_loader):.4f}")

        # --- Evaluation Loop ---
        model.eval()
        fold_preds, fold_labels, fold_uids = [], [], []
        
        with torch.no_grad():
            for keys, delta_t, labels, uids in test_loader:
                keys, delta_t = keys.to(device), delta_t.to(device)
                
                # We only need the classification logits for evaluation
                logits, _ = model(keys, delta_t) 
                preds = logits.argmax(dim=1).cpu().numpy()
                
                fold_preds.extend(preds)
                fold_labels.extend(labels.numpy())
                fold_uids.extend(uids)
                
        # 1. Window-Level Accuracy
        window_acc = np.mean(np.array(fold_preds) == np.array(fold_labels))
        fold_window_accs.append(window_acc)

        # 2. Session-Level (Majority Voting) Accuracy
        session_results = {}
        for p, l, u in zip(fold_preds, fold_labels, fold_uids):
            key = (u, l)
            if key not in session_results:
                session_results[key] = []
            session_results[key].append(p)

        voted_preds, voted_labels = [], []
        for (u, true_lab), p_list in session_results.items():
            vote = Counter(p_list).most_common(1)[0][0]
            voted_preds.append(vote)
            voted_labels.append(true_lab)
            
            all_voted_preds.append(vote)
            all_voted_labels.append(true_lab)

        session_acc = np.mean(np.array(voted_preds) == np.array(voted_labels))
        fold_session_accs.append(session_acc)

        print(f"Fold {fold+1} Window Acc: {window_acc:.4f} | Voted Session Acc: {session_acc:.4f}")

    # --- Global Results & Confusion Matrix ---
    print("\n" + "="*45)
    print(f"FINAL CV WINDOW ACC:  {np.mean(fold_window_accs):.4f}")
    print(f"FINAL CV SESSION ACC: {np.mean(fold_session_accs):.4f}")
    print("="*45)

    present_classes = np.unique(all_voted_labels)
    class_names = {0: "Bonafide", 1: "Paraphrase", 2: "Transcribe"}
    display_names = [class_names[i] for i in present_classes]

    cm = confusion_matrix(all_voted_labels, all_voted_preds, labels=present_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_names)
    
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("TPP Voted Session-Level Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return model
   
def run_cv_training(X, Y, ID, n_splits=5, epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running {n_splits}-Fold CV on {device}...")

    X_np = np.array(X, dtype=object)
    Y_np = np.array(Y)
    ID_np = np.array(ID)

    gkf = GroupKFold(n_splits=n_splits)
    
    num_keys = 3 # Adjust if your dataset has more key types
    num_classes = len(set(Y))
    
    all_voted_preds = []
    all_voted_labels = []
    fold_window_accs = []
    fold_session_accs = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_np, Y_np, groups=ID_np)):
        print(f"\n{'='*15} FOLD {fold + 1}/{n_splits} {'='*15}")
        
        # Split data
        train_dataset = KeystrokeDataset(X_np[train_idx], Y_np[train_idx], ID_np[train_idx])
        test_dataset = KeystrokeDataset(X_np[test_idx], Y_np[test_idx], ID_np[test_idx])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

        # Initialize fresh model for this fold
        model = PointProcessClassifier(num_keys=num_keys, hidden_dim=64, num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # --- Train Loop ---
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for keys, delta_t, labels, _ in train_loader:
                keys, delta_t, labels = keys.to(device), delta_t.to(device), labels.to(device)
                
                optimizer.zero_grad()
                logits = model(keys, delta_t)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss/len(train_loader):.4f}")

        # --- Evaluation Loop ---
        model.eval()
        fold_preds, fold_labels, fold_uids = [], [], []
        
        with torch.no_grad():
            for keys, delta_t, labels, uids in test_loader:
                keys, delta_t = keys.to(device), delta_t.to(device)
                logits = model(keys, delta_t)
                preds = logits.argmax(dim=1).cpu().numpy()
                
                fold_preds.extend(preds)
                fold_labels.extend(labels.numpy())
                fold_uids.extend(uids)
                
        # 1. Window-Level Accuracy
        window_acc = np.mean(np.array(fold_preds) == np.array(fold_labels))
        fold_window_accs.append(window_acc)

        # 2. Session-Level (Majority Voting) Accuracy
        # Group by (UID, True Label) in case a user has multiple tasks
        session_results = {}
        for p, l, u in zip(fold_preds, fold_labels, fold_uids):
            key = (u, l)
            if key not in session_results:
                session_results[key] = []
            session_results[key].append(p)

        voted_preds, voted_labels = [], []
        for (u, true_lab), p_list in session_results.items():
            # Get the most common prediction for this user/task combination
            vote = Counter(p_list).most_common(1)[0][0]
            voted_preds.append(vote)
            voted_labels.append(true_lab)
            
            # Save for the global confusion matrix
            all_voted_preds.append(vote)
            all_voted_labels.append(true_lab)

        session_acc = np.mean(np.array(voted_preds) == np.array(voted_labels))
        fold_session_accs.append(session_acc)

        print(f"Fold {fold+1} Window Acc: {window_acc:.4f} | Voted Session Acc: {session_acc:.4f}")

    # --- Global Results & Confusion Matrix ---
    print("\n" + "="*45)
    print(f"FINAL CV WINDOW ACC:  {np.mean(fold_window_accs):.4f}")
    print(f"FINAL CV SESSION ACC: {np.mean(fold_session_accs):.4f}")
    print("="*45)

    # Confusion Matrix
    present_classes = np.unique(all_voted_labels)
    class_names = {0: "Bonafide", 1: "Paraphrase", 2: "Transcribe"}
    display_names = [class_names[i] for i in present_classes]

    cm = confusion_matrix(all_voted_labels, all_voted_preds, labels=present_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_names)
    
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Voted Session-Level Confusion Matrix (All Folds)")
    plt.tight_layout()
    plt.show()

    return model
########################################
# 5. RUN
########################################
X, Y, ID = [], [], []
folder_path = "../dataset/viet_preprocessed"
user_folders = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
user2id = {user: i for i, user in enumerate(user_folders)}

for user_folder in user_folders:
    user_id = user2id[user_folder]
    user_path = os.path.join(folder_path, user_folder)
    for root, _, files in os.walk(user_path):
        for filename in files:
            if filename.endswith(".json"):
                filepath = os.path.join(root, filename)
                gin_make_windows(filepath, X, Y, ID, user_id, 300, 50)

model = run_cv_training_tpp(X, Y, ID, n_splits=5, epochs=100)