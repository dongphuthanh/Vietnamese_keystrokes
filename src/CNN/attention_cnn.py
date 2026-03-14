import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler

# Import only your window generator, we will define the CNN here for safety
from cnn_gen_data import make_windows

WIN_LENGTH = 200
STRIDE = 50

# ---------------------------------------------------------
# 1. MODEL DEFINITIONS
# ---------------------------------------------------------

class TemporalCNN(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=feature_dim, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=4, dilation=4)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.3)
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x, return_features=False):
        x = x.permute(0, 2, 1)  
        x = self.conv1(x)
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

class SessionAttentionModel(nn.Module):
    def __init__(self, feature_dim, num_classes, cnn_hidden_dim=256):
        super().__init__()
        # 1. The Pre-trained Backbone (Contains the perfect FC layer!)
        self.cnn = TemporalCNN(feature_dim=feature_dim, num_classes=num_classes)
        
        # 2. The Attention Judge (Learns from scratch)
        self.attention_judge = nn.Sequential(
            nn.Linear(cnn_hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # WE DELETED self.classifier! 

    def forward(self, list_of_sessions):
        device = next(self.parameters()).device
        session_logits = []
        
        for session_windows in list_of_sessions:
            session_windows = session_windows.to(device)
            
            # A. Extract perfect 256D features
            features = self.cnn(session_windows, return_features=True) 
            
            # B. Judge each window
            raw_scores = self.attention_judge(features) 
            attention_weights = torch.softmax(raw_scores, dim=0) 
            
            # C. Multiply features by weights and sum them up
            weighted_features = features * attention_weights
            session_vector = torch.sum(weighted_features, dim=0, keepdim=True) 
            
            # D. THE FIX: Use the perfectly trained CNN FC layer to make the final guess!
            session_logits.append(self.cnn.fc(session_vector))
            
        return torch.cat(session_logits, dim=0)

# ---------------------------------------------------------
# 2. DATA PREPARATION HELPERS
# ---------------------------------------------------------

def group_into_sessions(X_data, Y_data, ID_data):
    """Reconstructs the chronological sessions from the shuffled windows."""
    sessions = {}
    for i in range(len(X_data)):
        key = (ID_data[i], Y_data[i])
        if key not in sessions:
            sessions[key] = []
        sessions[key].append(X_data[i])
    
    session_list = []
    for k, v in sessions.items():
        session_list.append((np.array(v), k[1]))
    return session_list

def custom_collate(batch):
    """Tells PyTorch to return a Python list of variable-length sessions."""
    sessions = [torch.tensor(item[0], dtype=torch.float32) for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return sessions, labels

# ---------------------------------------------------------
# 3. DATA LOADING
# ---------------------------------------------------------

X, Y, ID = [], [], []

folder_path = "../dataset/viet_preprocessed"
user_folders = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
user2id = {user: i for i, user in enumerate(user_folders)}

print("Extracting windows...")
for user_folder in user_folders:
    user_id = user2id[user_folder]
    user_path = os.path.join(folder_path, user_folder)
    
    for root, _, files in os.walk(user_path):
        for filename in sorted(files):
            if filename.endswith(".json"):
                filepath = os.path.join(root, filename)
                make_windows(filepath, X, Y, ID, user_id, WIN_LENGTH, STRIDE)

X = np.array(X)
Y = np.array(Y)
ID = np.array(ID)

print(f"X: {X.shape} | Y: {Y.shape} | ID: {ID.shape}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

all_true_labels = []
all_pred_labels = []
fold_accuracies = []

# ---------------------------------------------------------
# 4. TRAINING & EVALUATION LOOP
# ---------------------------------------------------------

print(f"\nStarting {n_splits}-Fold Cross-Validation with Learnable Attention Pooling...\n")

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, Y, groups=ID)):
    print(f"--- Fold {fold + 1}/{n_splits} ---")
    
    X_train_fold, X_test_fold = X[train_idx].copy(), X[test_idx].copy()
    Y_train_fold, Y_test_fold = Y[train_idx], Y[test_idx]
    
    # Ensure we get both train and test IDs for grouping
    train_ids_fold = ID[train_idx]
    test_ids_fold = ID[test_idx]

    # --- Robust Scaling ---
    scaler = RobustScaler()
    train_dt = X_train_fold[:, :, 0].reshape(-1, 1)
    test_dt = X_test_fold[:, :, 0].reshape(-1, 1)
    
    X_train_fold[:, :, 0] = scaler.fit_transform(train_dt).reshape(X_train_fold.shape[0], X_train_fold.shape[1])
    X_test_fold[:, :, 0] = scaler.transform(test_dt).reshape(X_test_fold.shape[0], X_test_fold.shape[1])

    # --- Group back into sessions ---
    train_sessions = group_into_sessions(X_train_fold, Y_train_fold, train_ids_fold)
    test_sessions = group_into_sessions(X_test_fold, Y_test_fold, test_ids_fold)

    # --- Dataloaders (using custom collate and smaller batch size) ---
    train_loader = DataLoader(train_sessions, batch_size=16, shuffle=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_sessions, batch_size=16, shuffle=False, collate_fn=custom_collate)

    # --- Initialize Model, Optimizer, Scheduler ---
    model = SessionAttentionModel(feature_dim=X.shape[2], num_classes=len(np.unique(Y))).to(device)
    try:
        # Adjust this filename to whatever your Phase 1 weights are saved as!
        model.cnn.load_state_dict(torch.load(f"cnn_weights_fold_{fold + 1}.pth"))
        print("Loaded pre-trained CNN weights successfully.")
    except FileNotFoundError:
        print("WARNING: Could not find pre-trained CNN weights. Training from scratch.")

    # 2. FREEZE THE CNN (Protect it from the random Attention Judge)
    for param in model.cnn.parameters():
        param.requires_grad = False

    # 3. ONLY TRAIN THE ATTENTION JUDGE AND CLASSIFIER
    # We pass only the un-frozen parameters to the optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # --- Session-Level Training Loop ---
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for session_list, labels in train_loader:
            labels = labels.to(device)
            optimizer.zero_grad()
            
            out = model(session_list) 
            
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        scheduler.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1:03d}/{num_epochs} | Loss: {epoch_loss/len(train_loader):.4f}")

    # --- Direct Session Evaluation ---
    model.eval()
    fold_preds = []
    fold_labels = []
    
    with torch.no_grad():
        for session_list, labels in test_loader:
            labels = labels.to(device)
            out = model(session_list)
            
            preds = torch.argmax(out, dim=1).cpu().numpy()
            
            fold_preds.extend(preds)
            fold_labels.extend(labels.cpu().numpy())
            
            all_pred_labels.extend(preds)
            all_true_labels.extend(labels.cpu().numpy())

    fold_acc = np.mean(np.array(fold_preds) == np.array(fold_labels))
    fold_accuracies.append(fold_acc)
    print(f"Fold {fold+1} Session Accuracy: {fold_acc:.4f}\n")
    
    # Save the attention model weights

# --- FINAL SUMMARY ---
print("-" * 30)
print(f"Average CV Accuracy (Attention Pooling): {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
print("-" * 30)

cm = confusion_matrix(all_true_labels, all_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["bonafide", "paraphrase", "transcribe"])
disp.plot(cmap=plt.cm.Purples)
plt.title(f"Aggregated {n_splits}-Fold Confusion Matrix (Attention Pooling)")
plt.show()