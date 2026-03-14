import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler

# PyTorch Geometric imports
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, global_mean_pool
from torch.utils.data import DataLoader, Dataset

# Import your data generator
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

class FastGraphClassifier(nn.Module):
    def __init__(self, in_channels=256, hidden_channels=64, num_classes=3):
        super().__init__()
        mlp1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.gin1 = GINConv(mlp1)
        
        mlp2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.gin2 = GINConv(mlp2)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.LayerNorm(64), 
            nn.ReLU(),
            nn.Dropout(0.5), # High dropout to prevent memorization!
            nn.Linear(64, num_classes)
        )

    def forward(self, x, edge_index, batch):
        h = self.gin1(x, edge_index)
        h = F.relu(h)
        h = self.gin2(h, edge_index)
        h = F.relu(h)
        
        session_embedding = global_mean_pool(h, batch) 
        return self.classifier(session_embedding)

def build_macro_graph(window_features, similarity_threshold=0.8):
    num_nodes = window_features.size(0)
    edge_index = []

    for i in range(num_nodes - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])

    sim_matrix = F.cosine_similarity(
        window_features.unsqueeze(1), 
        window_features.unsqueeze(0), 
        dim=-1
    )

    for i in range(num_nodes):
        for j in range(i + 2, num_nodes): 
            if sim_matrix[i, j] > similarity_threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])

    if len(edge_index) == 0:
        edge_index = [[0], [0]]

    return torch.tensor(edge_index, dtype=torch.long, device=window_features.device).t().contiguous()

class EndToEndBatchedModel(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.cnn = TemporalCNN(feature_dim=feature_dim, num_classes=num_classes)
        self.gnn = FastGraphClassifier(in_channels=256, hidden_channels=64, num_classes=num_classes)
        
    def forward(self, list_of_sessions, similarity_threshold=0.8):
        device = next(self.parameters()).device
        data_list = []
        
        for session_windows in list_of_sessions:
            session_windows = session_windows.to(device)
            window_features = self.cnn(session_windows, return_features=True)
            edge_index = build_macro_graph(window_features, similarity_threshold)
            
            data = Data(x=window_features, edge_index=edge_index)
            data_list.append(data)
            
        mega_graph = Batch.from_data_list(data_list).to(device)
        return self.gnn(mega_graph.x, mega_graph.edge_index, mega_graph.batch)

# ---------------------------------------------------------
# 2. DATA PREPARATION & CUSTOM LOADER
# ---------------------------------------------------------

def group_into_sessions(X_data, Y_data, ID_data):
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
    """Prevents PyTorch from trying to stack variable-length sessions."""
    sessions = [torch.tensor(item[0], dtype=torch.float32) for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return sessions, labels

X, Y, ID = [], [], []

folder_path = "../dataset/viet_preprocessed"
user_folders = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
user2id = {user: i for i, user in enumerate(user_folders)}

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

print(f"Total Windows Extracted: {X.shape}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

all_true_labels = []
all_pred_labels = []
fold_accuracies = []

# ---------------------------------------------------------
# 3. END-TO-END TRAINING LOOP
# ---------------------------------------------------------

print(f"\nStarting {n_splits}-Fold End-to-End Batched Cross-Validation...\n")

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, Y, groups=ID)):
    print(f"--- Fold {fold + 1}/{n_splits} ---")
    
    X_train_fold, X_test_fold = X[train_idx].copy(), X[test_idx].copy()
    Y_train_fold, Y_test_fold = Y[train_idx], Y[test_idx]
    train_ids_fold, test_ids_fold = ID[train_idx], ID[test_idx]

    scaler = RobustScaler()
    train_dt = X_train_fold[:, :, 0].reshape(-1, 1)
    test_dt = X_test_fold[:, :, 0].reshape(-1, 1)
    
    X_train_fold[:, :, 0] = scaler.fit_transform(train_dt).reshape(X_train_fold.shape[0], X_train_fold.shape[1])
    X_test_fold[:, :, 0] = scaler.transform(test_dt).reshape(X_test_fold.shape[0], X_test_fold.shape[1])

    train_sessions = group_into_sessions(X_train_fold, Y_train_fold, train_ids_fold)
    test_sessions = group_into_sessions(X_test_fold, Y_test_fold, test_ids_fold)

    # Use standard DataLoader with our custom_collate function
    train_loader = DataLoader(train_sessions, batch_size=32, shuffle=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_sessions, batch_size=32, shuffle=False, collate_fn=custom_collate)

    # 1. Initialize the End-to-End Master Model
    model = EndToEndBatchedModel(feature_dim=X.shape[2], num_classes=len(np.unique(Y))).to(device)
    
    # 2. Optimizer & CRITERION
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss() # <--- HERE IT IS!
    
    num_epochs = 50
    
    print("Training End-to-End...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for sessions_list, labels in train_loader:
            labels = labels.to(device)
            
            optimizer.zero_grad()
            out = model(sessions_list) 
            
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch + 1:02d}/{num_epochs} | Average Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    fold_preds = []
    fold_labels = []
    
    with torch.no_grad():
        for sessions_list, labels in test_loader:
            labels = labels.to(device)
            out = model(sessions_list)
            preds = torch.argmax(out, dim=1).cpu().numpy()
            
            fold_preds.extend(preds)
            fold_labels.extend(labels.cpu().numpy())
            
            all_pred_labels.extend(preds)
            all_true_labels.extend(labels.cpu().numpy())

    fold_acc = np.mean(np.array(fold_preds) == np.array(fold_labels))
    fold_accuracies.append(fold_acc)
    print(f"Fold {fold+1} Session Accuracy: {fold_acc:.4f}\n")

# --- FINAL SUMMARY ---
print("-" * 30)
print(f"Average CV Accuracy (End-to-End GIN): {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
print("-" * 30)

cm = confusion_matrix(all_true_labels, all_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["bonafide", "paraphrase", "transcribe"])
disp.plot(cmap=plt.cm.Purples)
plt.title(f"Aggregated {n_splits}-Fold Confusion Matrix")
plt.show()