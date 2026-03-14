import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import copy
from collections import Counter
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# --- PyTorch Geometric Imports ---
from torch_geometric.loader import DataLoader as GNNDataLoader
from cnn_gen_data import make_windows, make_graph, GNNClassifier 

# --- 1. CONFIGURATION ---
WIN_LENGTH = 400
STRIDE = 100
NUM_EPOCHS = 200
BATCH_SIZE = 64
HIDDEN_DIM = 128
N_SPLITS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. DATA LOADING & WINDOW GENERATION ---
X_raw, Y_raw, ID_raw = [], [], []
folder_path = "../dataset/viet_preprocessed"
user_folders = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
user2id = {user: i for i, user in enumerate(user_folders)}

print(f"Loading data from {len(user_folders)} users...")

for user_folder in user_folders:
    user_id = user2id[user_folder]
    user_path = os.path.join(folder_path, user_folder)
    for root, _, files in os.walk(user_path):
        for filename in files:
            if filename.endswith(".json"):
                filepath = os.path.join(root, filename)
                make_windows(filepath, X_raw, Y_raw, ID_raw, user_id, WIN_LENGTH, STRIDE)

X_np = np.array(X_raw) 
Y_np = np.array(Y_raw)
ID_np = np.array(ID_raw)

print(f"Total Windows: {len(X_np)} | Class Balance: {dict(Counter(Y_np))}")

# --- 3. INITIAL GRAPH CONSTRUCTION ---
# We build them once here; we will deepcopy/regenerate in the loop to stay clean
all_graphs_base = make_graph(X_raw, Y_raw, ID_raw)

# --- 4. CROSS VALIDATION LOOP ---
gkf = GroupKFold(n_splits=N_SPLITS)
indices = np.arange(len(all_graphs_base))

fold_session_accs = []
fold_window_accs = []
all_true_labels, all_pred_labels = [], []

print(f"Starting {N_SPLITS}-Fold Group Cross-Validation on {device}...")

for fold, (train_idx, test_idx) in enumerate(gkf.split(indices, Y_np, groups=ID_np)):
    print(f"\n{'='*10} Fold {fold + 1}/{N_SPLITS} {'='*10}")
    
    # ISOLATION: Create deep copies so scaling in this fold doesn't affect others
    train_graphs = [copy.deepcopy(all_graphs_base[i]) for i in train_idx]
    test_graphs = [copy.deepcopy(all_graphs_base[i]) for i in test_idx]

    # --- Feature Scaling (Column 0: Log(dt) ONLY) ---
    scaler = RobustScaler()
    train_dts = np.concatenate([g.x[:, 0].numpy() for g in train_graphs]).reshape(-1, 1)
    scaler.fit(train_dts)
    
    for g_set in [train_graphs, test_graphs]:
        for g in g_set:
            dt_col = g.x[:, 0].reshape(-1, 1).numpy()
            g.x[:, 0] = torch.tensor(scaler.transform(dt_col).flatten(), dtype=torch.float32)

    train_loader = GNNDataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = GNNDataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Model & Training Tools
    num_classes = len(np.unique(Y_np))
    model = GNNClassifier(input_dim=X_np.shape[2], hidden_dim=HIDDEN_DIM, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # --- Training Phase ---
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            out = model(batch_data)
            loss = criterion(out, batch_data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch:02d} | Loss: {total_loss/len(train_loader):.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    # --- Evaluation Phase (Window & Session) ---
    model.eval()
    session_results = {} 
    correct_windows = 0
    total_windows = 0
    
    with torch.no_grad():
        for batch_data in test_loader:
            batch_data = batch_data.to(device)
            out = model(batch_data)
            preds = torch.argmax(out, dim=1).cpu().numpy()
            labels = batch_data.y.cpu().numpy()
            uids = batch_data.group_id.cpu().numpy()

            # Window-level tracking
            correct_windows += np.sum(preds == labels)
            total_windows += len(labels)

            # Session-level tracking
            for j in range(len(preds)):
                key = (uids[j], labels[j])
                if key not in session_results:
                    session_results[key] = []
                session_results[key].append(preds[j])

    # 1. Fold Window Accuracy
    f_window_acc = correct_windows / total_windows
    fold_window_accs.append(f_window_acc)

    # 2. Fold Session Accuracy (Majority Vote)
    f_session_preds, f_session_labels = [], []
    for (uid, true_lab), p_list in session_results.items():
        vote = Counter(p_list).most_common(1)[0][0]
        f_session_preds.append(vote)
        f_session_labels.append(true_lab)
        all_pred_labels.append(vote)
        all_true_labels.append(true_lab)

    f_session_acc = np.mean(np.array(f_session_preds) == np.array(f_session_labels))
    fold_session_accs.append(f_session_acc)

    print(f"\nFold {fold+1} Results:")
    print(f"  > Window Accuracy:  {f_window_acc:.4f}")
    print(f"  > Session Accuracy: {f_session_acc:.4f}")

# --- 5. FINAL SUMMARY & PLOTTING ---
print("\n" + "="*35)
print(f"FINAL CV WINDOW ACC:  {np.mean(fold_window_accs):.4f}")
print(f"FINAL CV SESSION ACC: {np.mean(fold_session_accs):.4f} (+/- {np.std(fold_session_accs):.4f})")
print("="*35)

# Build dynamic Confusion Matrix (Safe for missing classes)
present_classes = np.unique(all_true_labels)
all_class_names = ["bonafide", "paraphrase", "transcribe"]
display_names = [all_class_names[i] for i in present_classes]

cm = confusion_matrix(all_true_labels, all_pred_labels, labels=present_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_names)
disp.plot(cmap=plt.cm.Purples, xticks_rotation=45)
plt.title(f"GNN Aggregated Session Confusion Matrix")
plt.tight_layout()
plt.show()