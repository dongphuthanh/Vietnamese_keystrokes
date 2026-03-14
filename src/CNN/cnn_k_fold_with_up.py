import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from cnn_gen_data import make_windows_with_up
from cnn_gen_data import TemporalCNN

WIN_LENGTH = 200
STRIDE = 50

X=[] #Shape: (num_windows, window_length, feature)
Y=[] #label, Shape (num_windows)
ID=[]

"""
Generate windows from the dataset
"""
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
                # Make windows for this file, passing the correct user_id
                make_windows_with_up(filepath, X, Y, ID, user_id, WIN_LENGTH, STRIDE)



X = np.array(X)  # shape: (num_windows, window_size, feature_dim)
print("X: ", X.shape)
Y = np.array(Y)  # shape: (num_windows,)
print("Y: ",Y.shape)
ID = np.array(ID) # shape: (num_windows,)
print("ID: ", ID.shape)

    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_splits = 5
gkf = GroupKFold(n_splits=n_splits)
fold_accuracies = []

# Global results to build one final confusion matrix
all_true_labels = []
all_pred_labels = []

print(f"Starting {n_splits}-Fold Group Cross-Validation...\n")



for fold, (train_idx, test_idx) in enumerate(gkf.split(X, Y, groups=ID)):
    print(f"--- Fold {fold + 1}/{n_splits} ---")
    
    # Split
    X_train_fold, X_test_fold = X[train_idx].copy(), X[test_idx].copy()
    Y_train_fold, Y_test_fold = Y[train_idx], Y[test_idx]
    test_ids_fold = ID[test_idx]

    # --- Robust Scaling for Latency (Idx 0) and Dwell Time (Idx 1) ---
    scaler_dt = RobustScaler()
    scaler_dwell = RobustScaler()
    
    # Scale Latency (Feature 0)
    train_dt = X_train_fold[:, :, 0].reshape(-1, 1)
    test_dt = X_test_fold[:, :, 0].reshape(-1, 1)
    X_train_fold[:, :, 0] = scaler_dt.fit_transform(train_dt).reshape(X_train_fold.shape[0], X_train_fold.shape[1])
    X_test_fold[:, :, 0] = scaler_dt.transform(test_dt).reshape(X_test_fold.shape[0], X_test_fold.shape[1])

    # Scale Dwell Time (Feature 1)
    train_dwell = X_train_fold[:, :, 1].reshape(-1, 1)
    test_dwell = X_test_fold[:, :, 1].reshape(-1, 1)
    X_train_fold[:, :, 1] = scaler_dwell.fit_transform(train_dwell).reshape(X_train_fold.shape[0], X_train_fold.shape[1])
    X_test_fold[:, :, 1] = scaler_dwell.transform(test_dwell).reshape(X_test_fold.shape[0], X_test_fold.shape[1])

    # Dataloaders
    train_ds = TensorDataset(torch.tensor(X_train_fold, dtype=torch.float32), torch.tensor(Y_train_fold, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test_fold, dtype=torch.float32), torch.tensor(Y_test_fold, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # Initialize Model, Optimizer, Scheduler
    model = TemporalCNN(feature_dim=X.shape[2], num_classes=len(np.unique(Y))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Training Loop
    num_epochs = 100 # Reduced slightly for time, adjust as needed
    for epoch in range(num_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Evaluation for this Fold
    model.eval()
    session_results = {}
    current_idx = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds = torch.argmax(out, dim=1).cpu().numpy()
            labels = yb.cpu().numpy()
            
            batch_ids = test_ids_fold[current_idx : current_idx + len(xb)]
            current_idx += len(xb)

            for j in range(len(preds)):
                key = (batch_ids[j], labels[j])
                if key not in session_results: session_results[key] = []
                session_results[key].append(preds[j])

    # Aggregate session votes for this fold
    fold_preds = []
    fold_labels = []
    for (uid, true_lab), preds_list in session_results.items():
        majority_vote = Counter(preds_list).most_common(1)[0][0]
        fold_preds.append(majority_vote)
        fold_labels.append(true_lab)
        
        # Save for global confusion matrix
        all_pred_labels.append(majority_vote)
        all_true_labels.append(true_lab)

    fold_acc = np.mean(np.array(fold_preds) == np.array(fold_labels))
    fold_accuracies.append(fold_acc)
    print(f"Fold {fold+1} Session Accuracy: {fold_acc:.4f}\n")

# --- FINAL SUMMARY ---
print("-" * 30)
print(f"Average CV Accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
print("-" * 30)

# Plot Final Integrated Confusion Matrix
cm = confusion_matrix(all_true_labels, all_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["bonafide", "paraphrase", "transcribe"])
disp.plot(cmap=plt.cm.Purples)
plt.title(f"Aggregated {n_splits}-Fold Confusion Matrix")
plt.show()