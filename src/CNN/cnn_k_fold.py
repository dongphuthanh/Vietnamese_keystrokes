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
from cnn_gen_data import make_windows, make_windows_no_distinguish
from cnn_gen_data import TemporalCNN

WIN_LENGTH = 200
STRIDE = 50

# --- NEW PARAMETER ---
VOTING_MODE = 'soft'  # Change to 'hard' for majority voting, or 'soft' for probability averaging

X=[] #Shape: (num_windows, window_length, feature)
Y=[] #label, Shape (num_windows)
ID=[]

"""
Generate windows from the dataset
"""
folder_path = "../../dataset/viet_preprocessed"
user_folders = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
user2id = {user: i for i, user in enumerate(user_folders)}

for user_folder in user_folders:
    user_id = user2id[user_folder]
    user_path = os.path.join(folder_path, user_folder)
    
    for root, _, files in os.walk(user_path):
        for filename in sorted(files):
            if filename.endswith(".json"):
                filepath = os.path.join(root, filename)
                # Make windows for this file, passing the correct user_id
                make_windows(filepath, X, Y, ID, user_id, WIN_LENGTH, STRIDE)

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

print(f"Starting {n_splits}-Fold Group Cross-Validation with {VOTING_MODE.upper()} voting...\n")

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, Y, groups=ID)):
    print(f"--- Fold {fold + 1}/{n_splits} ---")
    
    # Split
    X_train_fold, X_test_fold = X[train_idx].copy(), X[test_idx].copy()
    Y_train_fold, Y_test_fold = Y[train_idx], Y[test_idx]
    test_ids_fold = ID[test_idx]

    # --- Robust Scaling (Selective for log(dt)) ---
    scaler = RobustScaler()
    train_dt = X_train_fold[:, :, 0].reshape(-1, 1)
    test_dt = X_test_fold[:, :, 0].reshape(-1, 1)
    
    X_train_fold[:, :, 0] = scaler.fit_transform(train_dt).reshape(X_train_fold.shape[0], X_train_fold.shape[1])
    X_test_fold[:, :, 0] = scaler.transform(test_dt).reshape(X_test_fold.shape[0], X_test_fold.shape[1])

    # Dataloaders
    train_ds = TensorDataset(torch.tensor(X_train_fold, dtype=torch.float32), torch.tensor(Y_train_fold, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test_fold, dtype=torch.float32), torch.tensor(Y_test_fold, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # Initialize Model, Optimizer, Scheduler
    model = TemporalCNN(feature_dim=X.shape[2], num_classes=len(np.unique(Y))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Training Loop
    num_epochs = 100 
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
            
            # --- EXTRACT PREDICTIONS BASED ON VOTING MODE ---
            if VOTING_MODE == 'soft':
                # Convert logits to probabilities: shape [batch_size, num_classes]
                store_vals = F.softmax(out, dim=1).cpu().numpy()
            else:
                # Get hard integer class labels: shape [batch_size]
                store_vals = torch.argmax(out, dim=1).cpu().numpy()
                
            labels = yb.cpu().numpy()
            
            batch_ids = test_ids_fold[current_idx : current_idx + len(xb)]
            current_idx += len(xb)

            for j in range(len(store_vals)):
                key = (batch_ids[j], labels[j])
                if key not in session_results: 
                    session_results[key] = []
                session_results[key].append(store_vals[j])

    # Aggregate session votes for this fold
    fold_preds = []
    fold_labels = []
    for (uid, true_lab), vals_list in session_results.items():
        
        # --- AGGREGATE BASED ON VOTING MODE ---
        if VOTING_MODE == 'soft':
            # vals_list is a list of arrays: [ [p0, p1, p2], [p0, p1, p2], ... ]
            # Take the mean across all windows, then find the class with the highest average probability
            mean_probs = np.mean(vals_list, axis=0)
            final_prediction = np.argmax(mean_probs)
        else:
            # vals_list is a list of integers: [0, 1, 0, 0, 2...]
            # Take the most common integer
            final_prediction = Counter(vals_list).most_common(1)[0][0]
            
        fold_preds.append(final_prediction)
        fold_labels.append(true_lab)
        
        # Save for global confusion matrix
        all_pred_labels.append(final_prediction)
        all_true_labels.append(true_lab)

    fold_acc = np.mean(np.array(fold_preds) == np.array(fold_labels))
    fold_accuracies.append(fold_acc)
    print(f"Fold {fold+1} Session Accuracy: {fold_acc:.4f}\n")
    torch.save(model.state_dict(), f"cnn_weights_fold_{fold + 1}.pth")

# --- FINAL SUMMARY ---
print("-" * 30)
print(f"Average CV Accuracy ({VOTING_MODE.upper()} Voting): {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
print("-" * 30)

# Plot Final Integrated Confusion Matrix
cm = confusion_matrix(all_true_labels, all_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["bonafide", "paraphrase", "transcribe"])
disp.plot(cmap=plt.cm.Purples)
plt.title(f"Aggregated {n_splits}-Fold Confusion Matrix ({VOTING_MODE.capitalize()} Voting)")
plt.show()