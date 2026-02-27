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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler
from cnn_gen_data import make_windows
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
                make_windows(filepath, X, Y, ID, user_id, WIN_LENGTH, STRIDE)



X = np.array(X)  # shape: (num_windows, window_size, feature_dim)
print("X: ", X.shape)
Y = np.array(Y)  # shape: (num_windows,)
print("Y: ",Y.shape)
ID = np.array(ID) # shape: (num_windows,)
print("ID: ", ID.shape)
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use GroupShuffleSplit to ensure windows from the same ID stay together
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# This returns indices that keep groups (IDs) separate
train_idx, test_idx = next(gss.split(X, Y, groups=ID))

X_train, X_test = X[train_idx], X[test_idx]
Y_train, Y_test = Y[train_idx], Y[test_idx]

scaler = RobustScaler() # Everything else stays the same

# 1. Isolate and flatten
train_dt_col = X_train[:, :, 0].reshape(-1, 1)
test_dt_col = X_test[:, :, 0].reshape(-1, 1)

# 2. Fit and transform
X_train[:, :, 0] = scaler.fit_transform(train_dt_col).reshape(X_train.shape[0], X_train.shape[1])
X_test[:, :, 0] = scaler.transform(test_dt_col).reshape(X_test.shape[0], X_test.shape[1])

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(Y_train, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(Y_test, dtype=torch.long))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = TemporalCNN(feature_dim=X.shape[2], num_classes=len(np.unique(Y))).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

num_epochs = 150
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()
    avg_loss = epoch_loss / len(train_loader)
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


# -------------------------------
# 6. Evaluation
# -------------------------------
model.eval()
# Dictionary key: (user_id, session_label) -> value: [list of window predictions]
session_results = {} 

with torch.no_grad():
    # Use the test_idx to align IDs with the test_loader
    test_ids = ID[test_idx]
    
    current_idx = 0
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        preds = torch.argmax(F.softmax(out, dim=1), dim=1).cpu().numpy()
        labels = yb.cpu().numpy()
        
        # Get the IDs for this specific batch
        batch_ids = test_ids[current_idx : current_idx + len(xb)]
        current_idx += len(xb)

        for j in range(len(preds)):
            uid = batch_ids[j]
            true_lab = labels[j]
            
            # Unique key for the specific user's specific session
            session_key = (uid, true_lab)
            
            if session_key not in session_results:
                session_results[session_key] = []
            session_results[session_key].append(preds[j])

# --- Aggregate and Show Results ---
all_session_preds = []
all_session_labels = []

print(f"{'User ID':<10} | {'True Session':<15} | {'Majority Pred':<15} | {'Status'}")
print("-" * 60)

for (uid, true_label), preds in session_results.items():
    # Majority vote for this specific session
    majority_vote = Counter(preds).most_common(1)[0][0]
    
    all_session_preds.append(majority_vote)
    all_session_labels.append(true_label)
    
    label_map = {0: "bonafide", 1: "paraphrase", 2: "transcribe"}
    status = "✅" if majority_vote == true_label else "❌"
    
    print(f"{uid:<10} | {label_map[true_label]:<15} | {label_map[majority_vote]:<15} | {status}")

# --- Final Metrics & Confusion Matrix ---
session_accuracy = np.mean(np.array(all_session_preds) == np.array(all_session_labels))
print(f"\nPer-Session Accuracy: {session_accuracy:.4f}")

# Plot Session-Level Confusion Matrix
cm = confusion_matrix(all_session_labels, all_session_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["bonafide", "paraphrase", "transcribe"])
disp.plot(cmap=plt.cm.Purples)
plt.title("Per-User-Session Confusion Matrix")
plt.show()
