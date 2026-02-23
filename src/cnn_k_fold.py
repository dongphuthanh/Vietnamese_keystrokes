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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler

X=[] #Shape: (num_windows, window_length, feature)
Y=[] #label, Shape (num_windows)
ID=[]

def make_windows(filepath,X,Y,ID,id,win_length=200,stride=50):
    with open(filepath, "r", encoding="utf8") as f:
        data = json.load(f)
    _keystrokes=data.get("keystrokes")
    keystrokes=[]
    for i in range(len(_keystrokes)):
        if _keystrokes[i]["event"]=="keydown":
            keystrokes.append(_keystrokes[i])
    session2_idx=-1
    session3_idx=-1
    for i in range(len(keystrokes)):
        if keystrokes[i]["session"]==2 and session2_idx==-1:
            session2_idx=i
        if keystrokes[i]["session"]==3 and session3_idx==-1:
            session3_idx=i
            break
    i=1
    while i<session2_idx-win_length:
        window=[]
        for j in range(i,i+win_length):
            a=[]
            dt=keystrokes[j]["timestamp"]-keystrokes[j-1]["timestamp"]
            dt = np.clip(dt, 1, 5000)
            a.append(np.log(dt))
            if keystrokes[j]["key"]==" ":
                a.append(1)
            else:
                a.append(0)
            if keystrokes[j]["key"]=="Backspace":
                a.append(1)
            else:
                a.append(0)
            window.append(a)
        X.append(window)
        Y.append(0)
        ID.append(id)
        i+=stride

    i=session2_idx+1
    while i<session3_idx-win_length:
        window=[]
        for j in range(i,i+win_length):
            a=[]
            dt=keystrokes[j]["timestamp"]-keystrokes[j-1]["timestamp"]
            dt = np.clip(dt, 1, 5000)
            a.append(np.log(dt))
            if keystrokes[j]["key"]==" ":
                a.append(1)
            else:
                a.append(0)
            if keystrokes[j]["key"]=="Backspace":
                a.append(1)
            else:
                a.append(0)
            window.append(a)
        X.append(window)
        Y.append(1)
        ID.append(id)
        i+=stride

    i=session3_idx+1
    while i<len(keystrokes)-win_length:
        window=[]
        for j in range(i,i+win_length):
            a=[]
            dt=keystrokes[j]["timestamp"]-keystrokes[j-1]["timestamp"]
            dt = np.clip(dt, 1, 5000)
            a.append(np.log(dt))
            if keystrokes[j]["key"]==" ":
                a.append(1)
            else:
                a.append(0)
            if keystrokes[j]["key"]=="Backspace":
                a.append(1)
            else:
                a.append(0)
            window.append(a)
        X.append(window)
        Y.append(2)
        ID.append(id)
        i+=stride
#folder_path="keystroke_sessions_only"
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
                make_windows(filepath, X, Y, ID, user_id)


#Split dataset
X = np.array(X)  # shape: (num_windows, window_size, feature_dim)
print(X.shape[0])
Y = np.array(Y)  # shape: (num_windows,)

ID = np.array(ID) 
class TemporalCNN(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=feature_dim, out_channels=64, kernel_size=3, padding=1)
        self.bn1=nn.BatchNorm1d(64)
        self.dropout1=nn.Dropout(0.3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=1)
        self.bn2=nn.BatchNorm1d(128)
        self.dropout2=nn.Dropout(0.3)
        self.conv3=nn.Conv1d(128,256,kernel_size=5,padding=1)
        self.bn3=nn.BatchNorm1d(256)
        self.dropout3=nn.Dropout(0.2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        x = x.permute(0, 2, 1)  # (batch, feature_dim, seq_len)
        x = self.conv1(x)
        x=F.relu(self.bn1(x))
        x=self.dropout1(x)
        x = self.conv2(x)
        x=F.relu(self.bn2(x))
        x=self.dropout2(x)
        x = self.conv3(x)
        x=F.relu(self.bn3(x))
        x=self.dropout3(x)
        x = self.pool(x).squeeze(-1)  # (batch, 128)
        x = self.fc(x)  # (batch, num_classes)
        return x
    

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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

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