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

X=[] #Shape: (num_windows, window_length, feature)
Y=[] #label, Shape (num_windows)
ID=[]

def make_windows(filepath,X,Y,ID,id,win_length=350,stride=90):
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
folder_path = "Combined"
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

# Use GroupShuffleSplit to ensure windows from the same ID stay together
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# This returns indices that keep groups (IDs) separate
train_idx, test_idx = next(gss.split(X, Y, groups=ID))

X_train, X_test = X[train_idx], X[test_idx]
Y_train, Y_test = Y[train_idx], Y[test_idx]

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(Y_train, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(Y_test, dtype=torch.long))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = TemporalCNN(feature_dim=X.shape[2], num_classes=len(np.unique(Y))).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
    scheduler.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

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
