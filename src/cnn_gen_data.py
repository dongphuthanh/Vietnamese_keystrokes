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

def make_windows(filepath,X,Y,ID,id,win_length,stride):
    with open(filepath, "r", encoding="utf8") as f:
        data = json.load(f)
    _keystrokes=data.get("keystrokes")

    sessions = {0 : [], 1 : [], 2 : []}
    for i in range(len(_keystrokes)):
        if _keystrokes[i]["event"]=="keydown":
            if _keystrokes[i]["session"] == 1:
                sessions[0].append(_keystrokes[i])
            elif _keystrokes[i]["session"] == 2:
                sessions[1].append(_keystrokes[i])
            else:
                sessions[2].append(_keystrokes[i])

    for label, keys in sessions.items():
        for i in range(1, len(keys) - win_length, stride):
            window = []
            for j in range(i, i + win_length):
                dt = keys[j]["timestamp"] - keys[j - 1]["timestamp"]
                dt = np.clip(dt, 1, 5000)

                is_space = 1 if (keys[j - 1]["key"] == " ") else 0
                is_backspace = 1 if (keys[j]["key"] == "Backspace") else 0
                window.append([
                    np.log(dt),
                    is_space
                    #is_backspace
                ])

            X.append(window)
            Y.append(label)
            ID.append(id)

class TemporalCNN(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=feature_dim, out_channels=64, kernel_size=3, padding=1)
        self.bn1=nn.BatchNorm1d(64)
        self.dropout1=nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=2, dilation=2)
        self.bn2=nn.BatchNorm1d(128)
        self.dropout2=nn.Dropout(0.2)
        self.conv3=nn.Conv1d(128,256,kernel_size=3,padding=4, dilation=4)
        self.bn3=nn.BatchNorm1d(256)
        self.dropout3=nn.Dropout(0.3)
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

