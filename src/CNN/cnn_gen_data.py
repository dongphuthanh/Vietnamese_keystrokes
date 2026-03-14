import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, BatchNorm, GINEConv, global_add_pool, GATv2Conv, GATConv, GINConv, GlobalAttention
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import matplotlib.pyplot as plt
from torch_geometric.utils import to_dense_adj
import networkx as nx
from torch_geometric.utils import to_networkx
import warnings
import itertools

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
                    is_space,
                    is_backspace
                ])

            X.append(window)
            Y.append(label)
            ID.append(id)

def make_windows_no_distinguish(filepath,X,Y,ID,id,win_length,stride):
    with open(filepath, "r", encoding="utf8") as f:
        data = json.load(f)
    _keystrokes=data.get("keystrokes")

    sessions = {0 : [], 1 : [], 2 : []}
    for i in range(len(_keystrokes)):
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
                    is_space,
                    is_backspace
                ])

            X.append(window)
            Y.append(label)
            ID.append(id)

def make_windows_with_up(filepath,X,Y,ID,id,win_length,stride):
    with open(filepath, "r", encoding="utf8") as f:
        data = json.load(f)
    _keystrokes=data.get("keystrokes")

    sessions = {0 : [], 1 : [], 2 : []}
    for i in range(len(_keystrokes)):
        if _keystrokes[i]["event"]=="keydown":
            _keystrokes[i]["dwell_time"] = np.log(np.clip(_keystrokes[i+1]["timestamp"] - _keystrokes[i]["timestamp"], 1, 2000))
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
                    keys[j-1]["dwell_time"],
                    is_space,
                    is_backspace
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
        
    def forward(self, x,return_features=False):
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
        features = self.pool(x).squeeze(-1)  # (batch, 128)

        if return_features:
            return features
    
        x = self.fc(features)  # (batch, num_classes)
        return x

def build_macro_graph(window_features, similarity_threshold=0.8):
    """Calculates edges for the PyG Data object."""
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

def create_graph_dataset(sessions, cnn_model, device, similarity_threshold=0.8):
    """Runs the CNN once and packages everything into PyG Mega-Graph format."""
    cnn_model.eval() 
    graph_dataset = []
    
    with torch.no_grad():
        for xb, yb in sessions:
            xb_tensor = torch.tensor(xb, dtype=torch.float32).to(device)
            window_features = cnn_model(xb_tensor, return_features=True)
            edge_index = build_macro_graph(window_features, similarity_threshold)
            
            data = Data(
                x=window_features.cpu(), 
                edge_index=edge_index.cpu(), 
                y=torch.tensor([yb], dtype=torch.long)
            )
            graph_dataset.append(data)
            
    return graph_dataset

class MacroGraphClassifier(nn.Module):
    def __init__(self, in_channels=256, hidden_channels=128, num_classes=3):
        super().__init__()
        # GAT allows windows to pay attention to the most important neighboring windows
        self.gat1 = GATConv(in_channels, hidden_channels, heads=4, concat=False)
        self.gat2 = GATConv(hidden_channels, hidden_channels, heads=4, concat=False)
        
        # The final decision maker
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, edge_index, batch):
        # 1. Windows talk to each other to understand the global session context
        h = self.gat1(x, edge_index)
        h = F.relu(h)
        h = self.gat2(h, edge_index)
        h = F.relu(h)
        
        # 2. Pool all the windows together into ONE master session vector
        session_embedding = global_mean_pool(h, batch)
        
        # 3. Classify the session
        return self.classifier(session_embedding)
class FastGraphClassifier(nn.Module):
    def __init__(self, in_channels=256, hidden_channels=128, num_classes=3):
        super().__init__()
        
        # GIN Aggregators
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
            nn.LayerNorm(64), # Safe for any batch size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, edge_index, batch):
        h = self.gin1(x, edge_index)
        h = F.relu(h)
        h = self.gin2(h, edge_index)
        h = F.relu(h)
        
        session_embedding = global_mean_pool(h, batch)
        return self.classifier(session_embedding)
    

class HierarchicalKeystrokeModel(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        # The Micro-Level Feature Extractor
        self.cnn = TemporalCNN(feature_dim=feature_dim, num_classes=num_classes)
        
        # The Macro-Level Graph Classifier
        # Input is 256 because that is what your TemporalCNN outputs
        self.gnn = MacroGraphClassifier(in_channels=256, hidden_channels=128, num_classes=num_classes)
        
    def forward(self, session_windows, similarity_threshold=0.8):
        """
        session_windows: A tensor of shape (num_windows_in_session, window_length, feature_dim)
        Note: We process ONE user's session at a time in this forward pass.
        """
        # 1. Extract 256D features from every window using the CNN
        window_features = self.cnn(session_windows, return_features=True)
        
        # 2. Dynamically build the graph based on how similar the windows are
        edge_index = build_macro_graph(window_features, similarity_threshold).to(window_features.device)
        
        # 3. Create a batch vector (all windows belong to User 0 in this pass)
        batch = torch.zeros(window_features.size(0), dtype=torch.long, device=window_features.device)
        
        # 4. Predict the entire session!
        session_logits = self.gnn(window_features, edge_index, batch)
        
        return session_logits
class GlobalAttentionClassifier(nn.Module):
    def __init__(self, in_channels=256, hidden_channels=64, num_classes=3):
        super().__init__()
        
        # 1. The "Regularity" Learner (Gate NN)
        # This tiny network looks at ONE window and outputs a single number:
        # "How important/anomalous is this window compared to the whole dataset?"
        self.gate_nn = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1) # Outputs a single score
        )
        
        # 2. The Smart Pooler
        self.attention_pool = GlobalAttention(gate_nn=self.gate_nn)
        
        # 3. The Final Classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5), # High dropout to prevent memorization
            nn.Linear(hidden_channels, num_classes)
        )

    def forward(self, x, edge_index, batch):
        # We completely ignore the graph edges! 
        # We just pass the raw CNN features (x) and the session groupings (batch)
        
        # The pooler automatically scores every window and creates a weighted session vector
        session_embedding = self.attention_pool(x, batch)
        
        return self.classifier(session_embedding)
    

class EndToEndBatchedModel(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.cnn = TemporalCNN(feature_dim=feature_dim, num_classes=num_classes)
        # Use your GIN or GlobalAttention classifier here!
        self.gnn = FastGraphClassifier(in_channels=256, hidden_channels=64, num_classes=num_classes) 
        
    def forward(self, list_of_sessions, similarity_threshold=0.8):
        """
        list_of_sessions: A list of tensors, where each tensor is a user's session of shape (num_windows, win_len, feature_dim)
        """
        device = next(self.parameters()).device
        data_list = []
        
        # 1. We process each session in the batch
        for session_windows in list_of_sessions:
            session_windows = session_windows.to(device)
            
            # Extract features dynamically (CNN is actively learning!)
            window_features = self.cnn(session_windows, return_features=True)
            
            # Build the graph dynamically (Edges change as CNN learns!)
            edge_index = build_macro_graph(window_features, similarity_threshold)
            
            # Create a temporary PyG Data object
            data = Data(x=window_features, edge_index=edge_index)
            data_list.append(data)
            
        # 2. Fuse them into a Mega-Graph instantly
        mega_graph = Batch.from_data_list(data_list).to(device)
        
        # 3. Classify the Mega-Graph
        return self.gnn(mega_graph.x, mega_graph.edge_index, mega_graph.batch)