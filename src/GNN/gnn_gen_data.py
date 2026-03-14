import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, BatchNorm, GINEConv, global_add_pool, GATv2Conv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from torch_geometric.utils import to_dense_adj
import networkx as nx
from torch_geometric.utils import to_networkx
import warnings
import itertools
from cnn_gen_data import *
from sklearn.metrics.pairwise import cosine_similarity

def gin_make_windows(filepath,X,Y,ID,id,win_length,stride):
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
                if keys[j]["key"] == " ":
                    window.append([0,keys[j]["timestamp"]])
                elif keys[j]["key"] == "Backspace":
                    window.append([1,keys[j]["timestamp"]])
                else:
                    window.append([2,keys[j]["timestamp"]])
            X.append(window)
            Y.append(label)
            ID.append(id)

def gin_make_windows_with_up(filepath,X,Y,ID,id,win_length,stride):
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
                if keys[j]["key"] == " ":
                    window.append([0,keys[j]["timestamp"],keys[j]["dwell_time"]])
                elif keys[j]["key"] == "Backspace":
                    window.append([1,keys[j]["timestamp"],keys[j]["dwell_time"]])
                elif keys[j]["key"].isalpha() and len(keys[j]["key"]) == 1:
                    window.append([2,keys[j]["timestamp"],keys[j]["dwell_time"]])
                else:
                    window.append([3,keys[j]["timestamp"],keys[j]["dwell_time"]])
            X.append(window)
            Y.append(label)
            ID.append(id)

"""
Graph config 1:
Connect every adjacent keystroke, then connect to other keystrokes base on a threshold
"""
def gin_make_graph(windows, labels, ID, percentile = 95, global_threshold = None):
    graphs = []
    b, b_count, p, p_count, t, t_count = 0, 0, 0, 0, 0, 0
    for idx in range(len(windows)):
        window = windows[idx]
        edge = []
        edge_feature = []
        length = len(window)
        if global_threshold is not None:
            # Fallback: Use the hardcoded global threshold
            threshold = global_threshold
        else:
            # Primary: Calculate the relative threshold for THIS window only
            window_ikis = [np.clip(abs(window[k+1][1] - window[k][1]),1,5000) for k in range(length - 1)]
            if len(window_ikis) > 0:
                threshold = np.percentile(window_ikis, percentile)
            else:
                threshold = 0
        for i in range(length):
            if i + 1 < length:
                diff_time = abs(window[i][1] - window[i+1][1])
                val = np.log(np.clip(diff_time, 1, 5000))
                
                edge.extend([[i, i+1], [i+1, i]])
                edge_feature.extend([[val], [val]])


            for j in range(i + 2, length):
                diff_time = abs(window[i][1] - window[j][1])
                
                if diff_time <= threshold:
                    val = np.log(np.clip(diff_time, 1, 5000))
                    edge.extend([[i, j], [j, i]])
                    edge_feature.extend([[val], [val]])
                else:
                    break
        z = len(edge_feature)
        if len(window[0])>=3:
            node_feature = [[sublist[0], sublist[2]] for sublist in window]
        else:
            node_feature = [[sublist[0]] for sublist in window]
        x = torch.tensor(node_feature, dtype = torch.float)
        edge = torch.tensor(edge, dtype = torch.long).t().contiguous()
        edge_feature = torch.tensor(edge_feature, dtype = torch.float)
        y = torch.tensor([labels[idx]], dtype = torch.long)
        data = Data(x = x, edge_index = edge, edge_attr = edge_feature, y = y)
        data.group_id = torch.tensor([ID[idx]], dtype=torch.long) 
        graphs.append(data)

        if labels[idx] == 0:
            b += z
            b_count += 1
        elif labels[idx] == 1:
            p += z 
            p_count += 1
        else:
            t += z
            t_count += 1
    
    print("Average edge of bonafide: ", b / b_count )
    print("Average edge of paraphrase: ", p / p_count )
    print("Average edge of transcribe: ", t / t_count )
    return graphs

"""
Graph config 2:
Connect to other keystrokes base on a threshold
"""
def gin_make_graph_no_backbone(windows, labels, ID, percentile = 95, global_threshold = 700):
    graphs = []
    b, b_count, p, p_count, t, t_count = 0, 0, 0, 0, 0, 0
    for idx in range(len(windows)):
        window = windows[idx]
        edge = []
        edge_feature = []
        length = len(window)
        if global_threshold is not None:
            # Fallback: Use the hardcoded global threshold
            threshold = global_threshold
        else:
            # Primary: Calculate the relative threshold for THIS window only
            window_ikis = [np.clip(abs(window[k+1][1] - window[k][1]),1,5000) for k in range(length - 1)]
            if len(window_ikis) > 0:
                threshold = np.percentile(window_ikis, percentile)
            else:
                threshold = 0
        for i in range(length):
            for j in range(i + 1, length):
                diff_time = abs(window[i][1] - window[j][1])
                
                if diff_time <= threshold:
                    val = np.log(np.clip(diff_time, 1, 5000))
                    edge.extend([[i, j], [j, i]])
                    edge_feature.extend([[val], [val]])
                else:
                    break
        z = len(edge_feature)
        if len(window[0]) >= 3:
            node_feature = [[sublist[0], sublist[2]] for sublist in window]
        else:
            node_feature = [[sublist[0]] for sublist in window]
        x = torch.tensor(node_feature, dtype = torch.float)
        edge = torch.tensor(edge, dtype = torch.long).t().contiguous()
        edge_feature = torch.tensor(edge_feature, dtype = torch.float)
        y = torch.tensor([labels[idx]], dtype = torch.long)
        data = Data(x = x, edge_index = edge, edge_attr = edge_feature, y = y)
        data.group_id = torch.tensor([ID[idx]], dtype=torch.long) 
        graphs.append(data)

        if labels[idx] == 0:
            b += z
            b_count += 1
        elif labels[idx] == 1:
            p += z 
            p_count += 1
        else:
            t += z
            t_count += 1
    
    print("Average edge of bonafide: ", b / b_count )
    print("Average edge of paraphrase: ", p / p_count )
    print("Average edge of transcribe: ", t / t_count )
    return graphs

"""
Graph config 3:
Using knn to connect the node
"""
def gin_make_graph_knn(windows, labels, ID, k=6):
    graphs = []
    b, b_count, p, p_count, t, t_count = 0, 0, 0, 0, 0, 0
    
    for idx in range(len(windows)):
        window = windows[idx]
        length = len(window)
        
        # Use a set to store unique undirected edges (u, v) where u < v
        unique_edges = set()
        
        for i in range(length):
            # 1. THE BACKBONE (Always keep sequential order intact)

                
            # 2. TEMPORAL K-NN (Find the k closest keys in time)
            left = i - 1
            right = i + 1
            count = 0
            
            # Expand outwards to find the k smallest time differences
            while count < k:
                if left >= 0 and right < length:
                    diff_left = window[i][1] - window[left][1]   # time is sorted, so i > left
                    diff_right = window[right][1] - window[i][1] # right > i
                    
                    if diff_left <= diff_right:
                        unique_edges.add((min(i, left), max(i, left)))
                        left -= 1
                    else:
                        unique_edges.add((min(i, right), max(i, right)))
                        right += 1
                elif left >= 0:
                    unique_edges.add((min(i, left), max(i, left)))
                    left -= 1
                elif right < length:
                    unique_edges.add((min(i, right), max(i, right)))
                    right += 1
                else:
                    break # Safety break if window is smaller than k
                count += 1
        
        # 3. CONVERT SET TO EDGE LISTS
        edge = []
        edge_feature = []
        for (u, v) in unique_edges:
            diff = abs(window[u][1] - window[v][1])
            val = np.log(np.clip(diff, 1, 5000))
            
            # Add bidirectional edges for PyTorch Geometric
            edge.extend([[u, v], [v, u]])
            edge_feature.extend([[val], [val]])
            
        # 4. PYTORCH GEOMETRIC DATA CONSTRUCTION
        z = len(edge_feature)
        if len(window[0]) >= 3:
            node_feature = [[sublist[0], sublist[2]] for sublist in window]
        else:
            node_feature = [[sublist[0]] for sublist in window]
        
        x = torch.tensor(node_feature, dtype = torch.float)
        
        # Safety net for completely empty windows
        if len(edge) == 0:
            edge = [[0, 0]]
            edge_feature = [[0.0]]
            
        edge_index = torch.tensor(edge, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_feature, dtype=torch.float32)
        y = torch.tensor([labels[idx]], dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.group_id = torch.tensor([ID[idx]], dtype=torch.long) 
        graphs.append(data)

        # 5. TRACK EDGE COUNTS
        if labels[idx] == 0:
            b += z
            b_count += 1
        elif labels[idx] == 1:
            p += z 
            p_count += 1
        else:
            t += z
            t_count += 1
    
    print("Average edge of bonafide:   ", b / b_count if b_count > 0 else 0)
    print("Average edge of paraphrase: ", p / p_count if p_count > 0 else 0)
    print("Average edge of transcribe: ", t / t_count if t_count > 0 else 0)
    
    return graphs

def gin_make_graph_islands(windows, labels, ID, threshold=150):
    """
    threshold: max flight time in ms (since your code uses np.clip(diff, 1, 5000))
    """
    graphs = []
    b, b_count, p, p_count, t, t_count = 0, 0, 0, 0, 0, 0
    
    for idx in range(len(windows)):
        window = windows[idx]
        length = len(window)
        unique_edges = set()
        
        # 1. IDENTIFY ISLANDS (Groups connected by flight time < threshold)
        islands = []
        if length > 0:
            current_island = [0]
            for i in range(length - 1):
                # Calculate flight time (FT) between sequential keys
                # window[i][1] is the timestamp
                ft = window[i+1][1] - window[i][1]
                
                if ft < threshold:
                    current_island.append(i + 1)
                else:
                    # Gap detected, close current island and start new one
                    if len(current_island) > 1:
                        islands.append(current_island)
                    current_island = [i + 1]
            
            # Catch the last island
            if len(current_island) > 1:
                islands.append(current_island)

        # 2. FULLY CONNECT EACH ISLAND (Create Cliques)
        for island_nodes in islands:
            # itertools.combinations(..., 2) generates undirected pairs (u, v)
            for u, v in itertools.combinations(island_nodes, 2):
                # Store as (min, max) to ensure uniqueness in the set
                unique_edges.add((u, v))
        
        # 3. CONVERT SET TO EDGE LISTS & COMPUTE FEATURES
        edge = []
        edge_feature = []
        for (u, v) in unique_edges:
            diff = abs(window[u][1] - window[v][1])
            # Keeping your log scaling logic
            val = np.log(np.clip(diff, 1, 5000))
            
            # Bidirectional edges for PyG
            edge.extend([[u, v], [v, u]])
            edge_feature.extend([[val], [val]])
            
        # 4. DATA CONSTRUCTION (Consistent with your provided format)
        z = len(edge_feature)
        
        # Node features: check for dwell time in column 2
        if len(window[0]) >= 3:
            node_feature = [[sublist[0], sublist[2]] for sublist in window]
        else:
            node_feature = [[sublist[0]] for sublist in window]
        
        x = torch.tensor(node_feature, dtype=torch.float)
        
        if len(edge) == 0:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.tensor([[0.0]], dtype=torch.float32)
        else:
            edge_index = torch.tensor(edge, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_feature, dtype=torch.float32)
            
        y = torch.tensor([labels[idx]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.group_id = torch.tensor([ID[idx]], dtype=torch.long) 
        graphs.append(data)

        # 5. TRACK EDGE COUNTS
        if labels[idx] == 0:
            b += z
            b_count += 1
        elif labels[idx] == 1:
            p += z 
            p_count += 1
        else:
            t += z
            t_count += 1
    
    print("Average edges of bonafide:   ", b / b_count if b_count > 0 else 0)
    print("Average edges of paraphrase: ", p / p_count if p_count > 0 else 0)
    print("Average edges of transcribe: ", t / t_count if t_count > 0 else 0)
    
    return graphs




def gin_make_graph_islands_v4(windows, labels, ID, island_th=150, burst_th=80, pause_th=800):
    graphs = []
    b, b_count, p, p_count, t, t_count = 0, 0, 0, 0, 0, 0
    
    for idx in range(len(windows)):
        window = windows[idx]
        length = len(window)
        edge_dict = {}
        
        # Helper to determine the 3 mutually exclusive category flags
        def get_category_flags(ft):
            if ft < burst_th:
                return [0, 1, 0] # Standard=0, Burst=1, Pause=0
            elif ft > pause_th:
                return [0, 0, 1] # Standard=0, Burst=0, Pause=1
            else:
                return [1, 0, 0] # Standard=1, Burst=0, Pause=0

        # --- 1. THE BACKBONE (Always connect adjacent keys) ---
        for i in range(length - 1):
            u, v = i, i + 1
            ft = window[v][1] - window[u][1]
            cat_flags = get_category_flags(ft)
            
            # Forward: Direction=0 | Backward: Direction=1
            edge_dict[(u, v)] = [ft] + cat_flags + [0]
            edge_dict[(v, u)] = [ft] + cat_flags + [1]

        # --- 2. IDENTIFY ISLANDS (Nodes connected by FT < island_th) ---
        islands = []
        if length > 0:
            current_island = [0]
            for i in range(length - 1):
                ft = window[i+1][1] - window[i][1]
                if ft < island_th:
                    current_island.append(i + 1)
                else:
                    if len(current_island) > 1: islands.append(current_island)
                    current_island = [i + 1]
            if len(current_island) > 1: islands.append(current_island)

        # --- 3. CLIQUE EDGES (Fully connect nodes within islands) ---
        for island_nodes in islands:
            for u, v in itertools.combinations(island_nodes, 2):
                if (u, v) not in edge_dict: # Don't overwrite backbone
                    ft = abs(window[v][1] - window[u][1])
                    cat_flags = get_category_flags(ft)
                    
                    edge_dict[(u, v)] = [ft] + cat_flags + [0]
                    edge_dict[(v, u)] = [ft] + cat_flags + [1]

        # --- 4. DATA CONSTRUCTION ---
        edge_index_list, edge_attr_list = [], []
        for (u, v), feats in edge_dict.items():
            ft_log = np.log(np.clip(feats[0], 1, 5000))
            edge_index_list.append([u, v])
            edge_attr_list.append([ft_log] + feats[1:])

        x = torch.tensor([[s[0], s[2]] if len(window[0]) >= 3 else [s[0]] for s in window], dtype=torch.float)
        
        if len(edge_index_list) == 0:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.tensor([[0.0, 0, 0, 0, 0]], dtype=torch.float32)
        else:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
            
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([labels[idx]], dtype=torch.long))
        data.group_id = torch.tensor([ID[idx]], dtype=torch.long) 
        graphs.append(data)

        # Stats tracking
        z = len(edge_attr_list)
        if labels[idx] == 0: b += z; b_count += 1
        elif labels[idx] == 1: p += z; p_count += 1
        else: t += z; t_count += 1
    
    print(f"Avg edges - Bonafide: {b/b_count:.1f} | Paraphrase: {p/p_count:.1f} | Transcribe: {t/t_count:.1f}")
    return graphs

def gin_make_graph_skip_backbone(windows, labels, ID, burst_th=200, pause_th=500, skip_th=100):
    graphs = []
    b, b_count, p, p_count, t, t_count = 0, 0, 0, 0, 0, 0
    
    for idx in range(len(windows)):
        window = windows[idx]
        length = len(window)
        # Using a set for undirected edges to ensure we don't duplicate backbone vs skip
        unique_edges = {} 
        
        def get_category_flags(ft):
            if ft < burst_th:
                return [0, 1, 0] # Standard=0, Burst=1, Pause=0
            elif ft > pause_th:
                return [0, 0, 1] # Standard=0, Burst=0, Pause=1
            else:
                return [1, 0, 0] # Standard=1, Burst=0, Pause=0

        # --- 1. THE BACKBONE (i to i+1) ---
        for i in range(length - 1):
            u, v = i, i + 1
            ft = window[v][1] - window[u][1]
            cat_flags = get_category_flags(ft)
            # Store as sorted tuple for undirected uniqueness
            pair = tuple(sorted((u, v)))
            unique_edges[pair] = [ft] + cat_flags

        # --- 2. THE SKIP CONNECTIONS (i to i+2) ---
        for i in range(length - 2):
            ft1 = window[i+1][1] - window[i][1]
            ft2 = window[i+2][1] - window[i+1][1]
            
            if ft1 < skip_th and ft2 < skip_th:
                u, v = i, i + 2
                ft_total = window[v][1] - window[u][1]
                pair = tuple(sorted((u, v)))
                # Only add if it doesn't exist (though i, i+2 shouldn't be in backbone)
                if pair not in unique_edges:
                    unique_edges[pair] = [ft_total] + get_category_flags(ft_total)

        # --- 3. DATA CONSTRUCTION ---
        edge_index_list, edge_attr_list = [], []
        for (u, v), feats in unique_edges.items():
            ft_log = np.log(np.clip(feats[0], 1, 5000))
            # PyG GINE expects directed-style edges even if the logic is undirected
            # We add both directions with the same attributes
            edge_index_list.extend([[u, v], [v, u]])
            edge_attr_list.extend([[ft_log] + feats[1:], [ft_log] + feats[1:]])

        # Node features (Identity and Dwell Time if available)
        x = torch.tensor([[s[0], s[2]] if len(window[0]) >= 3 else [s[0]] for s in window], dtype=torch.float)
        
        if len(edge_index_list) == 0:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.tensor([[0.0, 0, 0, 0]], dtype=torch.float32)
        else:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
            
        y = torch.tensor([labels[idx]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.group_id = torch.tensor([ID[idx]], dtype=torch.long) 
        graphs.append(data)

        # Stats
        z = len(edge_attr_list)
        if labels[idx] == 0: b += z; b_count += 1
        elif labels[idx] == 1: p += z; p_count += 1
        else: t += z; t_count += 1
    
    print(f"Avg edges - Bonafide: {b/b_count:.1f} | Paraphrase: {p/p_count:.1f} | Transcribe: {t/t_count:.1f}")
    return graphs

def make_burst_meta_graph(windows, labels, ID, burst_th=500, pause_th=2000, sim_th=0.95, sim_topk=3, sim_window=10):
    graphs = []
    stats = {
        0: {'n': [], 'e': []},
        1: {'n': [], 'e': []},
        2: {'n': [], 'e': []}
    }
    for idx, window in enumerate(windows):
        if len(window) == 0:
            continue
        # --------------------------------------------------
        # 1. CLUSTER INTO BURSTS
        # --------------------------------------------------
        bursts = []
        current_burst = [window[0]]
        for i in range(1, len(window)):
            ft = max(window[i][1] - window[i - 1][1], 0)
            if ft < burst_th:
                current_burst.append(window[i])
            else:
                bursts.append(current_burst)
                current_burst = [window[i]]

        bursts.append(current_burst)
        # --------------------------------------------------
        # 2. NODE FEATURES
        # --------------------------------------------------
        node_features = []

        for b in bursts:
            b_arr = np.array(b)
            if len(b) > 1:
                fts = np.diff(b_arr[:, 1])
                # FIX 1: Force all negative time-traveling glitches to 0.0!
                fts = np.maximum(fts, 0.0)
            else:
                fts = np.array([0.0])
                
            # FIX 2: Ignore NaNs in the mean/std calculation if the JSON was corrupted
            mean_ft = np.nanmean(fts)
            std_ft = np.nanstd(fts)
            burst_len = len(b)
            
            # FIX 3: Catch any lingering NaNs just in case an entire array was empty
            if np.isnan(mean_ft): mean_ft = 0.0
            if np.isnan(std_ft): std_ft = 0.0

            node_features.append([
                np.log(mean_ft + 1),
                np.log(std_ft + 1),
                np.log(burst_len + 1),
                1.0 if burst_len == 1 else 0.0
            ])

        x = torch.tensor(node_features, dtype=torch.float32)
        num_nodes = x.size(0)
        # --------------------------------------------------
        # 3. EDGE GENERATION
        # --------------------------------------------------

        edge_index = []
        edge_attr = []

        # ---------- Temporal backbone ----------

        for i in range(num_nodes - 1):
            pause_time = bursts[i + 1][0][1] - bursts[i][-1][1]
            pause_time = max(pause_time, 1)
            pause_log = np.log(pause_time)
            is_long = 1.0 if pause_time > pause_th else 0.0
            feat = [pause_log, 0.0, is_long, 0.0]
            edge_index.extend([[i, i + 1], [i + 1, i]])
            edge_attr.extend([feat, feat])

        # ---------- Similarity edges ----------

        if num_nodes > 2:
            feat_matrix = x[:, :3].cpu().numpy()
            sim_matrix = cosine_similarity(feat_matrix)
            for i in range(num_nodes):
                candidates = []
                for j in range(num_nodes):
                    if i == j:
                        continue
                    if abs(i - j) > sim_window:
                        continue
                    if j == i + 1 or j == i - 1:
                        continue
                    sim = sim_matrix[i, j]
                    if sim > sim_th:
                        candidates.append((j, sim))

                candidates.sort(key=lambda x: x[1], reverse=True)

                for j, sim in candidates[:sim_topk]:

                    feat = [0.0, float(sim), 0.0, 1.0]
                    edge_index.extend([[i, j], [j, i]])
                    edge_attr.extend([feat, feat])

        # --------------------------------------------------
        # 4. GRAPH SAFETY
        # --------------------------------------------------

        if num_nodes == 1:
            edge_index_tensor = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr_tensor = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

        elif len(edge_index) == 0:
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
            edge_attr_tensor = torch.empty((0, 4), dtype=torch.float32)

        else:
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)

        # --------------------------------------------------
        # 5. DATA OBJECT
        # --------------------------------------------------

        label = labels[idx]
        data = Data(
            x=x,
            edge_index=edge_index_tensor,
            edge_attr=edge_attr_tensor,
            y=torch.tensor([label], dtype=torch.long)
        )
        data.group_id = torch.tensor([ID[idx]], dtype=torch.long)

        graphs.append(data)

        stats[label]['n'].append(num_nodes)
        stats[label]['e'].append(edge_index_tensor.size(1))

    # --------------------------------------------------
    # 6. SUMMARY
    # --------------------------------------------------

    print("\n" + "-" * 40)
    print(f"{'Class':<12} | {'Avg Nodes':<10} | {'Avg Edges':<10}")
    print("-" * 40)

    for lab, name in {0: "Bonafide", 1: "Paraphrase", 2: "Transcribe"}.items():

        avg_n = np.mean(stats[lab]['n']) if stats[lab]['n'] else 0
        avg_e = np.mean(stats[lab]['e']) if stats[lab]['e'] else 0

        print(f"{name:<12} | {avg_n:<10.2f} | {avg_e:<10.2f}")

    print("-" * 40 + "\n")

    return graphs

class KeystrokeGINE(nn.Module):
    def __init__(self, num_node_types, edge_in_dim, hidden_dim, num_classes, num_layers=3):
        super(KeystrokeGINE, self).__init__()
        
        # --- 1. Node Feature Processing ---
        self.node_emb = nn.Embedding(num_node_types, hidden_dim)
        self.dwell_proj = nn.Linear(1, hidden_dim) 
        self.concat_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # --- 2. Convolutional Layers ---
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.convs.append(GINEConv(mlp, train_eps=True, edge_dim=edge_in_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
        # --- 3. Final Classifier ---
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # --- 1. NODE FEATURE HANDLING (Concatenation) ---
        if x.shape[1] > 1:
            flags = x[:, 0].long()       
            dwell = x[:, 1].unsqueeze(1) 
            
            x_emb = self.node_emb(flags)
            x_dwell = self.dwell_proj(dwell)
            
            # Concatenate [N, 64] and [N, 64] -> [N, 128], then project back to [N, 64]
            x_cat = torch.cat([x_emb, x_dwell], dim=-1)
            x = F.relu(self.concat_proj(x_cat))
        else:
            x = self.node_emb(x.squeeze(-1).long())
            
        # --- 2. EDGE FEATURE SAFETY NET ---
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)
        
        # --- 3. STANDARD MESSAGE PASSING ---
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(x)
            x = F.relu(x)
        
        # --- 4. STANDARD READOUT (Single Global Pool) ---
        # Pool the 400 node vectors into 1 graph vector at the very end
        x_global = global_add_pool(x, batch)
        
        return self.fc(x_global)


class BurstMetaGINE(nn.Module):
    def __init__(self, node_in_dim=5, edge_in_dim=3, hidden_dim=64, num_classes=3, num_layers=2):
        super(BurstMetaGINE, self).__init__()
        
        # --- 1. Node Feature Processing ---
        # No more Embedding! We project the 5D burst summary into hidden_dim.
        self.node_projection = nn.Linear(node_in_dim, hidden_dim)
        
        # --- 2. Convolutional Layers ---
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            # GINE requires an MLP to map aggregated messages
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            # edge_dim must match the 3D edge features (LogTime/Sim, LongFlag, TypeFlag)
            self.convs.append(GINEConv(mlp, train_eps=True, edge_dim=edge_in_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
        # --- 3. Final Classifier ---
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # --- 1. INITIAL PROJECTION ---
        # x is [Num_Bursts, 5] -> project to [Num_Bursts, hidden_dim]
        x = F.relu(self.node_projection(x))
            
        # --- 2. MESSAGE PASSING ---
        for conv, bn in zip(self.convs, self.batch_norms):
            # x_next = MLP( (1+eps) * x_i + sum( ReLU(x_j + edge_attr_ij) ) )
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(x)
            x = F.relu(x)
        
        # --- 3. GLOBAL READOUT ---
        # Since Meta-Graphs are smaller, 'add' or 'mean' pool both work well.
        x_global = global_add_pool(x, batch)
        
        return self.fc(x_global)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def plot_all_topological_distributions(graphs):
    # Dictionaries to hold all 5 metrics
    metrics = {
        'clustering': {0: [], 1: [], 2: []},
        'density':    {0: [], 1: [], 2: []},
        'assort':     {0: [], 1: [], 2: []},
        'path_len':   {0: [], 1: [], 2: []},
        'island_count': {0: [], 1: [], 2: []} # NEW METRIC
    }
    
    class_names = {0: "Bonafide", 1: "Paraphrase", 2: "Transcribe"}
    colors = {0: 'royalblue', 1: 'darkorange', 2: 'forestgreen'}
    
    print(f"Calculating 5 topological metrics for {len(graphs)} graphs...")
    
    for i, data in enumerate(graphs):
        label = data.y.item()
        nx_graph = to_networkx(data, to_undirected=True)
        
        # 1. Clustering Coefficient
        metrics['clustering'][label].append(nx.average_clustering(nx_graph))
        
        # 2. Graph Density
        metrics['density'][label].append(nx.density(nx_graph))
        
        # 3. Degree Assortativity
        try:
            assort = nx.degree_assortativity_coefficient(nx_graph)
            if not np.isnan(assort):
                metrics['assort'][label].append(assort)
        except Exception:
            pass
            
        # 4. Average Shortest Path Length
        if nx.is_connected(nx_graph) and len(nx_graph.nodes) > 1:
            metrics['path_len'][label].append(nx.average_shortest_path_length(nx_graph))
            
        # 5. NEW: Island Count (Connected Components)
        # Identifies separate bursts or "islands" in the window
        island_count = nx.number_connected_components(nx_graph)
        metrics['island_count'][label].append(island_count)
            
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1} / {len(graphs)} graphs...")

    # --- Plotting the 3x2 Grid (Now 5 plots total) ---
    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    axes = axes.flatten() 
    
    metric_keys = ['clustering', 'density', 'assort', 'path_len', 'island_count']
    titles = ['1. Clustering Coefficient', '2. Graph Density', 
              '3. Degree Assortativity', '4. Avg Shortest Path Length',
              '5. Island Count (Connected Components)']
    xlabels = ['Avg Clustering (Higher = More Triangles)', 
               'Density (Higher = More globally connected)', 
               'Assortativity (Higher = Fast bursts connect to each other)', 
               'Path Length (Lower = More "Jump-Edges")',
               'Number of Islands (Lower = Continuous typing)']
    
    for idx, m_key in enumerate(metric_keys):
        ax = axes[idx]
        for label in [0, 1, 2]:
            if len(metrics[m_key][label]) == 0:
                continue
                
            ax.hist(metrics[m_key][label], bins=35, alpha=0.6, 
                    label=class_names[label], color=colors[label], density=True)
            
            mean_val = np.mean(metrics[m_key][label])
            ax.axvline(mean_val, color=colors[label], linestyle='dashed', linewidth=2)
            
        ax.set_title(titles[idx], fontsize=15, fontweight='bold')
        ax.set_xlabel(xlabels[idx], fontsize=12)
        ax.set_ylabel("Density of Windows", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
    
    # Remove the 6th empty subplot
    fig.delaxes(axes[5])
        
    plt.tight_layout()
    plt.show()

def plot_all_topological_distributions_with_flag(graphs):
    # Dictionaries for the original 5 metrics + 3 new flag metrics
    metrics = {
        'clustering': {0: [], 1: [], 2: []},
        'density':    {0: [], 1: [], 2: []},
        'assort':     {0: [], 1: [], 2: []},
        'path_len':   {0: [], 1: [], 2: []},
        'island_count': {0: [], 1: [], 2: []},
        # Flag distributions (Percentage of edges per category)
        'std_pct':    {0: [], 1: [], 2: []},
        'burst_pct':  {0: [], 1: [], 2: []},
        'pause_pct':  {0: [], 1: [], 2: []}
    }
    
    class_names = {0: "Bonafide", 1: "Paraphrase", 2: "Transcribe"}
    colors = {0: 'royalblue', 1: 'darkorange', 2: 'forestgreen'}
    
    print(f"Calculating 8 metrics (Topological + Flag Dist) for {len(graphs)} graphs...")
    
    for i, data in enumerate(graphs):
        label = data.y.item()
        nx_graph = to_networkx(data, to_undirected=True)
        
        # --- 1. Topological Metrics ---
        metrics['clustering'][label].append(nx.average_clustering(nx_graph))
        metrics['density'][label].append(nx.density(nx_graph))
        metrics['island_count'][label].append(nx.number_connected_components(nx_graph))
        
        try:
            assort = nx.degree_assortativity_coefficient(nx_graph)
            if not np.isnan(assort): metrics['assort'][label].append(assort)
        except: pass
            
        if nx.is_connected(nx_graph) and len(nx_graph.nodes) > 1:
            metrics['path_len'][label].append(nx.average_shortest_path_length(nx_graph))
            
        # --- 2. Edge Flag Distributions ---
        # data.edge_attr is [LogFT, Standard, Burst, Pause, Direction]
        # We only count forward edges (Direction=0) to avoid double counting
        edge_attr = data.edge_attr
        relevant_edges = edge_attr
        
        if len(relevant_edges) > 0:
            # Sum up the binary flags and divide by total edges in that window
            std_sum   = torch.sum(relevant_edges[:, 1]).item()
            burst_sum = torch.sum(relevant_edges[:, 2]).item()
            pause_sum = torch.sum(relevant_edges[:, 3]).item()
            total = len(relevant_edges)
            
            metrics['std_pct'][label].append(std_sum / total)
            metrics['burst_pct'][label].append(burst_sum / total)
            metrics['pause_pct'][label].append(pause_sum / total)
            
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1} / {len(graphs)} graphs...")

    # --- Plotting the 4x2 Grid ---
    fig, axes = plt.subplots(4, 2, figsize=(18, 22))
    axes = axes.flatten() 
    
    metric_keys = [
        'clustering', 'density', 'assort', 'path_len', 
        'island_count', 'std_pct', 'burst_pct', 'pause_pct'
    ]
    titles = [
        '1. Clustering Coefficient', '2. Graph Density', 
        '3. Degree Assortativity', '4. Avg Shortest Path Length',
        '5. Island Count', '6. Standard Edge %', 
        '7. Burst Edge %', '8. Pause Edge %'
    ]
    
    for idx, m_key in enumerate(metric_keys):
        ax = axes[idx]
        for label in [0, 1, 2]:
            vals = metrics[m_key][label]
            if len(vals) == 0: continue
                
            ax.hist(vals, bins=35, alpha=0.6, label=class_names[label], 
                    color=colors[label], density=True)
            
            mean_val = np.mean(vals)
            ax.axvline(mean_val, color=colors[label], linestyle='dashed', linewidth=2)
            
        ax.set_title(titles[idx], fontsize=15, fontweight='bold')
        ax.set_ylabel("Density of Windows")
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
    plt.tight_layout()
    plt.show()

def plot_meta_graph_distributions(graphs):
    # Dictionaries for topological + meta-specific metrics
    metrics = {
        'node_count':   {0: [], 1: [], 2: []}, # Nodes per window
        'edge_count':   {0: [], 1: [], 2: []}, # Total edges (Backbone + Jumps)
        'avg_burst':    {0: [], 1: [], 2: []}, # Avg keys per node
        'lone_key_pct': {0: [], 1: [], 2: []}, # % of nodes that are lone keys
        'jump_pct':     {0: [], 1: [], 2: []}, # % of edges that are similarity jumps
        'avg_pause':    {0: [], 1: [], 2: []}, # Mean of backbone pause logs
        'clustering':   {0: [], 1: [], 2: []}, # Graph connectivity
        'density':      {0: [], 1: [], 2: []}  # Global density
    }
    
    class_names = {0: "Bonafide", 1: "Paraphrase", 2: "Transcribe"}
    colors = {0: 'royalblue', 1: 'darkorange', 2: 'forestgreen'}
    
    print(f"Analyzing meta-topology for {len(graphs)} windows...")
    
    for i, data in enumerate(graphs):
        label = data.y.item()
        
        # --- 1. Basic Counts ---
        num_nodes = data.x.size(0)
        num_edges = data.edge_index.size(1) // 2 # Undirected
        metrics['node_count'][label].append(num_nodes)
        metrics['edge_count'][label].append(num_edges)
        
        # --- 2. Node Features (x: [AvgSpeed, Std, Length, Dwell, LoneFlag]) ---
        avg_burst_len = torch.mean(data.x[:, 2]).item()
        lone_key_pct  = torch.sum(data.x[:, 3]).item() / num_nodes
        metrics['avg_burst'][label].append(avg_burst_len)
        metrics['lone_key_pct'][label].append(lone_key_pct)
        
        # --- 3. Edge Features (attr: [LogTime/Sim, LongFlag, TypeFlag]) ---
        if num_edges > 0:
            # TypeFlag is at index 2 (0=Backbone, 1=Jump)
            jump_count = torch.sum(data.edge_attr[:, 2]).item() // 2
            metrics['jump_pct'][label].append(jump_count / num_edges)
            
            # Backbone pause times (where TypeFlag == 0)
            backbone_mask = data.edge_attr[:, 2] == 0
            if torch.any(backbone_mask):
                avg_p = torch.mean(data.edge_attr[backbone_mask, 0]).item()
                metrics['avg_pause'][label].append(avg_p)
        
        # --- 4. NetworkX Topology ---
        nx_graph = to_networkx(data, to_undirected=True)
        metrics['clustering'][label].append(nx.average_clustering(nx_graph))
        metrics['density'][label].append(nx.density(nx_graph))
            
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1} / {len(graphs)} windows...")

    # --- Plotting 4x2 Grid ---
    fig, axes = plt.subplots(4, 2, figsize=(18, 22))
    axes = axes.flatten() 
    
    keys = ['node_count', 'edge_count', 'avg_burst', 'lone_key_pct', 
            'jump_pct', 'avg_pause', 'clustering', 'density']
    
    titles = ['1. Burst (Node) Count', '2. Total Edge Count',
              '3. Avg Keys per Burst', '4. Lone Key % (Fragmented Typing)',
              '5. Similarity Jump %', '6. Avg Pause Magnitude (Log)',
              '7. Clustering Coeff', '8. Graph Density']

    for idx, m_key in enumerate(keys):
        ax = axes[idx]
        for label in [0, 1, 2]:
            vals = metrics[m_key][label]
            if not vals: continue
            
            ax.hist(vals, bins=30, alpha=0.55, label=class_names[label], 
                    color=colors[label], density=True)
            
            mean_val = np.mean(vals)
            ax.axvline(mean_val, color=colors[label], linestyle='--', linewidth=2)
            
        ax.set_title(titles[idx], fontsize=14, fontweight='bold')
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(axis='y', alpha=0.2)
        
    plt.tight_layout()
    plt.show()

