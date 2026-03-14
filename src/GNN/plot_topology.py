import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

# Import your model architecture and data generators
from cnn_gnn import HierarchicalKeystrokeModel
from cnn_gen_data import make_windows

# ---------------------------------------------------------
# 1. HELPER FUNCTIONS
# ---------------------------------------------------------

def group_into_sessions(X_data, Y_data, ID_data):
    """Groups isolated windows back into their chronological sessions."""
    sessions = {}
    for i in range(len(X_data)):
        key = (ID_data[i], Y_data[i])
        if key not in sessions:
            sessions[key] = []
        sessions[key].append(X_data[i])
    
    session_list = []
    for k, v in sessions.items():
        session_list.append((np.array(v), k[1])) # (Windows_Array, Session_Label)
    return session_list

def plot_session_topology(window_features, label_name, similarity_threshold=0.8):
    """Takes the CNN 256D embeddings for a single session and draws the graph topology."""
    num_nodes = window_features.size(0)
    
    sim_matrix = F.cosine_similarity(
        window_features.unsqueeze(1), 
        window_features.unsqueeze(0), 
        dim=-1
    ).cpu().detach().numpy()

    G = nx.Graph()
    for i in range(num_nodes):
        G.add_node(i)

    chrono_edges = []
    sim_edges = []
    
    for i in range(num_nodes - 1):
        G.add_edge(i, i + 1)
        chrono_edges.append((i, i + 1))

    for i in range(num_nodes):
        for j in range(i + 2, num_nodes): 
            if sim_matrix[i, j] > similarity_threshold:
                G.add_edge(i, j)
                sim_edges.append((i, j))

    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, seed=42, iterations=15)
    
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=400, edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    
    nx.draw_networkx_edges(G, pos, edgelist=chrono_edges, edge_color='black', width=2.0)
    nx.draw_networkx_edges(G, pos, edgelist=sim_edges, edge_color='red', style='dashed', width=1.5, alpha=0.6)
    
    plt.title(f"Graph Topology for {label_name.capitalize()} Session\n({len(sim_edges)} Similarity Edges > {similarity_threshold})", fontsize=14)
    plt.axis('off')
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='Chronological Flow'),
        Line2D([0], [0], color='red', lw=1.5, linestyle='dashed', label=f'Similarity > {similarity_threshold}')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# 2. MINI DATA LOADER (Loads just 1 user for speed)
# ---------------------------------------------------------

WIN_LENGTH = 200
STRIDE = 50
folder_path = "../dataset/viet_preprocessed"

X, Y, ID = [], [], []

user_folders = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])

# JUST LOAD THE FIRST USER!
user_folder = user_folders[0] 
user_path = os.path.join(folder_path, user_folder)

print(f"Loading data for user: {user_folder}...")
for root, _, files in os.walk(user_path):
    for filename in sorted(files):
        if filename.endswith(".json"):
            filepath = os.path.join(root, filename)
            make_windows(filepath, X, Y, ID, 0, WIN_LENGTH, STRIDE) # user_id = 0

X = np.array(X)
Y = np.array(Y)
ID = np.array(ID)

# Group the windows into sessions
test_sessions = group_into_sessions(X, Y, ID)
print(f"Found {len(test_sessions)} sessions for this user.")

# ---------------------------------------------------------
# 3. INITIALIZE MODEL & PLOT
# ---------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_dim = X.shape[2] 
num_classes = len(np.unique(Y))

# 1. Initialize the empty shell
model = HierarchicalKeystrokeModel(feature_dim=feature_dim, num_classes=num_classes).to(device)

# 2. Inject the smart CNN brain! (Make sure this file exists)
# If you haven't saved weights yet, the plots will just look like random noise!
try:
    model.cnn.load_state_dict(torch.load("best_cnn_weights.pth"))
    print("Loaded pre-trained CNN weights!")
except FileNotFoundError:
    print("WARNING: 'best_cnn_weights.pth' not found. Plotting with random untrained weights.")

model.eval()

# 3. Generate the plots
class_names = {0: "Bonafide", 1: "Paraphrase", 2: "Transcribe"}
bonafide_plotted = False
paraphrase_plotted = False

with torch.no_grad():
    for xb, yb in test_sessions:
        label_int = int(yb)
        
        # Plot the first Bonafide session we find
        if label_int == 0 and not bonafide_plotted:
            print("Plotting Bonafide Session...")
            xb_tensor = torch.tensor(xb, dtype=torch.float32).to(device)
            features = model.cnn(xb_tensor, return_features=True)
            plot_session_topology(features, class_names[label_int])
            bonafide_plotted = True
            
        # Plot the first Paraphrase session we find
        elif label_int == 1 and not paraphrase_plotted:
            print("Plotting Paraphrase Session...")
            xb_tensor = torch.tensor(xb, dtype=torch.float32).to(device)
            features = model.cnn(xb_tensor, return_features=True)
            plot_session_topology(features, class_names[label_int])
            paraphrase_plotted = True
            
        if bonafide_plotted and paraphrase_plotted:
            break