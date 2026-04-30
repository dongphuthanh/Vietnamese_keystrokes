import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import os
import optuna
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler

# ---------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------
SCENARIO = 'M5'       
OPTUNA_TRIALS = 50     
OPTUNA_EPOCHS = 50    
FINAL_EPOCHS = 150

WIN_LENGTH = 100
STRIDE = 30

SCENARIO_TRAIN_CLASSES = {
    'M2': [0, 2],          
    'M3': [0, 1, 2],       
    'M4': [0, 1, 2, 4],    
    'M5': [0, 1, 2, 3, 4], 
    'M6': [0, 1]
}

# --- LEAVE-ONE-QUESTION-OUT (LOQO) GROUPS (3-Fold Setup) ---
QUESTION_GROUPS = [
    [1, 4],
    [2, 5],
    [3, 6]
]

# OPTIMIZATION TWEAKS
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.75, device=0)

# ---------------------------------------------------------
# 0. CUSTOM UTILITIES: TRIPLET MINER & CALLBACKS
# ---------------------------------------------------------
def batch_semi_hard_triplet_loss(labels, embeddings, margin=1.0):
    """Fully Vectorized Semi-Hard Triplet Loss (Zero CPU Overhead)"""
    dist_mat = torch.cdist(embeddings, embeddings, p=2)

    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    labels_ne = ~labels_eq
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    labels_eq = labels_eq & ~eye  

    # Find the HARDEST POSITIVE for each anchor
    hard_positives, _ = (dist_mat * labels_eq.float()).max(dim=1, keepdim=True)

    # Find the SEMI-HARD NEGATIVES
    neg_dist_mat = dist_mat.masked_fill(~labels_ne, float('inf'))
    semi_hard_mask = neg_dist_mat > hard_positives
    semi_hard_neg_dist = neg_dist_mat.masked_fill(~semi_hard_mask, float('inf'))
    semi_hard_negatives, _ = semi_hard_neg_dist.min(dim=1, keepdim=True)
    
    # Fallback to hard negative if no semi-hard exists
    hard_negatives, _ = neg_dist_mat.min(dim=1, keepdim=True)
    no_semi_hard_mask = torch.isinf(semi_hard_negatives)
    selected_negatives = torch.where(no_semi_hard_mask, hard_negatives, semi_hard_negatives)

    loss = F.relu(hard_positives - selected_negatives + margin)
    valid_losses = loss[loss > 0.0]
    
    return valid_losses.mean() if valid_losses.numel() > 0 else torch.tensor(0.0, device=embeddings.device, requires_grad=True)

class PeriodicLoggingCallback:
    def __init__(self, print_every=5, total_trials=100):
        self.print_every = print_every
        self.total_trials = total_trials

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        current_trial = trial.number + 1 
        if current_trial % self.print_every == 0:
            print(f"  -> [Optuna Progress] Trial {current_trial:03d}/{self.total_trials} "
                  f"| Current Acc: {trial.value:.4f} "
                  f"| Best Acc So Far: {study.best_value:.4f}")
            
class EarlyStoppingCallback:
    def __init__(self, patience: int):
        self.patience = patience

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        try:
            best_trial = study.best_trial
        except ValueError:
            return  
        trials_since_best = trial.number - best_trial.number
        if trials_since_best >= self.patience:
            print(f"\n[EARLY STOPPING TRIGGERED] No improvement in {self.patience} trials!")
            study.stop()
    __call__.experimental_disable_run_after_update = False

# ---------------------------------------------------------
# 1. METRIC LEARNING MODEL: EMBEDDING OUTPUT
# ---------------------------------------------------------
class TemporalCNN(nn.Module):
    def __init__(self, continuous_dim=1, hidden_dim=32, 
                 proj_dim=8, embedding_dim=8, 
                 dropout1=0.2, dropout2=0.2, dropout3=0.2, out_features=64):
        super().__init__()
        
        self.projector = nn.Linear(continuous_dim, proj_dim)
        self.embedding = nn.Embedding(num_embeddings=5, embedding_dim=embedding_dim) 
        conv_in_dim = proj_dim + embedding_dim
        
        self.conv1 = nn.Conv1d(in_channels=conv_in_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout1)
        
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout2)

        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout3 = nn.Dropout(dropout3)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
            
        self.embedding_head = nn.Linear(hidden_dim * 2, out_features)
        
    def forward(self, x):
        cont_feats = F.relu(self.projector(x[:, :, :-1]))
        embedded_keys = self.embedding(x[:, :, -1].long()) 
        
        x_combined = torch.cat([cont_feats, embedded_keys], dim=2).permute(0, 2, 1)  
        
        x = self.dropout1(F.relu(self.bn1(self.conv1(x_combined))))
        x = self.dropout2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout3(F.relu(self.bn3(self.conv3(x))))

        max_feats = self.max_pool(x).squeeze(-1)
        avg_feats = self.avg_pool(x).squeeze(-1)
        features = torch.cat([max_feats, avg_feats], dim=1)

        emb = self.embedding_head(features)
        return F.normalize(emb, p=2, dim=1)

# ---------------------------------------------------------
# 2. DATA GENERATORS
# ---------------------------------------------------------
def get_key_id(key_str):
    if len(key_str) == 1:
        if key_str.isalpha(): return 0
        elif key_str.isdigit(): return 1               
        elif key_str == ' ': return 2                               
    elif key_str == 'Backspace': return 3                                  
    return 4  

def make_windows(filepath, X, Y, ID, FILE_ID, Q_ID, user_id, file_id, win_length, stride):
    with open(filepath, "r", encoding="utf8") as f:
        data = json.load(f)
    _keystrokes = data.get("keystrokes", [])
    is_attack = filepath.endswith("attack.json")

    grouped_keys = {}
    for k in _keystrokes:
        if k["event"] == "keydown":
            q_idx_str = k.get("question_index")
            if not q_idx_str: continue 
            parts = str(q_idx_str).split('.')
            if len(parts) != 2: continue
            
            q_id, s_id = int(parts[1]), int(parts[0])
            label = None
            if not is_attack:
                if s_id == 1: label = 0    
                elif s_id == 2: label = 1  
                elif s_id == 3: label = 2  
            else:
                if s_id == 2: label = 3    
                elif s_id == 3: label = 4  

            if label is not None:
                group_key = (label, q_id)
                if group_key not in grouped_keys: grouped_keys[group_key] = []
                grouped_keys[group_key].append(k)

    for (label, q_id), keys in grouped_keys.items():
        if len(keys) == 0: continue
        for i in range(1, len(keys) - win_length, stride):
            window = []
            for j in range(i, i + win_length):
                dt = np.clip(keys[j]["timestamp"] - keys[j - 1]["timestamp"], 1, 5000)
                key_id = get_key_id(keys[j]["key"])
                window.append([np.log(dt), key_id])

            X.append(window)
            Y.append(label)
            ID.append(user_id)
            FILE_ID.append(file_id) 
            Q_ID.append(q_id) 

# ---------------------------------------------------------
# MAIN EXECUTION BLOCK (Required for num_workers on Windows)
# ---------------------------------------------------------
if __name__ == '__main__':
    # ---------------------------------------------------------
    # 3. DATA EXTRACTION
    # ---------------------------------------------------------
    X, Y, ID, FILE_ID, Q_ID = [], [], [], [], []

    folder_path = "../../dataset/Attack4"
    user_folders = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
    user2id = {user: i for i, user in enumerate(user_folders)}

    print("Generating standard context-agnostic windows...")
    file_counter = 0 

    for user_folder in user_folders:
        user_id = user2id[user_folder]
        user_path = os.path.join(folder_path, user_folder)
        for root, _, files in os.walk(user_path):
            for filename in sorted(files):
                if filename.endswith(".json"):
                    filepath = os.path.join(root, filename)
                    make_windows(filepath, X, Y, ID, FILE_ID, Q_ID, user_id, file_counter, WIN_LENGTH, STRIDE)
                    file_counter += 1

    X, Y, ID, FILE_ID, Q_ID = np.array(X), np.array(Y), np.array(ID), np.array(FILE_ID), np.array(Q_ID)
    NUM_CONTINUOUS_FEATURES = X.shape[2] - 1 

    print(f"Dataset Loaded -> X: {X.shape} | Y: {Y.shape} | Continuous Features Detected: {NUM_CONTINUOUS_FEATURES}")

    # ---------------------------------------------------------
    # 4. NESTED CROSS-CONTENT OPTUNA EVALUATION LOOP
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optuna.logging.set_verbosity(optuna.logging.WARNING) 
    n_splits = len(QUESTION_GROUPS)

    all_true_labels, all_pred_labels = [], []
    fold_cm_data = []

    allowed_train_classes = SCENARIO_TRAIN_CLASSES[SCENARIO]
    print(f"\nRunning Scenario {SCENARIO}: LOQO MULTI-CLASS METRIC LEARNING Pipeline")

    for fold in range(n_splits):
        test_questions = QUESTION_GROUPS[fold]
        train_questions = [q for i, group in enumerate(QUESTION_GROUPS) if i != fold for q in group]
        
        print(f"\n" + "="*50)
        print(f"OUTER FOLD {fold + 1}/{n_splits} - INITIALIZING OPTUNA EXPERIMENT")
        print(f"Train Questions: {train_questions} | Test Questions: {test_questions}")
        print("="*50)

        train_content_mask = np.isin(Q_ID, train_questions)
        test_content_mask = np.isin(Q_ID, test_questions)

        # ---------------------------------------------------------
        # INNER LOOP: OPTUNA TUNING (MAXIMIZE CENTROID ACCURACY)
        # ---------------------------------------------------------
        X_train_full = X[train_content_mask].copy()
        Y_train_full = Y[train_content_mask]
        FID_train_full = FILE_ID[train_content_mask]
        QID_train_full = Q_ID[train_content_mask]
        
        def objective(trial):
            lr = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
            hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
            proj_dim = trial.suggest_categorical('proj_dim', [4, 8, 16, 32])
            emb_dim = trial.suggest_categorical('embedding_dim', [4, 8, 16])
            drop1 = trial.suggest_float('dropout1', 0.1, 0.4)
            drop2 = trial.suggest_float('dropout2', 0.1, 0.4)
            drop3 = trial.suggest_float('dropout3', 0.1, 0.4)
            jitter_std = trial.suggest_float('jitter_std', 0.0, 0.1) 
            
            MARGIN_VAL = 0.4 # Locked Margin
            
            inner_gkf = GroupKFold(n_splits=4)
            inner_fold_scores = []
            
            for inner_train_idx, inner_val_idx in inner_gkf.split(X_train_full, Y_train_full, groups=QID_train_full):
                
                X_sub, Y_sub = X_train_full[inner_train_idx].copy(), Y_train_full[inner_train_idx]
                X_val, Y_val = X_train_full[inner_val_idx].copy(), Y_train_full[inner_val_idx]
                FID_val = FID_train_full[inner_val_idx]
                
                train_mask = np.isin(Y_sub, allowed_train_classes)
                X_sub, Y_sub = X_sub[train_mask], Y_sub[train_mask]
                
                scaler = RobustScaler()
                train_cont = X_sub[:, :, :NUM_CONTINUOUS_FEATURES].reshape(-1, NUM_CONTINUOUS_FEATURES)
                val_cont = X_val[:, :, :NUM_CONTINUOUS_FEATURES].reshape(-1, NUM_CONTINUOUS_FEATURES)
                
                X_sub[:, :, :NUM_CONTINUOUS_FEATURES] = scaler.fit_transform(train_cont).reshape(X_sub.shape[0], X_sub.shape[1], NUM_CONTINUOUS_FEATURES)
                X_val[:, :, :NUM_CONTINUOUS_FEATURES] = scaler.transform(val_cont).reshape(X_val.shape[0], X_val.shape[1], NUM_CONTINUOUS_FEATURES)
                
                train_ds = TensorDataset(torch.tensor(X_sub, dtype=torch.float32), torch.tensor(Y_sub, dtype=torch.long))
                val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.long))
                
                t_loader = DataLoader(train_ds, batch_size=128, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True)
                v_loader = DataLoader(val_ds, batch_size=128, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True)
                
                model = TemporalCNN(
                    continuous_dim=NUM_CONTINUOUS_FEATURES, hidden_dim=hidden_dim, 
                    proj_dim=proj_dim, embedding_dim=emb_dim,
                    dropout1=drop1, dropout2=drop2, dropout3=drop3, out_features=64
                ).to(device)
                
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
                
                for epoch in range(OPTUNA_EPOCHS):
                    model.train()
                    for xb, yb in t_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        optimizer.zero_grad(set_to_none=True)
                        
                        if jitter_std > 0.0:
                            noise_adder = torch.normal(mean=0.0, std=jitter_std, size=xb[:, :, :NUM_CONTINUOUS_FEATURES].shape).to(device)
                            xb[:, :, :NUM_CONTINUOUS_FEATURES] += torch.clamp(noise_adder, min=-(3 * jitter_std), max=(3 * jitter_std))
                            
                        embeddings = model(xb)
                        loss = batch_semi_hard_triplet_loss(yb, embeddings, margin=MARGIN_VAL)
                        loss.backward()
                        optimizer.step()
                        
                # --- EVALUATE INNER FOLD (5-CLASS NEAREST CENTROID) ---
                model.eval()
                class_centroids = []
                with torch.no_grad():
                    for class_idx in range(5):
                        class_embs = []
                        for xb, yb in t_loader:
                            mask = (yb == class_idx)
                            if mask.sum() > 0:
                                class_embs.append(model(xb.to(device)[mask]).cpu())
                        if len(class_embs) > 0:
                            centroid = torch.cat(class_embs).mean(dim=0)
                            centroid = F.normalize(centroid.unsqueeze(0), p=2, dim=1)
                        else:
                            centroid = torch.full((1, 64), float('inf'))
                        class_centroids.append(centroid)
                        
                all_centroids = torch.cat(class_centroids).to(device)
                
                inner_session_distances = {}
                current_idx = 0
                with torch.no_grad():
                    for xb, yb in v_loader:
                        test_embs = model(xb.to(device))
                        dists = torch.cdist(test_embs, all_centroids).cpu().numpy()
                        labels = yb.cpu().numpy()
                        batch_fids = FID_val[current_idx : current_idx + len(xb)]
                        current_idx += len(xb)

                        for j in range(len(dists)):
                            key = (batch_fids[j], labels[j]) 
                            if key not in inner_session_distances: inner_session_distances[key] = []
                            inner_session_distances[key].append(dists[j])
                            
                val_preds, val_labels = [], []
                for (fid, true_lab), dist_matrix in inner_session_distances.items():
                    mean_distances = np.mean(dist_matrix, axis=0)
                    val_preds.append(np.argmin(mean_distances))
                    val_labels.append(true_lab)
                    
                inner_fold_scores.append(accuracy_score(val_labels, val_preds))
                
            return np.mean(inner_fold_scores)

        study_name = f"context_scenario_{SCENARIO}_fold_{fold + 1}"
        study = optuna.create_study(
            study_name=study_name,
            storage="sqlite:///keystroke_optuna_metric.db",
            load_if_exists=True,
            direction="maximize" # SWAPPED TO MAXIMIZE FOR NEAREST CENTROID ACCURACY
        )
        
        study.optimize(
            objective, 
            n_trials=OPTUNA_TRIALS,
            callbacks=[
                EarlyStoppingCallback(patience=10),
                PeriodicLoggingCallback(print_every=5, total_trials=OPTUNA_TRIALS)
            ]
        )
        
        best_params = study.best_params
        BEST_MARGIN = 0.4
        
        print(f"Optuna Winning Params for Outer Fold {fold + 1}:")
        print(json.dumps(best_params, indent=4))
        
        # ---------------------------------------------------------
        # OUTER LOOP: FINAL TESTING WITH WINNING PARAMS
        # ---------------------------------------------------------
        print(f"--- OUTER FOLD {fold + 1} FINAL TEST RUN ---")
        
        X_train_final = X[train_content_mask].copy()
        Y_train_final = Y[train_content_mask]
        X_test_final = X[test_content_mask].copy()
        Y_test_final = Y[test_content_mask]
        FID_test = FILE_ID[test_content_mask]

        scenario_train_mask = np.isin(Y_train_final, allowed_train_classes)
        X_train_final = X_train_final[scenario_train_mask]
        Y_train_final = Y_train_final[scenario_train_mask]

        scaler = RobustScaler()
        train_cont = X_train_final[:, :, :NUM_CONTINUOUS_FEATURES].reshape(-1, NUM_CONTINUOUS_FEATURES)
        test_cont = X_test_final[:, :, :NUM_CONTINUOUS_FEATURES].reshape(-1, NUM_CONTINUOUS_FEATURES)
        
        X_train_final[:, :, :NUM_CONTINUOUS_FEATURES] = scaler.fit_transform(train_cont).reshape(X_train_final.shape[0], X_train_final.shape[1], NUM_CONTINUOUS_FEATURES)
        X_test_final[:, :, :NUM_CONTINUOUS_FEATURES] = scaler.transform(test_cont).reshape(X_test_final.shape[0], X_test_final.shape[1], NUM_CONTINUOUS_FEATURES)

        train_ds = TensorDataset(torch.tensor(X_train_final, dtype=torch.float32), torch.tensor(Y_train_final, dtype=torch.long))
        test_ds = TensorDataset(torch.tensor(X_test_final, dtype=torch.float32), torch.tensor(Y_test_final, dtype=torch.long))
        
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True)
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, pin_memory=True, num_workers=4, persistent_workers=True)

        model = TemporalCNN(
            continuous_dim=NUM_CONTINUOUS_FEATURES, hidden_dim=best_params['hidden_dim'], 
            proj_dim=best_params['proj_dim'], embedding_dim=best_params['embedding_dim'],
            dropout1=best_params['dropout1'], dropout2=best_params['dropout2'], dropout3=best_params['dropout3'], out_features=64
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINAL_EPOCHS)
        best_jitter = best_params.get('jitter_std', 0.0)

        for epoch in range(FINAL_EPOCHS):
            model.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                
                if best_jitter > 0.0:
                    noise_adder = torch.normal(mean=0.0, std=best_jitter, size=xb[:, :, :NUM_CONTINUOUS_FEATURES].shape).to(device)
                    xb[:, :, :NUM_CONTINUOUS_FEATURES] += torch.clamp(noise_adder, min=-(3 * best_jitter), max=(3 * best_jitter))
                    
                embeddings = model(xb)
                loss = batch_semi_hard_triplet_loss(yb, embeddings, margin=BEST_MARGIN)
                loss.backward()
                optimizer.step()
                scheduler.step()
                running_loss += loss.item() * xb.size(0)
                
            epoch_loss = running_loss / len(train_loader.dataset)
            if (epoch + 1) % 10 == 0 or (epoch + 1) == FINAL_EPOCHS:
                print(f"Final Train Epoch [{epoch + 1}/{FINAL_EPOCHS}], Loss: {epoch_loss:.4f}")

        # ---------------------------------------------------------
        # FINAL EVALUATION: 5-CLASS NEAREST CENTROID
        # ---------------------------------------------------------
        model.eval()
        
        class_centroids = []
        with torch.no_grad():
            for class_idx in range(5):
                class_embs = []
                for xb, yb in train_loader:
                    mask = (yb == class_idx)
                    if mask.sum() > 0:
                        class_embs.append(model(xb.to(device)[mask]).cpu())
                
                if len(class_embs) > 0:
                    centroid = torch.cat(class_embs).mean(dim=0)
                    centroid = F.normalize(centroid.unsqueeze(0), p=2, dim=1)
                else:
                    centroid = torch.zeros((1, 64)) 
                    
                class_centroids.append(centroid)
                
        all_centroids = torch.cat(class_centroids).to(device)

        session_distances = {}
        current_idx = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                test_embs = model(xb)
                
                dists = torch.cdist(test_embs, all_centroids).cpu().numpy()
                labels = yb.cpu().numpy()
                batch_fids = FID_test[current_idx : current_idx + len(xb)]
                current_idx += len(xb)

                for j in range(len(dists)):
                    key = (batch_fids[j], labels[j]) 
                    if key not in session_distances: session_distances[key] = []
                    session_distances[key].append(dists[j])

        fold_preds, fold_labels = [], []
        for (fid, true_lab), dist_matrix in session_distances.items():
            mean_distances = np.mean(dist_matrix, axis=0)
            final_prediction = np.argmin(mean_distances)
                
            fold_preds.append(final_prediction)
            fold_labels.append(true_lab)
            
            all_pred_labels.append(final_prediction)
            all_true_labels.append(true_lab)

        fold_acc = accuracy_score(fold_labels, fold_preds)
        fold_cm_data.append((fold_labels, fold_preds)) 

        print(f"-> Fold {fold+1} Session Accuracy: {fold_acc * 100:.2f}%")

    # ---------------------------------------------------------
    # 5. FINAL SUMMARY 
    # ---------------------------------------------------------
    print("=" * 40)
    print("METRIC LEARNING: 5-CLASS NEAREST CENTROID EVALUATION")
    print("=" * 40)

    global_acc = accuracy_score(all_true_labels, all_pred_labels)
    print(f"Global 5-Class Accuracy : {global_acc * 100:.2f}%\n")

    # ---------------------------------------------------------
    # 6. PLOT THREE LOQO CONFUSION MATRICES (1x3 Grid, Full 5x5)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6)) 
    axes = axes.flatten() 

    class_names = ["Bonafide", "Paraphrase", "Transcribe", "Fake_P", "Fake_T"]

    for i, (true_labs, pred_labs) in enumerate(fold_cm_data):
        cm = confusion_matrix(true_labs, pred_labs, labels=[0, 1, 2, 3, 4])
        
        ax = axes[i]
        cax = ax.matshow(cm, cmap=plt.cm.Purples, aspect='auto')
        
        ax.set_xticks(range(5))
        ax.set_xticklabels(class_names, fontsize=11, fontweight='bold', rotation=45)
        ax.set_yticks(range(5))
        ax.set_yticklabels(class_names, fontsize=11, fontweight='bold')
        ax.xaxis.set_ticks_position('bottom')
        
        for (r, c), val in np.ndenumerate(cm):
            color = "white" if val > (cm.max() / 2) else "black"
            ax.text(c, r, f"{val}", ha='center', va='center', color=color, fontsize=14)
        
        test_q = QUESTION_GROUPS[i][0]
        ax.set_title(f"Fold {i+1} (Tested blindly on Q{test_q})", pad=15, fontsize=14)

    plt.tight_layout()
    plt.savefig(f"Scenario_{SCENARIO}_LOQO_MetricLearning_5x5_CM.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Execution Complete. Matrices saved.")