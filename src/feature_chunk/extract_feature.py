import json
import numpy as np
import os
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

WIN_LENGTH = 200
STRIDE = 50

# --- NEW PARAMETERS ---
VOTING_MODE = 'soft'  
SCENARIO = 'M5'       # CHANGE THIS TO 'M2', 'M3', 'M4', or 'M5'

# Define what classes the model is allowed to see during training
SCENARIO_TRAIN_CLASSES = {
    'M2': [0, 2],          # B, T
    'M3': [0, 1, 2],       # B, P, T
    'M4': [0, 1, 2, 4],    # B, P, T, F_T
    'M5': [0, 1, 2, 3, 4]  # B, P, T, F_P, F_T
}

# ---------------------------------------------------------
# 1. MODIFIED WINDOW GENERATOR (Fixed IndexError)
# ---------------------------------------------------------
def make_windows(filepath, X, Y, ID, user_id, win_length, stride):
    with open(filepath, "r", encoding="utf8") as f:
        data = json.load(f)
    _keystrokes = data.get("keystrokes", [])
    
    is_attack = filepath.endswith("attack.json")

    # 0: B, 1: P, 2: T, 3: F_P, 4: F_T
    sessions = {0: [], 1: [], 2: [], 3: [], 4: []}
    
    # FIX: Loop to len - 1 to safely peek ahead at i+1
    for i in range(len(_keystrokes) - 1):
        if _keystrokes[i]["event"] == "keydown":
            session_id = _keystrokes[i]["session"]
            
            # Safely capture the time to the next recorded event
            _keystrokes[i]["time_to_next"] = np.clip(_keystrokes[i+1]["timestamp"] - _keystrokes[i]["timestamp"], 1, 2000)
            
            if not is_attack:
                if session_id == 1: sessions[0].append(_keystrokes[i])
                elif session_id == 2: sessions[1].append(_keystrokes[i])
                elif session_id == 3: sessions[2].append(_keystrokes[i])
            else:
                if session_id == 2: sessions[3].append(_keystrokes[i])
                elif session_id == 3: sessions[4].append(_keystrokes[i])

    for label, keys in sessions.items():
        if len(keys) == 0: continue
        
        for i in range(1, len(keys) - win_length, stride):
            window = []
            for j in range(i, i + win_length):
                flight_time = keys[j]["timestamp"] - keys[j - 1]["timestamp"]
                flight_time = np.clip(flight_time, 1, 5000)
                
                # Fetch time_to_next safely with a fallback
                time_to_next = keys[j].get("time_to_next", 100) 
                
                is_space = 1 if (keys[j - 1]["key"] == " ") else 0
                is_backspace = 1 if (keys[j]["key"] == "Backspace") else 0
                
                window.append([
                    flight_time,
                    time_to_next,
                    is_space,
                    is_backspace
                ])

            X.append(window)
            Y.append(label)
            ID.append(user_id)

# ---------------------------------------------------------
# 2. FEATURE EXTRACTOR
# ---------------------------------------------------------
def extract_xgboost_features(X_windows):
    """
    Transforms raw windows (N, 200, 4) into statistical features (N, 13).
    """
    flight_times = X_windows[:, :, 0]
    time_to_next = X_windows[:, :, 1]
    spaces = X_windows[:, :, 2]
    backspaces = X_windows[:, :, 3]

    ft_mean = np.mean(flight_times, axis=1)
    ft_median = np.median(flight_times, axis=1)
    ft_std = np.std(flight_times, axis=1)
    ft_max = np.max(flight_times, axis=1)
    ft_90th = np.percentile(flight_times, 90, axis=1)

    tn_mean = np.mean(time_to_next, axis=1)
    tn_median = np.median(time_to_next, axis=1)
    tn_std = np.std(time_to_next, axis=1)
    tn_max = np.max(time_to_next, axis=1)

    pause_counts = np.sum(flight_times > 1000, axis=1)
    pause_time_total = np.sum(np.where(flight_times > 1000, flight_times, 0), axis=1)

    word_counts = np.sum(spaces, axis=1)
    backspace_counts = np.sum(backspaces, axis=1)

    win_len = flight_times.shape[1] # Usually 200
    p1 = np.sum((flight_times >= 0) & (flight_times < 150), axis=1) / win_len
    p2 = np.sum((flight_times >= 150) & (flight_times < 500), axis=1) / win_len
    p3 = np.sum((flight_times >= 500) & (flight_times < 1000), axis=1) / win_len
    p4 = np.sum((flight_times >= 1000), axis=1) / win_len
    
    # Stack probabilities into a matrix: shape (num_windows, 4)
    probs = np.column_stack((p1, p2, p3, p4))
    
    # Step 3: Prevent mathematical errors
    # log2(0) is undefined and will crash Python. We add a microscopic number to 0.
    epsilon = 1e-9
    probs = np.clip(probs, epsilon, 1.0)
    
    # Step 4: Calculate Shannon Entropy: -sum(p * log2(p))
    pause_entropy = -np.sum(probs * np.log2(probs), axis=1)

    X_features = np.column_stack((
        ft_mean, ft_median, ft_std,
        tn_mean, tn_std,
        pause_counts, pause_time_total,
        word_counts, backspace_counts, pause_entropy
    ))
    return X_features

# ---------------------------------------------------------
# 3. DATA LOAD & EXTRACTION
# ---------------------------------------------------------
X, Y, ID = [], [], []

folder_path = "../../dataset/Attack4" # Update to your Attack4 folder
user_folders = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
user2id = {user: i for i, user in enumerate(user_folders)}

print("Generating windows...")
for user_folder in user_folders:
    user_id = user2id[user_folder]
    user_path = os.path.join(folder_path, user_folder)
    
    for root, _, files in os.walk(user_path):
        for filename in sorted(files):
            if filename.endswith(".json"):
                filepath = os.path.join(root, filename)
                make_windows(filepath, X, Y, ID, user_id, WIN_LENGTH, STRIDE)

X_raw = np.array(X)
Y = np.array(Y)
ID = np.array(ID)

print(f"Raw Dataset Loaded -> X: {X_raw.shape} | Y: {Y.shape} | ID: {ID.shape}")

print("Extracting XGBoost Features...")
X_flat = extract_xgboost_features(X_raw)
print(f"Features Extracted -> X_flat: {X_flat.shape}")

# ---------------------------------------------------------
# 4. TRAINING LOOP WITH SCENARIO FILTERING
# ---------------------------------------------------------
n_splits = 3
gkf = GroupKFold(n_splits=n_splits)

fold_accuracies = []
all_true_labels = []
all_pred_labels = []

allowed_train_classes = SCENARIO_TRAIN_CLASSES[SCENARIO]
print(f"\nRunning Scenario {SCENARIO}: Training on classes {allowed_train_classes}, Testing on ALL 5 classes.")

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_flat, Y, groups=ID)):
    print(f"\n--- Fold {fold + 1}/{n_splits} ---")
    
    X_train_fold, X_test_fold = X_flat[train_idx].copy(), X_flat[test_idx].copy()
    Y_train_fold, Y_test_fold = Y[train_idx], Y[test_idx]
    test_ids_fold = ID[test_idx]

    # --- THE SCENARIO FILTER ---
    train_mask = np.isin(Y_train_fold, allowed_train_classes)
    X_train_fold = X_train_fold[train_mask]
    Y_train_fold = Y_train_fold[train_mask]

    # NOTE: RobustScaler is removed because XGBoost (and tree models in general) 
    # are entirely scale-invariant. Scaling them wastes computation time.

    # --- INITIALIZE AND TRAIN XGBOOST ---
    # We use relatively shallow trees to prevent overfitting the statistical noise
    encoder = LabelEncoder()
    Y_train_encoded = encoder.fit_transform(Y_train_fold)

    # --- PREPARE SAFE EVALUATION SET ---
    eval_mask = np.isin(Y_test_fold, allowed_train_classes)
    X_eval_safe = X_test_fold[eval_mask]
    
    # We must also encode the eval labels so they match the training labels!
    Y_eval_safe_encoded = encoder.transform(Y_test_fold[eval_mask])

    num_train_classes = len(encoder.classes_)
    safe_metric = 'logloss' if num_train_classes == 2 else 'mlogloss'
    # --- INITIALIZE XGBOOST (GPU ENABLED) ---
    xgb_model = XGBClassifier(
        tree_method='hist',
        device='cuda',
        n_estimators=1500,           # 1. Massive increase in total trees
        max_depth=8,                 # 2. Deeper trees for complex interactions
        learning_rate=0.02,          # 3. Slower learning to absorb nuances safely
        min_child_weight=1,          # 4. Allows leaves to map rare edge-cases
        subsample=0.8,               
        colsample_bytree=0.8,        
        random_state=42,
        eval_metric=safe_metric,
        early_stopping_rounds=70     # 5. THE SHIELD: Stops training if it overfits
    )

    print("\nTraining High-Capacity XGBoost Trees...")
    xgb_model.fit(
        X_train_fold, 
        Y_train_encoded,
        eval_set=[(X_train_fold, Y_train_encoded), (X_eval_safe, Y_eval_safe_encoded)],
        verbose=50 # Prints every 50 trees
    )

    # --- EVALUATION AND PADDING ---
    raw_probs = xgb_model.predict_proba(X_test_fold)
    
    # We map the probabilities back to the correct 5-column slots using encoder.classes_
    # For M2, encoder.classes_ is exactly [0, 2].
    padded_probs = np.zeros((len(X_test_fold), 5))
    for idx, true_class_label in enumerate(encoder.classes_):
        padded_probs[:, true_class_label] = raw_probs[:, idx]
    if VOTING_MODE == 'soft':
        store_vals = padded_probs
    else:
        # Get the index of the highest probability from our padded array
        store_vals = np.argmax(padded_probs, axis=1)
        
    session_results = {}
    for j in range(len(store_vals)):
        key = (test_ids_fold[j], Y_test_fold[j])
        if key not in session_results: 
            session_results[key] = []
        session_results[key].append(store_vals[j])

    # --- AGGREGATE SESSION VOTES ---
    fold_preds = []
    fold_labels = []
    for (uid, true_lab), vals_list in session_results.items():
        if VOTING_MODE == 'soft':
            mean_probs = np.mean(vals_list, axis=0)
            final_prediction = np.argmax(mean_probs)
        else:
            final_prediction = Counter(vals_list).most_common(1)[0][0]
            
        fold_preds.append(final_prediction)
        fold_labels.append(true_lab)
        all_pred_labels.append(final_prediction)
        all_true_labels.append(true_lab)

    fold_acc = np.mean(np.array(fold_preds) == np.array(fold_labels))
    fold_accuracies.append(fold_acc)
    print(f"Fold {fold+1} Session Accuracy: {fold_acc:.4f}")

# --- FINAL SUMMARY ---
print("-" * 30)
print(f"Scenario {SCENARIO} Average CV Accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
print("-" * 30)

cm = confusion_matrix(all_true_labels, all_pred_labels, labels=[0, 1, 2, 3, 4])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["B", "P", "T", "F_P", "F_T"])
disp.plot(cmap=plt.cm.Purples)
plt.title(f"{SCENARIO} Confusion Matrix ({VOTING_MODE.capitalize()} Voting)")
plt.show()