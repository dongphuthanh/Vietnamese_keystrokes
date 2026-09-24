"""
Central configuration for the Vietnamese Keystroke pipeline.
By default the data is read from <repo>/dataset/Attack4; edit NORMAL_FOLDER and
ATTACK_FOLDER to use another location.
"""

from pathlib import Path

# ============================================================
# INPUT DATA PATHS  (default: <repo>/dataset/Attack4)
# ============================================================
REPO_ROOT     = Path(__file__).resolve().parents[2]
NORMAL_FOLDER = REPO_ROOT / "dataset" / "Attack4"
ATTACK_FOLDER = REPO_ROOT / "dataset" / "Attack4"

# ============================================================
# OUTPUT PATHS  (relative to this file's parent directory)
# ============================================================
BASE_DIR       = Path(__file__).parent
PKL_DIR        = BASE_DIR / "pkl"
CONTEXT_DIR    = BASE_DIR / "context_indep_datasets"   # split by cognitive level
USER_DIR       = BASE_DIR / "user_indep_datasets"      # split by user + cognitive level
RUNS_CONTEXT   = BASE_DIR / "runs_context_indep"
RUNS_USER      = BASE_DIR / "runs_user_indep"

# Context-indep PKLs: features extracted per (session, level-group {1,4}/{2,5}/{3,6})
NORMAL_PKL_CONTEXT = PKL_DIR / "full_context.pkl"
ATTACK_PKL_CONTEXT = PKL_DIR / "attack_context.pkl"

# User-indep PKLs: features extracted per session (all 6 levels combined)
NORMAL_PKL_USER = PKL_DIR / "full_user.pkl"
ATTACK_PKL_USER = PKL_DIR / "attack_user.pkl"

# Aliases kept for backward compatibility
NORMAL_PKL = NORMAL_PKL_CONTEXT
ATTACK_PKL = ATTACK_PKL_CONTEXT

# Path to the XGBoost + genetic-algorithm feature-selection classifier
XGB_SCRIPT = BASE_DIR / "xgb_ga_classifier.py"

# ============================================================
# FEATURE EXTRACTION
# ============================================================
TOP_BIGRAMS_N = 50       # number of most-frequent bigrams to keep

# ============================================================
# SCENARIOS
# source = "normal" or "attack", sessions = list of int
# Labels:  normal/s1=0(B)  normal/s2=1(P)  normal/s3=2(T)
#          attack/s2=3(F_P) attack/s3=4(F_T)
# ============================================================
SCENARIOS = {
    "M2": [("normal", [1, 3])],
    "M3": [("normal", [1, 2, 3])],
    "M4": [("normal", [1, 2, 3]), ("attack", [3])],
    "M5": [("normal", [1, 2, 3]), ("attack", [2, 3])],
}

# ============================================================
# CONTEXT-INDEPENDENT FOLDS
# Split by level group (group 1={1,4}, 2={2,5}, 3={3,6}).
# Fold i: train on train_groups, test on test_groups (all users).
# ============================================================
CONTEXT_FOLDS = [
    {"fold": 1, "train_groups": [2, 3], "test_groups": [1]},
    {"fold": 2, "train_groups": [1, 3], "test_groups": [2]},
    {"fold": 3, "train_groups": [1, 2], "test_groups": [3]},
]

# ============================================================
# USER-INDEPENDENT FOLDS
# KFold(n_splits=3, shuffle=True, random_state=42) on unique users.
# No cognitive-level filtering — all groups used in train and test.
# ============================================================

# ============================================================
# XGB RUNNER SETTINGS  (matches xgb_ga_classifier.py defaults)
# ============================================================
FEATURE_PERCENTAGE = 50
SEED               = 42
CV_SPLITS          = 5
POPULATION         = 20
GENERATIONS        = 10
CM_LABELS          = [0, 1, 2, 3, 4]   # fixed for all scenarios (test set has all 5 labels)

# ============================================================
# LABEL NAMES
# ============================================================
LABEL_NAMES = {0: "B", 1: "P", 2: "T", 3: "F_P", 4: "F_T"}
BONAFIDE_LABEL = 0
