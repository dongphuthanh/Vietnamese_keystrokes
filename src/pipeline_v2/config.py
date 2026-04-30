"""
Central configuration for the Vietnamese Keystroke pipeline.
Edit NORMAL_FOLDER and ATTACK_FOLDER to point to your data.
"""

from pathlib import Path

# ============================================================
# INPUT DATA PATHS  (change these to match your machine)
# ============================================================
NORMAL_FOLDER = Path(r"C:\Users\ADMIN\Documents\Vietnamese_keystrokes\dataset\viet_preprocessed")
ATTACK_FOLDER = Path(r"C:\Users\ADMIN\Documents\Vietnamese_keystrokes\dataset\Attack4")

# ============================================================
# OUTPUT PATHS  (relative to this file's parent directory)
# ============================================================
BASE_DIR       = Path(__file__).parent
PKL_DIR        = BASE_DIR / "pkl"
CONTEXT_DIR    = BASE_DIR / "context_indep_datasets"   # split by cognitive level
USER_DIR       = BASE_DIR / "user_indep_datasets"      # split by user + cognitive level
RUNS_CONTEXT   = BASE_DIR / "runs_context_indep"
RUNS_USER      = BASE_DIR / "runs_user_indep"

NORMAL_PKL = PKL_DIR / "full.pkl"
ATTACK_PKL = PKL_DIR / "attack.pkl"

# Path to the XGB.py script (in src/)
XGB_SCRIPT = BASE_DIR.parent / "XGB.py"

# ============================================================
# FEATURE EXTRACTION
# ============================================================
TOP_BIGRAMS_N = 200       # number of most-frequent bigrams to keep

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
# Split by cognitive level (question_index = session.level)
# Fold i: train on train_levels, test on test_levels (all users)
# ============================================================
CONTEXT_FOLDS = [
    {"fold": 1, "train_levels": [2, 3, 5, 6], "test_levels": [1, 4]},
    {"fold": 2, "train_levels": [1, 3, 4, 6], "test_levels": [2, 5]},
    {"fold": 3, "train_levels": [1, 2, 4, 5], "test_levels": [3, 6]},
]

# ============================================================
# USER-INDEPENDENT FOLDS
# Fixed user split (30 train / 15 test), then cognitive level
# varies across folds (same as context folds above).
# ============================================================
N_TEST_USERS = 15

USER_FOLDS = [
    {"fold": 1, "train_levels": [2, 3, 5, 6], "test_levels": [1, 4]},
    {"fold": 2, "train_levels": [1, 3, 4, 6], "test_levels": [2, 5]},
    {"fold": 3, "train_levels": [1, 2, 4, 5], "test_levels": [3, 6]},
]

# ============================================================
# XGB RUNNER SETTINGS  (matches src/XGB.py defaults)
# ============================================================
FEATURE_PERCENTAGE = 50
SEED               = 42
CV_SPLITS          = 5
POPULATION         = 50
GENERATIONS        = 10
CM_LABELS          = [0, 1, 2, 3, 4]   # fixed for all scenarios (test set has all 5 labels)

# ============================================================
# LABEL NAMES
# ============================================================
LABEL_NAMES = {0: "B", 1: "P", 2: "T", 3: "F_P", 4: "F_T"}
BONAFIDE_LABEL = 0
