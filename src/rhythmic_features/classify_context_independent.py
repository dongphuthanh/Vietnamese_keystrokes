"""Rhythmic-feature classifier, context-independent (CIE) (Figs. 1-2, Rhythmic rows).

XGBoost on the 107 rhythmic features: top-50% features by mutual information, then a
randomized search followed by a local grid search. Folds: one question pair ({1,4}, {2,5}, {3,6}) held out per fold.
Scenarios S1-S4 are the paper's M2-M5 (S3 trains on B, P, T and FT).

Input:  output/merged_data.csv from extract_rhythmic_features.py
Output: output/xgb_window_based/<scenario>_percentage_report.csv and <scenario>_cm.png
"""

import os
import warnings
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore")

# ----------------------------
# 1. CONFIGURATION & MAPPING
# ----------------------------
LABEL_MAP = {1: 'B', 2: 'P', 3: 'T', 4: 'F_P', 5: 'F_T'}
GLOBAL_LABELS = [1, 2, 3, 4, 5]
DISPLAY_ORDER = ['B', 'P', 'T', 'F_P', 'F_T']

RANDOM_PARAM_SPACE = {
    "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
    "n_estimators": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
}

SCENARIOS = [
    {"name": "S1_Train13", "train_classes": [1, 3], "attack_classes": [2, 4, 5]},
    {"name": "S2_Train123", "train_classes": [1, 2, 3], "attack_classes": [4, 5]},
    {"name": "S3_Train1235", "train_classes": [1, 2, 3, 5], "attack_classes": [4]},
    {"name": "S4_TrainAll", "train_classes": [1, 2, 3, 4, 5], "attack_classes": []},
]

# Window-based folds
# Window-based folds matching actual CSV window labels
WINDOW_FOLDS = [
    {
        "train_windows": [
            "Questions_S1_Q1_4", "Questions_S1_Q2_5",
            "Questions_S2_Q1_4", "Questions_S2_Q2_5",
            "Questions_S3_Q1_4", "Questions_S3_Q2_5",
        ],
        "test_windows": [
            "Questions_S1_Q3_6",
            "Questions_S2_Q3_6",
            "Questions_S3_Q3_6",
        ],
    },
    {
        "train_windows": [
            "Questions_S1_Q1_4", "Questions_S1_Q3_6",
            "Questions_S2_Q1_4", "Questions_S2_Q3_6",
            "Questions_S3_Q1_4", "Questions_S3_Q3_6",
        ],
        "test_windows": [
            "Questions_S1_Q2_5",
            "Questions_S2_Q2_5",
            "Questions_S3_Q2_5",
        ],
    },
    {
        "train_windows": [
            "Questions_S1_Q2_5", "Questions_S1_Q3_6",
            "Questions_S2_Q2_5", "Questions_S2_Q3_6",
            "Questions_S3_Q2_5", "Questions_S3_Q3_6",
        ],
        "test_windows": [
            "Questions_S1_Q1_4",
            "Questions_S2_Q1_4",
            "Questions_S3_Q1_4",
        ],
    },
]

# ----------------------------
# 2. TWO-STEP TUNING LOGIC
# ----------------------------
def find_best_params_two_step(X_train, y_train, seed):
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    clf = XGBClassifier(
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=-1
    )

    rs = RandomizedSearchCV(
        estimator=clf,
        param_distributions=RANDOM_PARAM_SPACE,
        n_iter=10,
        scoring="accuracy",
        cv=3,
        random_state=seed,
        n_jobs=-1
    )

    rs.fit(X_train, y_enc)
    best_rs = rs.best_params_

    grid_space = {
        "max_depth": sorted({
            max(3, best_rs["max_depth"] - 1),
            best_rs["max_depth"],
            best_rs["max_depth"] + 1
        }),
        "n_estimators": sorted({
            max(50, best_rs["n_estimators"] - 50),
            best_rs["n_estimators"],
            best_rs["n_estimators"] + 50
        }),
        "learning_rate": sorted({
            max(0.01, best_rs["learning_rate"] - 0.02),
            best_rs["learning_rate"],
            best_rs["learning_rate"] + 0.02
        }),
    }

    gs = GridSearchCV(
        estimator=XGBClassifier(
            eval_metric="mlogloss",
            subsample=best_rs["subsample"],
            colsample_bytree=best_rs["colsample_bytree"],
            random_state=seed,
            n_jobs=-1
        ),
        param_grid=grid_space,
        scoring="accuracy",
        cv=2,
        n_jobs=-1
    )

    gs.fit(X_train, y_enc)

    best_final = gs.best_params_
    best_final["subsample"] = best_rs["subsample"]
    best_final["colsample_bytree"] = best_rs["colsample_bytree"]

    return best_final


# ----------------------------
# 3. VISUALIZATION
# ----------------------------
def save_confusion_matrix_image(pct_cm, scenario, fold_accs, out_dir):
    labels = list(pct_cm.index)
    data = pct_cm.values
    n = len(labels)

    color_img = np.ones((n, n, 3), dtype=float) * 0.96
    attack_classes_str = [LABEL_MAP[c] for c in scenario["attack_classes"]]

    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            intensity = data[i, j] / 100.0

            if true_label == pred_label:
                color_img[i, j] = [
                    0.6 - 0.3 * intensity,
                    0.95,
                    0.6 - 0.3 * intensity
                ]

            elif true_label in attack_classes_str and pred_label == 'B':
                if intensity > 0:
                    color_img[i, j] = [
                        1.0,
                        1.0 - 0.8 * intensity,
                        1.0 - 0.8 * intensity
                    ]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(color_img, aspect="equal")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_yticklabels(labels, fontweight="bold")

    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")

    avg_acc = np.mean(fold_accs) * 100 if fold_accs else 0

    ax.set_title(
        f"XGB Window-Based {scenario['name']}\nBonafide Accuracy: {avg_acc:.2f}%",
        fontweight="bold",
        pad=12
    )

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{data[i, j]:.1f}%",
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=11
            )

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{scenario['name']}_cm.png"), dpi=300)
    plt.close()


# ----------------------------
# 4. EXPERIMENT RUNNER
# ----------------------------
def run_experiment(csv_path, out_dir):
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    df["label"] = (
        pd.to_numeric(
            df["section"].str.extract(r"(\d+)")[0],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Ensure window is numeric
    df["window"] = df["window"].astype(str).str.strip()
    df = df.dropna(subset=["window"])

    for sc in SCENARIOS:
        print(f"\nProcessing Scenario: {sc['name']} (Window-Based)")

        sc_df = df[df["label"].isin(GLOBAL_LABELS)].copy()

        combined_cm = pd.DataFrame(
            0,
            index=GLOBAL_LABELS,
            columns=GLOBAL_LABELS
        )

        fold_accs = []

        for fold_id, fold in enumerate(WINDOW_FOLDS, start=1):
            print(
                f"  Fold {fold_id}: "
                f"Train windows {fold['train_windows']} | "
                f"Test windows {fold['test_windows']}"
            )

            train_set = sc_df[
                sc_df["window"].isin(fold["train_windows"]) &
                sc_df["label"].isin(sc["train_classes"])
            ].copy()

            test_set = sc_df[
                sc_df["window"].isin(fold["test_windows"])
            ].copy()

            if train_set.empty or test_set.empty:
                print("    Skipping empty fold.")
                continue

            drop_cols = [
                "user_id",
                "window",
                "section",
                "label",
                "session"
            ]

            X_train = train_set.drop(columns=drop_cols)
            y_train = train_set["label"]

            X_test = test_set.drop(columns=drop_cols)
            y_test = test_set["label"]

            mi = mutual_info_classif(X_train, y_train, random_state=seed)

            n_selected = max(1, int(len(mi) * 0.5))
            sel_feats = X_train.columns[np.argsort(mi)[-n_selected:]].tolist()

            best_params = find_best_params_two_step(
                X_train[sel_feats],
                y_train,
                seed
            )

            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train)

            model = XGBClassifier(
                **best_params,
                eval_metric="mlogloss",
                random_state=seed,
                n_jobs=-1
            )

            model.fit(X_train[sel_feats], y_train_enc)

            y_pred_enc = model.predict(X_test[sel_feats])
            y_pred = le.inverse_transform(y_pred_enc)

            fold_cm = pd.DataFrame(
                confusion_matrix(y_test, y_pred, labels=GLOBAL_LABELS),
                index=GLOBAL_LABELS,
                columns=GLOBAL_LABELS
            )

            combined_cm += fold_cm

            bon_mask = y_test.isin(sc["train_classes"])

            if bon_mask.sum() > 0:
                fold_accs.append(
                    accuracy_score(y_test[bon_mask], y_pred[bon_mask])
                )

        row_sums = combined_cm.sum(axis=1)
        pct_cm = combined_cm.div(row_sums, axis=0).fillna(0) * 100

        report_cm = (
            pct_cm
            .rename(index=LABEL_MAP, columns=LABEL_MAP)
            .reindex(index=DISPLAY_ORDER, columns=DISPLAY_ORDER)
            .fillna(0)
        )

        report_cm.to_csv(
            os.path.join(out_dir, f"{sc['name']}_percentage_report.csv")
        )

        save_confusion_matrix_image(
            report_cm,
            sc,
            fold_accs,
            out_dir
        )


if __name__ == "__main__":
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    csv_path = os.path.join(OUTPUT_DIR, "merged_data.csv")
    if os.path.exists(csv_path):
        run_experiment(csv_path, os.path.join(OUTPUT_DIR, "xgb_window_based"))
    else:
        print(f"Error: {csv_path} not found. Run extract_rhythmic_features.py first.")