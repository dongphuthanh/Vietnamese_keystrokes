"""Rhythmic-feature classifier, user-independent (UIE) (Figs. 1-2, Rhythmic rows).

XGBoost on the 107 rhythmic features: top-50% features by mutual information, then a
randomized search followed by a local grid search. Folds: 3-fold KFold over users (shuffle, seed 42).
Scenarios S1-S4 are the paper's M2-M5 (S3 trains on B, P, T and FT).

Input:  output/merged_data.csv from extract_rhythmic_features.py
Output: output/xgb_user_kfold/<scenario>_percentage_report.csv and <scenario>_cm.png
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
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore")

# ----------------------------
# 1. CONFIGURATION & MAPPING
# ----------------------------
LABEL_MAP = {1: 'B', 2: 'P', 3: 'T', 4: 'F_P', 5: 'F_T'}
GLOBAL_LABELS = [1, 2, 3, 4, 5]
DISPLAY_ORDER = ['B', 'P', 'T', 'F_P', 'F_T']

RANDOM_PARAM_SPACE = {
    "max_depth": [3, 4, 5, 6, 7, 8, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "n_estimators": [100, 200, 300, 400],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

SCENARIOS = [
    {"name": "S1_Train13", "train_classes": [1, 3], "attack_classes": [2, 4, 5]},
    {"name": "S2_Train123", "train_classes": [1, 2, 3], "attack_classes": [4, 5]},
    {"name": "S3_Train1235", "train_classes": [1, 2, 3, 5], "attack_classes": [4]},
    {"name": "S4_TrainAll", "train_classes": [1, 2, 3, 4, 5], "attack_classes": []},
]


# ----------------------------
# 2. TUNING LOGIC
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
        n_iter=8,
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
        })
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
# 3. SAVE IMAGE
# ----------------------------
def save_confusion_matrix_image(pct_cm, scenario, fold_accs, out_dir):
    labels = list(pct_cm.index)
    data = pct_cm.values
    n = len(labels)

    color_img = np.ones((n, n, 3), dtype=float) * 0.96

    attack_classes_str = [LABEL_MAP[c] for c in scenario["attack_classes"]]

    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            val = data[i, j]

            if true_label == pred_label:
                intensity = val / 100.0
                color_img[i, j] = [
                    0.6 - 0.3 * intensity,
                    0.95,
                    0.6 - 0.3 * intensity
                ]

            elif true_label in attack_classes_str and pred_label == 'B':
                if val > 0:
                    intensity = val / 100.0
                    color_img[i, j] = [
                        1.0,
                        1.0 - 0.7 * intensity,
                        1.0 - 0.7 * intensity
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
        f"{scenario['name']} (Percentages)\nBonafide Accuracy: {avg_acc:.2f}%",
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
    n_splits = 3

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    df["label"] = pd.to_numeric(
        df["section"].str.extract(r"(\d+)")[0],
        errors="coerce"
    ).fillna(0).astype(int)

    for sc in SCENARIOS:
        print(f"Processing Scenario: {sc['name']}")

        sc_df = df[df["label"].isin(GLOBAL_LABELS)].copy()

        unique_users = np.unique(sc_df["user_id"])

        if len(unique_users) < 2:
            continue

        actual_splits = min(n_splits, len(unique_users))

        kf = KFold(
            n_splits=actual_splits,
            shuffle=True,
            random_state=seed
        )

        combined_cm = pd.DataFrame(
            0,
            index=GLOBAL_LABELS,
            columns=GLOBAL_LABELS
        )

        fold_accs = []

        for fold, (train_user_idx, test_user_idx) in enumerate(kf.split(unique_users), start=1):
            train_users = unique_users[train_user_idx]
            test_users = unique_users[test_user_idx]

            train_set = sc_df[
                sc_df["user_id"].isin(train_users)
                & sc_df["label"].isin(sc["train_classes"])
            ]

            test_set = sc_df[
                sc_df["user_id"].isin(test_users)
            ]

            if train_set.empty or test_set.empty:
                continue

            X_train = train_set.drop(
                columns=["user_id", "window", "section", "label", "session"]
            )

            y_train = train_set["label"]

            X_test = test_set.drop(
                columns=["user_id", "window", "section", "label", "session"]
            )

            y_test = test_set["label"]

            mi = mutual_info_classif(
                X_train,
                y_train,
                random_state=seed
            )

            sel_feats = X_train.columns[
                np.argsort(mi)[-int(len(mi) * 0.5):]
            ].tolist()

            params = find_best_params_two_step(
                X_train[sel_feats],
                y_train,
                seed
            )

            le = LabelEncoder()

            model = XGBClassifier(
                **params,
                eval_metric="mlogloss",
                random_state=seed
            )

            model.fit(
                X_train[sel_feats],
                le.fit_transform(y_train)
            )

            y_pred = le.inverse_transform(
                model.predict(X_test[sel_feats])
            )

            combined_cm += pd.DataFrame(
                confusion_matrix(
                    y_test,
                    y_pred,
                    labels=GLOBAL_LABELS
                ),
                index=GLOBAL_LABELS,
                columns=GLOBAL_LABELS
            )

            bon_mask = y_test.isin(sc["train_classes"])

            if bon_mask.any():
                fold_accs.append(
                    accuracy_score(
                        y_test[bon_mask],
                        y_pred[bon_mask]
                    )
                )

        row_sums = combined_cm.sum(axis=1)
        pct_cm = combined_cm.div(row_sums, axis=0).fillna(0) * 100

        pct_cm = pct_cm.rename(
            index=LABEL_MAP,
            columns=LABEL_MAP
        ).reindex(
            index=DISPLAY_ORDER,
            columns=DISPLAY_ORDER
        ).fillna(0)

        pct_cm.to_csv(
            os.path.join(out_dir, f"{sc['name']}_percentage_report.csv")
        )

        save_confusion_matrix_image(
            pct_cm,
            sc,
            fold_accs,
            out_dir
        )


if __name__ == "__main__":
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    csv_path = os.path.join(OUTPUT_DIR, "merged_data.csv")
    if os.path.exists(csv_path):
        run_experiment(csv_path, os.path.join(OUTPUT_DIR, "xgb_user_kfold"))
    else:
        print(f"Error: {csv_path} not found. Run extract_rhythmic_features.py first.")