#!/usr/bin/env python3
"""
Run GA-tuned XGBoost on a train/test CSV pair.

Expected columns in CSV:
- label (required)
- user_id, part (optional; will be dropped if present)
All other columns are treated as features.

Example:
  python run_xgb_ga.py --train /path/train.csv --test /path/test.csv --outdir ./runs/run1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import joblib
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class GAConfig:
    population: int = 50
    generations: int = 10
    cxpb: float = 0.5
    mutpb: float = 0.2
    indpb: float = 0.2        # per-gene mutation probability
    tournsize: int = 3
    cv_splits: int = 5
    n_jobs_cv: int = -1


@dataclass(frozen=True)
class ExperimentConfig:
    feature_percentage: float = 50.0
    random_state: int = 42
    ga: GAConfig = GAConfig()


# Hyperparameter search space (same as yours)
PARAM_SPACE: Dict[str, List[Any]] = {
    "max_depth": list(range(3, 11)),
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "n_estimators": list(range(50, 551, 50)),
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
}


# ----------------------------
# Utilities
# ----------------------------

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def load_xy(csv_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError(f"{csv_path} missing required column 'label'.")

    y = df["label"]
    drop_cols = [c for c in ["user_id", 'session','section', "label"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    if X.shape[1] == 0:
        raise ValueError(f"{csv_path} has no feature columns after dropping {drop_cols}.")
    return X, y


def select_features_mutual_info(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_percentage: float,
    random_state: int,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Returns (selected_feature_names, mi_ranking_df)
    """
    if not (0 < feature_percentage <= 100):
        raise ValueError("feature_percentage must be in (0, 100].")

    mi = mutual_info_classif(
        X_train,
        y_train,
        discrete_features=False,
        random_state=random_state,
    )
    mi_df = (
        pd.DataFrame({"feature": X_train.columns, "mi_score": mi})
        .sort_values("mi_score", ascending=False)
        .reset_index(drop=True)
    )

    k = max(1, int(len(mi_df) * (feature_percentage / 100.0)))
    selected = mi_df["feature"].iloc[:k].tolist()
    return selected, mi_df


# ----------------------------
# DEAP setup (safe / idempotent)
# ----------------------------

def ensure_deap_creators() -> None:
    """
    DEAP 'creator' is global and will error if you create the same classes twice.
    Make it idempotent so the module is reusable.
    """
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)


def custom_mutation(individual: creator.Individual, indpb: float):
    # layout: [max_depth, learning_rate, n_estimators, subsample, colsample_bytree]
    if random.random() < indpb:
        individual[0] = random.choice(PARAM_SPACE["max_depth"])
    if random.random() < indpb:
        individual[1] = random.choice(PARAM_SPACE["learning_rate"])
    if random.random() < indpb:
        individual[2] = random.choice(PARAM_SPACE["n_estimators"])
    if random.random() < indpb:
        individual[3] = random.choice(PARAM_SPACE["subsample"])
    if random.random() < indpb:
        individual[4] = random.choice(PARAM_SPACE["colsample_bytree"])
    return (individual,)


def xgb_genetic_algorithm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: ExperimentConfig,
) -> Tuple[Dict[str, Any], float]:
    """
    GA-based hyperparameter tuning for XGBClassifier using StratifiedKFold CV.
    Returns (best_params, best_cv_accuracy).
    """
    ensure_deap_creators()
    ga = cfg.ga

    skf = StratifiedKFold(
        n_splits=ga.cv_splits,
        shuffle=True,
        random_state=cfg.random_state,
    )

    def evaluate(individual):
        clf = XGBClassifier(
            max_depth=individual[0],
            learning_rate=individual[1],
            n_estimators=individual[2],
            subsample=individual[3],
            colsample_bytree=individual[4],
            eval_metric="mlogloss",
            random_state=cfg.random_state,
        )
        pipeline = Pipeline(
            [
                ("scaler", MinMaxScaler()),
                ("classifier", clf),
            ]
        )
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=skf,
            scoring="accuracy",
            n_jobs=ga.n_jobs_cv,
        )
        return (float(scores.mean()),)

    toolbox = base.Toolbox()

    toolbox.register("attr_max_depth", random.choice, PARAM_SPACE["max_depth"])
    toolbox.register("attr_learning_rate", random.choice, PARAM_SPACE["learning_rate"])
    toolbox.register("attr_n_estimators", random.choice, PARAM_SPACE["n_estimators"])
    toolbox.register("attr_subsample", random.choice, PARAM_SPACE["subsample"])
    toolbox.register("attr_colsample", random.choice, PARAM_SPACE["colsample_bytree"])

    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        (
            toolbox.attr_max_depth,
            toolbox.attr_learning_rate,
            toolbox.attr_n_estimators,
            toolbox.attr_subsample,
            toolbox.attr_colsample,
        ),
        n=1,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", custom_mutation, indpb=ga.indpb)
    toolbox.register("select", tools.selTournament, tournsize=ga.tournsize)

    population = toolbox.population(n=ga.population)

    print("Starting GA optimization for XGBoost...")
    start = time.time()
    result, _log = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=ga.cxpb,
        mutpb=ga.mutpb,
        ngen=ga.generations,
        verbose=True,
    )
    print(f"Finished GA in {time.time() - start:.2f}s")

    best = tools.selBest(result, k=1)[0]
    best_params = {
        "max_depth": best[0],
        "learning_rate": best[1],
        "n_estimators": best[2],
        "subsample": best[3],
        "colsample_bytree": best[4],
    }
    best_cv_acc = float(best.fitness.values[0])
    return best_params, best_cv_acc


def train_final_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: Dict[str, Any],
    random_state: int,
) -> Pipeline:
    pipe = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("classifier", XGBClassifier(
                **best_params,
                eval_metric="mlogloss",
                random_state=random_state,
                use_label_encoder=False,
            )),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="weighted"))
    cm = confusion_matrix(y_test, y_pred, labels=labels) if labels is not None else confusion_matrix(y_test, y_pred)

    return {
        "test_accuracy": acc,
        "test_weighted_f1": f1,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


# ----------------------------
# Main experiment runner
# ----------------------------

def run_experiment(
    train_csv: Path,
    test_csv: Path,
    outdir: Path,
    cfg: ExperimentConfig,
    cm_labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    ensure_outdir(outdir)

    X_train_all, y_train = load_xy(train_csv)
    X_test_all, y_test = load_xy(test_csv)

    # Feature selection
    selected_features, mi_df = select_features_mutual_info(
        X_train_all, y_train,
        feature_percentage=cfg.feature_percentage,
        random_state=cfg.random_state,
    )
    X_train = X_train_all[selected_features]
    X_test = X_test_all[selected_features]

    # GA tuning
    best_params, best_cv_acc = xgb_genetic_algorithm(X_train, y_train, cfg)

    # Train final and evaluate
    model = train_final_pipeline(X_train, y_train, best_params, cfg.random_state)
    metrics = evaluate_model(model, X_test, y_test, labels=cm_labels)

    # Save artifacts
    mi_df.to_csv(outdir / "mutual_info_ranking.csv", index=False)
    (outdir / "selected_features.txt").write_text("\n".join(selected_features), encoding="utf-8")
    joblib.dump(model, outdir / "model.joblib")

    summary = {
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features_total": int(X_train_all.shape[1]),
        "n_features_selected": int(len(selected_features)),
        "feature_percentage": cfg.feature_percentage,
        "random_state": cfg.random_state,
        "ga_best_cv_accuracy": best_cv_acc,
        "best_params": best_params,
        **metrics,
    }

    with open(outdir / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Console output
    print("\n=== Final Evaluation ===")
    print(f"Train: {train_csv.name}")
    print(f"Test : {test_csv.name}")
    print(f"Selected features: {len(selected_features)} / {X_train_all.shape[1]} ({cfg.feature_percentage}%)")
    print(f"GA best CV accuracy: {best_cv_acc:.4f}")
    print(f"Test accuracy     : {summary['test_accuracy']:.4f}")
    print(f"Test weighted F1  : {summary['test_weighted_f1']:.4f}")
    print("Confusion matrix:")
    print(np.array(summary["confusion_matrix"]))
    print(f"\nSaved outputs to: {outdir}")

    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GA-tuned XGBoost on train/test feature CSVs.")
    p.add_argument("--train", required=True, type=Path, help="Path to training CSV.")
    p.add_argument("--test", required=True, type=Path, help="Path to testing CSV.")
    p.add_argument("--outdir", required=True, type=Path, help="Output directory for artifacts.")

    p.add_argument("--feature-percentage", type=float, default=50.0, help="Top %% features by mutual info to keep.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--cv-splits", type=int, default=5, help="StratifiedKFold splits.")
    p.add_argument("--population", type=int, default=50, help="GA population size.")
    p.add_argument("--generations", type=int, default=10, help="GA generations.")
    p.add_argument("--cxpb", type=float, default=0.5, help="GA crossover probability.")
    p.add_argument("--mutpb", type=float, default=0.2, help="GA mutation probability.")
    p.add_argument("--indpb", type=float, default=0.2, help="Per-gene mutation probability.")
    p.add_argument("--tournsize", type=int, default=3, help="Tournament size.")
    p.add_argument("--n-jobs-cv", type=int, default=-1, help="n_jobs for cross_val_score.")

    # Optional: fixed label order for confusion matrix
    p.add_argument("--cm-labels", type=int, nargs="*", default=None,
                   help="Optional explicit label order for confusion matrix, e.g. --cm-labels 0 1 2")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    cfg = ExperimentConfig(
        feature_percentage=args.feature_percentage,
        random_state=args.seed,
        ga=GAConfig(
            population=args.population,
            generations=args.generations,
            cxpb=args.cxpb,
            mutpb=args.mutpb,
            indpb=args.indpb,
            tournsize=args.tournsize,
            cv_splits=args.cv_splits,
            n_jobs_cv=args.n_jobs_cv,
        ),
    )

    if not args.train.exists():
        raise FileNotFoundError(f"Train file not found: {args.train}")
    if not args.test.exists():
        raise FileNotFoundError(f"Test file not found: {args.test}")

    run_experiment(
        train_csv=args.train,
        test_csv=args.test,
        outdir=args.outdir,
        cfg=cfg,
        cm_labels=args.cm_labels,
    )


if __name__ == "__main__":
    main()
    
#python run_xgb_ga.py \
#  --train /path/to/training.csv \
#  --test /path/to/testing.csv \
#  --outdir ./runs/exp1 \
#  --feature-percentage 50 \
#  --seed 42 \
#  --cm-labels 0 1 2
