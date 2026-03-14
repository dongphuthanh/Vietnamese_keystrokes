#!/usr/bin/env python3
"""
Run GA-tuned MLPClassifier on a train/test CSV pair.

Expected CSV columns:
- label (required)
- user_id, part (optional; will be dropped if present)
All remaining columns are treated as numeric features.

Example:
  python run_mlp_ga.py --train /path/train.csv --test /path/test.csv --outdir ./runs/mlp_exp1
"""

from __future__ import annotations

import argparse
import json
import random
import time
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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class GAConfig:
    population: int = 50
    generations: int = 10
    cxpb: float = 0.5
    mutpb: float = 0.2
    indpb: float = 0.2
    tournsize: int = 3
    cv_splits: int = 5
    n_jobs_cv: int = -1


@dataclass(frozen=True)
class ExperimentConfig:
    feature_percentage: float = 50.0
    random_state: int = 42
    max_iter: int = 500
    ga: GAConfig = GAConfig()


# ----------------------------
# MLP Hyperparameter Space
# ----------------------------

PARAM_SPACE: Dict[str, List[Any]] = {
    "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
    "activation": ["relu", "tanh", "logistic"],
    "alpha": [0.0001, 0.001, 0.01],
    "learning_rate": ["constant", "adaptive"],
}


# ----------------------------
# Utilities
# ----------------------------

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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
# DEAP (safe / idempotent)
# ----------------------------

def ensure_deap_creators() -> None:
    """
    DEAP creator is global; creating the same class twice throws.
    Make it safe for repeated runs.
    """
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)


def custom_mutation(individual, indpb: float):
    if random.random() < indpb:
        individual[0] = random.choice(PARAM_SPACE["hidden_layer_sizes"])
    if random.random() < indpb:
        individual[1] = random.choice(PARAM_SPACE["activation"])
    if random.random() < indpb:
        individual[2] = random.choice(PARAM_SPACE["alpha"])
    if random.random() < indpb:
        individual[3] = random.choice(PARAM_SPACE["learning_rate"])
    return (individual,)


# ----------------------------
# GA for MLP
# ----------------------------

def mlp_genetic_algorithm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: ExperimentConfig,
) -> Tuple[Dict[str, Any], float]:
    ensure_deap_creators()
    ga = cfg.ga

    skf = StratifiedKFold(
        n_splits=ga.cv_splits,
        shuffle=True,
        random_state=cfg.random_state,
    )

    def evaluate(individual):
        clf = MLPClassifier(
            hidden_layer_sizes=individual[0],
            activation=individual[1],
            alpha=individual[2],
            learning_rate=individual[3],
            max_iter=cfg.max_iter,
            random_state=cfg.random_state,
        )

        pipeline = Pipeline([
            ("scaler", MinMaxScaler()),
            ("classifier", clf),
        ])

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

    toolbox.register("attr_hidden", random.choice, PARAM_SPACE["hidden_layer_sizes"])
    toolbox.register("attr_act", random.choice, PARAM_SPACE["activation"])
    toolbox.register("attr_alpha", random.choice, PARAM_SPACE["alpha"])
    toolbox.register("attr_lr", random.choice, PARAM_SPACE["learning_rate"])

    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        (toolbox.attr_hidden, toolbox.attr_act, toolbox.attr_alpha, toolbox.attr_lr),
        n=1,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", custom_mutation, indpb=ga.indpb)
    toolbox.register("select", tools.selTournament, tournsize=ga.tournsize)

    population = toolbox.population(n=ga.population)

    print("Starting GA optimization for MLPClassifier...")
    start = time.time()
    result, _ = algorithms.eaSimple(
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
        "hidden_layer_sizes": best[0],
        "activation": best[1],
        "alpha": best[2],
        "learning_rate": best[3],
    }
    best_cv_acc = float(best.fitness.values[0])
    return best_params, best_cv_acc


# ----------------------------
# Train / Eval
# ----------------------------

def train_final_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: Dict[str, Any],
    cfg: ExperimentConfig,
) -> Pipeline:
    pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        ("classifier", MLPClassifier(
            **best_params,
            max_iter=cfg.max_iter,
            random_state=cfg.random_state,
        )),
    ])
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
    f1w = float(f1_score(y_test, y_pred, average="weighted"))
    cm = confusion_matrix(y_test, y_pred, labels=labels) if labels is not None else confusion_matrix(y_test, y_pred)

    return {
        "test_accuracy": acc,
        "test_weighted_f1": f1w,
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
    outdir.mkdir(parents=True, exist_ok=True)

    X_train_all, y_train = load_xy(train_csv)
    X_test_all, y_test = load_xy(test_csv)

    selected_features, mi_df = select_features_mutual_info(
        X_train_all,
        y_train,
        feature_percentage=cfg.feature_percentage,
        random_state=cfg.random_state,
    )

    X_train = X_train_all[selected_features]
    X_test = X_test_all[selected_features]

    best_params, best_cv_acc = mlp_genetic_algorithm(X_train, y_train, cfg)

    model = train_final_pipeline(X_train, y_train, best_params, cfg)
    metrics = evaluate_model(model, X_test, y_test, labels=cm_labels)

    # Save artifacts
    joblib.dump(model, outdir / "model.joblib")
    mi_df.to_csv(outdir / "mutual_info_ranking.csv", index=False)
    (outdir / "selected_features.txt").write_text("\n".join(selected_features), encoding="utf-8")

    summary = {
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features_total": int(X_train_all.shape[1]),
        "n_features_selected": int(len(selected_features)),
        "feature_percentage": cfg.feature_percentage,
        "random_state": cfg.random_state,
        "max_iter": cfg.max_iter,
        "ga_best_cv_accuracy": best_cv_acc,
        "best_params": best_params,
        **metrics,
    }

    with open(outdir / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Console summary
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


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GA-tuned MLPClassifier on train/test feature CSVs.")
    p.add_argument("--train", required=True, type=Path, help="Path to training CSV.")
    p.add_argument("--test", required=True, type=Path, help="Path to testing CSV.")
    p.add_argument("--outdir", required=True, type=Path, help="Output directory for artifacts.")

    p.add_argument("--feature-percentage", type=float, default=50.0, help="Top %% features by mutual info to keep.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--max-iter", type=int, default=500, help="MLP max_iter.")

    # GA knobs (optional overrides)
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--population", type=int, default=50)
    p.add_argument("--generations", type=int, default=10)
    p.add_argument("--cxpb", type=float, default=0.5)
    p.add_argument("--mutpb", type=float, default=0.2)
    p.add_argument("--indpb", type=float, default=0.2)
    p.add_argument("--tournsize", type=int, default=3)
    p.add_argument("--n-jobs-cv", type=int, default=-1)

    p.add_argument("--cm-labels", type=int, nargs="*", default=None,
                   help="Optional label order for confusion matrix, e.g. --cm-labels 0 1 2")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    cfg = ExperimentConfig(
        feature_percentage=args.feature_percentage,
        random_state=args.seed,
        max_iter=args.max_iter,
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
    
#python run_mlp_ga.py \
#  --train /path/to/training.csv \
#  --test /path/to/testing.csv \
#  --outdir ./runs/mlp_exp1 \
#  --feature-percentage 50 \
#  --seed 42 \
#  --cm-labels 0 1 2
