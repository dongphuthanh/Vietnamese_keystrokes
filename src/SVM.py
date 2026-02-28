#!/usr/bin/env python3
"""
Run GA-tuned SVM on a train/test CSV pair.

Example:
  python run_svm_ga.py \
    --train /path/train.csv \
    --test /path/test.csv \
    --outdir ./runs/svm_exp1
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
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.pipeline import Pipeline


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
    ga: GAConfig = GAConfig()


# ----------------------------
# SVM Hyperparameter Space
# ----------------------------

PARAM_SPACE: Dict[str, List[Any]] = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["rbf", "poly"],
    "gamma": ["scale", "auto", 0.0001, 0.001, 0.01, 0.1, 1],
    "degree": [2, 3, 4, 5],  # only relevant if kernel="poly"
}


# ----------------------------
# Utilities
# ----------------------------

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def load_xy(csv_path: Path):
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError(f"{csv_path} missing required column 'label'.")

    y = df["label"]
    drop_cols = [c for c in ["user_id", 'session','section' "label"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    return X, y


def select_features_mutual_info(X, y, feature_percentage, random_state):
    mi = mutual_info_classif(
        X, y,
        discrete_features=False,
        random_state=random_state
    )

    mi_df = (
        pd.DataFrame({"feature": X.columns, "mi_score": mi})
        .sort_values("mi_score", ascending=False)
        .reset_index(drop=True)
    )

    k = max(1, int(len(mi_df) * (feature_percentage / 100)))
    selected = mi_df["feature"].iloc[:k].tolist()

    return selected, mi_df


# ----------------------------
# DEAP setup (safe)
# ----------------------------

def ensure_deap_creators():
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)


def custom_mutation(individual, indpb):
    if random.random() < indpb:
        individual[0] = random.choice(PARAM_SPACE["C"])
    if random.random() < indpb:
        individual[1] = random.choice(PARAM_SPACE["kernel"])
    if random.random() < indpb:
        individual[2] = random.choice(PARAM_SPACE["gamma"])
    if random.random() < indpb:
        individual[3] = random.choice(PARAM_SPACE["degree"])
    return (individual,)


# ----------------------------
# GA for SVM
# ----------------------------

def svm_genetic_algorithm(X_train, y_train, cfg: ExperimentConfig):

    ensure_deap_creators()
    ga = cfg.ga

    skf = StratifiedKFold(
        n_splits=ga.cv_splits,
        shuffle=True,
        random_state=cfg.random_state
    )

    def evaluate(individual):
        C, kernel, gamma, degree = individual

        clf = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            degree=degree if kernel == "poly" else 3,
            class_weight="balanced",
            random_state=cfg.random_state
        )

        pipeline = Pipeline([
            ("scaler", MinMaxScaler()),
            ("classifier", clf)
        ])

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=skf,
            scoring="accuracy",
            n_jobs=ga.n_jobs_cv
        )

        return (float(scores.mean()),)

    toolbox = base.Toolbox()

    toolbox.register("attr_C", random.choice, PARAM_SPACE["C"])
    toolbox.register("attr_kernel", random.choice, PARAM_SPACE["kernel"])
    toolbox.register("attr_gamma", random.choice, PARAM_SPACE["gamma"])
    toolbox.register("attr_degree", random.choice, PARAM_SPACE["degree"])

    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        (
            toolbox.attr_C,
            toolbox.attr_kernel,
            toolbox.attr_gamma,
            toolbox.attr_degree
        ),
        n=1
    )

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", custom_mutation, indpb=ga.indpb)
    toolbox.register("select", tools.selTournament, tournsize=ga.tournsize)

    population = toolbox.population(n=ga.population)

    print("Starting GA optimization for SVM...")
    start = time.time()

    result, _ = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=ga.cxpb,
        mutpb=ga.mutpb,
        ngen=ga.generations,
        verbose=True
    )

    print(f"Finished GA in {time.time() - start:.2f}s")

    best = tools.selBest(result, k=1)[0]

    best_params = {
        "C": best[0],
        "kernel": best[1],
        "gamma": best[2],
        "degree": best[3] if best[1] == "poly" else 3
    }

    best_cv_acc = float(best.fitness.values[0])
    return best_params, best_cv_acc


# ----------------------------
# Training & Evaluation
# ----------------------------

def train_final_pipeline(X_train, y_train, best_params, random_state):
    pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        ("classifier", SVC(
            **best_params,
            class_weight="balanced",
            random_state=random_state
        ))
    ])
    pipe.fit(X_train, y_train)
    return pipe


def evaluate_model(model, X_test, y_test, labels=None):
    y_pred = model.predict(X_test)

    return {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist()
    }


# ----------------------------
# Main Experiment Runner
# ----------------------------

def run_experiment(train_csv, test_csv, outdir, cfg, cm_labels=None):

    outdir.mkdir(parents=True, exist_ok=True)

    X_train_all, y_train = load_xy(train_csv)
    X_test_all, y_test = load_xy(test_csv)

    selected_features, mi_df = select_features_mutual_info(
        X_train_all, y_train,
        cfg.feature_percentage,
        cfg.random_state
    )

    X_train = X_train_all[selected_features]
    X_test = X_test_all[selected_features]

    best_params, best_cv_acc = svm_genetic_algorithm(X_train, y_train, cfg)

    model = train_final_pipeline(X_train, y_train, best_params, cfg.random_state)

    metrics = evaluate_model(model, X_test, y_test, labels=cm_labels)

    # Save artifacts
    joblib.dump(model, outdir / "model.joblib")
    mi_df.to_csv(outdir / "mutual_info_ranking.csv", index=False)
    (outdir / "selected_features.txt").write_text("\n".join(selected_features))

    summary = {
        "ga_best_cv_accuracy": best_cv_acc,
        "best_params": best_params,
        **metrics
    }

    with open(outdir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Final Evaluation ===")
    print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test weighted F1: {metrics['test_weighted_f1']:.4f}")
    print("Confusion matrix:")
    print(np.array(metrics["confusion_matrix"]))

    print(f"\nSaved outputs to: {outdir}")


# ----------------------------
# CLI
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--test", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--feature-percentage", type=float, default=50.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cm-labels", type=int, nargs="*", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    set_seeds(args.seed)

    cfg = ExperimentConfig(
        feature_percentage=args.feature_percentage,
        random_state=args.seed
    )

    run_experiment(
        train_csv=args.train,
        test_csv=args.test,
        outdir=args.outdir,
        cfg=cfg,
        cm_labels=args.cm_labels
    )


if __name__ == "__main__":
    main()
#python run_svm_ga.py \
#  --train /path/to/train.csv \
#  --test /path/to/test.csv \
#  --outdir ./runs/svm_exp1 \
#  --feature-percentage 50 \
#  --seed 42 \
#  --cm-labels 0 1 2
