"""
Plot top-20 feature importance aggregated across all folds for a given
scenario and experiment type.

Usage:
  python plot_importance.py --scenario M5 --type context
  python plot_importance.py --scenario all --type context        # one plot per scenario
  python plot_importance.py --scenario all --type context --combined  # all scenarios in one figure
"""

import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from config import RUNS_CONTEXT, RUNS_USER, SCENARIOS

FOLDS = [1, 2, 3]


# ============================================================
# CATEGORIZE FEATURE NAME
# ============================================================
def categorize(fname: str) -> str:
    if "_rkd_" in fname:
        return "RUKD (Release→Key Down)"
    if "->" in fname:
        return "KIT (Key Down→Key Down)"
    return "KHT (Key Hold Time)"


PALETTE = {
    "KIT (Key Down→Key Down)":   "#1f77b4",
    "KHT (Key Hold Time)":        "#ff7f0e",
    "RUKD (Release→Key Down)":   "#2ca02c",
}


# ============================================================
# AGGREGATE IMPORTANCE ACROSS FOLDS
# ============================================================
def aggregate_importance(runs_dir: Path, scenario: str) -> pd.DataFrame | None:
    all_imp = []

    for fold in FOLDS:
        run_dir      = runs_dir / f"xgb_{scenario}_fold{fold}"
        model_path   = run_dir / "model.joblib"
        features_path = run_dir / "selected_features.txt"

        if not model_path.exists() or not features_path.exists():
            continue

        pipeline   = joblib.load(model_path)
        xgb_model  = pipeline.named_steps["classifier"]
        importances = xgb_model.feature_importances_

        feature_names = features_path.read_text(encoding="utf-8").splitlines()
        feature_names = [f.strip() for f in feature_names if f.strip()]

        all_imp.append(pd.DataFrame({
            "feature":    feature_names,
            "importance": importances,
            "fold":       fold,
        }))

    if not all_imp:
        return None

    df = pd.concat(all_imp, ignore_index=True)
    # Mean importance across folds (only features that appear in all folds)
    df_mean = (
        df.groupby("feature")["importance"]
        .mean()
        .reset_index()
        .rename(columns={"importance": "mean_importance"})
        .sort_values("mean_importance", ascending=False)
    )
    df_mean["category"] = df_mean["feature"].apply(categorize)
    return df_mean


# ============================================================
# PLOT
# ============================================================
def plot_top20(df_mean: pd.DataFrame, title: str, out_path: Path):
    df_top20 = df_mean.head(20).copy()

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.set_theme(style="whitegrid")

    sns.barplot(
        data=df_top20,
        x="mean_importance",
        y="feature",
        hue="category",
        dodge=False,
        palette=PALETTE,
        ax=ax,
    )

    # Value labels on bars
    for i, val in enumerate(df_top20["mean_importance"]):
        ax.text(val + 0.001, i, f"{val:.4f}", va="center", fontsize=9)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Mean Feature Importance (Gain, averaged across folds)", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    ax.legend(title="Feature Type", fontsize=10, title_fontsize=11, loc="lower right")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"  Saved: {out_path}")
    print(f"  Saved: {out_path.with_suffix('.pdf')}")
    plt.close()


# ============================================================
# POOL ALL SCENARIOS → SINGLE TOP-20 PLOT
# ============================================================
def plot_pooled(dfs: dict, exp_type: str, runs_dir: Path):
    """
    dfs: {scenario: df_mean}
    Average mean_importance across all scenarios (equal weight per scenario),
    then plot single top-20.
    """
    combined = pd.concat(dfs.values(), ignore_index=True)
    df_pool = (
        combined.groupby("feature")["mean_importance"]
        .mean()
        .reset_index()
        .rename(columns={"mean_importance": "mean_importance"})
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )
    df_pool["category"] = df_pool["feature"].apply(categorize)

    scenario_list = ", ".join(dfs.keys())
    title = f"Top 20 Important Features"
    out_path = runs_dir / f"pooled_top20_{exp_type}.png"
    plot_top20(df_pool, title, out_path)


# ============================================================
# COMBINED PLOT  (2×2 grid, one subplot per scenario)
# ============================================================
def plot_combined(dfs: dict, exp_type: str, runs_dir: Path):
    """
    dfs: {scenario_name: df_mean}  — only scenarios with data included
    """
    scenarios_ordered = [s for s in ["M2", "M3", "M4", "M5"] if s in dfs]
    n = len(scenarios_ordered)
    if n == 0:
        print("  No data to plot.")
        return

    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, nrows * 9))
    sns.set_theme(style="whitegrid")
    axes_flat = axes.flat if n > 1 else [axes]

    legend_handles = {}

    for ax, scenario in zip(axes_flat, scenarios_ordered):
        df_top20 = dfs[scenario].head(20).copy()

        bars = sns.barplot(
            data=df_top20,
            x="mean_importance",
            y="feature",
            hue="category",
            dodge=False,
            palette=PALETTE,
            ax=ax,
        )

        for i, val in enumerate(df_top20["mean_importance"]):
            ax.text(val + ax.get_xlim()[1] * 0.005, i,
                    f"{val:.4f}", va="center", fontsize=8)

        ax.set_title(f"{scenario} ({exp_type}-independent)",
                     fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Mean Feature Importance (Gain)", fontsize=10)
        ax.set_ylabel("Feature", fontsize=10)
        ax.tick_params(axis="y", labelsize=9)

        # Collect legend handles from first subplot that has them
        if not legend_handles:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                legend_handles[label] = handle
        ax.legend_.remove() if ax.get_legend() else None

    # Hide unused subplots
    for ax in list(axes_flat)[n:]:
        ax.set_visible(False)

    # Single shared legend below the plots
    if legend_handles:
        fig.legend(
            legend_handles.values(), legend_handles.keys(),
            title="Feature Type", fontsize=11, title_fontsize=12,
            loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02),
        )

    fig.suptitle(
        f"Top 20 Feature Importances by Scenario — {exp_type.capitalize()}-Independent",
        fontsize=15, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    out_path = runs_dir / f"combined_top20_{exp_type}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"  Saved: {out_path}")
    print(f"  Saved: {out_path.with_suffix('.pdf')}")
    plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="M5",
                   help="Scenario name (M2/M3/M4/M5) or 'all'")
    p.add_argument("--type", choices=["context", "user", "both"], default="both")
    p.add_argument("--combined", action="store_true",
                   help="Save one combined 2×2 figure (one subplot per scenario)")
    p.add_argument("--pooled", action="store_true",
                   help="Pool all scenarios into a single top-20 plot")
    args = p.parse_args()

    scenarios = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]
    runs_map  = {}
    if args.type in ("context", "both"):
        runs_map["context"] = RUNS_CONTEXT
    if args.type in ("user", "both"):
        runs_map["user"] = RUNS_USER

    for exp_type, runs_dir in runs_map.items():
        collected = {}

        for scenario in scenarios:
            print(f"\n[{exp_type.upper()} | {scenario}] Aggregating importance...")
            df_mean = aggregate_importance(runs_dir, scenario)

            if df_mean is None or df_mean.empty:
                print(f"  No results found in {runs_dir}")
                continue

            collected[scenario] = df_mean

            if not args.combined and not args.pooled:
                title    = f"Top 20 Features — {scenario} ({exp_type}-independent)"
                out_path = runs_dir / f"top20_{scenario}.png"
                plot_top20(df_mean, title, out_path)

            # Print summary table
            top10_str = df_mean[["feature", "mean_importance", "category"]].head(10).to_string(index=False)
            print(f"\n  Top 10 features:")
            print(top10_str.encode("ascii", errors="replace").decode("ascii"))

            # KIT vs KHT vs RUKD breakdown
            breakdown = df_mean.groupby("category")["mean_importance"].agg(["count", "sum", "mean"])
            print(f"\n  Category breakdown (all selected features):")
            print(breakdown.to_string().encode("ascii", errors="replace").decode("ascii"))

        if args.combined and collected:
            print(f"\n[{exp_type.upper()}] Saving combined 2x2 plot...")
            plot_combined(collected, exp_type, runs_dir)

        if args.pooled and collected:
            print(f"\n[{exp_type.upper()}] Saving pooled single plot...")
            plot_pooled(collected, exp_type, runs_dir)


if __name__ == "__main__":
    main()
