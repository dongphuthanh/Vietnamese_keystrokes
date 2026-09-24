"""
Step 5: Aggregate results across 3 folds for each experiment type.

Metrics:
  - Accuracy (mean ± std across folds)
  - Weighted F1
  - FAR = non-Bonafide predicted as Bonafide / total non-Bonafide
  - FRR = Bonafide predicted as non-Bonafide / total Bonafide
  - Combined confusion matrix (sum across folds)

Outputs:
  runs_context_indep/summary.json  +  summary_table printed
  runs_user_indep/summary.json     +  summary_table printed
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    RUNS_CONTEXT, RUNS_USER,
    SCENARIOS, LABEL_NAMES, BONAFIDE_LABEL,
)

FOLDS = [1, 2, 3]


# ============================================================
# FAR / FRR
# ============================================================
def calc_far_frr(cm: np.ndarray):
    b = BONAFIDE_LABEL
    total_bonafide = cm[b, :].sum()
    frr = cm[b, np.arange(cm.shape[1]) != b].sum() / total_bonafide if total_bonafide > 0 else 0.0

    non_b = np.delete(cm, b, axis=0)
    total_non_b = non_b.sum()
    far = non_b[:, b].sum() / total_non_b if total_non_b > 0 else 0.0
    return float(far), float(frr)


# ============================================================
# PRINT CONFUSION MATRIX
# ============================================================
def print_cm(cm: np.ndarray, title: str):
    n = cm.shape[0]
    labels = [LABEL_NAMES.get(i, str(i)) for i in range(n)]
    col_w = 8
    print(f"\n  {title}  (rows=True, cols=Pred)")
    print("  " + "".join(f"{lb:>{col_w}}" for lb in [""] + labels))
    for i, row in enumerate(cm):
        print("  " + f"{labels[i]:>{col_w}}" + "".join(f"{int(v):>{col_w}}" for v in row))


# ============================================================
# COLLECT ONE EXPERIMENT
# ============================================================
def collect(runs_dir: Path, label: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Runs dir: {runs_dir}")
    print(f"{'='*60}")

    summary = {}
    rows = []

    for scenario in SCENARIOS:
        accs, f1s = [], []
        total_cm  = None
        found     = 0

        for fold in FOLDS:
            json_path = runs_dir / f"xgb_{scenario}_fold{fold}" / "results.json"
            if not json_path.exists():
                print(f"  [missing] {json_path}")
                continue
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"  [error] {json_path}: {e}")
                continue

            acc = data.get("test_accuracy", 0.0)
            f1  = data.get("test_weighted_f1", 0.0)
            cm  = np.array(data.get("confusion_matrix", []))
            far, frr = calc_far_frr(cm) if cm.size > 0 else (0.0, 0.0)

            accs.append(acc)
            f1s.append(f1)
            total_cm = cm if total_cm is None else total_cm + cm
            found += 1

            rows.append({
                "Scenario": scenario, "Fold": fold,
                "Accuracy": acc, "F1": f1,
                "FAR(%)": far * 100, "FRR(%)": frr * 100,
                "n_train": data.get("n_train", 0),
                "n_test":  data.get("n_test", 0),
                "n_features_selected": data.get("n_features_selected", 0),
            })
            print(f"  {scenario} fold{fold}: acc={acc:.4f}  f1={f1:.4f}  FAR={far*100:.2f}%  FRR={frr*100:.2f}%")

        if found == 0:
            continue

        far_c, frr_c = calc_far_frr(total_cm)
        summary[scenario] = {
            "accuracy_mean_%":  float(np.mean(accs) * 100),
            "accuracy_std_%":   float(np.std(accs)  * 100),
            "f1_mean_%":        float(np.mean(f1s)  * 100),
            "f1_std_%":         float(np.std(f1s)   * 100),
            "far_%":            float(far_c * 100),
            "frr_%":            float(frr_c * 100),
            "folds_found":      found,
            "combined_cm":      total_cm.tolist(),
        }

    # ---- Save ----
    out_json = runs_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    out_csv = runs_dir / "detailed_results.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # ---- Print summary table ----
    print(f"\n  Summary ({label})")
    print(f"  {'Scenario':<10} {'Accuracy':>14} {'F1':>14} {'FAR%':>8} {'FRR%':>8}")
    print(f"  {'-'*56}")
    for sc in SCENARIOS:
        if sc not in summary:
            continue
        s = summary[sc]
        print(f"  {sc:<10} {s['accuracy_mean_%']:>6.2f} +/- {s['accuracy_std_%']:<5.2f}  "
              f"{s['f1_mean_%']:>6.2f} +/- {s['f1_std_%']:<5.2f}  "
              f"{s['far_%']:>6.2f}  {s['frr_%']:>6.2f}")

    # ---- Print combined confusion matrices ----
    for sc in SCENARIOS:
        if sc in summary:
            print_cm(np.array(summary[sc]["combined_cm"]), sc)

    print(f"\n  Saved: {out_json}")
    print(f"  Saved: {out_csv}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--type", choices=["context", "user", "both"], default="both")
    args = p.parse_args()

    if args.type in ("context", "both"):
        collect(RUNS_CONTEXT, "CONTEXT-INDEPENDENT")

    if args.type in ("user", "both"):
        collect(RUNS_USER, "USER-INDEPENDENT")

    print("\n[Step 5] Done.")


if __name__ == "__main__":
    main()
