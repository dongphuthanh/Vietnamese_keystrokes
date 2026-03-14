import json
import numpy as np
import pandas as pd
from pathlib import Path

models = ["svm", "mlp", "xgb"]
ms = ["m2", "m3", "m4", "m5"]
runs = Path("/home/mtcd001/PycharmProjects/Vietnamese_keystrokes/runs")

# Label mapping
label_names = {0: "B", 1: "P", 2: "T"}
source_map = {"normal": "", "attack": "F_"}  # attack P→F_P, attack T→F_T

for model in models:
    for m in ms:
        print(f"\n{'='*40}")
        print(f"=== {model.upper()} {m.upper()} ===")
        print(f"{'='*40}")

        accs = []
        # Confusion matrix: rows = B, P, T, F_P, F_T, cols = B, P, T
        cm_total = np.zeros((5, 3), dtype=int)
        row_map = {"B": 0, "P": 1, "T": 2, "F_P": 3, "F_T": 4}

        for fold in range(1, 6):
            result_path = runs / f"{model}_{m}_fold{fold}" / "results.json"
            pred_path = runs / f"{model}_{m}_fold{fold}" / "predictions.csv"

            if not result_path.exists():
                print(f"  Missing fold {fold}")
                continue

            with open(result_path) as f:
                results = json.load(f)
            accs.append(results["test_accuracy"])

            if pred_path.exists():
                df = pd.read_csv(pred_path)
                for _, row in df.iterrows():
                    label = int(row['label'])
                    pred = int(row['predicted'])
                    source = row.get('source', 'normal')

                    # Xác định row name
                    if source == 'attack':
                        row_name = f"F_{label_names[label]}"
                    else:
                        row_name = label_names[label]

                    if row_name in row_map:
                        cm_total[row_map[row_name], pred] += 1

        if accs:
            print(f"Acc: {np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}%")
            print(f"\nConfusion Matrix (rows=actual, cols=predicted B/P/T):")
            print(f"{'':6} {'B':>6} {'P':>6} {'T':>6}")
            for name, idx in row_map.items():
                print(f"{name:6} {cm_total[idx,0]:>6} {cm_total[idx,1]:>6} {cm_total[idx,2]:>6}")