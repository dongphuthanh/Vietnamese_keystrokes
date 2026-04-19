#!/usr/bin/env python3
"""
Tổng hợp kết quả từ các experiments với cognitive level split

FAR = số non-Bonafide bị đoán là Bonafide / tổng non-Bonafide
FRR = số Bonafide bị đoán thành class khác / tổng Bonafide
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================
# CẤU HÌNH
# ==========================================
RUNS_DIR  = Path('runs_cognitive_level')
SCENARIOS = ['M2', 'M3', 'M4', 'M5']
FOLDS     = [1, 2, 3]
BONAFIDE_LABEL = 0  # index của Bonafide trong confusion matrix

LABEL_NAMES = {
    0: "B",
    1: "P",
    2: "T",
    3: "F_P",
    4: "F_T",
}

# ==========================================
# FAR / FRR — chỉ tính theo class Bonafide
# ==========================================
def calculate_far_frr(cm):
    """
    cm: confusion matrix 5x5, hàng = True label, cột = Predicted label
    Bonafide = label 0 (hàng 0, cột 0)

    FAR = non-Bonafide bị predict thành Bonafide / tổng non-Bonafide
        = sum(cm[1:, 0]) / sum(cm[1:, :])

    FRR = Bonafide bị predict thành class khác / tổng Bonafide
        = sum(cm[0, 1:]) / sum(cm[0, :])
    """
    cm = np.array(cm, dtype=float)
    b  = BONAFIDE_LABEL

    # FRR: Bonafide bị đoán sai
    total_bonafide = cm[b, :].sum()
    frr = cm[b, np.arange(cm.shape[1]) != b].sum() / total_bonafide if total_bonafide > 0 else 0

    # FAR: non-Bonafide bị đoán thành Bonafide
    non_bonafide_rows = np.delete(cm, b, axis=0)          # bỏ hàng Bonafide
    total_non_bonafide = non_bonafide_rows.sum()
    far = non_bonafide_rows[:, b].sum() / total_non_bonafide if total_non_bonafide > 0 else 0

    return far, frr

# ==========================================
# IN CONFUSION MATRIX ĐẸP
# hàng dọc  = True label
# hàng ngang = Predicted label
# ==========================================
def print_confusion_matrix(cm, scenario):
    cm     = np.array(cm, dtype=int)
    n      = cm.shape[0]
    labels = [LABEL_NAMES.get(i, str(i)) for i in range(n)]
    col_w  = 8
    row_lw = 6

    print(f"\n📊 {scenario}  (hàng = True label | cột = Predicted)")
    header = f"{'':>{row_lw}}  " + "".join(f"{lb:>{col_w}}" for lb in labels)
    print(header)
    print(f"{'':>{row_lw}}  " + "-" * (col_w * n))
    for i, row in enumerate(cm):
        print(f"{labels[i]:>{row_lw}}  " + "".join(f"{v:>{col_w}}" for v in row))

# ==========================================
# MAIN
# ==========================================
def main():
    print("="*70)
    print("   TỔNG HỢP KẾT QUẢ (COGNITIVE LEVEL SPLIT)")
    print("="*70)

    if not RUNS_DIR.exists():
        print(f"❌ Không tìm thấy thư mục {RUNS_DIR}")
        return

    all_results = []
    summary     = {}

    print("\n📊 Đang thu thập kết quả...\n")

    for scenario in SCENARIOS:
        accuracies, f1_scores = [], []
        total_cm    = None
        folds_found = 0

        for fold in FOLDS:
            json_path = RUNS_DIR / f"xgb_{scenario}_fold{fold}" / "results.json"
            if not json_path.exists():
                print(f"⚠️  Không tìm thấy: {json_path}")
                continue

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                acc = data.get('test_accuracy', 0)
                f1  = data.get('test_weighted_f1', 0)
                cm  = data.get('confusion_matrix', [])

                accuracies.append(acc)
                f1_scores.append(f1)

                if cm:
                    cm_arr   = np.array(cm)
                    total_cm = cm_arr if total_cm is None else total_cm + cm_arr
                    far, frr = calculate_far_frr(cm)
                else:
                    far, frr = 0, 0

                all_results.append({
                    'Scenario':            scenario,
                    'Fold':                fold,
                    'Accuracy':            acc,
                    'F1_Score':            f1,
                    'FAR(%)':              far * 100,
                    'FRR(%)':              frr * 100,
                    'n_train':             data.get('n_train', 0),
                    'n_test':              data.get('n_test', 0),
                    'n_features_selected': data.get('n_features_selected', 0),
                })

                folds_found += 1
                print(f"✅ {scenario} Fold {fold}: ACC={acc:.4f}, F1={f1:.4f}, "
                      f"FAR={far*100:.2f}%, FRR={frr*100:.2f}%")

            except Exception as e:
                print(f"❌ Lỗi đọc {json_path}: {e}")

        if folds_found > 0:
            far_c, frr_c = calculate_far_frr(total_cm) if total_cm is not None else (0, 0)
            summary[scenario] = {
                'accuracy_mean':             np.mean(accuracies) * 100,
                'accuracy_std':              np.std(accuracies)  * 100,
                'f1_mean':                   np.mean(f1_scores)  * 100,
                'f1_std':                    np.std(f1_scores)   * 100,
                'far_percent':               far_c * 100,
                'frr_percent':               frr_c * 100,
                'combined_confusion_matrix': total_cm.tolist() if total_cm is not None else [],
                'folds_found':               folds_found,
            }

    # ==========================================
    # LƯU FILE
    # ==========================================
    df_results = pd.DataFrame(all_results)
    if not df_results.empty:
        out = RUNS_DIR / "detailed_results.csv"
        df_results.to_csv(out, index=False)
        print(f"\n📄 Chi tiết: {out}")

    out_json = RUNS_DIR / "summary_results.json"
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"📄 JSON: {out_json}")

    # ==========================================
    # BẢNG TỔNG HỢP
    # ==========================================
    print("\n" + "="*70)
    print("   KẾT QUẢ TỔNG HỢP (TRUNG BÌNH 3 FOLDS)")
    print("   FAR = non-Bonafide đoán thành Bonafide / tổng non-Bonafide")
    print("   FRR = Bonafide đoán thành class khác   / tổng Bonafide")
    print("="*70)

    rows = []
    for sc in SCENARIOS:
        if sc in summary:
            s = summary[sc]
            rows.append({
                'Scenario':    sc,
                'Accuracy(%)': f"{s['accuracy_mean']:.2f} ± {s['accuracy_std']:.2f}",
                'F1(%)':       f"{s['f1_mean']:.2f} ± {s['f1_std']:.2f}",
                'FAR(%)':      f"{s['far_percent']:.2f}",
                'FRR(%)':      f"{s['frr_percent']:.2f}",
                'Folds':       s['folds_found'],
            })
    print(pd.DataFrame(rows).to_string(index=False))

    # ==========================================
    # CONFUSION MATRICES
    # ==========================================
    print("\n" + "="*70)
    print("   COMBINED CONFUSION MATRICES")
    print("   (hàng dọc = True label | hàng ngang = Predicted label)")
    print("="*70)

    for sc in SCENARIOS:
        if sc in summary and summary[sc]['combined_confusion_matrix']:
            print_confusion_matrix(summary[sc]['combined_confusion_matrix'], sc)

    print("\n" + "="*70)
    print("✅ HOÀN TẤT!")
    print("="*70)

if __name__ == "__main__":
    main()