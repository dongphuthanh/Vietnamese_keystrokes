#!/usr/bin/env python3
"""
Chạy XGBoost cho TẤT CẢ các kịch bản (M2, M3, M4, M5) và 3 folds

Fix:
  - sys.executable thay vì "python" → chạy đúng venv
  - --cm-labels luôn là 0 1 2 3 4 vì test set chứa tất cả 5 nhãn
"""

import os
import sys
import subprocess
from pathlib import Path

# ==========================================
# CẤU HÌNH
# ==========================================
#DATA_DIR = Path("cognitive_level_datasets")
#RUNS_DIR = Path("runs_cognitive_level") Context-Independent + Cognitive Level Split
DATA_DIR = Path("user_indep_cognitive_datasets")
RUNS_DIR = Path("runs_user_indep_cognitive") # User-Independent + Cognitive Level Split
XGB_SCRIPT = "XGB.py"

SCENARIOS = ["M2", "M3", "M4", "M5"]
FOLDS = [1, 2, 3]

FEATURE_PERCENTAGE = 50
SEED = 42
CV_SPLITS = 3
POPULATION = 20
GENERATIONS = 10

# Test set luôn có đủ 5 nhãn → cm-labels cố định cho tất cả kịch bản
CM_LABELS = [0, 1, 2, 3, 4]

# ==========================================
# KIỂM TRA FILE
# ==========================================
if not DATA_DIR.exists():
    print(f"❌ Không tìm thấy thư mục {DATA_DIR}")
    print("   Vui lòng chạy generate_cognitive_level_splits.py trước!")
    exit(1)

if not os.path.exists(XGB_SCRIPT):
    print(f"❌ Không tìm thấy {XGB_SCRIPT}")
    exit(1)

RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# CHẠY XGB CHO TẤT CẢ KỊCH BẢN
# ==========================================
print("="*70)
print("   CHẠY XGB CHO TẤT CẢ KỊCH BẢN (COGNITIVE LEVEL SPLIT 14/25/36)")
print(f"   Python: {sys.executable}")
print("="*70)

total_runs = len(SCENARIOS) * len(FOLDS)
current_run = 0

for scenario in SCENARIOS:
    for fold in FOLDS:
        current_run += 1

        train_csv = DATA_DIR / f"train_{scenario}_fold{fold}.csv"
        test_csv  = DATA_DIR / f"test_{scenario}_fold{fold}.csv"
        outdir    = RUNS_DIR / f"xgb_{scenario}_fold{fold}"

        if not train_csv.exists() or not test_csv.exists():
            print(f"⚠️  Bỏ qua {scenario} Fold {fold}: Thiếu file dữ liệu")
            continue

        print(f"\n{'='*70}")
        print(f"🚀 [{current_run}/{total_runs}] {scenario} - Fold {fold}")
        print(f"{'='*70}")
        print(f"   Train: {train_csv.name}")
        print(f"   Test:  {test_csv.name}")
        print(f"   Output: {outdir}")

        cmd = [
            sys.executable, XGB_SCRIPT,
            "--train",              str(train_csv),
            "--test",               str(test_csv),
            "--outdir",             str(outdir),
            "--feature-percentage", str(FEATURE_PERCENTAGE),
            "--seed",               str(SEED),
            "--cv-splits",          str(CV_SPLITS),
            "--population",         str(POPULATION),
            "--generations",        str(GENERATIONS),
            "--cm-labels",
        ] + [str(l) for l in CM_LABELS]

        print(f"   📝 Command: {' '.join(cmd)}\n")

        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Hoàn thành: {scenario} Fold {fold}")
        except subprocess.CalledProcessError as e:
            print(f"❌ LỖI khi chạy {scenario} Fold {fold}: {e}")
            continue
        except KeyboardInterrupt:
            print("\n⚠️  Người dùng dừng chương trình")
            exit(1)

print("\n" + "="*70)
print("🎉 HOÀN TẤT TẤT CẢ!")
print(f"   Kết quả được lưu trong: {RUNS_DIR}")
print("="*70)
print("\n💡 Chạy collect_results_cognitive.py để tổng hợp kết quả:")
print("   python collect_results_cognitive.py")