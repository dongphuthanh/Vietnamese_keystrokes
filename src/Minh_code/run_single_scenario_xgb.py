#!/usr/bin/env python3
"""
Chạy XGBoost cho MỘT kịch bản cụ thể (M2, M3, M4, hoặc M5) với 3 folds
Usage:
    python run_single_scenario.py M2
    python run_single_scenario.py M3
    python run_single_scenario.py M4
    python run_single_scenario.py M5
"""

import os
import sys
import subprocess
from pathlib import Path

# ==========================================
# KIỂM TRA THAM SỐ
# ==========================================
VALID_SCENARIOS = ["M2", "M3", "M4", "M5"]

if len(sys.argv) < 2:
    print("❌ Thiếu tham số kịch bản!")
    print("   Cách dùng: python run_single_scenario.py <SCENARIO>")
    print(f"   Ví dụ:     python run_single_scenario.py M2")
    print(f"   Kịch bản hợp lệ: {', '.join(VALID_SCENARIOS)}")
    exit(1)

SCENARIO = sys.argv[1].upper()
if SCENARIO not in VALID_SCENARIOS:
    print(f"❌ Kịch bản '{SCENARIO}' không hợp lệ!")
    print(f"   Kịch bản hợp lệ: {', '.join(VALID_SCENARIOS)}")
    exit(1)

# ==========================================
# CẤU HÌNH
# ==========================================
DATA_DIR = Path("cognitive_level_datasets")
RUNS_DIR = Path("runs_cognitive_level")
XGB_SCRIPT = "XGB.py"

FOLDS = [1, 2, 3]

FEATURE_PERCENTAGE = 50
SEED = 42
CV_SPLITS = 3
POPULATION = 20
GENERATIONS = 10

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
# CHẠY
# ==========================================
print("="*70)
print(f"   CHẠY KỊCH BẢN: {SCENARIO} (3 FOLDS)")
print(f"   Python: {sys.executable}")
print("="*70)

for fold in FOLDS:
    train_csv = DATA_DIR / f"train_{SCENARIO}_fold{fold}.csv"
    test_csv  = DATA_DIR / f"test_{SCENARIO}_fold{fold}.csv"
    outdir    = RUNS_DIR / f"xgb_{SCENARIO}_fold{fold}"

    if not train_csv.exists() or not test_csv.exists():
        print(f"⚠️  Bỏ qua Fold {fold}: Thiếu file dữ liệu")
        continue

    print(f"\n{'='*70}")
    print(f"🚀 {SCENARIO} - Fold {fold}")
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
        print(f"✅ Hoàn thành: {SCENARIO} Fold {fold}")
    except subprocess.CalledProcessError as e:
        print(f"❌ LỖI khi chạy {SCENARIO} Fold {fold}: {e}")
        continue
    except KeyboardInterrupt:
        print("\n⚠️  Người dùng dừng chương trình")
        exit(1)

print("\n" + "="*70)
print(f"🎉 HOÀN TẤT KỊCH BẢN {SCENARIO}!")
print(f"   Kết quả: {RUNS_DIR}/xgb_{SCENARIO}_fold*/")
print("="*70)
print("\n💡 Tổng hợp kết quả:")
print("   python collect_results_cognitive.py")