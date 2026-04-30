#!/usr/bin/env python3
"""
MASTER SCRIPT - Chạy toàn bộ pipeline tự động
1. Tạo datasets theo cognitive level
2. Chạy XGBoost cho tất cả kịch bản
3. Tổng hợp kết quả
"""

import subprocess
import sys
from pathlib import Path
import os

def run_script(script_name, description):
    """Chạy một script Python và xử lý lỗi"""
    print("\n" + "="*70)
    print(f"▶️  {description}")
    print("="*70)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✅ Hoàn thành: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ LỖI: {description} thất bại!")
        print(f"   Exit code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Không tìm thấy script: {script_name}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Người dùng dừng chương trình")
        sys.exit(1)

def check_input_files():
    """Kiểm tra file input có tồn tại không"""
    required_files = ['full.pkl', 'attack.pkl', 'XGB.py']
    missing = []
    
    for f in required_files:
        if not os.path.exists(f):
            missing.append(f)
    
    if missing:
        print("❌ Thiếu các file sau:")
        for f in missing:
            print(f"   - {f}")
        return False
    
    print("✅ Tất cả file input đã sẵn sàng")
    return True

def main():
    print("="*70)
    print("   🚀 MASTER PIPELINE - COGNITIVE LEVEL EXPERIMENTS")
    print("="*70)
    
    # Kiểm tra file đầu vào
    print("\n📋 Kiểm tra file đầu vào...")
    if not check_input_files():
        print("\n⚠️  Vui lòng chuẩn bị đầy đủ file trước khi chạy!")
        return
    
    # Danh sách các bước
    steps = [
        {
            "script": "generate_cognitive_level_splits.py",
            "description": "BƯỚC 1: Tạo datasets theo cognitive level",
            "skip_on_error": False
        },
        {
            "script": "run_all_xgb_cognitive.py",
            "description": "BƯỚC 2: Chạy XGBoost cho tất cả kịch bản (CÓ THỂ MẤT VÀI GIỜ)",
            "skip_on_error": False
        },
        {
            "script": "collect_results_cognitive.py",
            "description": "BƯỚC 3: Tổng hợp kết quả",
            "skip_on_error": True  # Có thể chạy riêng sau
        }
    ]
    
    # Chạy từng bước
    for i, step in enumerate(steps, 1):
        success = run_script(step["script"], step["description"])
        
        if not success and not step["skip_on_error"]:
            print(f"\n❌ Pipeline dừng tại bước {i}")
            print(f"   Vui lòng kiểm tra lỗi và chạy lại!")
            return
    
    # Hoàn thành
    print("\n" + "="*70)
    print("   🎉 HOÀN TẤT TOÀN BỘ PIPELINE!")
    print("="*70)
    print("\n📊 Kết quả được lưu tại:")
    print("   - Datasets: cognitive_level_datasets/")
    print("   - Models:   runs_cognitive_level/")
    print("   - Summary:  runs_cognitive_level/summary_results.json")
    print("\n💡 Xem chi tiết trong: runs_cognitive_level/detailed_results.csv")

if __name__ == "__main__":
    main()