import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================
# CẤU HÌNH
# ==========================================
# Trỏ đến thư mục experiment bạn muốn phân tích
RUN_DIR = Path("runs_cognitive_level/xgb_M5_fold1") 

def main():
    print("="*60)
    print(f"🎻 VẼ VIOLIN PLOT: PHÂN PHỐI FEATURE IMPORTANCE")
    print("="*60)

    model_path = RUN_DIR / "model.joblib"
    features_path = RUN_DIR / "selected_features.txt"

    if not model_path.exists() or not features_path.exists():
        print(f"❌ Không tìm thấy model hoặc features tại {RUN_DIR}")
        return

    # 1. Load pipeline và features
    pipeline = joblib.load(model_path)
    with open(features_path, 'r', encoding='utf-8') as f:
        feature_names = [line.strip() for line in f if line.strip()]

    # 2. Lấy Feature Importances từ XGBoost
    xgb_model = pipeline.named_steps["classifier"]
    importances = xgb_model.feature_importances_

    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # 3. Phân loại KIT và KHT
    def categorize_feature(fname):
        if fname == 'question_index':
            return 'Other'
        elif '->' in fname:
            return 'KIT (Key Interval Time)'
        else:
            return 'KHT (Key Hold Time)'

    df_imp['Group'] = df_imp['Feature'].apply(categorize_feature)

    # Lọc bỏ nhóm 'Other' để biểu đồ tập trung so sánh KIT và KHT
    df_plot = df_imp[df_imp['Group'].isin(['KIT (Key Interval Time)', 'KHT (Key Hold Time)'])]

    # 4. Vẽ Violin Plot
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")
    
    # Bảng màu: Xanh cho KIT, Đỏ cho KHT
    my_pal = {'KIT (Key Interval Time)': '#3498db', 'KHT (Key Hold Time)': '#e74c3c'}
    
    # Lớp 1: Cây đàn Violin (Hiển thị phân phối và mật độ)
    # inner='quartile' sẽ vẽ các đường gạch ngang thể hiện 25%, 50% (median), 75% dữ liệu
    sns.violinplot(data=df_plot, x='Group', y='Importance', 
                   palette=my_pal, inner='quartile', alpha=0.6, linewidth=1.5)
    
    # Lớp 2: Các chấm đen (Mỗi chấm là 1 feature thực tế)
    sns.stripplot(data=df_plot, x='Group', y='Importance', 
                  color='black', alpha=0.5, size=4, jitter=True)

    # Trang trí biểu đồ
    plt.title('Distribution of Feature Importance: KIT vs KHT', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('XGBoost Feature Importance (Gain)', fontsize=13, fontweight='bold')
    plt.xlabel('Feature Category', fontsize=13, fontweight='bold')
    plt.xticks(fontsize=12)
    
    # Lưu file
    out_violin = RUN_DIR / "importance_violin_plot.png"
    plt.tight_layout()
    plt.savefig(out_violin, dpi=300)
    print(f"✅ Đã lưu biểu đồ Violin tuyệt đẹp tại: {out_violin}")
    plt.close()

if __name__ == "__main__":
    main()