import joblib
import pandas as pd
from pathlib import Path

# ==========================================
# CẤU HÌNH
# ==========================================
# Trỏ đến thư mục experiment bạn muốn phân tích
RUN_DIR = Path("runs_cognitive_level/xgb_M5_fold1") 

def main():
    print("="*70)
    print("   📊 XUẤT THỐNG KÊ FEATURE IMPORTANCE (KIT vs KHT)")
    print("="*70)

    model_path = RUN_DIR / "model.joblib"
    features_path = RUN_DIR / "selected_features.txt"

    if not model_path.exists() or not features_path.exists():
        print(f"❌ Không tìm thấy model hoặc features tại {RUN_DIR}")
        return

    # 1. Load pipeline và features
    pipeline = joblib.load(model_path)
    with open(features_path, 'r', encoding='utf-8') as f:
        feature_names = [line.strip() for line in f if line.strip()]

    # 2. Lấy Feature Importances (Gain) từ XGBoost
    xgb_model = pipeline.named_steps["classifier"]
    importances = xgb_model.feature_importances_

    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # 3. Phân loại KIT và KHT chuẩn xác
    def categorize_feature(fname):
        if fname == 'question_index':
            return 'Other'
        elif '->' in fname:
            return 'KIT (Key Interval Time)'
        else:
            return 'KHT (Key Hold Time)'

    df_imp['Group'] = df_imp['Feature'].apply(categorize_feature)

    # 4. Tính toán các chỉ số thống kê
    stats = df_imp.groupby('Group')['Importance'].agg(
        Count='count',             # Số lượng features
        Total_Gain='sum',          # Tổng độ quan trọng
        Mean_Gain='mean',          # Trung bình
        Median_Gain='median',      # Trung vị
        Max_Gain='max',            # Feature mạnh nhất
        Min_Gain='min'             # Feature yếu nhất
    ).reset_index()

    # Tính thêm tỷ lệ % đóng góp tổng thể
    total_gain_sum = stats['Total_Gain'].sum()
    stats['Contribution (%)'] = (stats['Total_Gain'] / total_gain_sum) * 100

    # Sắp xếp lại thứ tự cột cho hợp lý và làm tròn số
    cols_order = ['Group', 'Count', 'Contribution (%)', 'Total_Gain', 'Mean_Gain', 'Median_Gain', 'Max_Gain', 'Min_Gain']
    stats = stats[cols_order]
    
    # Làm tròn số cho dễ nhìn
    stats = stats.round({
        'Contribution (%)': 2,
        'Total_Gain': 4,
        'Mean_Gain': 4,
        'Median_Gain': 4,
        'Max_Gain': 4,
        'Min_Gain': 4
    })

    # 5. Lưu ra file CSV
    out_csv = RUN_DIR / "importance_stats.csv"
    stats.to_csv(out_csv, index=False)

    # In kết quả ra màn hình
    print("\nBẢNG THỐNG KÊ TÓM TẮT:")
    print(stats.to_string(index=False))
    print(f"\n✅ Đã xuất bảng báo cáo chi tiết ra file: {out_csv}")
    print("="*70)

if __name__ == "__main__":
    main()