import os
import json
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from deap import base, creator, tools, algorithms
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. CẤU HÌNH THUẬT TOÁN GA & ĐƯỜNG DẪN
# ==========================================
DATA_DIR = Path("prepared_datasets")
OUT_DIR = Path("runs/svm_final_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Bạn có thể giảm population và generations xuống nếu máy chạy quá lâu
POPULATION = 20  
GENERATIONS = 5 
CV_SPLITS = 3
RANDOM_STATE = 42

PARAM_SPACE = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["rbf", "poly", "linear"],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    "degree": [2, 3, 4] 
}

# ==========================================
# 2. HÀM PHỤ TRỢ (LOAD DATA, TÍNH FAR/FRR)
# ==========================================
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)

def load_xy(csv_path):
    df = pd.read_csv(csv_path)
    y = df["label"]
    # Bỏ các cột thông tin, chỉ lấy đúng 30 cột features đã lọc ở bước trước
    drop_cols = [c for c in ["user_id", 'session', 'session_file', "label", "file", "source", "question_index", "cognitive_level"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    return X, y

def calculate_far_frr(y_true, y_pred):
    """Hàm tính False Acceptance Rate (FAR) và False Rejection Rate (FRR) từ Confusion Matrix"""
    cm = confusion_matrix(y_true, y_pred)
    far_list, frr_list = [], []
    
    for i in range(len(cm)):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (FP + FN + TP)
        
        FAR = FP / (FP + TN) if (FP + TN) > 0 else 0
        FRR = FN / (FN + TP) if (FN + TP) > 0 else 0
        far_list.append(FAR)
        frr_list.append(FRR)
        
    return np.mean(far_list), np.mean(frr_list)

# ==========================================
# 3. SETUP THUẬT TOÁN DI TRUYỀN (GA) CHO SVM
# ==========================================
def ensure_deap_creators():
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

def custom_mutation(individual, indpb):
    if random.random() < indpb: individual[0] = random.choice(PARAM_SPACE["C"])
    if random.random() < indpb: individual[1] = random.choice(PARAM_SPACE["kernel"])
    if random.random() < indpb: individual[2] = random.choice(PARAM_SPACE["gamma"])
    if random.random() < indpb: individual[3] = random.choice(PARAM_SPACE["degree"])
    return (individual,)

def svm_genetic_algorithm(X_train, y_train):
    ensure_deap_creators()
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    def evaluate(individual):
        C, kernel, gamma, degree = individual
        clf = SVC(C=C, kernel=kernel, gamma=gamma, degree=degree if kernel == "poly" else 3, 
                  class_weight="balanced", random_state=RANDOM_STATE)
        pipe = Pipeline([("scaler", MinMaxScaler()), ("classifier", clf)])
        scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
        return (float(scores.mean()),)

    toolbox = base.Toolbox()
    toolbox.register("attr_C", random.choice, PARAM_SPACE["C"])
    toolbox.register("attr_kernel", random.choice, PARAM_SPACE["kernel"])
    toolbox.register("attr_gamma", random.choice, PARAM_SPACE["gamma"])
    toolbox.register("attr_degree", random.choice, PARAM_SPACE["degree"])
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_C, toolbox.attr_kernel, toolbox.attr_gamma, toolbox.attr_degree), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", custom_mutation, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=POPULATION)
    result, _ = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=GENERATIONS, verbose=False)
    
    best = tools.selBest(result, k=1)[0]
    best_params = {
        "C": best[0], "kernel": best[1], "gamma": best[2], 
        "degree": best[3] if best[1] == "poly" else 3
    }
    return best_params

# ==========================================
# 4. CHẠY VÒNG LẶP CHO TẤT CẢ DATASET
# ==========================================
def main():
    set_seeds(RANDOM_STATE)
    if not DATA_DIR.exists():
        print(f"❌ Không tìm thấy thư mục {DATA_DIR}!")
        return

    scenarios = ["M2", "M3", "M4", "M5"]
    folds = [1, 2, 3]
    all_results = []

    print("🚀 BẮT ĐẦU HUẤN LUYỆN SVM CHO TẤT CẢ KỊCH BẢN...\n")

    for scenario in scenarios:
        for fold in folds:
            train_csv = DATA_DIR / f"train_{scenario}_fold{fold}.csv"
            test_csv = DATA_DIR / f"test_{scenario}_fold{fold}.csv"

            if not train_csv.exists() or not test_csv.exists():
                continue
                
            print(f"⏳ Đang xử lý: {scenario} - Fold {fold}...")
            start_time = time.time()

            X_train, y_train = load_xy(train_csv)
            X_test, y_test = load_xy(test_csv)

            # 1. Chạy GA tìm siêu tham số tốt nhất
            best_params = svm_genetic_algorithm(X_train, y_train)

            # 2. Train model cuối cùng với tham số tốt nhất
            model = Pipeline([
                ("scaler", MinMaxScaler()),
                ("classifier", SVC(**best_params, class_weight="balanced", random_state=RANDOM_STATE))
            ])
            model.fit(X_train, y_train)

            # 3. Dự đoán và đánh giá
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            far, frr = calculate_far_frr(y_test, y_pred)
            
            run_time = time.time() - start_time
            print(f"   ✅ Xong trong {run_time:.1f}s | ACC: {acc:.4f} | FAR: {far*100:.2f}% | FRR: {frr*100:.2f}%")

            all_results.append({
                "Scenario": scenario,
                "Fold": fold,
                "Accuracy": acc,
                "F1_Score": f1,
                "FAR(%)": far * 100,
                "FRR(%)": frr * 100,
                "Best_Params": str(best_params)
            })

    # Lưu và in báo cáo tổng hợp
    df_results = pd.DataFrame(all_results)
    
    print("\n" + "="*60)
    print("🏆 KẾT QUẢ TỔNG HỢP CÁC KỊCH BẢN (TRUNG BÌNH 3 FOLDS)")
    print("="*60)
    summary = df_results.groupby("Scenario")[["Accuracy", "FAR(%)", "FRR(%)"]].mean().reset_index()
    print(summary.to_string(index=False))
    
    df_results.to_csv(OUT_DIR / "svm_all_results.csv", index=False)
    print(f"\n📁 Chi tiết từng Fold đã được lưu tại: {OUT_DIR}/svm_all_results.csv")

if __name__ == "__main__":
    main()