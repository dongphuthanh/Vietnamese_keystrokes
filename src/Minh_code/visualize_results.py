#!/usr/bin/env python3
"""
Tạo visualization cho kết quả experiments
- Bar charts so sánh accuracy giữa các kịch bản
- Heatmaps cho confusion matrices
- Line plots cho FAR/FRR
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================
# CẤU HÌNH
# ==========================================
RUNS_DIR = Path('runs_cognitive_level')
OUTPUT_DIR = RUNS_DIR / 'visualizations'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = {
    0: "B",
    1: "P", 
    2: "T",
    3: "F_P",
    4: "F_T"
}

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# ==========================================
# LOAD DỮ LIỆU
# ==========================================
def load_summary():
    """Load summary results từ JSON"""
    json_path = RUNS_DIR / "summary_results.json"
    
    if not json_path.exists():
        print(f"❌ Không tìm thấy {json_path}")
        print("   Vui lòng chạy collect_results_cognitive.py trước!")
        return None
    
    with open(json_path, 'r') as f:
        return json.load(f)

# ==========================================
# PLOT 1: Accuracy Comparison
# ==========================================
def plot_accuracy_comparison(summary):
    """So sánh accuracy giữa các kịch bản"""
    scenarios = []
    accuracies = []
    errors = []
    
    for scenario in ['M2', 'M3', 'M4', 'M5']:
        if scenario in summary:
            scenarios.append(scenario)
            accuracies.append(summary[scenario]['accuracy_mean'])
            errors.append(summary[scenario]['accuracy_std'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(scenarios))
    bars = ax.bar(x, accuracies, yerr=errors, capsize=5, 
                   color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'],
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Thêm giá trị lên đầu cột
    for i, (acc, err) in enumerate(zip(accuracies, errors)):
        ax.text(i, acc + err + 1, f'{acc:.2f}±{err:.2f}%', 
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Scenarios', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Comparison Across Scenarios', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=11)
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'accuracy_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

# ==========================================
# PLOT 2: F1 Score Comparison
# ==========================================
def plot_f1_comparison(summary):
    """So sánh F1 score giữa các kịch bản"""
    scenarios = []
    f1_scores = []
    errors = []
    
    for scenario in ['M2', 'M3', 'M4', 'M5']:
        if scenario in summary:
            scenarios.append(scenario)
            f1_scores.append(summary[scenario]['f1_mean'])
            errors.append(summary[scenario]['f1_std'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(scenarios))
    bars = ax.bar(x, f1_scores, yerr=errors, capsize=5, 
                   color=['#9b59b6', '#1abc9c', '#e67e22', '#c0392b'],
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for i, (f1, err) in enumerate(zip(f1_scores, errors)):
        ax.text(i, f1 + err + 1, f'{f1:.2f}±{err:.2f}%', 
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Scenarios', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('F1 Score Comparison Across Scenarios', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=11)
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'f1_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

# ==========================================
# PLOT 3: FAR/FRR Comparison
# ==========================================
def plot_far_frr_comparison(summary):
    """So sánh FAR và FRR giữa các kịch bản"""
    scenarios = []
    far_values = []
    frr_values = []
    
    for scenario in ['M2', 'M3', 'M4', 'M5']:
        if scenario in summary:
            scenarios.append(scenario)
            far_values.append(summary[scenario]['far_percent'])
            frr_values.append(summary[scenario]['frr_percent'])
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, far_values, width, label='FAR', 
                   color='#e74c3c', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, frr_values, width, label='FRR', 
                   color='#3498db', alpha=0.8, edgecolor='black')
    
    # Thêm giá trị
    for i, (far, frr) in enumerate(zip(far_values, frr_values)):
        ax.text(i - width/2, far + 0.1, f'{far:.2f}%', 
                ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, frr + 0.1, f'{frr:.2f}%', 
                ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Scenarios', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('FAR and FRR Comparison Across Scenarios', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=11)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'far_frr_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

# ==========================================
# PLOT 4: Confusion Matrices
# ==========================================
def plot_confusion_matrices(summary):
    """Vẽ confusion matrices cho tất cả kịch bản"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, scenario in enumerate(['M2', 'M3', 'M4', 'M5']):
        if scenario not in summary:
            continue
        
        cm = np.array(summary[scenario]['combined_confusion_matrix'])
        
        # Normalize confusion matrix (theo hàng)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
        
        # Labels
        labels = [LABEL_NAMES.get(i, str(i)) for i in range(cm.shape[0])]
        
        # Plot
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=labels, yticklabels=labels,
                   ax=axes[idx], cbar_kws={'label': 'Normalized Count'},
                   linewidths=0.5, linecolor='gray')
        
        axes[idx].set_title(f'{scenario} - Confusion Matrix (Normalized)', 
                           fontsize=12, fontweight='bold', pad=10)
        axes[idx].set_xlabel('Predicted', fontsize=10, fontweight='bold')
        axes[idx].set_ylabel('True', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'confusion_matrices.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

# ==========================================
# PLOT 5: Overall Comparison
# ==========================================
def plot_overall_comparison(summary):
    """Vẽ biểu đồ so sánh tổng thể tất cả metrics"""
    scenarios = ['M2', 'M3', 'M4', 'M5']
    metrics = {
        'Accuracy': [],
        'F1 Score': [],
        'FAR': [],
        'FRR': []
    }
    
    for scenario in scenarios:
        if scenario in summary:
            metrics['Accuracy'].append(summary[scenario]['accuracy_mean'])
            metrics['F1 Score'].append(summary[scenario]['f1_mean'])
            metrics['FAR'].append(summary[scenario]['far_percent'])
            metrics['FRR'].append(summary[scenario]['frr_percent'])
    
    # Normalize metrics to 0-100 scale for comparison
    # (FAR and FRR are already percentages, but small values)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(scenarios))
    width = 0.2
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    for i, (metric, values) in enumerate(metrics.items()):
        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, values, width, label=metric, 
                      color=colors[i], alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Scenarios', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value (%)', fontsize=12, fontweight='bold')
    ax.set_title('Overall Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'overall_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

# ==========================================
# MAIN
# ==========================================
def main():
    print("="*70)
    print("   📊 TẠO VISUALIZATION CHO EXPERIMENTS")
    print("="*70)
    
    # Load data
    print("\n📂 Đang load dữ liệu...")
    summary = load_summary()
    
    if summary is None:
        return
    
    print(f"✅ Đã load {len(summary)} kịch bản\n")
    
    # Tạo các plots
    print("🎨 Đang tạo visualizations...\n")
    
    plot_accuracy_comparison(summary)
    plot_f1_comparison(summary)
    plot_far_frr_comparison(summary)
    plot_confusion_matrices(summary)
    plot_overall_comparison(summary)
    
    print("\n" + "="*70)
    print(f"✅ HOÀN TẤT! Đã lưu visualizations tại: {OUTPUT_DIR}")
    print("="*70)
    print("\n📁 Các file được tạo:")
    for f in OUTPUT_DIR.glob("*.png"):
        print(f"   - {f.name}")

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"❌ Thiếu thư viện: {e}")
        print("\nCài đặt thư viện cần thiết:")
        print("   pip install matplotlib seaborn")