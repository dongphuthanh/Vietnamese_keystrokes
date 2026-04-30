import os
import json
import numpy as np

# Define the models and scenarios to scan
MODELS = ['svm', 'xgb', 'mlp']
SCENARIOS = ['M2', 'M3', 'M4', 'M5']
RUNS_DIR = 'runs'

def main():
    print("=================================================")
    print("      AGGREGATING RESULTS ACROSS 3 FOLDS         ")
    print("=================================================")
    
    if not os.path.exists(RUNS_DIR):
        print(f"[-] Directory '{RUNS_DIR}' not found. Please run the training scripts first.")
        return

    # Dictionary to save the final aggregated results
    summary = {}

    for model in MODELS:
        for scenario in SCENARIOS:
            total_cm = None
            accuracies = [] # List to store accuracy of each fold
            folds_found = 0
            
            # Loop through fold 1, 2, 3
            for fold in [1, 2, 3]:
                # Construct folder name (e.g., svm_M2_fold1 or svm_m2_fold1)
                folder_name = f"{model}_{scenario}_fold{fold}"
                json_path = os.path.join(RUNS_DIR, folder_name, "results.json")
                
                if not os.path.exists(json_path):
                    folder_name_lower = f"{model}_{scenario.lower()}_fold{fold}"
                    json_path = os.path.join(RUNS_DIR, folder_name_lower, "results.json")

                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                        # Extract Confusion Matrix
                        if 'confusion_matrix' in data:
                            cm = np.array(data['confusion_matrix'])
                            
                            # 1. Add to the total confusion matrix
                            if total_cm is None:
                                total_cm = np.zeros_like(cm)
                            total_cm += cm
                            
                            # 2. Calculate accuracy for this specific fold
                            # Accuracy = (Sum of correctly predicted / Total predictions)
                            fold_accuracy = np.trace(cm) / np.sum(cm)
                            accuracies.append(fold_accuracy)
                            
                            folds_found += 1
                                
                    except Exception as e:
                        print(f"[-] Error reading {json_path}: {e}")

            # If we found at least one fold for this model + scenario
            if folds_found > 0:
                print(f"\n[+] {model.upper()} - {scenario} (Aggregated from {folds_found} folds):")
                
                # Calculate Mean and Standard Deviation of Accuracy (converted to %)
                mean_acc = np.mean(accuracies) * 100 if accuracies else 0.0
                std_acc = np.std(accuracies) * 100 if accuracies else 0.0
                
                print(f"    Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
                
                if total_cm is not None:
                    print("    Combined Confusion Matrix:")
                    # Print formatted matrix
                    for row in total_cm:
                        print("    " + str(row))
                
                # Store in summary dictionary
                summary[f"{model.upper()}_{scenario}"] = {
                    "accuracy_mean_percent": float(mean_acc),
                    "accuracy_std_percent": float(std_acc),
                    "combined_confusion_matrix": total_cm.tolist() if total_cm is not None else []
                }

    # Save everything to a final summary JSON
    summary_path = "final_aggregated_results.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
        
    print("\n=================================================")
    print(f" All results aggregated and saved to: {summary_path}")
    print("=================================================")

if __name__ == "__main__":
    main()