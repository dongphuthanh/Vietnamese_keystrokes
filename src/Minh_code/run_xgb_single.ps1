# Prompt user for the scenario they want to run
$m = Read-Host "Enter the scenario you want to run (M2, M3, M4, or M5)"
$m = $m.ToUpper().Trim()

# Validate user input
if ($m -notin @("M2", "M3", "M4", "M5")) {
    Write-Host "[-] Invalid input! Please run the script again and type exactly M2, M3, M4, or M5." -ForegroundColor Red
    exit
}

$folds = 1..3

Write-Host "`n=================================================" -ForegroundColor Green
Write-Host " STARTING XGBOOST TRAINING FOR SCENARIO: $m (3 FOLDS) " -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green

# Loop through 3 folds for the selected scenario
foreach ($f in $folds) {
    
    $train_file = "cognitive_level_datasets\train_${m}_fold${f}.csv"
    $test_file = "cognitive_level_datasets\test_${m}_fold${f}.csv"
    $out_dir = "runs_cognitive_level\xgb_${m}_fold${f}"
    
    Write-Host "`n[+] Running: XGBoost | Scenario: $m | Fold: $f" -ForegroundColor Yellow
    Write-Host "Train file: $train_file" -ForegroundColor Gray
    Write-Host "Test file : $test_file" -ForegroundColor Gray
    
    # Execute Python script for XGBoost
    python XGB.py --train $train_file --test $test_file --outdir $out_dir --feature-percentage 50 --seed 42 --cm-labels 0 1 2 3 4
}

Write-Host "`n=================================================" -ForegroundColor Green
Write-Host " SUCCESSFULLY COMPLETED 3 FOLDS FOR $m (XGBOOST)! " -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green