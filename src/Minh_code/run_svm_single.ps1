# Hoi nguoi dung muon chay kich ban nao
$m = Read-Host "Nhap kich ban ban muon chay (M2, M3, M4 hoac M5)"
$m = $m.ToUpper().Trim()

# Kiem tra xem nhap co dung khong
if ($m -notin @("M2", "M3", "M4", "M5")) {
    Write-Host "[-] Nhap sai! Vui long chay lai va go dung M2, M3, M4 hoac M5." -ForegroundColor Red
    exit
}

$folds = 1..3

Write-Host "`n=================================================" -ForegroundColor Green
Write-Host "   BAT DAU CHAY SVM CHO KICH BAN: $m (3 FOLDS)   " -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green

# Vong lap chay 3 fold cho dung kich ban
foreach ($f in $folds) {
    
    $train_file = "prepared_datasets\train_${m}_fold${f}.csv"
    $test_file = "prepared_datasets\test_${m}_fold${f}.csv"
    $out_dir = "runs\svm_${m}_fold${f}"
    
    Write-Host "`n[+] Dang chay: SVM | Kich ban: $m | Fold: $f" -ForegroundColor Yellow
    Write-Host "Train: $train_file" -ForegroundColor Gray
    Write-Host "Test : $test_file" -ForegroundColor Gray
    
    # Lenh goi Python chay SVM
    python SVM.py --train $train_file --test $test_file --outdir $out_dir --feature-percentage 50 --seed 42 --cm-labels 0 1 2 3 4
}

Write-Host "`n=================================================" -ForegroundColor Green
Write-Host "   HOAN TAT CHAY 3 FOLDS CUA KICH BAN $m!        " -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green