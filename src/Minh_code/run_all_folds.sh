#!/bin/bash

MODELS=("SVM" "MLP" "XGB")
MS=("m2" "m3" "m4" "m5")
SRC="C:/Users/ADMIN/Documents/Vietnamese_keystrokes/src/Minh code"
RUNS="C:/Users/ADMIN/Documents/Vietnamese_keystrokes/runs"

for model in "${MODELS[@]}"; do
    for m in "${MS[@]}"; do
        for fold in 1 2 3; do
            echo "Running $model $m fold$fold..."
            python "$SRC/$model.py" \
                --train "$SRC/train_${m}_fold${fold}.csv" \
                --test "$SRC/test_${m}_fold${fold}.csv" \
                --outdir "$RUNS/${model,,}_${m}_fold${fold}" \
                --feature-percentage 50 \
                --seed 42 \
                --cm-labels 0 1 2
        done
    done
done