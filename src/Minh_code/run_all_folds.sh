#!/bin/bash

MODELS=("SVM" "MLP" "XGB")
MS=("m2" "m3" "m4" "m5")
SRC="/home/mtcd001/PycharmProjects/Vietnamese_keystrokes/src"
RUNS="/home/mtcd001/PycharmProjects/Vietnamese_keystrokes/runs"

for model in "${MODELS[@]}"; do
    for m in "${MS[@]}"; do
        for fold in 1 2 3 4 5; do
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