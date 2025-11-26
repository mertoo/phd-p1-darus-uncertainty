#!/bin/bash

echo "=========================================="
echo "  TRAINING ALL BASELINE MODELS"
echo "=========================================="

CONFIG_DIR="experiments/configs"
LOG_DIR="logs"
mkdir -p $LOG_DIR

LOG_FILE="$LOG_DIR/train_all.log"
echo "Training started: $(date)" > $LOG_FILE

CONFIGS=(
    "p1_lstm_seq2seq.yaml"
    "p1_gru_seq2seq.yaml"
    "p1_tcn.yaml"
    "p1_mlp.yaml"
    "p1_linear.yaml"
    "p1_naive.yaml"
)

for CFG in "${CONFIGS[@]}"; do
    echo ""
    echo "------------------------------------------"
    echo " Training: $CFG"
    echo "------------------------------------------"

    CMD="python3 -m src.training.train_baseline --config $CONFIG_DIR/$CFG"

    echo "[RUN] $CMD" | tee -a $LOG_FILE

    # Run and append to log file
    $CMD 2>&1 | tee -a $LOG_FILE

    # Check exit code
    if [ $? -ne 0 ]; then
        echo "❌ Error: Training failed for $CFG" | tee -a $LOG_FILE
        echo "Stopping training pipeline." | tee -a $LOG_FILE
        exit 1
    fi

    echo "✔ Completed $CFG" | tee -a $LOG_FILE
done

echo ""
echo "=========================================="
echo " ALL MODELS TRAINED SUCCESSFULLY"
echo " Logs saved to $LOG_FILE"
echo "=========================================="
