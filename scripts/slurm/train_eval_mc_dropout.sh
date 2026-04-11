#!/bin/bash
#SBATCH --job-name=darus_mc_dropout
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40:1

module load rocky8-spack/master
module load cuda/12.2.2-gcc-10.3.0-5rec
module load viz-spack/0.19.2
module load python/3.10.8-gcc-10.3.0-56wj

cd ~/phd-p1-darus-uncertainty
source venv/bin/activate
export PYTHONPATH=$PWD

MODEL_DIR="experiments/results/uncertainty/mc_dropout_lstm"
CONFIG_SRC="experiments/configs/uncertainty/p2_lstm_mc_dropout.yaml"
CONFIG_DST="${MODEL_DIR}/config.yaml"

mkdir -p "$MODEL_DIR"
cp "$CONFIG_SRC" "$CONFIG_DST"

echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME  GPU: $SLURM_GPUS_ON_NODE"
echo "Started: $(date)"

echo "=== Training MC Dropout LSTM (dropout=0.2) ==="
python -u -m src.training.train_baseline --config "$CONFIG_DST"

echo "=== MC Dropout eval: TEST split ==="
python -u -m src.evaluation.run_mc_dropout_eval \
    --model_dir "$MODEL_DIR" \
    --config "$CONFIG_DST" \
    --n_samples 200 \
    --split test

echo "=== MC Dropout eval: OOD split ==="
python -u -m src.evaluation.run_mc_dropout_eval \
    --model_dir "$MODEL_DIR" \
    --config "$CONFIG_DST" \
    --n_samples 200 \
    --split ood

echo "Done: $(date)"
