#!/bin/bash
#SBATCH --job-name=darus_baselines
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
# Train all 6 P1 baselines in parallel.
# Submit with: sbatch --array=0-5 scripts/slurm/train_baselines.sh
#SBATCH --array=0-5

source venv/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH}"

CONFIGS=(
    "experiments/configs/p1_lstm_seq2seq.yaml"
    "experiments/configs/p1_gru_seq2seq.yaml"
    "experiments/configs/p1_tcn.yaml"
    "experiments/configs/p1_mlp.yaml"
    "experiments/configs/p1_linear.yaml"
    "experiments/configs/p1_naive.yaml"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "Job array: $SLURM_ARRAY_JOB_ID  Task: $SLURM_ARRAY_TASK_ID"
echo "Config: $CONFIG"
echo "Started: $(date)"

mkdir -p logs/slurm

python -u -m src.training.train_baseline --config "$CONFIG"

echo "Finished: $(date)"
