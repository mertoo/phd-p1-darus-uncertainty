#!/bin/bash
#SBATCH --job-name=darus_gaussian_lstm
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
# Adjust the partition name to match your cluster:
#SBATCH --partition=gpu

# ── Environment ────────────────────────────────────────────────────────────────
# Load any cluster modules your site requires, e.g.:
#   module load python/3.11 cuda/12.1 cudnn/8.9
# Then activate the project virtualenv:
source venv/bin/activate

# Ensure the project root is on PYTHONPATH so `src.*` imports resolve
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG="experiments/configs/uncertainty/p2_lstm_gaussian.yaml"

# ── Run ────────────────────────────────────────────────────────────────────────
echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"
echo "Config: $CONFIG"
echo "Started: $(date)"

mkdir -p logs/slurm

python -u -m src.uncertainty.train_lstm_gaussian --config "$CONFIG"

echo "Finished: $(date)"
