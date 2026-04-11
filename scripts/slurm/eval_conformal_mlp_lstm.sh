#!/bin/bash
#SBATCH --job-name=darus_conformal_cmp
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40:1

module load rocky8-spack/master
module load cuda/12.2.2-gcc-10.3.0-5rec
module load viz-spack/0.19.2
module load python/3.10.8-gcc-10.3.0-56wj

cd ~/phd-p1-darus-uncertainty
source venv/bin/activate
export PYTHONPATH=$PWD

# task 0 = LSTM baseline, task 1 = MLP baseline
MODELS=(lstm mlp)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
MODEL_DIR="experiments/results/p1_${MODEL}_baseline"
CONFIG="${MODEL_DIR}/config.yaml"
ALPHA=0.10

echo "Job: $SLURM_JOB_NAME  Array: $SLURM_ARRAY_JOB_ID[$SLURM_ARRAY_TASK_ID]"
echo "Node: $SLURMD_NODENAME  GPU: $SLURM_GPUS_ON_NODE"
echo "Backbone: $MODEL  dir: $MODEL_DIR"
echo "Started: $(date)"

echo "=== Conformal: TEST ==="
python -u -m src.uncertainty.eval_conformal_metrics \
    --model_dir "$MODEL_DIR" \
    --config    "$CONFIG" \
    --alpha     $ALPHA \
    --split     test

echo "=== Conformal: OOD ==="
python -u -m src.uncertainty.eval_conformal_metrics \
    --model_dir "$MODEL_DIR" \
    --config    "$CONFIG" \
    --alpha     $ALPHA \
    --split     ood

echo "Done: $(date)"
