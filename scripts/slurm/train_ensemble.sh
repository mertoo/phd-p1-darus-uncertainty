#!/bin/bash
#SBATCH --job-name=darus_ensemble
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40:1
#SBATCH --array=0-4

module load rocky8-spack/master
module load cuda/12.2.2-gcc-10.3.0-5rec
module load viz-spack/0.19.2
module load python/3.10.8-gcc-10.3.0-56wj

cd ~/phd-p1-darus-uncertainty
source venv/bin/activate
export PYTHONPATH=$PWD

MEMBER=$SLURM_ARRAY_TASK_ID
MODEL_DIR="experiments/results/uncertainty/ensemble_lstm/model_${MEMBER}"
CONFIG_SRC="experiments/configs/uncertainty/p2_lstm_ensemble.yaml"
CONFIG_DST="${MODEL_DIR}/config.yaml"

mkdir -p "$MODEL_DIR"
cp "$CONFIG_SRC" "$CONFIG_DST"

echo "Job: $SLURM_JOB_NAME  Array: $SLURM_ARRAY_JOB_ID  Member: $MEMBER"
echo "Node: $SLURMD_NODENAME  GPU: $SLURM_GPUS_ON_NODE"
echo "Started: $(date)"

python -u -m src.training.train_baseline --config "$CONFIG_DST"

echo "Done member ${MEMBER}: $(date)"
