#!/bin/bash
#SBATCH --job-name=darus_eval_ensemble_mlp
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
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

ENSEMBLE_DIR="experiments/results/uncertainty/ensemble_mlp"
CONFIG="experiments/configs/uncertainty/p2_mlp_ensemble.yaml"

echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME  GPU: $SLURM_GPUS_ON_NODE"
echo "Started: $(date)"

echo "=== MLP Ensemble: TEST ==="
python -u -m src.uncertainty.eval_ensemble \
    --ensemble_dir "$ENSEMBLE_DIR" \
    --config       "$CONFIG" \
    --split        test

echo "=== MLP Ensemble: OOD ==="
python -u -m src.uncertainty.eval_ensemble \
    --ensemble_dir "$ENSEMBLE_DIR" \
    --config       "$CONFIG" \
    --split        ood

echo "Done: $(date)"
