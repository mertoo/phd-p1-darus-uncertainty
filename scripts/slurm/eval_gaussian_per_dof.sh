#!/bin/bash
#SBATCH --job-name=darus_eval_gauss_per_dof
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

echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME  GPU: $SLURM_GPUS_ON_NODE"
echo "Started: $(date)"

echo "=== Per-DoF diagnostics: TEST split ==="
python -u -m src.uncertainty.eval_gaussian_diagnostics_per_dof \
    --model_dir experiments/results/p2_lstm_gaussian \
    --config experiments/configs/uncertainty/p2_lstm_gaussian.yaml \
    --split test \
    --calibrate_per_dof

echo "=== Per-DoF diagnostics: OOD split ==="
python -u -m src.uncertainty.eval_gaussian_diagnostics_per_dof \
    --model_dir experiments/results/p2_lstm_gaussian \
    --config experiments/configs/uncertainty/p2_lstm_gaussian.yaml \
    --split ood \
    --calibrate_per_dof

echo "Done: $(date)"
