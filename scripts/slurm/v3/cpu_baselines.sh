#!/bin/bash
# CPU-only: closed-form ridge (penalty chosen on validation) and the naive baseline.
# DO NOT SUBMIT without authorisation for stage B.
#SBATCH --job-name=darus_v3_cpu_baselines
#SBATCH --output=logs/slurm/v3/%x_%j.out
#SBATCH --error=logs/slurm/v3/%x_%j.err
#SBATCH --partition=common
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --no-requeue
set -euo pipefail
cd ~/phd-p1-darus-uncertainty
ENV="${DARUS_ENV:-$HOME/envs/darus-v3}"
[ -x "$ENV/bin/python" ] || { echo "ERROR: environment $ENV missing"; exit 1; }
export PATH="$ENV/bin:$PATH"
export PYTHONPATH=$PWD
echo "commit $(git rev-parse HEAD) node=$SLURMD_NODENAME start=$(date -Is)"
python -u -m src.training.fit_ridge --config experiments/configs/v3/linear.yaml --run_name ridge
python -u -m src.training.train --config experiments/configs/v3/naive.yaml --seed 0 --run_name naive
echo "done $(date -Is)"
