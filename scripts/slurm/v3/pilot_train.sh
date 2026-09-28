#!/bin/bash
# TIMING PILOT, training part. DO NOT SUBMIT without compute authorisation.
#   python -m scripts.pilot_preflight            # must print PREFLIGHT OK
#   TRAIN=$(sbatch --parsable --array=1-7 scripts/slurm/v3/pilot_train.sh)
#   sbatch --dependency=afterany:$TRAIN scripts/slurm/v3/pilot_eval.sh
# 3 epochs, no early stopping, full real data, primary v3 split and scaling.
# Hard cap: 7 tasks x 30 min x 1 GPU = 3.5 GPU-h. No automatic requeue.
#SBATCH --job-name=darus_pilot_train
#SBATCH --output=logs/slurm/v3/%x_%A_%a.out
#SBATCH --error=logs/slurm/v3/%x_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40:1
#SBATCH --no-requeue
set -euo pipefail
module load rocky8-spack/master
module load cuda/12.2.2-gcc-10.3.0-5rec
module load python/3.10.8-gcc-10.3.0-56wj
cd ~/phd-p1-darus-uncertainty
source venv/bin/activate
export PYTHONPATH=$PWD
mkdir -p logs/slurm/v3
IFS=$'\t' read -r _ CFG SEED RUN < <(awk -F'\t' -v t="$SLURM_ARRAY_TASK_ID" 'NR>1 && $1==t' scripts/slurm/v3/pilot_runs.tsv)
echo "commit $(git rev-parse HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l) node=$SLURMD_NODENAME gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "config=$CFG seed=$SEED run=$RUN start=$(date -Is)"
python -u -m src.training.train --config "experiments/configs/v3/${CFG}.yaml" --seed "$SEED" \
    --run_name "$RUN" --epochs 3 --no_patience
echo "done $(date -Is)"
