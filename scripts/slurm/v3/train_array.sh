#!/bin/bash
# Train one row of run_list.tsv per array task. DO NOT SUBMIT before the
# protocol (EXPERIMENT_PROTOCOL.md) is signed off.
#   sbatch --array=<rows> scripts/slurm/v3/train_array.sh [STAGE]
# Array index i selects data row i (1-based, header excluded) among rows of STAGE.
#SBATCH --job-name=darus_v3_train
#SBATCH --output=logs/slurm/v3/%x_%A_%a.out
#SBATCH --error=logs/slurm/v3/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40:1
set -euo pipefail
module load rocky8-spack/master
module load cuda/12.2.2-gcc-10.3.0-5rec
module load python/3.10.8-gcc-10.3.0-56wj
cd ~/phd-p1-darus-uncertainty
source venv/bin/activate
export PYTHONPATH=$PWD
mkdir -p logs/slurm/v3

STAGE="${1:?stage (0, 1a, 1b, 1c, 1d)}"
ROW=$(awk -F'\t' -v s="$STAGE" 'NR>1 && $1==s' scripts/slurm/v3/run_list.tsv | sed -n "${SLURM_ARRAY_TASK_ID}p")
[ -n "$ROW" ] || { echo "no row ${SLURM_ARRAY_TASK_ID} for stage ${STAGE}"; exit 1; }
IFS=$'\t' read -r _ CFG SEED RUN FEAT <<< "$ROW"
CONFIG="experiments/configs/v3/${CFG}.yaml"
if [ -n "$FEAT" ]; then
  mkdir -p experiments/configs/v3/generated
  sed "s/^  features: .*/  features: \"${FEAT}\"/" "$CONFIG" > "experiments/configs/v3/generated/${CFG}_${FEAT}.yaml"
  CONFIG="experiments/configs/v3/generated/${CFG}_${FEAT}.yaml"
fi
echo "commit $(git rev-parse HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
echo "stage=$STAGE config=$CONFIG seed=$SEED run=$RUN node=$SLURMD_NODENAME start=$(date -Is)"
python -u -m src.training.train --config "$CONFIG" --seed "$SEED" --run_name "$RUN"
echo "done $(date -Is)"
