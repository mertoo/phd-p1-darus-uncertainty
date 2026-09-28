#!/bin/bash
# Evaluate one method on one run (or one ensemble). DO NOT SUBMIT before sign-off.
#   sbatch scripts/slurm/v3/eval.sh point      experiments/runs/v3/lstm_ens/rep0/model_0  lstm_single_rep0
#   N_MEMBERS=5 sbatch --export=ALL scripts/slurm/v3/eval.sh ensemble 'experiments/runs/v3/lstm_ens/rep0/model_*' lstm_ens_rep0
#   sbatch scripts/slurm/v3/eval.sh mc_dropout experiments/runs/v3/lstm_dropout/rep0      lstm_mcd_rep0
#   sbatch scripts/slurm/v3/eval.sh gaussian   experiments/runs/v3/lstm_gaussian/rep0     lstm_gauss_rep0
#SBATCH --job-name=darus_v3_eval
#SBATCH --output=logs/slurm/v3/%x_%j.out
#SBATCH --error=logs/slurm/v3/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40:1
set -euo pipefail
cd ~/phd-p1-darus-uncertainty
ENV="${DARUS_ENV:-$HOME/envs/darus-v3}"      # built by scripts/slurm/v3/setup_env.sh
[ -x "$ENV/bin/python" ] || { echo "ERROR: environment $ENV missing"; exit 1; }
export PATH="$ENV/bin:$PATH"
export PYTHONPATH=$PWD
METHOD="$1"; RUNS="$2"; TAG="$3"
echo "commit $(git rev-parse HEAD) method=$METHOD runs=$RUNS start=$(date -Is)"
python -u -m src.evaluation.benchmark_eval --method "$METHOD" --runs "$RUNS" \
    --passes 200 --mc_seed 0 ${N_MEMBERS:+--n_members $N_MEMBERS} --out "experiments/eval/v3/${TAG}"
echo "done $(date -Is)"
