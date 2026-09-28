#!/bin/bash
# TIMING PILOT, evaluation part (runs after every pilot_train.sh task ends,
# successful or not, so failures are reported). Scores the VALIDATION split
# only (--eval_splits val): test/OOD are not touched by the pilot.
# Hard cap: 1 job x 2 h x 1 GPU = 2 GPU-h. No automatic requeue.
#SBATCH --job-name=darus_pilot_eval
#SBATCH --output=logs/slurm/v3/%x_%j.out
#SBATCH --error=logs/slurm/v3/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40:1
#SBATCH --no-requeue
set -uo pipefail
FAILED=0
cd ~/phd-p1-darus-uncertainty
ENV="${DARUS_ENV:-$HOME/envs/darus-v3}"      # built by scripts/slurm/v3/setup_env.sh
[ -x "$ENV/bin/python" ] || { echo "ERROR: environment $ENV missing"; exit 1; }
export PATH="$ENV/bin:$PATH"
export PYTHONPATH=$PWD
R=experiments/runs/v3/pilot; E=experiments/eval/v3/pilot
ev() { local t0=$(date +%s); python -u -m src.evaluation.benchmark_eval --eval_splits val "$@" || { echo "EVAL FAILED: $*"; FAILED=1; }; echo "WALL $* $(( $(date +%s) - t0 ))s"; }
echo "commit $(git rev-parse HEAD) node=$SLURMD_NODENAME start=$(date -Is)"
ev --method point      --runs $R/lstm_s9001                  --out $E/point_lstm
ev --method ensemble   --runs "$R/lstm_s900[12]" --n_members 2 --out $E/ens_lstm
ev --method mc_dropout --runs $R/lstm_dropout_s9001 --passes 200 --mc_seed 0 --out $E/mcd_lstm
ev --method gaussian   --runs $R/lstm_gaussian_s9001          --out $E/gauss_lstm
ev --method gaussian   --runs $R/mlp_gaussian_s9001           --out $E/gauss_mlp
ev --method point      --runs $R/mlp_s9001                    --out $E/point_mlp
python -u -m src.training.fit_ridge --config experiments/configs/v3/linear.yaml --run_name pilot/ridge || { echo "RIDGE FAILED"; FAILED=1; }
cat > $E/bootstrap_spec.yaml <<SPEC
level: 90
n_boot: 2000
seed: 0
splits: [val]
methods:
  conformal_lstm: {tag: conformal_horizon_channel, evals: ["$E/point_lstm"]}
  ensemble_lstm:  {tag: ensemble_horizon_channel,  evals: ["$E/ens_lstm"]}
  mcd_lstm:       {tag: mc_dropout_horizon_channel, evals: ["$E/mcd_lstm"]}
  gaussian_lstm:  {tag: gaussian_horizon_channel,  evals: ["$E/gauss_lstm"]}
  gaussian_mlp:   {tag: gaussian_horizon_channel,  evals: ["$E/gauss_mlp"]}
pairs:
  - [ensemble_lstm, conformal_lstm]
SPEC
t0=$(date +%s); python -u -m src.analysis.bootstrap --spec $E/bootstrap_spec.yaml --out $E/bootstrap || { echo "BOOTSTRAP FAILED"; FAILED=1; }
echo "WALL bootstrap $(( $(date +%s) - t0 ))s"
python -u -m scripts.pilot_report --runs $R --evals $E --out $E/pilot_report.json || FAILED=1
du -sh $E/* experiments/runs/v3/pilot/* | sort -h | tail -20
echo "done $(date -Is) failed=$FAILED"
exit $FAILED
