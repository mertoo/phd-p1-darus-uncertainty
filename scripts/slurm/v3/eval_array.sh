#!/bin/bash
# Full-benchmark evaluations, one row of eval_list.tsv per array task.
# DO NOT SUBMIT without authorisation for stage C.
#   sbatch --array=1-35 scripts/slurm/v3/eval_array.sh
# A missing checkpoint (or a wrong number of ensemble members) SKIPS the task
# with exit code 3; historical checkpoints are never used as a fallback.
#SBATCH --job-name=darus_v3_eval
#SBATCH --output=logs/slurm/v3/%x_%A_%a.out
#SBATCH --error=logs/slurm/v3/%x_%A_%a.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40:1
#SBATCH --no-requeue
set -uo pipefail
cd "${DARUS_REPO:-$HOME/phd-p1-darus-uncertainty}"
ENV="${DARUS_ENV:-$HOME/envs/darus-v3}"
[ -x "$ENV/bin/python" ] || { echo "ERROR: environment $ENV missing"; exit 1; }
export PATH="$ENV/bin:$PATH"
export PYTHONPATH=$PWD
# '|' keeps empty fields (a tab IFS would collapse the empty n_members column)
IFS='|' read -r METHOD RUNS NMEM TAG < <(awk -F'\t' -v t="$SLURM_ARRAY_TASK_ID" 'NR>1 && $1==t {print $2"|"$3"|"$4"|"$5}' scripts/slurm/v3/eval_list.tsv)
[ -n "${TAG:-}" ] || { echo "ERROR: no eval_list row for task $SLURM_ARRAY_TASK_ID"; exit 1; }
ROOT="${DARUS_RUNS:-experiments/runs/v3}"
shopt -s nullglob
CKPTS=()
for f in $ROOT/$RUNS/best_model.pt; do [ -f "$f" ] && CKPTS+=("$f"); done   # literal paths bypass nullglob
WANT=${NMEM:-1}
if [ ${#CKPTS[@]} -ne "$WANT" ]; then
  echo "SKIPPED_MISSING_CHECKPOINT tag=$TAG method=$METHOD found=${#CKPTS[@]} expected=$WANT pattern=$ROOT/$RUNS"
  exit 3
fi
echo "commit $(git rev-parse HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l) node=$SLURMD_NODENAME tag=$TAG start=$(date -Is)"
python -u -m src.evaluation.benchmark_eval --method "$METHOD" --runs "$ROOT/$RUNS" \
    ${NMEM:+--n_members $NMEM} --passes 200 --mc_seed 0 --out "${DARUS_EVAL_OUT:-experiments/eval/v3}/$TAG"
RC=$?
echo "done $(date -Is) rc=$RC"
exit $RC
