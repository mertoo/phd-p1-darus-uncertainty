#!/bin/bash
# One-off CPU-only job that builds the v3 Python environment (no GPU requested).
# The April 2026 venv points at a Python install that is no longer accessible.
#   sbatch scripts/slurm/v3/setup_env.sh
# Creates $DARUS_ENV (default ~/envs/darus-v3), refuses to overwrite it, then
# runs the test suite and the pilot preflight inside the new environment.
#SBATCH --job-name=darus_v3_setup_env
#SBATCH --output=logs/slurm/v3/%x_%j.out
#SBATCH --error=logs/slurm/v3/%x_%j.err
#SBATCH --partition=common
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --no-requeue
set -euo pipefail
cd ~/phd-p1-darus-uncertainty
ENV="${DARUS_ENV:-$HOME/envs/darus-v3}"
echo "node=$SLURMD_NODENAME commit=$(git rev-parse HEAD) start=$(date -Is) env=$ENV"
if [ -e "$ENV" ]; then echo "ERROR: $ENV already exists; refusing to overwrite"; exit 1; fi
curl -sSf -o /dev/null --max-time 20 https://pypi.org/simple/pip/ \
  || { echo "ERROR: no outbound internet on compute node; download on the login node instead"; exit 1; }

module load rocky8/all
module load micromamba/2.6.1
export MAMBA_ROOT_PREFIX="$HOME/.micromamba"
micromamba create -y -p "$ENV" -c conda-forge --override-channels python=3.11 pip
PY="$ENV/bin/python"
"$PY" -m pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
"$PY" -m pip install --no-cache-dir -r requirements.txt pytest==8.3.5
"$PY" -m pip freeze > "logs/slurm/v3/env_darus-v3_freeze_${SLURM_JOB_ID}.txt"
"$PY" -c "import sys, torch, numpy, pandas; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda build', torch.version.cuda, 'numpy', numpy.__version__, 'pandas', pandas.__version__)"

export PYTHONPATH=$PWD
"$PY" -m pytest tests -q
"$PY" -m scripts.pilot_preflight
echo "done $(date -Is)"
