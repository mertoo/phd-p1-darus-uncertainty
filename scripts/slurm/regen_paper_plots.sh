#!/bin/bash
#SBATCH --job-name=darus_paper_plots
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

BASELINE_DIR="experiments/results/uncertainty/ensemble_lstm/model_3"
BASELINE_CFG="experiments/configs/p1_lstm_seq2seq.yaml"
GAUSSIAN_DIR="experiments/results/p2_lstm_gaussian"
GAUSSIAN_CFG="experiments/configs/uncertainty/p2_lstm_gaussian.yaml"
ENSEMBLE_DIR="experiments/results/uncertainty/ensemble_lstm"
ENSEMBLE_CFG="experiments/configs/uncertainty/p2_lstm_ensemble.yaml"
MC_DIR="experiments/results/uncertainty/mc_dropout_lstm"
MC_CFG="${MC_DIR}/config.yaml"

echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME  GPU: $SLURM_GPUS_ON_NODE"
echo "Started: $(date)"

# ── 1. Baseline LSTM prediction plots (→ prediction_phi_test.png etc.) ────────
echo "=== Baseline LSTM: prediction plots ==="
python -u -m src.evaluation.run_eval \
    --model_dir "$BASELINE_DIR" \
    --config "$BASELINE_CFG"

# ── 2. Gaussian LSTM (→ gaussian_phi_test.png, gaussian_phi_ood.png) ──────────
echo "=== Gaussian LSTM: test ==="
python -u -m src.uncertainty.eval_lstm_gaussian \
    --model_dir "$GAUSSIAN_DIR" \
    --config "$GAUSSIAN_CFG" \
    --split test

echo "=== Gaussian LSTM: OOD ==="
python -u -m src.uncertainty.eval_lstm_gaussian \
    --model_dir "$GAUSSIAN_DIR" \
    --config "$GAUSSIAN_CFG" \
    --split ood

# ── 3. Deep Ensemble (→ ensemble_u_test.png, ensemble_u_ood.png) ──────────────
echo "=== Deep Ensemble: test ==="
python -u -m src.uncertainty.eval_ensemble \
    --ensemble_dir "$ENSEMBLE_DIR" \
    --config "$ENSEMBLE_CFG" \
    --split test

echo "=== Deep Ensemble: OOD ==="
python -u -m src.uncertainty.eval_ensemble \
    --ensemble_dir "$ENSEMBLE_DIR" \
    --config "$ENSEMBLE_CFG" \
    --split ood

# ── 4. MC Dropout (→ mc_u_test.png, mc_u_ood.png) ────────────────────────────
echo "=== MC Dropout: test ==="
python -u -m src.evaluation.run_mc_dropout_eval \
    --model_dir "$MC_DIR" \
    --config "$MC_CFG" \
    --n_samples 200 \
    --split test

echo "=== MC Dropout: OOD ==="
python -u -m src.evaluation.run_mc_dropout_eval \
    --model_dir "$MC_DIR" \
    --config "$MC_CFG" \
    --n_samples 200 \
    --split ood

# ── 5. Conformal (→ conformal_phi_test.png) ───────────────────────────────────
echo "=== Conformal: test ==="
python -u -m src.uncertainty.run_conformal_eval \
    --model_dir "$BASELINE_DIR" \
    --config "$BASELINE_CFG" \
    --alpha 0.10 \
    --split test

echo "=== Conformal: OOD ==="
python -u -m src.uncertainty.run_conformal_eval \
    --model_dir "$BASELINE_DIR" \
    --config "$BASELINE_CFG" \
    --alpha 0.10 \
    --split ood

echo "Done: $(date)"
echo ""
echo "=== Paper figures expected at ==="
echo "  prediction_phi_test.png  → $BASELINE_DIR/plots/"
echo "  gaussian_phi_test.png    → $GAUSSIAN_DIR/plots/"
echo "  gaussian_phi_ood.png     → $GAUSSIAN_DIR/plots/"
echo "  ensemble_u_test.png      → $ENSEMBLE_DIR/plots/test/"
echo "  ensemble_u_ood.png       → $ENSEMBLE_DIR/plots/ood/"
echo "  mc_u_test.png            → $MC_DIR/mc_dropout_plots/test/"
echo "  conformal_phi_test.png   → $BASELINE_DIR/conformal/test/"
