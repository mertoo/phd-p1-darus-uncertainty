# Uncertainty-Aware Multi-Step Prediction of Ship Motion Using the DaRUS Dataset

This repository contains the implementation and analysis framework for Phase 1 (P1) of a broader research effort on uncertainty-aware forecasting of marine vessel dynamics. The primary objective of this phase is to establish deterministic baseline models for multi-step prediction of ship motion under environmental disturbances using the DaRUS (Dynamic Response of an Underactuated Ship) dataset. The baselines developed here form the foundation for subsequent work on probabilistic modeling and uncertainty quantification.

## Overview

Reliable prediction of ship motion plays a crucial role in autonomous navigation, dynamic positioning, health monitoring, and risk-aware control. Traditional forecasting approaches provide point estimates without assessing the uncertainty associated with predictions, which limits their usefulness in safety-critical scenarios and under conditions not represented in training data.

This repository provides:  
1. A reproducible pipeline for model training and evaluation.  
2. A set of deterministic baseline models implemented consistently within the same framework.  
3. Tools for assessing performance in both in-distribution (ID) and out-of-distribution (OOD) sea-state regimes.  
4. Visual and numerical outputs that will serve as benchmarks for future uncertainty-aware methods.

## Repository Structure

```
phd-p1-darus-uncertainty/
│
├── experiments/
│   ├── configs/              # YAML configuration files
│   │   └── uncertainty/      # P2 uncertainty method configs
│   └── results/              # Saved checkpoints and evaluation plots
│
├── scripts/
│   └── slurm/                # SLURM job scripts for HPC cluster
│
├── src/
│   ├── data_loading/         # DaRUS dataset loader and parser
│   ├── models/               # Baseline and probabilistic model architectures
│   ├── training/             # Training script for deterministic baselines
│   ├── evaluation/           # Evaluation, metrics, and plotting utilities
│   └── uncertainty/          # P2: UQ training and evaluation pipelines
│
├── requirements.txt
├── train_all.sh              # Sequential launcher for all P1 baselines
└── README.md
```

## Dataset

The DaRUS dataset provides multi-sensor measurements of vessel dynamics, environmental disturbances, and control inputs. The dataset includes several sea-state conditions, enabling a clear separation of ID and OOD scenarios.

For this benchmark:

- **Input sequence length:** 30 steps  
- **Prediction horizon:** 30 steps  
- **Target variables:** 5-DOF motion components  
- **Evaluation:** performed on held-out ID and OOD subsets  

## Baseline Models

The following deterministic forecasting models are implemented:

- LSTM Encoder–Decoder  
- GRU Encoder–Decoder  
- Temporal Convolutional Network (TCN)  
- Multi-Layer Perceptron (MLP)  
- Linear autoregressive baseline  
- Naive last-value baseline  

All models share a unified interface and are trained using a consistent protocol.

## Training

To train a model:

```bash
python3 -m src.training.train_baseline --config experiments/configs/<config>.yaml
```

To train all baseline models sequentially:

```bash
chmod +x train_all.sh
./train_all.sh
```

Checkpoints and logs are written to:

```
experiments/results/<model_name>/
```

## Evaluation

Each model can be evaluated using:

```bash
python3 -m src.evaluation.run_eval --model_dir experiments/results/<model_name>
```

The evaluation stage:

- Computes RMSE for ID and OOD test sets  
- Produces prediction–truth comparison plots  
- Generates multi-horizon RMSE curves  
- Saves outputs into `experiments/results/<model_name>/plots/`  

## Baseline Performance Summary

Verified on GPU (SLURM jobs 918426/918432), history=30, horizon=30.

| Model  | Best Val Loss | Test RMSE | OOD RMSE |
|--------|--------------|-----------|----------|
| LSTM   | 0.01599      | 0.120     | 0.537    |
| MLP    | 0.01561      | **0.121** | **0.330** |
| GRU    | 0.01880      | 0.132     | 0.506    |
| TCN    | 0.02480      | 0.155     | 0.347    |
| Naive  | —            | 0.173     | 0.411    |
| Linear | 3.291        | 1.791     | 2.297    |

LSTM is best in-distribution; MLP generalises best OOD. Checkpoints saved to
`experiments/results/p1_<model>_baseline/best_model.pt`.

These values serve as deterministic reference points for uncertainty-aware modeling.

## Phase 2: Uncertainty Quantification

P2 extends the deterministic baselines with UQ methods. Results are in
`experiments/results/RESULTS.md`. Quick summary (α=0.10, nominal 90% coverage):

| Method | Backbone | Test RMSE | Test Coverage | OOD Coverage |
|--------|----------|-----------|---------------|--------------|
| Gaussian LSTM (calibrated) | LSTM | 0.133 | 89.9% ✅ | 44.6% |
| Conformal Prediction | MLP  | 0.121 | 90.1% ✅ | **68.0%** |
| Conformal Prediction | LSTM | 0.120 | 90.5% ✅ | 66.2% |
| MLP Deep Ensemble ±2σ | MLP  | 0.123 | 81.3% | 57.4% |
| LSTM Deep Ensemble ±2σ | LSTM | **0.111** | 75.8% | 49.0% |
| MC Dropout ±2σ | LSTM | 0.123 | 35.7% | 16.5% |

### Training UQ methods

```bash
# Gaussian LSTM
python -m src.uncertainty.train_lstm_gaussian \
    --config experiments/configs/uncertainty/p2_lstm_gaussian.yaml

# Deep ensemble — 5 members via SLURM array (LSTM or MLP)
sbatch scripts/slurm/train_ensemble.sh       # LSTM
sbatch scripts/slurm/train_ensemble_mlp.sh   # MLP

# MC Dropout — standard train_baseline with dropout=0.2
sbatch scripts/slurm/train_eval_mc_dropout.sh
```

### Evaluating UQ methods

```bash
# Gaussian LSTM — NLL, RMSE, coverage/width per DoF with per-DoF calibration
python -m src.uncertainty.eval_gaussian_per_dof \
    --model_dir experiments/results/p2_lstm_gaussian \
    --config experiments/configs/uncertainty/p2_lstm_gaussian.yaml \
    --split test --calibrate_per_dof

# Deep ensemble (works for any model type via config model.type)
python -m src.uncertainty.eval_ensemble \
    --ensemble_dir experiments/results/uncertainty/ensemble_mlp \
    --config experiments/configs/uncertainty/p2_mlp_ensemble.yaml \
    --split test

# Conformal prediction (works for any model type)
python -m src.uncertainty.eval_conformal_metrics \
    --model_dir experiments/results/p1_mlp_baseline \
    --config experiments/results/p1_mlp_baseline/config.yaml \
    --alpha 0.10 --split test

# MC Dropout
python -m src.evaluation.run_mc_dropout_eval \
    --model_dir experiments/results/uncertainty/mc_dropout_lstm \
    --config experiments/results/uncertainty/mc_dropout_lstm/config.yaml \
    --n_samples 200 --split test
```

## Running on an HPC Cluster (SLURM)

### 1. Environment setup

```bash
# Clone and enter the repo
git clone <repo-url>
cd phd-p1-darus-uncertainty

# Create a virtualenv (or use your site's conda/module environment)
python -m venv venv
source venv/bin/activate

# Install PyTorch with the CUDA version matching your cluster, e.g. CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

> **Note:** If your cluster uses `module` commands, load Python and CUDA before creating the venv, e.g. `module load python/3.11 cuda/12.1`. Add the relevant `module load` lines inside the SLURM scripts under the `# Load any cluster modules` comment.

### 2. Data

Copy the processed DaRUS data to the cluster so the following structure is present:

```
data/processed/darus/
    patrol_ship_routine/train/        # 57 CSV files
    patrol_ship_routine/validation/   # 9 CSV files
    patrol_ship_routine/test/         # 30 CSV files
    patrol_ship_ood/test/             # 29 CSV files
```

### 3. Submit jobs

All SLURM scripts live in `scripts/slurm/`. Edit the `--partition` line to match your cluster's GPU partition name before submitting.

**Train the Gaussian LSTM (single GPU job):**

```bash
sbatch scripts/slurm/train_gaussian_lstm.sh
```

**Train all P1 baselines in parallel (job array, one GPU per model):**

```bash
sbatch scripts/slurm/train_baselines.sh       # array 0-5 defined inside the script
```

**Train the deep ensemble in parallel (5 members simultaneously):**

```bash
sbatch scripts/slurm/train_ensemble.sh        # LSTM ensemble (array 0-4)
sbatch scripts/slurm/train_ensemble_mlp.sh    # MLP ensemble  (array 0-4)
```

### 4. Monitor and retrieve results

```bash
# Watch queue
squeue -u $USER

# Tail a running job's log
tail -f logs/slurm/darus_gaussian_lstm_<JOBID>.out

# Checkpoints are written to:
#   experiments/results/<run_name>/best_model.pt
```

### Resource requirements

| Job | GPUs | RAM | Typical wall time |
|---|---|---|---|
| Single baseline (20 epochs) | 1 | 16 GB | ~5 min (MLP) / ~10 min (LSTM) |
| Ensemble member (20 epochs) | 1 | 16 GB | ~5–10 min |
| Gaussian LSTM (20 epochs) | 1 | 16 GB | ~30 min |
| Conformal / ensemble eval | 1 | 16 GB | < 2 min |
| MC Dropout eval (200 samples) | 1 | 16 GB | ~5 min |

## Contact

**Vedat Mert Asan**  
Naval Architecture and Hydrodynamics Research Group  
TalTech – Kuressaare College  
Email: mert.asan@taltech.ee
