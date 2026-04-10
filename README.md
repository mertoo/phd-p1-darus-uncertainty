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

| Model  | Best Validation Loss | Test RMSE | OOD RMSE |
|--------|-----------------------|-----------|----------|
| LSTM   | 0.0151                | 0.118     | 0.511    |
| GRU    | 0.0151                | 1.638     | 2.493    |
| MLP    | 0.0157                | 0.122     | 0.324    |
| TCN    | 0.0258                | —         | —        |
| Linear | 12.608                | 3.554     | 4.239    |
| Naive  | —                     | 0.300     | 0.585    |

These values serve as deterministic reference points for later uncertainty-aware modeling.

## Phase 2: Uncertainty Quantification

P2 extends the deterministic baselines with four uncertainty quantification methods, all using the LSTM backbone:

| Method | Script | Config |
|---|---|---|
| Gaussian LSTM | `src/uncertainty/train_lstm_gaussian.py` | `p2_lstm_gaussian.yaml` |
| Deep Ensemble (5×) | `src/uncertainty/train_ensemble.py` | `p2_lstm_ensemble.yaml` |
| MC Dropout | `src/evaluation/run_mc_dropout_eval.py` | `p1_lstm_seq2seq.yaml` |
| Conformal Prediction | `src/uncertainty/run_conformal_eval.py` | any baseline config |

Train the Gaussian LSTM:

```bash
python -m src.uncertainty.train_lstm_gaussian \
    --config experiments/configs/uncertainty/p2_lstm_gaussian.yaml
```

Train the deep ensemble (5 members sequentially):

```bash
python -m src.uncertainty.train_ensemble \
    --config experiments/configs/uncertainty/p2_lstm_ensemble.yaml \
    --num_models 5
```

Evaluate with calibrated uncertainty intervals:

```bash
# Gaussian LSTM — NLL, RMSE, coverage/width per DoF
python -m src.uncertainty.eval_gaussian_diagnostics \
    --model_dir experiments/results/p2_lstm_gaussian \
    --config experiments/configs/uncertainty/p2_lstm_gaussian.yaml \
    --split test --calibrate_sigma_temp

# Deep ensemble
python -m src.uncertainty.eval_ensemble \
    --ensemble_dir experiments/results/uncertainty/ensemble_lstm \
    --config experiments/configs/uncertainty/p2_lstm_ensemble.yaml \
    --split test

# Conformal prediction
python -m src.uncertainty.run_conformal_eval \
    --model_dir experiments/results/p1_lstm_baseline \
    --config experiments/configs/p1_lstm_seq2seq.yaml
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
sbatch --array=0-5 scripts/slurm/train_baselines.sh
```

**Train the deep ensemble in parallel (5 members simultaneously):**

```bash
sbatch --array=0-4 scripts/slurm/train_ensemble.sh
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
| Gaussian LSTM (20 epochs) | 1 | 16 GB | ~30 min |
| Single baseline (20 epochs) | 1 | 16 GB | ~20 min |
| Ensemble member | 1 | 16 GB | ~20 min |

## Contact

**Vedat Mert Asan**  
Naval Architecture and Hydrodynamics Research Group  
TalTech – Kuressaare College  
Email: mert.asan@taltech.ee
