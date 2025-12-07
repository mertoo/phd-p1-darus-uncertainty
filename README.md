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
│   ├── configs/          # YAML configuration files
│   └── results/          # Saved checkpoints, logs, evaluation plots
│
├── src/
│   ├── data_loading/     # DaRUS dataset preprocessing and loaders
│   ├── models/           # Implementations of baseline architectures
│   ├── training/         # Training utilities for baseline models
│   ├── evaluation/       # Evaluation routines and plotting utilities
│   └── uncertainty/      # Scripts for UQ methods (ensembles, MC dropout)
│
├── train_all.sh          # Batch launcher for training all baselines
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

These values serve as deterministic reference points for the uncertainty-aware models.

## Phase 2: Uncertainty-Aware Models

This repository has been updated to include methods for uncertainty quantification (UQ). The following UQ techniques are implemented.

### Deep Ensembles

A deep ensemble is a collection of independently trained models. The variance of their predictions serves as a measure of uncertainty.

**Training:**

To train a 5-member LSTM ensemble:
```bash
python3 -m src.uncertainty.train_ensemble \
    --config experiments/configs/uncertainty/p2_lstm_ensemble.yaml \
    --num_models 5
```
This script will train and save 5 independent models to `experiments/results/uncertainty/ensemble_lstm/`.

**Evaluation:**

To evaluate the trained ensemble and generate prediction plots with uncertainty intervals:
```bash
python3 -m src.uncertainty.eval_ensemble \
    --ensemble_dir experiments/results/uncertainty/ensemble_lstm \
    --config experiments/configs/uncertainty/p2_lstm_ensemble.yaml \
    --split ood  # or "test"
```
Plots will be saved to `experiments/results/uncertainty/ensemble_lstm/plots/`.

### Monte Carlo Dropout (MC Dropout)

MC Dropout is a technique where dropout layers are kept active during inference. Making multiple stochastic forward passes provides a distribution of predictions, from which uncertainty can be estimated.

**Training:**

No special training script is needed. Simply train a baseline model (e.g., LSTM) with a dropout rate greater than 0. This is configured in the corresponding `.yaml` file (e.g., `model.dropout: 0.2`).

**Evaluation:**

MC Dropout is an evaluation-time technique. The functions in `src/uncertainty/mc_dropout.py` can be integrated into an evaluation script. At evaluation time, call `enable_mc_dropout(model)` and then use `mc_dropout_predict(model, x, n_samples)` to get a mean and standard deviation for the model's predictions.

## Contact

**Vedat Mert Asan**  
Naval Architecture and Hydrodynamics Research Group  
TalTech – Kuressaare College  
Email: mert.asan@taltech.ee
