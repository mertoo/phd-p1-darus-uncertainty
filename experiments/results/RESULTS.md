# Uncertainty Quantification — Evaluation Results

**Dataset:** DaRUS patrol-ship motion (5 DoF: u, v, p, r, φ)  
**Task:** Seq2Seq trajectory prediction (history=30, horizon=30 steps)  
**Architecture:** LSTM (hidden=128, layers=2)  
**Evaluated:** 2026-04-11 on SLURM/L40 GPU

---

## 1. Gaussian LSTM (calibrated)

Trained with negative log-likelihood loss outputting (μ, σ) per DoF.  
Sigma temperature T_σ fitted on validation set per DoF (grid search, α=0.10).  
SLURM job: 918386 — `scripts/slurm/eval_gaussian_per_dof.sh`

**Calibrated temperatures (from VAL, nominal coverage = 90%)**

| DoF | T_σ |
|-----|-----|
| u   | 1.000 |
| v   | 0.700 |
| p   | 0.700 |
| r   | 0.700 |
| φ   | 1.000 |

### TEST split

| Metric | Value |
|--------|-------|
| RMSE (overall) | 0.1331 |
| Avg NLL | −2.3686 |
| Overall coverage | **89.9%** ✅ (nominal 90%) |
| Overall width | 0.201 |

| DoF | Coverage | Width | RMSE |
|-----|----------|-------|------|
| u   | 92.0% | 0.643 | 0.2797 |
| v   | 92.3% | 0.297 | 0.1009 |
| p   | 90.8% | 0.017 | 0.0061 |
| r   | 85.7% | 0.016 | 0.0067 |
| φ   | 88.5% | 0.032 | 0.0125 |

### OOD split

| Metric | Value |
|--------|-------|
| RMSE (overall) | 0.5497 |
| Avg NLL | 7.4713 |
| Overall coverage | **44.6%** ⚠️ (nominal 90%) |
| Overall width | 0.248 |

| DoF | Coverage | Width | RMSE |
|-----|----------|-------|------|
| u   | 29.9% | 0.799 | 1.1364 |
| v   | 35.8% | 0.363 | 0.4654 |
| p   | 74.7% | 0.016 | 0.0080 |
| r   | 35.7% | 0.019 | 0.0234 |
| φ   | 47.0% | 0.046 | 0.0440 |

---

## 2. Deep Ensemble (5 members)

Five independent `LSTMSeq2Seq` models trained with different random initialisations.  
Uncertainty from variance across member predictions. Intervals reported as ±2σ (≈95.4% nominal).  
SLURM jobs: 918397 (training array), 918406 (eval) — `scripts/slurm/train_ensemble.sh`, `eval_ensemble.sh`

**Member val losses (best checkpoint per member)**

| Member | Best Val Loss |
|--------|--------------|
| 0 | 0.01558 |
| 1 | 0.01572 |
| 2 | 0.01579 |
| 3 | **0.01449** |
| 4 | 0.01560 |

### TEST split

| Metric | Value |
|--------|-------|
| RMSE (overall) | 0.1106 |
| Overall coverage ±2σ | **75.8%** ⚠️ (nominal 95.4%) |
| Overall width | — |

| DoF | Coverage | RMSE |
|-----|----------|------|
| u   | 77.1% | 0.2290 |
| v   | 75.8% | 0.0887 |
| p   | 76.9% | 0.0061 |
| r   | 79.9% | 0.0077 |
| φ   | 69.2% | 0.0186 |

### OOD split

| Metric | Value |
|--------|-------|
| RMSE (overall) | 0.4924 |
| Overall coverage ±2σ | **49.0%** ⚠️ (nominal 95.4%) |
| Overall width | — |

| DoF | Coverage | RMSE |
|-----|----------|------|
| u   | 32.1% | 1.0066 |
| v   | 47.8% | 0.4374 |
| p   | 67.8% | 0.0080 |
| r   | 49.8% | 0.0227 |
| φ   | 47.6% | 0.0512 |

---

## 3. Conformal Prediction

Split-conformal intervals calibrated on validation residuals (marginal, per-DoF quantile).  
α = 0.10 → nominal coverage = 90%.  
SLURM jobs: 918407 (orig LSTM), 918441 (LSTM vs MLP comparison) — `scripts/slurm/eval_conformal_mlp_lstm.sh`

### 3a. Original result: ensemble member 3 backbone (val loss = 0.01449)

| Split | Coverage | Width |
|-------|----------|-------|
| TEST  | **90.1%** ✅ | 0.210 |
| OOD   | **65.9%** ⚠️ | 0.210 |

### 3b. LSTM baseline backbone (val loss = 0.01599, SLURM 918441)

#### TEST split

| Metric | Value |
|--------|-------|
| Overall coverage | **90.5%** ✅ (nominal 90%) |
| Overall width | 0.234 |

| DoF | Coverage | Width |
|-----|----------|-------|
| u   | 91.3% | 0.548 |
| v   | 88.9% | 0.283 |
| p   | 92.2% | 0.020 |
| r   | 90.1% | 0.028 |
| φ   | 90.5% | 0.071 |

#### OOD split

| Metric | Value |
|--------|-------|
| Overall coverage | **66.2%** ⚠️ (nominal 90%) |
| Overall width | 0.234 |

| DoF | Coverage | Width |
|-----|----------|-------|
| u   | 25.5% | 0.548 |
| v   | 28.2% | 0.283 |
| p   | 81.1% | 0.020 |
| r   | 46.8% | 0.028 |
| φ   | 46.5% | 0.071 |

### 3c. MLP baseline backbone (val loss = 0.01561, SLURM 918441)

#### TEST split

| Metric | Value |
|--------|-------|
| Overall coverage | **90.1%** ✅ (nominal 90%) |
| Overall width | 0.212 |

| DoF | Coverage | Width |
|-----|----------|-------|
| u   | 89.2% | 0.487 |
| v   | 88.9% | 0.258 |
| p   | 92.0% | 0.022 |
| r   | 90.4% | 0.033 |
| φ   | 87.4% | 0.080 |

#### OOD split

| Metric | Value |
|--------|-------|
| Overall coverage | **68.0%** ⚠️ (nominal 90%) |
| Overall width | 0.212 |

| DoF | Coverage | Width |
|-----|----------|-------|
| u   | 39.7% | 0.487 |
| v   | 30.3% | 0.258 |
| p   | 78.7% | 0.022 |
| r   | 56.3% | 0.033 |
| φ   | 53.4% | 0.080 |

### Conformal backbone comparison

Both backbones achieve near-exact 90% test coverage (conformal guarantee holds). The MLP backbone
gives **+1.8% better OOD coverage** (68.0% vs 66.2%) with slightly **narrower intervals** (0.212 vs 0.234),
consistent with its better point-prediction accuracy on OOD data. The improvement comes from smaller
calibration residuals on the val set, which translate to tighter quantiles that still cover more OOD points.

---

## 4. MC Dropout

Single `LSTMSeq2Seq` trained with dropout=0.2. At inference, dropout remains active and 200 stochastic
forward passes are averaged to estimate mean and epistemic std. Intervals reported as ±2σ (≈95.4% nominal).  
SLURM jobs: 918411 (train), 918413 (eval) — `scripts/slurm/train_eval_mc_dropout.sh`, `eval_mc_dropout.sh`

**Training:** Best val loss = 0.01629 (epoch 13, dropout=0.2)

### TEST split

| Metric | Value |
|--------|-------|
| RMSE (overall) | 0.1230 |
| Overall coverage ±2σ | **35.7%** ⚠️ (nominal 95.4%) |

| DoF | Coverage | RMSE |
|-----|----------|------|
| u   | 44.4% | 0.2507 |
| v   | 43.0% | 0.1083 |
| p   | 35.9% | 0.0063 |
| r   | 20.7% | 0.0117 |
| φ   | 34.5% | 0.0229 |

### OOD split

| Metric | Value |
|--------|-------|
| RMSE (overall) | 0.5201 |
| Overall coverage ±2σ | **16.5%** ⚠️ (nominal 95.4%) |

| DoF | Coverage | RMSE |
|-----|----------|------|
| u   | 15.0% | 1.0731 |
| v   | 19.0% | 0.4384 |
| p   | 20.2% | 0.0083 |
| r   | 13.4% | 0.0264 |
| φ   | 15.1% | 0.0542 |

---

## 5. Baseline Model Comparison (point prediction only)

All models trained with history=30, horizon=30. Evaluated on test and OOD splits.  
SLURM job: 918426 (train array), 918432 (eval array)

| Model | Test RMSE | OOD RMSE |
|-------|-----------|----------|
| **LSTM** (hidden=128, layers=2) | **0.120** | 0.537 |
| MLP (hidden=256, layers=3) | 0.121 | **0.330** |
| GRU (hidden=128, layers=2) | 0.132 | 0.506 |
| Naive (last-step repeat) | 0.173 | 0.411 |
| TCN | 0.155 | 0.347 |
| Linear | 1.791 | 2.297 |

LSTM is best in-distribution; MLP generalises best OOD. Linear fails to capture nonlinear dynamics.

---

## Summary Comparison

### TEST split (nominal coverage 90%)

| Method | RMSE | Coverage | Width |
|--------|------|----------|-------|
| Gaussian LSTM (calibrated) | 0.133 | **89.9%** ✅ | 0.201 |
| Conformal (LSTM backbone) | 0.120 | **90.5%** ✅ | 0.234 |
| Conformal (MLP backbone) | 0.121 | **90.1%** ✅ | 0.212 |
| Deep Ensemble ±2σ | **0.111** | 75.8% ⚠️ | — |
| MC Dropout ±2σ | 0.123 | 35.7% ⚠️ | — |

### OOD split (nominal coverage 90%)

| Method | RMSE | Coverage | Width |
|--------|------|----------|-------|
| Conformal (MLP backbone) | **0.330** | **68.0%** ⚠️ | 0.212 |
| Conformal (LSTM backbone) | 0.537 | 66.2% ⚠️ | 0.234 |
| Gaussian LSTM (calibrated) | 0.550 | 44.6% ⚠️ | 0.248 |
| Deep Ensemble ±2σ | 0.492 | 49.0% ⚠️ | — |
| MC Dropout ±2σ | 0.520 | 16.5% ⚠️ | — |

---

## Key Observations

1. **In-distribution calibration:** Both the calibrated Gaussian LSTM and conformal prediction achieve near-exact 90% coverage on the test set (89.9% and 90.1%). The deep ensemble and MC Dropout use raw ±2σ intervals without calibration and are severely under-covered (75.8% and 35.7% respectively).

2. **MC Dropout underperforms across the board:** Despite 200 stochastic passes, the epistemic variance is far smaller than the actual prediction error. The model learns to route around dropout, producing near-deterministic outputs — a well-known failure mode. Temperature scaling or a larger dropout rate would be needed to calibrate it.

3. **OOD coverage collapse:** All methods degrade under distribution shift. Conformal prediction degrades least (65.9%), MC Dropout worst (16.5%), with Gaussian LSTM (44.6%) and ensemble (49.0%) in between. Conformal's advantage comes from its fixed-width intervals being relatively wider for the high-error DoFs (u, v).

4. **Per-DoF patterns:** DoF `u` (surge) and `v` (sway) are consistently the hardest — highest RMSE and lowest OOD coverage across all methods. DoF `p` (roll) is best-covered OOD, likely because roll dynamics are more self-similar across sea states.

5. **Interval width vs. coverage trade-off:** Conformal intervals are fixed-width (constant across input), which is efficient on test but cannot adapt to locally harder OOD inputs. Gaussian and ensemble methods produce input-adaptive widths but are poorly calibrated OOD without explicit post-hoc calibration.

6. **Next steps:** Covariate-adaptive conformal (e.g. locally-weighted conformal, RAPS) or ensemble calibration via temperature scaling could close the OOD coverage gap while maintaining adaptive interval widths.
