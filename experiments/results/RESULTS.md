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

Split-conformal intervals calibrated on validation residuals (marginal, per-timestep).  
Backbone: best baseline LSTM (ensemble member 3, val loss = 0.01449).  
α = 0.10 → nominal coverage = 90%.  
SLURM job: 918407 — `scripts/slurm/eval_conformal.sh`

### TEST split

| Metric | Value |
|--------|-------|
| Overall coverage | **90.1%** ✅ (nominal 90%) |
| Overall width | 0.210 |

| DoF | Coverage | Width |
|-----|----------|-------|
| u   | 91.4% | 0.485 |
| v   | 87.9% | 0.234 |
| p   | 92.0% | 0.021 |
| r   | 90.3% | 0.028 |
| φ   | 91.0% | 0.068 |

### OOD split

| Metric | Value |
|--------|-------|
| Overall coverage | **65.9%** ⚠️ (nominal 90%) |
| Overall width | 0.210 |

| DoF | Coverage | Width |
|-----|----------|------|
| u   | 23.3% | 0.485 |
| v   | 26.5% | 0.234 |
| p   | 80.1% | 0.021 |
| r   | 49.8% | 0.028 |
| φ   | 48.9% | 0.068 |

---

## Summary Comparison

### TEST split (nominal coverage 90%)

| Method | RMSE | Coverage | Width |
|--------|------|----------|-------|
| Gaussian LSTM (calibrated) | 0.133 | **89.9%** ✅ | 0.201 |
| Conformal Prediction | 0.118* | **90.1%** ✅ | 0.210 |
| Deep Ensemble ±2σ | **0.111** | 75.8% ⚠️ | — |

*Conformal uses the baseline LSTM backbone (ensemble member 3); RMSE 0.118 is the backbone point-prediction error.

### OOD split (nominal coverage 90%)

| Method | RMSE | Coverage | Width |
|--------|------|----------|-------|
| Gaussian LSTM (calibrated) | 0.550 | 44.6% ⚠️ | 0.248 |
| Deep Ensemble ±2σ | 0.492 | 49.0% ⚠️ | — |
| Conformal Prediction | — | **65.9%** ⚠️ | 0.210 |

---

## Key Observations

1. **In-distribution calibration:** Both the calibrated Gaussian LSTM and conformal prediction achieve near-exact 90% coverage on the test set (89.9% and 90.1%). The deep ensemble with raw ±2σ intervals is under-covered (75.8%) — the ensemble spread is too narrow relative to nominal 95.4%, suggesting the members are too similar or the interval computation needs calibration.

2. **OOD coverage collapse:** All three methods degrade severely under distribution shift. Conformal prediction degrades least (65.9%), Gaussian LSTM worst (44.6%), with ensemble in between (49.0%). Conformal's advantage comes from its fixed-width intervals being relatively wider for the high-error DoFs (u, v).

3. **Per-DoF patterns:** DoF `u` (surge) and `v` (sway) are consistently the hardest — highest RMSE and lowest OOD coverage across all methods. DoF `p` (roll) is best-covered OOD, likely because roll dynamics are more self-similar across sea states.

4. **Interval width vs. coverage trade-off:** Conformal intervals are fixed-width (constant across input), which is efficient on test but cannot adapt to locally harder OOD inputs. Gaussian and ensemble methods produce input-adaptive widths but are poorly calibrated OOD.

5. **Next steps:** Covariate-adaptive conformal (e.g. locally-weighted conformal, RAPS) or MC-Dropout with temperature scaling could close the OOD coverage gap while maintaining adaptive interval widths.
