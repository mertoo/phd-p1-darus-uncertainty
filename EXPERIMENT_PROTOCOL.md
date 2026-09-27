# Experiment protocol v3 (DRAFT: requires author sign-off before any HPC run)

Status: **draft, not frozen.** The items marked *decision* need a yes/no from the authors. Nothing in this protocol has been run on the full data.

## 1. Data roles (fixed before any test/OOD evaluation)
| Role | Recordings | Used for |
|---|---|---|
| Train (fit) | 48 of the 57 deposit `train` recordings | gradient updates; scaler fitting |
| Calibration | 9 of the 57 `train` recordings, drawn with `numpy.random.default_rng(20261001)` from the sorted names (list written into every checkpoint and `metrics.json`) | conformal thresholds and spread calibration only; never model selection |
| Model selection | deposit `validation` (9) | best-epoch choice, early stopping, ridge penalty, time-feature ablation |
| ID test | deposit routine `test` (30) | reporting only |
| OOD test | deposit `patrol_ship_ood/test` (29) | reporting only. Exploratory: these recordings already influenced the historical backbone choice |

*Decision D1:* hold calibration recordings out of `train` (proposed) instead of splitting `validation` (only 9 recordings, which would leave about 4–5 for each role).

Windows: H = T = 30 s at 1 Hz, stride 1, built only inside recordings (3542 per recording). All windows overlap heavily; the independent unit is the **recording** (48/9/9/30/29).

## 2. Preprocessing
- Inputs and targets are standardised with per-channel mean/std fitted on the 48 fit recordings (`normalize: true`). Loss is MSE (or Gaussian NLL) in standardised units, so all five targets carry equal weight. *Decision D2* (changes the historical loss weighting; needs retraining anyway).
- Metrics are computed in physical units after inverse transform. Dimensionless summaries divide by the training-target std.

## 3. Feature set: validation-only ablation (Stage 0)
`time` restarts at 0 in every file and encodes position within a recording. We train LSTM and MLP with `features: legacy` (12 inputs) and `no_time` (11 inputs), seed 1 each. **Rule:** keep `no_time` unless `legacy` lowers the normalised validation MSE by more than 2%. The rule is fixed now; test/OOD results of the ablation are not looked at. The chosen set is then frozen for Stage 1.

## 4. Training
Adam, lr 1e-3, batch 256, max 100 epochs, early stopping with patience 10 on the validation objective (MSE, or Gaussian NLL including 0.5·log 2π), best-validation checkpoint. Seeds are explicit and recorded. Linear: SGD version (as before) **plus** closed-form ridge with the penalty selected on validation (`src/training/fit_ridge.py`).

## 5. UQ methods and scoring
| Method | Backbones | Raw output | Calibrated variants (same calibration recordings) |
|---|---|---|---|
| Split conformal | each point model | point | abs-residual, `channel` and `horizon_channel` |
| Deep ensemble, N=5 | LSTM, MLP | mean, member std (ddof=1) | raw mean±zσ; spread-normalised `channel`/`horizon_channel`; residual conformal on ensemble mean |
| MC dropout, M=200, p=0.2 | LSTM (inter-layer), MLP (after hidden layers) | mean, pass std (ddof=1) | same as ensemble |
| Gaussian likelihood | LSTM (MLP: *decision D3*) | μ, σ | raw Gaussian; spread-normalised (= fitted temperature per group) |

- Nominal levels, predeclared: 0.5, 0.8, **0.9 (primary)**, 0.95.
- Spread floor for normalised scores: 1e-3 × mean calibration spread per channel (fixed; sensitivity reported only if the floor is ever active for >1% of calibration scores).
- Primary conformal construction: **`horizon_channel`** (marginal coverage per forecast step and channel), chosen because error grows with lead time. `channel` is reported as secondary.
- Reported per split: RMSE per channel (physical units), pooled and dimensionless mean RMSE; marginal coverage overall/per channel/per horizon; mean width per channel (physical and normalised); interval (Winkler) score; simultaneous trajectory coverage per channel and over all channels (reported separately, never called marginal).
- Uncertainty: cluster bootstrap over **recordings** (2000 resamples) for ID (30) and OOD (29) metrics, plus between-seed spread across repeats. Paired method comparisons use the same recordings.
- No claim of a finite-sample guarantee: windows overlap and recordings differ. We report empirical marginal calibration in distribution, and describe OOD behaviour without guarantees.

## 6. Repeats
3 independent repeats of every trained configuration (seeds listed in the run manifest). An ensemble repeat = 5 fresh members; member 0 of each repeat doubles as that repeat's single-model backbone for conformal/point results. The 5 members of one ensemble are **not** 5 repeats of the ensemble experiment.

## 7. Proposed run list and compute
Timings are placeholders until a real-data timing run exists. Historical L40 runs took about 1 min/epoch with the old loader (20 epochs ≈ 20 min). With early stopping we expect 20–60 epochs.

| Stage | Runs | Trainings | GPU-h (est.) |
|---|---|---|---|
| 0 time ablation | LSTM, MLP × {legacy, no_time} × seed 1 | 4 | 2–7 |
| 1a ensembles (incl. single-model backbones) | LSTM, MLP × 3 repeats × 5 members | 30 | 15–50 |
| 1b other baselines | GRU, TCN, Linear-SGD × 3 seeds; ridge ×1; naive ×1 | 9 (+2 CPU) | 5–15 |
| 1c MC dropout | LSTM-p0.2, MLP-p0.2 × 3 seeds | 6 | 3–10 |
| 1d Gaussian | LSTM × 3 seeds (+ MLP × 3 if D3) | 3 (6) | 2–10 |
| 2 evaluation | all methods, MC 200 passes over cal+test+OOD ≈ 240k windows | — | 5–10 |
| **Total** | | **52 (55)** | **≈ 32–100 L40 GPU-h** |

Storage: float32 predictions ≈ 0.3–0.5 GB per evaluated method-run, so about 10–15 GB in total. Memory: under 8 GB host RAM per job. All jobs are single-GPU and independent (SLURM arrays).

## 8. Selection hygiene
No hyperparameter, seed, calibration setting, floor, or example window is chosen from ID-test or OOD results. Example trajectories for figures are chosen by a fixed rule (median per-window RMSE of the primary model, same window across methods), and their window IDs are reported.
