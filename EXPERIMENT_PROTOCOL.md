# Experiment protocol v3 (pre-run draft)

Status (2026-09-28): **not frozen, nothing run on the full data.**
- D1 and D2 are the preferred protocol (author instruction, 2026-09-28).
- The calibration-recording choice and the feature set remain provisional (§1, §3).
- The timing pilot (§7a) is prepared and **awaits compute authorisation**; the full run list (§7) will not be submitted until after the pilot.

The historical-checkpoint recovery track (old-vs-new comparison) is separate from this benchmark and is described in REVISION_TRACKER.md.

## 1. Data roles
| Role | Recordings | Used for |
|---|---|---|
| Train (fit) | 48 of the 57 deposit `train` recordings | gradient updates; scaler fitting (D2) |
| Calibration (D1) | 9 of the 57 `train` recordings (list below) | conformal thresholds and spread calibration only; never model selection |
| Model selection | deposit `validation` (9) | best epoch, early stopping, ridge penalty, feature decision |
| ID test | deposit routine `test` (30) | reporting only |
| OOD test | deposit `patrol_ship_ood/test` (29) | reporting only. **Exploratory**: these recordings informed the historical backbone choice, and no untouched OOD recordings are known |

Windows: H = T = 30 s at 1 Hz, stride 1, within recordings only (3542 per recording; train 170,016, cal 31,878, val 31,878, test 106,260, OOD 102,718). The independent unit is the recording.

### Calibration recordings: selection and representativeness check
Script: `python -m scripts.check_calibration_split`. Outputs: `experiments/manifests/calibration_selection.json` and `calibration_representativeness.csv`. It reads only the 57 training recordings (per-recording mean and SD of the 11 physical inputs/targets). No model, validation, test or OOD data is used.

- **Predeclared criterion:** |standardised mean difference| ≤ 0.5 for all 22 variables, and the calibration recordings' mean `u` and `n` within the fit-recording range. Fallback if the random draw fails: stratified draw, one recording per equal-count stratum of mean shaft speed `n`, same seed.
- **Outcome:**
  - Random draw (seed 20261001): fails. Max |SMD| 0.84 on `deltal_std`; mean `u`/`n` in range, but the calibration set is slower (SMD −0.52 for `u`, −0.49 for `n`).
  - Stratified draw: also fails. Max |SMD| 0.83 on `Vw_mean`; speed matched (SMD 0.03 for both `u` and `n`).
- **Diagnosis:** the criterion was mis-specified. Over 5000 random 9-of-57 draws it passes only 7.1% of the time (chance max |SMD|: median 0.75, 95th percentile 1.14). Both actual selections are typical draws (≈67th percentile). We do not redraw until something passes.
- **Provisional choice:** `stratified_n` (the predeclared fallback), recorded in every v3 config:
  `20190805-095929, -100322, -100351, -101342, -101925, -102210, -102939, -104006, -104542` (.csv).
- *Author decision C1:* accept `stratified_n` as is, or replace the criterion with the chance-referenced one (max |SMD| below the 95th percentile of random draws, which both selections satisfy). Either way the post-hoc change is disclosed.

## 2. Preprocessing (D2)
Inputs and targets are standardised with statistics from the 48 fit recordings; the scalers are stored in each checkpoint and re-verified at evaluation. Loss is MSE (or Gaussian NLL with 0.5·log 2π) in standardised units, so the five targets carry equal weight. Metrics are reported in physical units, plus dimensionless summaries divided by the training-target SD.

## 3. Feature set: validation-only rule
Script: `python -m scripts.select_features`. It reads only `train_log.json` best validation losses; test and OOD are never read.

- **Stage 0:** LSTM and MLP × {`legacy` (12 inputs, with `time`), `no_time` (11 inputs)}, seed 1.
- **Gain per backbone:** gain_b = (L_no_time − L_legacy) / L_no_time, using the mean best-validation loss over the available seeds. Losses are comparable because the target scaler is identical.
- **Borderline:** any gain in [1%, 3%], or the backbones disagree about whether gain > 2%.
- **Round 1:** if not borderline, use `legacy` iff gain > 2% for **both** backbones, else `no_time`. If borderline, the decision is "repeat": run stage 0b (seeds 2 and 3 for all four configurations).
- **Round 2:** the same rule on the 3-seed means. A remaining disagreement defaults to `no_time`, the parsimonious set, since `time` encodes position within a recording. The disagreement is reported as a finding. There are no further rounds.
- **One common feature set** is used for every model in the primary comparison. The non-selected set is not evaluated on test/OOD.

## 4. Training
Adam, lr 1e-3, batch 256, max 100 epochs, early stopping with patience 10 on the validation objective, and the best-validation checkpoint is kept. Seeds are explicit and recorded. A best epoch equal to the last epoch is reported as "stopped by epoch budget". It is not interpreted as proof of (non-)convergence either way. Linear: the SGD model plus a closed-form ridge with the penalty chosen on validation.

## 5. UQ methods and scoring
| Method | Backbones | Raw output | Calibrated variants (same calibration recordings) |
|---|---|---|---|
| Split conformal | each point model | point | abs-residual, `channel` / `horizon_channel` |
| Deep ensemble, N=5 | LSTM, MLP | mean, member SD (ddof=1) | raw mean ± zσ; spread-normalised `channel` / `horizon_channel`; residual conformal on the mean |
| MC dropout, M=200, p=0.2 | LSTM (inter-layer only), MLP (after hidden layers) | mean, pass SD (ddof=1) | as ensemble |
| Gaussian likelihood | LSTM, MLP | μ, σ | raw Gaussian; spread-normalised (= fitted temperature per group); residual conformal on μ |

- **Levels** (predeclared): 0.5, 0.8, **0.9 (primary)**, 0.95. Primary construction: `horizon_channel`.
- **Spread floor:** floor_d = 1e-3 × max(mean calibration spread_d, SD of calibration targets_d). It is strictly positive and in the channel's physical scale. If both terms are zero (constant target and no spread), evaluation stops with `DegenerateSpreadError`. NaN inputs or bounds are errors. Infinite conformal thresholds (k > n) are kept and reported (`frac_infinite`). The fraction of floored scores is recorded per channel; a sensitivity analysis is added only if more than 1% of calibration scores are floored.
- **Metrics:** RMSE per channel (physical), pooled, and dimensionless mean; marginal coverage (overall/channel/horizon); width (physical and normalised); interval score; simultaneous trajectory coverage, reported separately.
- **Provenance:** every checkpoint of every evaluation is checked (count = declared ensemble size, distinct checkpoints and seeds, identical model config, parameter shapes, features, target order, split files, split hash and scalers).
- **Uncertainty:** `python -m src.analysis.bootstrap --spec … --out …` does a recording-level cluster bootstrap (2000 draws, shared across methods) giving 95% CIs and paired differences on identical recordings. Between-seed SD across the 3 repeats is reported separately. No finite-sample guarantee is claimed.

## 6. Repeats
3 independent repeats per trained configuration. An ensemble repeat is 5 fresh members; member 0 of each repeat is that repeat's single-model backbone. The 5 members are not 5 repeats.

## 7. Full run list (after the pilot; not submitted)
`scripts/slurm/v3/run_list.tsv`: 63 rows = 55 unconditional trainings + 8 conditional (stage 0b).

| Stage | Trainings |
|---|---|
| 0 feature ablation | 4 |
| 0b repeat round (only if rule returns "repeat") | 8 |
| 1a LSTM/MLP ensembles, 3 × 5 | 30 |
| 1b GRU, TCN, Linear-SGD × 3 | 9 (+ ridge, naive on CPU) |
| 1c MC dropout LSTM/MLP × 3 | 6 |
| 1d LSTM Gaussian × 3 | 3 |
| **1e MLP Gaussian × 3 (new, separate increment)** | **3** |

**Compute**, to be replaced by the pilot's measured projection. A local CPU benchmark (Apple laptop) gave about 1 min per LSTM epoch, about 1.4 s per MLP epoch, and about 240k windows/s from the loader, so data loading is not the bottleneck. As an upper bound, assume L40 ≥ laptop CPU speed. The 30 recurrent trainings (LSTM incl. stage 0b, GRU, LSTM-dropout, LSTM-Gaussian) plus 3 TCN runs (assumed no slower) then need at most 33 × 100 epochs × 1 min ≈ 55 GPU-h worst case, about 22 GPU-h at 40 epochs. The feed-forward trainings are negligible and evaluation is a few GPU-h. **Increment for the MLP Gaussian (stage 1e):** 3 runs × ≤ 100 epochs of an MLP-size model plus 3 evaluations, ≤ 1 GPU-h; the pilot measures it directly (task 6).

## 7a. Timing pilot (awaiting authorisation)
- **Jobs:** `scripts/slurm/v3/pilot_train.sh` (array 1–7, `pilot_runs.tsv`) then `pilot_eval.sh` (`--dependency=afterok`).
  - Training: LSTM seeds 9001/9002, MLP, LSTM-dropout 0.2, LSTM-Gaussian, MLP-Gaussian and TCN on the full data, 3 epochs each, early stopping off.
  - Evaluation: point (LSTM, MLP), a 2-member LSTM ensemble, MC dropout with 200 passes, LSTM- and MLP-Gaussian, ridge fit, bootstrap, `scripts/pilot_report.py`.
  - The evaluations score the **validation split only** (`--eval_splits val`), so no test/OOD number is produced.
- **Maximum GPU-hours (SLURM hard limits):** 7 × 0.5 h + 1 × 2 h = **5.5 GPU-h**. Expected ≈ 1–2 GPU-h.
- **Expected outputs:**
  - per-run `train_log.json` (s/epoch, windows/s, peak GPU memory, peak RSS);
  - per-eval `metrics.json` (inference seconds per split, GPU memory, member validation);
  - `bootstrap/summary_val.csv` and `paired_val.csv`;
  - `pilot_report.json` with every check and the projected full-plan GPU-hours per config.
- **Pass criteria (mechanical, `pilot_report.py`):**
  - all jobs exit 0 within limits;
  - every run has 3 finite epochs, and train loss decreases from epoch 1 to 3;
  - GPU memory < 40 GB and host RSS < 14 GB;
  - identical split hash across runs;
  - every evaluation has no NaN in coverage/width/score, scored val only, used exactly the manifest's calibration recordings, and wrote < 1 GB;
  - the bootstrap output exists;
  - projected full plan ≤ 40 GPU-h at 40 epochs and ≤ 100 GPU-h at 100 epochs;
  - every planned config has a timing proxy.
- **On failure:** stop, report, and revise before any stage-0/1 submission. Pilot checkpoints are not reused as benchmark runs.
- The chain was dry-run end-to-end on trimmed recordings (CPU, 2026-09-28) and all checks passed. That only shows the scripts work; the dry-run numbers are meaningless.

## 8. Selection hygiene
No hyperparameter, seed, calibration setting, floor, feature set or example window is chosen from ID-test or OOD results. The pilot never scores test/OOD. Example trajectories are chosen by a fixed rule (median per-window RMSE of the primary model, same window across methods) and their IDs are reported.
