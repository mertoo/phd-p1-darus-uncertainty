# Experiment protocol v3.0 (frozen 2026-09-28; Stages A–D executed at commit 83585da)

Status (2026-09-28): **nothing run on the full data yet.**
- D1 and D2 are the protocol; the primary calibration split is fixed (§1).
- Feature set: **`no_time`**, decided by the §3 rule in Stage A (2026-09-28). A2 was not triggered.
- The timing pilot (§7a) **ran on 2026-09-28 and PASSED** (0.075 billed GPU-h).
- The full run list (§7) needs **separate authorisation** after the pilot report.

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
- **Decision (author, 2026-09-28): the fixed-seed random draw is the PRIMARY calibration split.**
  `20190805-100228, -100852, -101342, -102826, -104006, -104246, -104247, -105120, -110729` (.csv)
  - No provenance, duplication or eligibility problem was found: all 9 are unique deposit-training files listed in the MANIFEST, complete (3601 rows, no gaps or missing values), with no content duplicates among the 125 files.
  - The failed 0.5-SD check is reported descriptively: max |SMD| 0.84 (`deltal_std`); speed variables `u_mean` −0.52 and `n_mean` −0.49, i.e. somewhat slower calibration recordings. It is **not** replaced by a post-hoc threshold. The chance-reference numbers are descriptive only.
- **Sensitivity analysis (optional, not in the primary run list):** the speed-stratified draw (`method: stratified_n`; overlaps the primary in 2 recordings) can be run by retraining with that config setting, because it changes the fit set.
- The selection history is recorded in `calibration_selection.json` (`history`), including the brief provisional use of `stratified_n` in commit 1970dd9.

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
- **Spread floor (FROZEN, v3.0):** for each method, channel d and calibration set,
  `floor_d = 1e-3 × max( mean over calibration windows/steps of spread_d , SD over calibration windows/steps of y_d )`.
  - Scores are `|y − μ| / max(spread, floor_d)`, and intervals are `μ ± q · max(spread, floor_d)`.
  - If both terms are 0, evaluation stops with `DegenerateSpreadError`.
  - NaN inputs or bounds are errors; ±inf bounds (k > n) are legitimate and reported as `frac_infinite`.
  - In practice the target-SD term sets the floor, because every spread measured so far is well below the target SD (P10).
- **Floor sensitivity (FROZEN):** floor_rel ∈ {1e-4, **1e-3 (primary)**, 1e-2}.
  - Triggered for any method whose floored calibration fraction exceeds 1% in any channel. P10 is **open**, so the floored fractions of all spread methods are reported regardless.
  - The analysis is recomputed on CPU from the saved calibration and evaluation predictions (no retraining, no GPU). It reports how thresholds, widths, coverage and interval scores change.
  - It does **not** select a floor: the primary result stays at 1e-3.
- **"Chance" reference for floor activation (provisional comparison only):** P10 compared the pilot's floored fraction with a half-normal reference. The reference assumes the two members' prediction difference is Gaussian, mean zero, with one variance shared by all windows and horizon steps of a channel; the 2-member SD |a−b|/√2 is then half-normal with its scale fixed by the channel's mean spread. Real member disagreement varies between windows and lead times, and a scale mixture puts more mass near zero than a single half-normal. The windows also overlap and are not independent. The reference is therefore descriptive only, not a test, and does not apply to 5-member ensembles or MC dropout.
- **Metrics:** RMSE per channel (physical), pooled, and dimensionless mean; marginal coverage (overall/channel/horizon); width (physical and normalised); interval score; simultaneous trajectory coverage, reported separately.
- **Provenance:** every checkpoint of every evaluation is checked (count = declared ensemble size, distinct checkpoints and seeds, identical model config, parameter shapes, features, target order, split files, split hash and scalers).
- **Uncertainty:** `python -m src.analysis.bootstrap --spec … --out …` does a recording-level cluster bootstrap (2000 draws, shared across methods) giving 95% CIs and paired differences on identical recordings. Between-seed SD across the 3 repeats is reported separately. No finite-sample guarantee is claimed.

## 6. Repeats
3 independent repeats per trained configuration. An ensemble repeat is 5 fresh members; member 0 of each repeat is that repeat's single-model backbone. The 5 members are not 5 repeats.

## 7. Full run list (not submitted)
`scripts/slurm/v3/run_list.tsv`: 63 training rows, plus `eval_list.tsv` with 35 evaluations (ridge and naive trained by the CPU job `cpu_baselines.sh`) = 55 unconditional trainings + 8 conditional (stage 0b).

| Stage | Trainings |
|---|---|
| 0 feature ablation | 4 |
| 0b repeat round (only if rule returns "repeat") | 8 |
| 1a LSTM/MLP ensembles, 3 × 5 | 30 |
| 1b GRU, TCN, Linear-SGD × 3 | 9 (+ ridge, naive on CPU) |
| 1c MC dropout LSTM/MLP × 3 | 6 |
| 1d LSTM Gaussian × 3 | 3 |
| **1e MLP Gaussian × 3 (new, separate increment)** | **3** |

**Compute:** superseded by the pilot measurements; see §9 and `experiments/manifests/full_plan_estimate.json` (`python -m scripts.estimate_full_plan`).

## 7a. Timing/correctness pilot (authorised 2026-09-28: ≤ 5.5 L40 GPU-h combined, no retries)
- **Preflight** (login node, no GPU): `python -m scripts.pilot_preflight` must print `PREFLIGHT OK`. It checks:
  - identical data blocks in all pilot configs, with the primary random calibration (seed 20261001) and training-only standardisation;
  - the loader's calibration list equals the manifest's;
  - summed SLURM limits across **every array task and the evaluation job** are ≤ 5.5 GPU-h (7 × 0.5 h + 2 h = 5.5 h);
  - `--no-requeue` on both scripts;
  - no earlier pilot outputs;
  - a clean tree at a pushed commit.
- **Jobs:**
  - `TRAIN=$(sbatch --parsable --array=1-7 scripts/slurm/v3/pilot_train.sh)`: seven 3-epoch trainings, early stopping off: LSTM ×2, MLP, LSTM-dropout, LSTM-Gaussian, MLP-Gaussian, TCN.
  - `sbatch --dependency=afterany:$TRAIN scripts/slurm/v3/pilot_eval.sh`: point LSTM/MLP, 2-member LSTM ensemble, MC dropout (200 passes), LSTM/MLP Gaussian, ridge, bootstrap, report. It runs even if a training task failed, so the failure is reported.
- **Scope:** the validation split only (`--eval_splits val`). Test and OOD are never scored. No pilot coverage or loss is used to choose methods, calibration settings or the feature set. Pilot checkpoints are not reused.
- **Report** (`scripts/pilot_report.py`, exit 1 on FAIL):
  - *Correctness (failure):*
    - exactly 3 epochs and all losses finite;
    - identical split hash, scaler hash and features across runs;
    - calibration files equal the primary list, and standardisation is on;
    - every evaluation present, with no NaN coverage/width/score, validation-only scoring, the primary calibration list, and member provenance validated;
    - bootstrap output present.

    Invalid intervals and provenance mismatches also raise inside the evaluator and are reported as a failed step.
  - *Resources (failure):* peak GPU memory < 40 GB, peak host RSS < 14 GB, each evaluation's artifacts < 1 GB. The job limits enforce the time cap.
  - *Diagnostics (inspect, never automatic failures):* non-monotonic train or validation loss over the 3 epochs; > 1% of calibration spreads floored; infinite interval bounds.
  - *Planning (separate):* projected GPU-h for the 63-row run list at 40 and 100 epochs, including evaluation. Exceeding 40 / 100 GPU-h blocks the full benchmark until it is re-planned; it is not a pilot failure.
- **Result (2026-09-28): PASS.**
  - Jobs 984243 (array 1–7) and 984244 on `ada1` (L40), commit `54a69d5`; environment `~/envs/darus-v3` built by CPU job 984242. Record: `experiments/manifests/pilot_2026-09-28/`.
  - Billed: 271 GPU-seconds = **0.075 GPU-h** of the 5.5 authorised.
  - Training speed: 3.4–3.7 s/epoch for LSTM variants, 2.4–2.8 for MLP variants, 3.4 for TCN (46–70k windows/s).
  - Memory: ≤ 0.22 GB GPU, about 1.1 GB peak RSS in training; eval job RSS 1.6 GB.
  - Evaluation (cal + val, 63.8k windows): 1–2.5 s per method, 23 s per split for MC dropout with 200 passes.
  - Artifacts: 59–125 MB per method at cal + val scale, i.e. about 0.23–0.47 GB per method at full scale (< 1 GB).
  - Numerical issues: none (no non-finite losses, no NaN metrics).
  - Diagnostics:
    - TCN validation loss rose slightly at epoch 3 (0.1911 → 0.1919); ordinary fluctuation, and the best epoch (2) was kept.
    - The 2-member, 3-epoch pilot ensemble floored 1.0–1.9% of calibration spreads per channel, above the 1% trigger. In the full run the floor sensitivity analysis will be run if the 5-member ensembles also exceed 1%.
  - **Revised full-run estimate** (63 rows, compute only): 2.1 GPU-h training at 40 epochs, 5.2 at 100 epochs; evaluation ≈ 0.2–0.5 GPU-h (35 evaluation jobs). With about 15–20 s SLURM start-up per job, the total is ≈ 3 GPU-h expected and ≈ 6 GPU-h worst case, compared with the earlier 22–55 GPU-h estimate.
- **After the pilot,** report: measured GPU-hours from `sacct -j <ids> --format=JobID,Elapsed,AllocTRES%40,MaxRSS,State`; peak memory; throughput (s/epoch, windows/s, inference s); artifact sizes; numerical issues; and the revised full-run estimate. Then wait for separate authorisation of the full benchmark.

## 8. Selection hygiene
No hyperparameter, seed, calibration setting, floor, feature set or example window is chosen from ID-test or OOD results. The pilot never scores test/OOD. Example trajectories are chosen by a fixed rule (median per-window RMSE of the primary model, same window across methods) and their IDs are reported.

## 9. Staged execution proposal (awaiting author approval)
Everything runs from one pushed commit (the "v3.0 commit", reported with the approval request); stage B may use a follow-up commit that only records the feature decision (§3). Each stage has a preflight before submission and a gate after it (`scripts/check_stage.py`: finite losses, 665 optimiser steps per epoch, 170,016 training windows, seeds, primary calibration files, identical split and scaler hashes, member counts, evaluation validity, and the floor-activation table). A failed gate stops the next stage. No automatic retries (`--no-requeue`). Skipped or failed tasks are reported, never replaced by historical checkpoints. The per-task limit is 15 min, against a measured worst case of about 6.5 min.

| Stage | Jobs | Hard ceiling (limits × GPUs) | Estimate (measured, max epochs) | Gate |
|---|---|---|---|---|
| A: feature ablation (stage 0) | 4 GPU trainings | 1.00 GPU-h | 0.34 GPU-h | `check_stage --stages 0`; `select_features` decision committed |
| A2: conditional repeat (0b), only if A says "repeat" | 8 GPU trainings | 2.00 GPU-h | 0.69 GPU-h | same, round 2 |
| B: all benchmark trainings (1a–1e) + CPU baselines | 51 GPU trainings + 1 CPU job | 12.75 GPU-h (+ 0.5 CPU-h) | 4.47 GPU-h | `check_stage --stages 1a 1b 1c 1d 1e` |
| C: evaluations (test + OOD, first time) | 35 GPU eval tasks | 8.75 GPU-h | 1.09 GPU-h | `check_stage --evals` + floor table |
| D: analysis | CPU only (bootstrap per `bootstrap_full.yaml`, floor sensitivity, tables) | ≤ 2 CPU-h | < 1 CPU-h | outputs complete |
| **Total** | | **24.5 GPU-h ceiling** | **6.6 GPU-h (9.9 with ×1.5 contingency)** | |

Storage ≈ 12.3 GB (evaluation artifacts) + 0.1 GB (checkpoints), against 468 GB free in the home quota. The ceiling is what SLURM can bill at most. The estimate uses measured L40 epoch times with every run at 100 epochs (early stopping can only lower it). GRU, linear and MLP-dropout are timed by proxy. Cumulative billed GPU-h is read from `sacct` at every gate, and remaining stages are not submitted if the authorised total would be exceeded.

## 10. Analysis rules (FROZEN 2026-09-28, before Stage C)
Fixed before any full-benchmark test/OOD result exists. Any later change is logged as a deviation.
- **Tables and figures:** `python -m src.analysis.report --rules experiments/configs/v3/analysis_rules.yaml`.
  - Tables: baselines (per-channel physical RMSE and normalised RMSE with 95% CI and seed SD); UQ summary at 90% (coverage, normalised width, normalised interval score, all with CI); per-channel coverage and width; paired differences for the pairs predeclared in `bootstrap_full.yaml`.
  - Figures: coverage vs nominal level (mean and min–max over repeats); horizon-resolved coverage and width; one example window.
  - Every number is read from bootstrap outputs, `metrics.json` or saved predictions. `analysis_manifest.json` records the SHA-256 of all inputs and outputs.
- **Example window:** reference evaluation `lstm_single_rep0`, ID test. Chosen as the window whose mean over channels of per-window RMSE / training-target SD equals the median; ties go to the smallest window ID. The same window is used for every panel, and its ID and recording are printed on the figure. It is an illustration only.
- **Missing inputs block:** any `eval_list.tsv` evaluation without `metrics.json`, any missing bootstrap summary, or a method with ≠ 3 repeats stops both the bootstrap (`required_repeats`) and the report. No partial tables are produced. The Stage C gate (`check_stage --evals`) fails on the same conditions.
- **Floor sensitivity:** `python -m src.analysis.floor_sensitivity` runs for all spread-method evaluations at floor_rel {1e-4, 1e-3, 1e-2}, for both variants, all levels and all scored splits. It fails if the rebuilt primary intervals differ from the stored metrics (coverage > 1e-4 absolute, normalised width > 1e-4 relative). A result is marked `triggered` when the primary floor clamps > 1% of calibration spreads in any channel. The primary floor is never replaced.
- **Budget rule:** before every submission batch, the preflight requires consumed GPU-h (ledger `experiments/manifests/gpu_ledger.tsv`, from `sacct`) + the batch's maximum allocation (SLURM limits × GPUs × tasks) ≤ the authorised total (`--scope benchmark --cap_total X`).
