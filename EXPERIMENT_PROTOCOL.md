# Experiment protocol v3 (pre-run draft)

Status (2026-09-28): **nothing run on the full data yet.**
- D1 and D2 are the protocol; the primary calibration split is fixed (§1).
- The feature set is decided by the §3 rule after stage 0; the pilot uses `no_time`.
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
- **Spread floor:** floor_d = 1e-3 × max(mean calibration spread_d, SD of calibration targets_d). It is strictly positive and in the channel's physical scale. If both terms are zero (constant target and no spread), evaluation stops with `DegenerateSpreadError`. NaN inputs or bounds are errors. Infinite conformal thresholds (k > n) are kept and reported (`frac_infinite`). The fraction of floored scores is recorded per channel; a sensitivity analysis (floor_rel 1e-4 / 1e-3 / 1e-2) is added if more than 1% of calibration scores are floored for any method. Note (pilot): because every method's mean spread is well below the target SD, the target-SD term sets the floor in practice (floor ≈ 1e-3 × SD(y)).
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
