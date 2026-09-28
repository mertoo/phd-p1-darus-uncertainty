# Revision tracker — UQ benchmark (manuscript baseline `main v2.2.tex`)

Last updated: 2026-09-28 (pre-run corrections round). Branch `revision/v2.3` (pushed; PR mertoo/phd-p1-darus-uncertainty#2), created from `origin/main` = `9419c66` (the audited snapshot).

Status vocabulary: **confirmed** (verified against source/log/data), **contradicted**, **modified** (true in part), **unverified**. Work status: `open`, `implemented – awaiting experiments`, `verified complete`, `blocked`, `deferred`.

**Completion levels** (the summary table below uses these four columns, each yes/no/n.a.):
1. **Code**: implemented in the repository.
2. **Tests**: targeted synthetic unit tests pass (`pytest tests`, 54 tests on 2026-09-28).
3. **E2E**: connected end-to-end and exercised by a CLI run on trimmed *real* recordings (2 to 3 epochs, 3 recordings per split). This shows the pipeline runs; it produces no scientific result.
4. **Full**: the corrected full-data experiment has finished and its outputs have been checked. As of 2026-09-28 (Stages A–D, commit `83585da`), items marked `yes` in the Full column have reached this level. This does not by itself establish the manuscript's final conclusions (see RESULTS_CHANGELOG.md).
Evidence types: *src* = source read, *log* = committed historical log, *data* = computed on the local DaRUS files, *test* = synthetic unit test. A passing test is never a paper result.

---

## R0 inventory (verified complete)

### Source and manuscript identity
| Item | Finding |
|---|---|
| Local branch at start | `p2-uncertainty` @ `8d16deb`, **13 commits behind** the audited `origin/main` `9419c66` (fast-forward; no divergent local commits). |
| Revision branch | `revision/v2.3` from `9419c66`. User's uncommitted files carried over untouched: `experiments/hidden_state_analysis.py`, `experiments/prediction_setup_figure.py`, `paper/figs/hidden_state_analysis.{pdf,png}`, modified `experiments/results/.DS_Store`. A stash `stash@{0}` (small edits to `train_baseline.py`, `train_ensemble.py`) exists and was left alone. |
| Manuscript baseline | `~/Downloads/main v2.2.tex`, 496 lines, SHA-256 `34dfb52d…0d841a59` (matches brief). Not copied into the repo yet; the repo's `paper/main.tex` and the nested `paper/Uncertainty_Aware_…/main.tex` are older versions. |
| Bibliography | `paper/Uncertainty_Aware_…/bibliography.bib` (84 lines) is the only .bib; its correspondence to v2.2 citation keys is **unverified**. |

### Data (evidence: *data*, `scripts/audit_data.py` → `experiments/manifests/`)
| Item | Finding |
|---|---|
| Files | 125 processed CSVs; names and byte sizes match the deposit `MANIFEST.TXT` exactly. `data/raw/darus/` is empty: no raw-to-processed pipeline or raw files exist locally. |
| Split | The deposit's own partition: train 57 / validation 9 / ID test 30 / OOD test 29 recordings. No re-split in code. |
| Recordings | Every recording is exactly 3601 rows, `time` = 0…3600 s, Δt = 1.000 s everywhere (**1 Hz confirmed**), no gaps, repeats, missing values, duplicate rows, or duplicate contents across splits. |
| Columns | `time, n, deltal, deltar, Vw, alpha_x, alpha_y, u, v, p, r, phi` (units in the deposit README.txt). No wave elevation/force, significant wave height or period in the files. |
| Valid windows (H=T=30, stride 1) | 3542 per recording: train 201,894; val 31,878; test 106,260; OOD 102,718. |
| Legacy windows crossing a recording join | train 3,304 / 205,197 (1.61%); val 472 (1.46%); test 1,711 (1.58%); OOD 1,652 (1.58%). |
| Operating-condition shift | OOD `u` range 3.45–13.44 m/s vs train 0.95–9.25; mean `u` 8.50 vs 5.64. OOD shaft speed `n` mean 1582 vs 1040 1/s, max 2233 vs 1612, so OOD requires **extrapolation in a control input**. Wind `Vw` similar (mean 1.81 vs 1.97). `Vw` takes small negative values (min −0.70 m/s) despite being a speed (unexplained). Sea-state parameters are not in the files. |

### Checkpoints, logs, figures
| Artifact | Status |
|---|---|
| Checkpoints that produced the paper numbers (HPC, Apr 2026: jobs 918426 baselines, 918397 LSTM ensemble, 918445 MLP ensemble, 918411 MC dropout, Gaussian 9183xx) | **Not present locally.** Local `experiments/results/*/best_model.pt` are Nov–Dec 2025 CPU runs from older code, so they cannot reproduce paper tables. |
| Logs in repo | `logs/slurm/`: Gaussian training/eval, LSTM ensemble train+eval, conformal eval of **ensemble member 3** (918407), MC-dropout eval, paper plots. **Missing:** baseline train/eval (918426/918432), MLP ensemble (918445/918450), LSTM-vs-MLP conformal (918441), MC-dropout training (918411). Values from those jobs exist only as hand-copied numbers in `experiments/results/RESULTS.md`. |
| Saved predictions | None anywhere. The limitation text ("reprocessing already-saved per-member predictions") is **contradicted**. |
| Seeds | Never set or recorded in any historical path; historical seeds are unrecoverable. |
| Environment | `requirements.txt` (pins except torch). HPC: Python 3.10.8, CUDA 12.2 module (the manuscript says CUDA 12.1 and PyTorch 2.5.1; torch version **unverified**). Local: macOS, Python 3.14, CPU only, 10 cores, 16 GB RAM. |
| Figure 1 `prediction_setup_figure.pdf` | Generator found (untracked `experiments/prediction_setup_figure.py`). It uses the **local Nov-2025 checkpoint**, searches ID test windows for the *best-tracked* `r`/`v` window, draws **yaw rate r**, and draws **no interval**. The v2.2 caption (surge `u`, 90% interval, "representative") does not describe this figure. |
| Figure 2 `hidden_state_analysis.pdf` | Generator found (untracked `experiments/hidden_state_analysis.py`). It uses the local Nov-2025 checkpoint, the first 50 batches only, and **feeds `fc(output)` back as decoder input**, whereas the model's decoder receives zeros. The plotted states are therefore not the states of the model that was evaluated. |
| Figures 3–7 (`ensemble_u_test`, `mc_u_test`, `conformal_phi_test`, `ensemble_u_ood`, `gaussian_phi_ood`) | From `scripts/slurm/regen_paper_plots.sh` (log 918414). Every panel shows window 0 (first window of the first recording in the split), which is not a representative-selection rule. `conformal_phi_test` uses **ensemble member 3** with `run_conformal_eval.py` **horizon×channel** quantiles, which is neither of Table 5's constructions. `gaussian_phi_ood` uses `eval_lstm_gaussian.py`: one **global** T_σ = 0.958 fitted by **NLL** on grid 0.5–5, with a **±2σ** band, not Table 4's per-channel coverage-fitted temperatures at 90%. In the same job, the "baseline" member 3 gives test/OOD RMSE 0.115/0.505, not Table 2's 0.120/0.537. |

### Table/figure → generator map
| Paper item | Generator | Checkpoint | Reproducible now? |
|---|---|---|---|
| Tab. 2 baseline RMSE | `run_eval.py` via `eval_baselines.sh` (918432, log missing) | HPC `p1_*_baseline` | no (log and ckpt absent) |
| Tab. 3 ensembles | `eval_ensemble.py` (918406 LSTM; 918450 MLP, log missing) | HPC ensembles | no |
| Tab. 4 Gaussian | `eval_gaussian_diagnostics_per_dof.py` (918386) | HPC `p2_lstm_gaussian` | no |
| Tab. 5 conformal | **mixed** (see R4) | ensemble member 3 + LSTM baseline + MLP baseline | no |
| Tab. 6 conformal backbone | `eval_conformal_metrics.py` (918441, log missing) | HPC baselines | no |
| Tab. 7 summary | hand-assembled from RESULTS.md | mixed | no |
| Fig. 1, Fig. 2 | untracked scripts above | local Nov-2025 LSTM | yes, but not the evaluated model |

---

## Completion summary (four levels)

| Item | Code | Tests | E2E | Full | Notes |
|---|---|---|---|---|---|
| R1 recording-aware windows, correct counts | yes | yes | yes | yes | Stages A–D |
| R2 splits/roles: calibration hold-out (D1), train-only scaling (D2) | yes | yes | yes | no | D1/D2 adopted; primary calibration = fixed-seed random draw (author decision) |
| R2 calibration representativeness check | yes | n.a. | yes (full training data) | n.a. | predeclared criterion failed for both draws; reported descriptively, no post-hoc threshold; stratified draw kept as optional sensitivity |
| R3 named interval variants, one object per result | yes | yes | yes | yes | |
| R3b finite-sample order statistic, legitimate +inf | yes | yes | yes | yes | no infinite bounds occurred (n = 31,878 calibration windows) |
| R4 regenerate tables from structured records | yes | yes | yes | yes | `experiments/analysis/v3/report/` (manuscript not yet edited) |
| R5 feature set (`time`) rule + selection script | yes | yes | yes | yes | Stage A: `no_time` |
| R5 standardised loss/metrics | yes | yes | yes | yes | |
| R6 raw vs calibrated UQ at common levels | yes | yes | yes | yes | |
| R6 zero/near-zero spread floor, NaN policy | yes | yes | yes | yes | P1; floor inactive at full scale (P10) |
| R7 ddof=1, count-weighted metrics, RMSE conventions | yes | yes | yes | yes | |
| R8 seeds, provenance, no-overwrite, strict configs | yes | yes | yes | yes | |
| R8 all-member provenance validation | yes | yes | yes | yes | Stage C: 35/35 validated |
| R9 ridge sanity baseline | yes | n.a. | yes | yes | ridge ≈ networks; the v2.2 linear failure did not reproduce |
| R9 MLP Gaussian (backbone pairing) | yes | yes | yes | yes | |
| R11 bootstrap CIs + paired differences entry point | yes | yes | yes | yes | Stage D |
| R11 channel/horizon/level analyses | yes | yes | yes | yes | Stage D report |
| Timing pilot (preflight, jobs, report) | yes | yes (`test_pilot_report.py`) | yes (HPC, full data, 3 epochs) | pilot complete — PASS | 0.075 GPU-h billed; validation split only; not a benchmark result |
| R10 core benchmark rerun | yes | yes | yes | yes | Stages A–D; 3 repeats |
| R12 backbone pairings complete; OOD confirmation | yes | yes | yes | partial | all 4 UQ families × LSTM/MLP done; no untouched OOD set, so OOD stays exploratory |
| R13 manuscript alignment | no | no | no | no | pending author review of RESULTS_CHANGELOG.md |
| R14 packaging / release | no | no | no | no | |

## Pre-run corrections round (2026-09-28, author request)

| ID | Request | What was done | Evidence |
|---|---|---|---|
| P1 | Zero-spread calibration produced NaN | The floor is now `1e-3 × max(mean calibration spread, SD of calibration targets)` per channel, strictly positive and scale-aware. A degenerate channel (constant target and zero spread) raises `DegenerateSpreadError`. Non-finite inputs raise; NaN bounds are rejected by `Intervals`; ±inf bounds (k > n) remain legitimate and are reported via `frac_infinite`. Floored fractions are recorded per channel. | `tests/test_intervals.py` (zero-spread channel, near-zero spread, degenerate channel, NaN in μ/s/y, NaN bounds, infinite conformal and spread-normalised intervals) |
| P2 | Validate every ensemble member | `src/evaluation/provenance.py` checks all members: declared size (`--n_members`, now required for ensembles), distinct checkpoint SHA-256 and seeds, identical model config and parameter shapes, identical data state (features, target order, split files/hash, scalers); mixed legacy sets are rejected. The rebuilt data state is checked against **every** member. `--method gaussian` is checked against the model type. | `tests/test_provenance.py`; smoke: wrong size and wrong method correctly refused |
| P3 | Time-feature rule | Protocol §3 and `scripts/select_features.py` (validation losses only). Both backbones must agree for `legacy`; borderline or disagreement means one repeat round (stage 0b, seeds 2 and 3), then default `no_time`; common feature set for all models. | `tests/test_select_features.py` |
| P4 | Bootstrap entry point | `python -m src.analysis.bootstrap --spec … --out …`: recording-level cluster bootstrap with draws shared across methods, CIs, paired differences, and between-seed SD reported separately. Refuses unpaired windows or mismatched scales. The helper in `uq_metrics` was removed (single implementation). | `tests/test_bootstrap.py`; smoke run on 4 methods |
| P5 | D1/D2 preferred; document and check calibration recordings | `scripts/check_calibration_split.py` produces the documented list and SMD table. The predeclared criterion failed for the random draw (max \|SMD\| 0.84) **and** the stratified fallback (0.83). A diagnostic added afterwards shows the criterion passes only 7.1% of random draws, and both selections are typical (≈67th percentile). `stratified_n` is adopted provisionally; the author decides (C1). | `experiments/manifests/calibration_selection.json`, `calibration_representativeness.csv`; loader reproduces the documented list |
| P6 | MLP Gaussian | `src/models/mlp_gaussian.py`, factory entry, config `v3/mlp_gaussian.yaml`, run-list stage 1e (3 seeds). Increment ≤ 1 GPU-h (to be measured by the pilot). | `tests/test_models.py` (NLL constant, learns heteroscedastic scale, logvar clamp); smoke train + eval |
| P7 | Timing pilot | `scripts/slurm/v3/pilot_{train,eval}.sh`, `pilot_runs.tsv`, `scripts/pilot_report.py`. 7 trainings × 3 epochs + evaluation on the validation split only; hard cap 5.5 GPU-h; mechanical pass/fail. | full chain dry-run on trimmed data: all checks pass (numbers meaningless) |

| P8 | Calibration decision + pilot authorisation | Primary calibration = fixed-seed random draw, after an eligibility check (unique deposit-training files, complete, no content duplicates); balance reported descriptively; stratified draw kept as optional sensitivity; history in `calibration_selection.json`. Pilot report split into correctness failures, resource failures, diagnostics (non-monotonic losses are diagnostics only) and a separate planning section; exits 1 on failure. `scripts/pilot_preflight.py` checks the frozen split and scaling in every pilot config and the aggregate SLURM cap over all array tasks + eval (5.50 h). `--no-requeue` on both jobs; eval runs `afterany` and reports failed steps. | `tests/test_pilot_report.py`; dry run on trimmed data; local preflight passes except for uncommitted changes |

| P9 | Pilot execution | The cluster's April venv was unusable (its Python install is no longer accessible). A new environment `~/envs/darus-v3` was built by CPU-only job 984242 (Python 3.11.16, torch 2.5.1+cu121, numpy 2.3.5, pandas 2.3.3; freeze archived); **60/60 tests pass on the cluster**; preflight OK. Pilot jobs 984243/984244 **PASS**: 0.075 GPU-h billed, no numerical issues, 2 diagnostics (TCN val loss +0.4% at epoch 3; 2-member pilot ensemble floors 1.0–1.9% of calibration spreads). Revised full-run estimate ≈ 3 GPU-h expected, ≈ 6 worst case. | `experiments/manifests/pilot_2026-09-28/` (report, sacct, pip freeze) |

| P10 (pilot) | Floor-activation diagnostic (pilot 2-member ensemble); superseded by the closed P10 entry below | Investigated using calibration-split predictions only (no val/test/OOD, no coverage). (1) The fraction below the floor is 1.00–1.91% per channel (recomputed = stored). There are no exact zeros, and float32 resolution is far below the floor, so this is not an artifact. (2) In every channel the floor is set by the **target-SD term**, 1e-3 × SD(y), not the spread term: mean ensemble spread is only 3–6% of target SD. So the floor is about 18–30× larger than "1e-3 × mean spread" would be. (3) For u, v, r the floored fraction matches a half-normal reference for a 2-member SD (1.12/1.25/1.00% vs 1.13/1.12/0.91%). For p and φ it exceeds it (1.51 vs 1.02%, 1.91 vs 1.38%). p's floored mass rises with lead time (2.7% at steps 28–29); the other channels concentrate at early steps. (4) Calibration-step sensitivity (level 0.9, `channel` variant): with floor_rel 1e-4 vs 1e-3, thresholds and median half-widths change by ≤ 1.6%; with floor_rel 1e-2 (10–19% floored), half-widths shrink by 23–37%. **Not labelled harmless:** this is a 2-member, 3-epoch model, and effects on coverage were not examined. Full run: report floored fractions for every spread method; if > 1%, run the predeclared floor sensitivity (floor_rel 1e-4 / 1e-3 / 1e-2) on calibration and report its effect on evaluation metrics. The p/φ excess over the reference remains to be explained. | local analysis of `ens_lstm/predictions_cal.npz` from pilot job 984244 |
| P11 | Commit used for the pilot | Author asked for `abd0909`; the pilot ran from `54a69d5`. The diff is limited to the 5 v3 SLURM scripts (environment activation: the dead April venv replaced by `~/envs/darus-v3`) plus the new `setup_env.sh`. `src/`, configs, tests, manifests, run lists and SLURM limits are identical. `abd0909` could not have run (its venv Python is inaccessible). The substitution should have been confirmed with the author first; it is disclosed here. | `git diff --stat abd0909 54a69d5` |

| P12 | Pre-authorisation verification (author items 1–5) | **(1)** All 7 pilot trainings: `base_path` `data/processed/darus` (no smoke data); the fit set is exactly the 48 expected recordings, and the calibration set equals the primary 9; windows train 170,016 / val 31,878 / cal 31,878 / test 106,260 / OOD 102,718; batch 256, stride 1, H=T=30, 3 epochs, no patience. Optimiser steps: **derived** as 665/epoch and 1,995 total (`drop_last=False`); they were *not measured* in the pilot. Training now logs measured steps and fails if an epoch's step or window count differs from the expected value. **(2)** Scheduler: 271 allocated GPU-s (8 jobs × 1 GPU) = 0.0753 GPU-h; measured compute 130.6 s (67.9 s training + 62.7 s inference), 48% of allocation; the rest is start-up, data building, metrics, I/O, ridge, bootstrap, report. Setup job: 174 s CPU. **(3)** `scripts/estimate_full_plan.py` → `full_plan_estimate.json`: 5.50 GPU-h training bound (63 rows at 100 epochs, incl. conditional 0b) + 1.09 GPU-h for 35 evaluations = 6.6 GPU-h; 9.9 with ×1.5 contingency; < 1 CPU-h; about 12.3 GB storage. **(4)** Floor formula and sensitivity levels frozen (protocol §5); chance reference documented as provisional (assumptions stated); P10 stays open. **(5)** Protocol v3.0 with the staged plan (§9). New: `eval_list.tsv`, `eval_array.sh` (skips missing checkpoints with exit 3, no fallback; exit codes propagate; both tested locally), `cpu_baselines.sh`, `check_stage.py` (tested), predeclared `bootstrap_full.yaml`, generalised preflight; 15-min per-task limits. | `tests/test_check_stage.py`, local eval-array skip tests, 66 tests pass |

| P13 | **Stage A (feature ablation) — executed 2026-09-28** at `dde1848` | Cluster preflight (CPU job 984251) passed 12/12. Array **984252** (4 tasks × 1 L40, 15-min limit, ceiling 1.00 GPU-h) completed on `ada1`: **203 GPU-s = 0.056 GPU-h**. Gate (`check_stage --stages 0`, CPU job 984256): **PASS**; measured 665 optimiser steps/epoch everywhere. Validation-only decision: best val loss LSTM legacy 0.18484 vs no_time 0.18448 (gain −0.19%); MLP 0.18665 vs 0.18600 (gain −0.35%). Not borderline and no disagreement → **`no_time`; A2 not triggered**. Diagnostic (not a finding): validation loss is lowest early (epoch 4 in three runs, 9 for MLP no_time) and then rises (e.g. LSTM no_time 0.1845 → 0.2171 by epoch 14), so early stopping ended training at epochs 14–19; the historical protocol trained a fixed 20 epochs. | `experiments/manifests/stageA_2026-09-28/` (sacct, gate, decision, train logs) |
| P14 | Analysis tooling (while Stage A ran) | `src/analysis/rebuild.py` (exact interval rebuild from saved predictions), `floor_sensitivity.py` (frozen rules; fails on rebuild mismatch), `report.py` + frozen `analysis_rules.yaml` (tables/figures from structured results only; missing evaluations or repeats block output; SHA-256 manifest). The bootstrap now requires the exact repeat count (`required_repeats`); ridge and naive were added to the predeclared spec. GPU ledger + preflight cap check (consumed + batch maximum ≤ authorised total). Validated on smoke-data evaluations only (numbers meaningless). | `tests/test_report.py`, `tests/test_floor_sensitivity.py`, `tests/test_bootstrap.py`; 72 tests pass |

### Remaining blockers and open items before/within the full benchmark
- **Before stage A:** none known in code. Needs author approval, then the preflight on the cluster at the v3.0 commit.
- ~~Before stage B: feature decision~~ recorded: `no_time` (Stage A).
- ~~Before stage D: floor-sensitivity script and table/figure generators~~ written, tested on smoke data, rules frozen (protocol §10).
- **Open, not blocking:** P10 (floor activation); GRU, linear-SGD and MLP-dropout epoch times are proxies; the ridge pilot selected alpha_rel = 0 (the unpenalised end of the grid), to be checked in stage B; no untouched OOD data, so OOD stays exploratory.

| P15 | **Stages B–D executed** (commit `83585da`) | **B:** preflights 5/5; arrays 984258–984262 (51 × 1 L40, 15 min) + CPU 984263 (ridge, naive); 52/52 completed; **0.772 GPU-h**; gate PASS; every run early-stopped (best epochs: recurrent/TCN 2–5, MLP 3–9, MLP-dropout 16–32, linear-SGD 22–49); ridge alpha_rel = 0. **C:** preflight (commit, clean tree, checkpoint counts, no earlier outputs, cap 0.828 + 8.75 ≤ 21.56) OK; array 984312 (35 × 1 L40); 35/35 completed; **0.717 GPU-h**; gate PASS; P10 floor activation ≤ 0.001%. **D:** CPU job 984348 (1 core = 2 threads, 2 h limit), 41 min 57 s = 1.40 thread-hours; bootstrap (19 methods; 30 ID / 29 OOD recordings), report (11 outputs), floor sensitivity (18 evaluations, not triggered). Cumulative benchmark allocation **1.546 GPU-h of 21.56**. | `experiments/manifests/stageB_2026-09-28/`, `stageC_2026-09-28/` (metrics, gate, sacct), `experiments/analysis/v3/`, `gpu_ledger.tsv` |
| P16 | Old-vs-new comparison | `RESULTS_CHANGELOG.md` + `scripts/results_changelog.py`: v2.2 values quoted, v3 values generated; claim-by-claim status (supported / weakened / contradicted / untested). Manuscript unchanged. | `experiments/analysis/v3/results_changelog_values.csv` |
| P10 | Floor activation — **CLOSED: resolved for this benchmark** (2026-09-28) | Verified from `experiments/analysis/v3/floor_sensitivity.csv` over 18 spread-method evaluations (EN/DN/GN-SCP × LSTM/MLP × 3 repeats), 5 channels, 2 variants, 4 levels and 2 splits. Primary floor (1e-3 × max(mean calibration spread, calibration-target SD)): max floored calibration fraction 6.3e-6, eval 3.8e-6; trigger (> 1%) never reached. floor_rel 1e-4: \|Δcoverage\| ≤ 7e-7. floor_rel 1e-2 (up to 15% of DN calibration spreads floored): \|Δcoverage\| ≤ 0.0025, \|Δnorm. width\| ≤ 0.049, \|Δnorm. interval score\| ≤ 0.076. Formula and sensitivity levels unchanged. The 2-member pilot's p/φ excess is not relevant to the reported models. **Not generalised beyond these runs.** | floor_sensitivity.csv |
| O1 | ID over-coverage — **documented, open** (no benchmark change) | Descriptive, from stored outputs (post hoc; `experiments/analysis/v3/posthoc/o1_coverage_breakdown.csv`). **Levels:** every SCP construction over-covers ID at every level: 0.61–0.65 at 50%, 0.86–0.87 at 80%, 0.93 at 90%, 0.96 at 95%. **Channels (90%):** u 0.94–0.96, v 0.985–0.99, p 0.89–0.91, r 0.90–0.91, φ 0.89–0.91. **Horizon (90%):** ≈ 0.934–0.945 at step 1, falling to ≈ 0.911–0.922 at step 30. **Seed variation:** ≤ 0.3 pp across 3 repeats (calibrated constructions). **Recording-level uncertainty:** AR-SCP LSTM 92.8% [90.9, 94.4] (bootstrap over 30 ID recordings), wider than the seed spread. **Width / score at 90%:** normalised width 0.90–1.20, interval score 1.30–1.63. **Marginal vs simultaneous:** whole-trajectory coverage 0.16–0.40. **Hypothesis only:** the calibration recordings are descriptively slower (SMD −0.52 on mean u, −0.49 on n). Per-recording coverage breakdown **not produced**: it needs the per-window files on the cluster, which was unreachable (VPN) during this revision. No calibration change was made. | o1_coverage_breakdown.csv; bootstrap summaries |
| O2 | Presentation | The `coverage_vs_level` legend (14 series) is crowded; any layout change will be logged as a deviation from the frozen analysis rules (numbers unchanged). | figure |

## Separate track: historical checkpoint recovery (not part of the new benchmark)
Purpose: an old-vs-new comparison for RESULTS_CHANGELOG.md only. Requires the April 2026 HPC checkpoints and logs (author question 1). If recovered, they are evaluated with `benchmark_eval --legacy_config` (recording-aware windows, validation calibration, flagged `legacy_checkpoint: true`). They are never mixed into benchmark tables. Status: **unblocked, not started.** The April artifacts exist on the cluster (`~/phd-p1-darus-uncertainty/experiments/results/`: baselines, Gaussian, LSTM/MLP ensembles, MC dropout; 106 logs in `logs/slurm/`, including jobs missing from GitHub such as 918426). Left untouched.


## Findings

| ID | Finding | Evidence | Verdict | Imp. | Diff. | Dep. | Work status | Files changed | Validation | Remaining |
|---|---|---|---|---|---|---|---|---|---|---|
| R1a | Windows slide across concatenated recordings | *src* `darus_dataset.py` (old), *data* 1.5–1.6% of windows per split | confirmed | 5 | 4 | R0 | implemented – awaiting experiments | `src/data_loading/darus_dataset.py`, `darus_parser.py` | `tests/test_windowing.py` (82 windows for 2×100 rows, gap splitting, short recordings, alignment) | retrain everything (R10) |
| R1b | Off-by-one window count (`len-H-T`) | *src*, *data* | confirmed | 3 | 1 | R0 | implemented – awaiting experiments | same | test | — |
| R2a | "≈70/15/15 split … differs from the dataset's originally documented partition" | *data*: deposit split used unchanged, 57/9/30 = 59.4/9.4/31.3% | **contradicted** | 5 | 1 | R0 | open (manuscript) | — | manifest | rewrite `sec:splits` |
| R2b | Validation used for checkpoint selection, conformal calibration and Gaussian temperature | *src* | confirmed | 5 | 3 | R0 | implemented – awaiting experiments: `data.calibration` holds out training recordings (v3: 9 of 57, seed 20261001) as a calibration-only split | `darus_dataset.py`, `benchmark_eval.py` | smoke test | protocol freeze + reruns |
| R2c | No preprocessing/scaling in loader | *src* | confirmed (no external scaler found either) | 5 | 3 | R0 | implemented – awaiting experiments: optional train-only `Standardizer`, stored in checkpoint and re-checked at evaluation | same | `test_scaling_round_trip` | decide in protocol |
| R2d | Raw→processed transformations | `data/raw` empty | blocked | 4 | 3 | author | blocked | — | — | ask for DaRUS DOIs / processing description (Q2) |
| R2e | OOD informed backbone choice (manuscript §3.1) | manuscript text | confirmed | 5 | 4 | R10 | open | — | — | no untouched OOD set exists (Q4) |
| R3 | Conformal "overall" used a pooled threshold `q_all`; per-DoF used `q_per_dim`; plots used horizon×channel | *src* `eval_conformal_metrics.py`, `run_conformal_eval.py` | confirmed | 5 | 3 | R0 | implemented – awaiting experiments: named variants `channel`, `horizon_channel`, one `Intervals` object per result, all metrics derived from it; legacy script relabelled | `src/uncertainty/intervals.py`, `src/evaluation/uq_metrics.py`, `benchmark_eval.py` | identity tests (overall = mean channel = mean horizon; masked weighted identity) | reruns |
| R3b | Plain empirical quantile, no finite-sample order statistic | *src* | confirmed | 4 | 1 | — | implemented: k = ⌈(n+1)(1−α)⌉, +inf when k>n | `intervals.py` | `test_order_statistic`, exchangeable coverage test | state n = overlapping windows in paper |
| R4 | Tab. 5 mixes runs: ID channels [91.4, 87.9, 92.0, 90.3, 91.0] = member-3 log 918407; overall ID 90.5/0.234 = LSTM-baseline summary; OOD `u` 25.5 = LSTM baseline; OOD v–φ [30.3, 78.7, 56.3, 53.4] = MLP baseline | *log* 918407 vs RESULTS.md §3b/3c vs v2.2 lines 332–333 | confirmed | 5 | 2 | R3, R10 | open | — | — | regenerate from structured metrics; also Tab. 5 declares 9 columns for 8 fields |
| R5a | `time` input = elapsed seconds 0–3600 restarting in every file | *data* | confirmed (not labelled leakage: identical distribution in all splits, but it encodes recording progress/start-up transient) | 5 | 3 | R2 | implemented: `features: legacy | no_time` | `darus_dataset.py` | test | validation-only ablation (protocol §3) |
| R5b | Aggregate MSE loss/RMSE dominated by `u`,`v` | *src* (no scaling), RESULTS.md | confirmed | 5 | 3 | R2 | implemented: standardised targets → equal-weight loss; metrics per channel in physical units plus dimensionless mean | `uq_metrics.py` | tests | retraining |
| R6 | ±2σ raw spread vs 90% calibrated intervals | *src*, manuscript | confirmed | 5 | 3 | R3 | implemented: every spread method reported raw (Gaussian-reference) **and** spread-normalised-calibrated on the same calibration recordings at predeclared levels 0.5/0.8/0.9/0.95, plus interval score | `intervals.py`, `benchmark_eval.py` | `test_raw_gaussian_band_and_calibrated_spread` | reruns |
| R7a | Ensemble std ddof=0 vs manuscript N−1 | *src* | confirmed | 4 | 1 | — | implemented: ddof=1 in `deep_ensemble.py`, MC-dropout paths, `benchmark_eval` | tests | `test_ensemble_std_uses_n_minus_1` | reruns |
| R7b | Gaussian per-DoF evaluator averages per batch | *src* | confirmed | 4 | 1 | — | implemented (legacy script now full-array); new evaluator count-weighted | `eval_gaussian_diagnostics_per_dof.py` | `test_metrics_are_batch_invariant` | — |
| R7c | RMSE definitions differ across entry points (pooled vs mean-of-channel in `eval_ensemble_metrics.py`) | *src* | confirmed | 4 | 1 | — | implemented: three named RMSEs, never interchanged | `uq_metrics.py` | — | — |
| R8a | `train_baseline.py` saves next to the config; `train_all.sh` would overwrite one checkpoint | *src* | confirmed | 4 | 2 | — | implemented: `src/training/train.py` writes to `save_dir/run_name`, refuses overwrite; `train_baseline.py` delegates | `train.py`, `train_baseline.py`, `train_ensemble.py` | smoke test (pending) | update SLURM scripts |
| R8b | No seeds | *src*, *log* | confirmed | 4 | 2 | — | implemented: required `seed`, recorded; ensemble seeds base+i | same | — | — |
| R8c | Config keys silently ignored (TCN hidden_dim/num_layers; MLP num_layers; `base_path`) | *src* | confirmed | 4 | 2 | — | implemented: strict `src/models/factory.py`, `base_path` honoured, unknown data/training keys raise | `factory.py`, `mlp.py` | `test_unused_config_key_fails` | — |
| R8d | README references absent module `eval_gaussian_per_dof` | *src* | confirmed | 2 | 1 | — | fixed (name only; command not run with HPC ckpt) | `README.md` | — | — |
| R8e | Checkpoints lack provenance | *src* | confirmed | 4 | 2 | — | implemented: config, seed, split files+hash, scalers, git commit/dirty, versions, best epoch | `train.py` | — | — |
| R9a | MC dropout "disabled" accusation | *src*: `model.train()` keeps dropout on | contradicted (dropout is active); but note: `nn.LSTM` dropout acts **only between stacked layers** (encoder/decoder layer-1 outputs), not on inputs or recurrent connections | 3 | 1 | — | documented | — | `test_mc_dropout_varies_and_eval_is_deterministic` | describe placement in paper |
| R9b | Decoder is zero-input, non-autoregressive | *src* | confirmed; the hidden-state script contradicts it (feedback) | 4 | 1 | — | documented | — | `test_lstm_decoder_input_is_zero_not_feedback` | rewrite §2.2/§3.1 text |
| R9c | Gaussian training may have been stopped by the epoch budget: best val NLL −3.2197 occurred at the final epoch (20/20). This is consistent with, but not proof of, non-convergence. | *log* 918306 | confirmed (as a budget-limited run, not as proven non-convergence) | 4 | 2 | R8 | implemented: epochs 100 + patience 10 for all trainable models; `stopped_by` recorded; NLL includes 0.5·log 2π in training and evaluation | `train.py` | — | reruns |
| R9d | Historical per-DoF temperature grid tops out at 1.0; `u` and `φ` were assigned T=1.0 (grid edge) | *src*, *log* 918386 | confirmed | 3 | 1 | — | superseded (spread-normalised calibration has no grid) | — | — | — |
| R9e | Linear baseline poorly optimised? | RESULTS: Linear 1.79 RMSE vs naive 0.17 | plausible, unverified | 4 | 2 | R2 | implemented: closed-form ridge with validation-selected penalty | `src/training/fit_ridge.py` | smoke test (pending) | run on full data |
| R9f | Manuscript GRU "330K" parameters | *test*: implementation has 305,157 (LSTM 407K, MLP 197K, TCN 353K match) | contradicted | 2 | 1 | — | open (manuscript) | — | `test_historical_parameter_counts` | — |
| R10 | Rerun core benchmark | — | — | 5 | 4 | R1–R9 | blocked: timing pilot awaits authorisation; full run list only after the pilot passes | — | — | see EXPERIMENT_PROTOCOL.md |
| R11 | Channel/horizon/nominal-level/interval-score/recording-bootstrap analyses | — | — | 4 | 3 | R10 | implemented (metrics + per-window files + `src/analysis/bootstrap.py` entry point) – awaiting experiments | `uq_metrics.py`, `src/analysis/bootstrap.py` | — | analysis scripts for tables/figures |
| R12 | Independent shift confirmation / MLP Gaussian & MLP dropout | — | — | 4–5 | 4 | R10 | MLP dropout and MLP Gaussian implemented (stages 1c/1e); no untouched OOD set known, so OOD stays exploratory | configs, `mlp_gaussian.py` | tests + smoke | runs |
| R13 | Align figures/claims | see claim list below | — | 4 | 2–3 | R10 | open | — | — | after R10 |
| R14 | Packaging | — | — | 4 | 2–3 | all | open | — | — | — |
| R15 | Transformer/adaptive/full-scale | — | — | 2–4 | 4–5 | core | deferred (brief §2) | — | — | — |

### Manuscript claims already known to be wrong or unsupported (no replacement values yet)
| v2.2 anchor (orig. lines) | Claim | Status |
|---|---|---|
| `sec:splits` 113 | ≈70/15/15, differs from documented partition, "exact recording IDs … listed" | contradicted (deposit split 57/9/30; no list published until now) |
| `sec:dataset` 86 | 12 inputs listed only generically | `time` not disclosed as an input; OOD also differs in propeller speed (control-input extrapolation) |
| `sec:task` 99 | "decoding predictions one step at a time" | modified: zero-input decoder, all steps jointly, no feedback |
| `fig:pred-setup` 101–105 | surge `u`, representative, 90% interval shown | contradicted by generator (best-tracked `r`, no interval) |
| `sec:backbones` 130 | MLP "makes each prediction independently" | contradicted (one joint output vector) |
| `sec:backbones` 131 | linear "from past states" | modified (all 12 inputs incl. `time`) |
| `tab:baseline-rmse` | GRU 330K params | contradicted (305K) |
| `sec:training` 220 | dropout in [0.0, 0.1]; "early stopping" | modified: executed baselines 0.0, MC-dropout model 0.2; fixed 20-epoch budget with best-val checkpoint, no early stopping |
| `fig:hidden-state` 257–261 | decoder hidden states of the evaluated LSTM | contradicted: generator feeds predictions back, uses a different (local) checkpoint and a subset of windows |
| `sec:gaussian-method` 192 | temperature minimises coverage deviation; global and per-channel both reported | modified: the table path uses per-channel coverage-gap grid search; the plotting path (`eval_lstm_gaussian.py`) fits a global T by NLL on grid 0.5–5; only per-channel values are reported |
| `tab:gaussian-perdof` caption | per-channel diagnostics | rows are aggregates |
| `tab:conformal` | one run, one interval definition | contradicted (R4) |
| `sec:results-ood` 353, fig. caption 357 | ensemble mean anchors near the "training mean (≈8.6 m/s)" | contradicted: training mean `u` = 5.64 m/s (8.50 is the OOD mean) |
| `sec:limitations` 466, 474 | saved per-member/per-pass predictions allow reprocessing | contradicted (none saved) |
| Data availability 489 | split IDs and seeds released | contradicted (no seeds exist; split list created only now) |

---

## Author questions that block valid corrections
1. ~~**HPC artifacts:**~~ answered: they exist on the cluster (see the separate track). Original question: do `~/phd-p1-darus-uncertainty/experiments/results/` and `logs/slurm/` still exist on the TalTech cluster (April 2026 checkpoints, logs 918426/918432/918441/918445/918450/918411)? They are needed only for the old-versus-new comparison (evaluate the historical checkpoints with corrected windows/intervals via `benchmark_eval --legacy_config`). They are not needed for the new benchmark.
2. **Dataset provenance:** the DaRUS DOI(s)/versions of the routine and OOD deposits, and any documented simulation settings (Hs, Tp, wave direction, speed/propeller set-points). The processed files contain no sea-state parameters; without them the shift must be called a *combined operating-condition shift*.
3. **Compute authorisation and protocol sign-off:** see EXPERIMENT_PROTOCOL.md (≈45–100 L40 GPU-hours). Also: add an MLP Gaussian head to complete the pairing (R12), or narrow the backbone claim to ensembles/conformal/dropout?
4. **Confirmation data:** all 29 OOD recordings already informed backbone selection. Is any other DaRUS patrol-vessel condition available as an untouched confirmation set? If not, the OOD results are reported as exploratory (current default).
5. ~~C1~~ decided: random draw is primary (2026-09-28).
6. ~~Pilot authorisation~~ granted: ≤ 5.5 L40 GPU-h combined, no retries. Full benchmark needs separate authorisation.


## Manuscript integration (v3 manuscript, 2026-09-28)

Manuscript-integration commit: `eaa44a6` (local; not pushed). Results archive: `630baad`. Benchmark code: Stage A `dde1848`, Stages B–D `83585da`.

Status levels: **code** = analysis/generation code exists and is tested; **experiments** = corrected benchmark outputs exist (Stages A–D); **manuscript** = integrated into `paper/revision_v3/main_v3.tex` and checked against the source outputs.

| Item | Code | Experiments | Manuscript | Notes |
|---|---|---|---|---|
| Methods match executed benchmark (roles, windows, scaling, `no_time`, training, repeats, constructions, floor, bootstrap) | yes | yes | yes | Stage A at `dde1848`, B–D at `83585da` stated explicitly |
| Tables (9, all generated) and Figures 1–2 | yes (`src/analysis/report_pub.py`) | yes | yes | provenance: `paper/revision_v3/generated/manifest.json` + `SHA256SUMS` |
| Figure 3 (example window) | yes (regeneration code tested on smoke data) | yes | **frozen copy** | relabelled regeneration needs the cluster predictions (VPN); rule and window unchanged |
| Post hoc analyses | yes (`scripts/posthoc_analyses.py`; reproduces the CSVs exactly) | yes | yes, labelled post hoc | no recording-level CIs for the matched contrasts |
| Claim-to-evidence record | — | — | yes | `experiments/analysis/v3/claim_evidence.md` (items 1–23) |
| Bibliography | — | — | yes | cited entries only, complete authors, notes moved to `paper/revision_v3/SOURCE_VERIFICATION.md` |
| Citation rendering / page total | — | — | fixed in source | natbib (authoryear) per the Elsevier template; `\citep`/`\citet`; page total via LastPage; **compiled visual check pending** |
| v2.2 error catalogue | — | — | moved out of the paper | this tracker + `RESULTS_CHANGELOG.md`; the paper keeps prior OOD exposure and scientific limitations |

### Open questions for the authors
1. **Propeller shaft speed unit.** The README states 1/s; the magnitudes suggest rpm; no primary source was found (details: claim_evidence #23). No result depends on it.
2. **Public availability.** Commits after `630baad` are local only; checkpoints and predictions (~16 GB) are on the cluster only. The availability statement says so. Decide whether to push/merge and whether to archive (e.g. with a DOI) before submission.
3. **Figure 3 regeneration.** Needs TalTech VPN access to copy four evaluation directories (predictions only), then `python -m src.analysis.report_pub` and `scripts/sync_paper_assets.sh`.
4. **O1 (ID over-coverage in surge and sway)** remains unexplained; per-recording breakdown pending cluster access.
5. **Compiled-PDF inspection** of the revised source (Overleaf recompile), in particular citations, page totals, dense tables and figures.
