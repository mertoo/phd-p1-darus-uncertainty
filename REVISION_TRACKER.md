# Revision tracker — UQ benchmark (manuscript baseline `main v2.2.tex`)

Last updated: 2026-09-28. Branch `revision/v2.3` (local only, not pushed), created from `origin/main` = `9419c66` (the audited snapshot).

Status vocabulary: **confirmed** (verified against source/log/data), **contradicted**, **modified** (true in part), **unverified**. Work status: `open`, `implemented – awaiting experiments`, `verified complete`, `blocked`, `deferred`.
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
| R9c | Gaussian still improving at epoch 20 (best val NLL −3.2197 at epoch 20/20) | *log* 918306 | confirmed | 4 | 2 | R8 | implemented: epochs 100 + patience 10 for all trainable models; NLL now includes 0.5·log 2π in training and evaluation | `train.py` | — | reruns |
| R9d | Historical per-DoF temperature grid tops out at 1.0; `u` and `φ` were assigned T=1.0 (grid edge) | *src*, *log* 918386 | confirmed | 3 | 1 | — | superseded (spread-normalised calibration has no grid) | — | — | — |
| R9e | Linear baseline poorly optimised? | RESULTS: Linear 1.79 RMSE vs naive 0.17 | plausible, unverified | 4 | 2 | R2 | implemented: closed-form ridge with validation-selected penalty | `src/training/fit_ridge.py` | smoke test (pending) | run on full data |
| R9f | Manuscript GRU "330K" parameters | *test*: implementation has 305,157 (LSTM 407K, MLP 197K, TCN 353K match) | contradicted | 2 | 1 | — | open (manuscript) | — | `test_historical_parameter_counts` | — |
| R10 | Rerun core benchmark | — | — | 5 | 4 | R1–R9 | blocked on compute authorisation and protocol sign-off | — | — | see EXPERIMENT_PROTOCOL.md |
| R11 | Channel/horizon/nominal-level/interval-score/recording-bootstrap analyses | — | — | 4 | 3 | R10 | implemented (metrics + per-window files + `recording_bootstrap`) – awaiting experiments | `uq_metrics.py` | — | analysis scripts for tables/figures |
| R12 | Independent shift confirmation / MLP Gaussian & MLP dropout | — | — | 4–5 | 4 | R10 | `mlp_dropout` config added; MLP Gaussian head not implemented (decision Q3) | configs | — | — |
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
1. **HPC artifacts:** do `~/phd-p1-darus-uncertainty/experiments/results/` and `logs/slurm/` still exist on the TalTech cluster (April 2026 checkpoints, logs 918426/918432/918441/918445/918450/918411)? They are needed only for the old-versus-new comparison (evaluate the historical checkpoints with corrected windows/intervals via `benchmark_eval --legacy_config`). They are not needed for the new benchmark.
2. **Dataset provenance:** the DaRUS DOI(s)/versions of the routine and OOD deposits, and any documented simulation settings (Hs, Tp, wave direction, speed/propeller set-points). The processed files contain no sea-state parameters; without them the shift must be called a *combined operating-condition shift*.
3. **Compute authorisation and protocol sign-off:** see EXPERIMENT_PROTOCOL.md (≈45–100 L40 GPU-hours). Also: add an MLP Gaussian head to complete the pairing (R12), or narrow the backbone claim to ensembles/conformal/dropout?
4. **Confirmation data:** all 29 OOD recordings already informed backbone selection. Is any other DaRUS patrol-vessel condition available as an untouched confirmation set? If not, the OOD results must be reported as exploratory.
