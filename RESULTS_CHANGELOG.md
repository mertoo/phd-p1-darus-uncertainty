# Results changelog: main v2.2 → corrected benchmark (protocol v3.0)

Created 2026-09-28 after Stages A–D at commit `83585da`. **Nothing in `main v2.2.tex` has been changed.** Historical values are quoted from v2.2; corrected values come only from generated files:
- `experiments/analysis/v3/report/tables/*.csv` (bootstrap estimates with 95% recording-level CIs)
- `experiments/analysis/v3/results_changelog_values.csv` (`scripts/results_changelog.py`)
- `experiments/manifests/stageC_2026-09-28/metrics/*.json`

OOD results are **exploratory**: the same 29 OOD recordings informed the historical backbone choice, and no untouched OOD set exists. A frozen rerun does not undo that earlier use.

**Revision 2026-09-28 (interpretation audit):** corrected after the claim-to-evidence audit, `experiments/analysis/v3/claim_evidence.md`, which is the authoritative claim table. Construction names used below:
- **AR-SCP**: split conformal, absolute-residual score.
- **EN-SCP / DN-SCP / GN-SCP**: split conformal with a score normalised by the ensemble SD, the MC-dropout SD or the Gaussian σ.
- **Raw**: uncalibrated μ ± z·s.

The "calibrated ensemble / MC dropout / Gaussian" of the first version of this file **are** conformal constructions (EN/DN/GN-SCP), so "ensemble vs conformal" is not a contrast between calibrated and uncalibrated methods.

## Why values changed
Every historical number was produced under a different protocol. **No ablation isolates any single correction, so no change below is attributed to one cause.** Changes that occurred together:
- windows no longer cross recording boundaries, and the window count is correct (R1);
- inputs and targets are standardised on training data only, so the loss weights all five channels equally, and `time` is dropped (R5);
- calibration uses 9 held-out training recordings instead of the validation split (R2);
- early stopping replaces the fixed 20 epochs (R9);
- seeds are explicit, with 3 repeats per configuration (R8);
- ensemble and dropout spread use N−1 (R7);
- all conformal and spread intervals use named variants with the finite-sample order statistic (R3).

## Point accuracy (pooled RMSE over the five channels, physical units: the v2.2 definition)
Mean over 3 repeats [min, max]; single runs where noted.

| Model | v2.2 ID | v3 ID | v2.2 OOD | v3 OOD | Comparability |
|---|---|---|---|---|---|
| LSTM seq2seq | 0.120 | 0.087 [0.080, 0.097] | 0.537 | 0.359 [0.346, 0.380] | same metric, new protocol |
| MLP | 0.121 | 0.079 [0.076, 0.084] | 0.330 | 0.314 [0.311, 0.316] | same |
| GRU seq2seq | 0.132 | 0.070 [0.069, 0.071] | 0.506 | 0.314 [0.309, 0.323] | same |
| TCN | 0.155 | 0.081 [0.078, 0.083] | 0.347 | 0.330 [0.326, 0.337] | same |
| Naive (last value) | 0.173 | 0.149 (1 run) | 0.411 | 0.406 (1 run) | same |
| Linear (SGD) | **1.791** | **0.068** [0.067, 0.070] | **2.297** | **0.304** [0.304, 0.304] | same metric; the v2.2 failure did not reproduce (cause not isolated) |
| Linear (ridge, new) | — | 0.060 (1 run) | — | 0.292 (1 run) | new sanity baseline |

Pooled RMSE is dominated by surge and sway. The primary metric is now the channel-averaged normalised RMSE (tab `baselines`). ID: all learned models 0.314–0.336 with overlapping CIs, naive 0.494 [0.419, 0.568]. OOD: learned models 0.927–1.01, naive 1.24.

## Interval quality at nominal 90% (primary construction: `horizon_channel`; tab `uq_summary`)

| Claim quantity | v2.2 | v3 (95% CI) | Comparability |
|---|---|---|---|
| AR-SCP LSTM coverage, ID / OOD | 90.5% / 66.2% (pooled threshold; Tab. 5 mixed three runs) | 92.8 [90.9, 94.4] / 51.5 [49.8, 53.2] | not like-for-like (construction, calibration split) |
| AR-SCP MLP coverage, ID / OOD | 90.1% / 68.0% | 92.8 [91.0, 94.4] / 52.3 [50.6, 54.1] | not like-for-like |
| Gaussian LSTM recalibrated (v2.2) vs GN-SCP (v3), ID / OOD | 89.9% / 44.6% | 92.7 [92.4, 93.1] / 49.9 [47.9, 51.8] | closest analogue |
| Gaussian LSTM **raw**, ID | (not reported; described as needing recalibration) | 93.7 [93.4, 93.9] | new |
| Ensemble LSTM raw ±zσ, ID / OOD | 75.8% / 49.0% at ±2σ (≈95.4%, ddof 0) | at 90%: 59.6 [57.8, 61.4] / 32.1 [30.7, 33.6]; at 95%: 64.5% / 36.8% (mean of repeats) | not like-for-like |
| Ensemble MLP raw, ID / OOD | 81.3% / 57.4% (±2σ) | at 90%: 61.0 / 33.2; at 95%: 66.0% / 37.8% | not like-for-like |
| MC dropout LSTM raw, ID / OOD | 35.7% / 16.5% (±2σ) | at 90%: 25.7 / 12.2; at 95%: 30.0% / 14.3% | not like-for-like |
| EN-SCP (LSTM / MLP), OOD | not evaluated | 76.2 [75.2, 77.2] / 77.7 [76.8, 78.7] | new (R6) |
| DN-SCP (LSTM / MLP), OOD | not evaluated | 70.4 [69.2, 71.5] / 74.3 [73.1, 75.6] | new |
| MC dropout / Gaussian on MLP | not evaluated | see `uq_summary` | new (R12) |

## Claims in v2.2 and their status
Supported = consistent with corrected evidence; weakened = direction holds but the magnitude or scope is much smaller or needs qualification; contradicted = corrected evidence disagrees; untested = no corrected evidence.

| v2.2 anchor | Claim | Status | Evidence |
|---|---|---|---|
| Abstract, contribution 1, §3.1 | MLP achieves 38% lower OOD RMSE than LSTM (0.330 vs 0.537); near-identical ID accuracy | **weakened** (the 38% and 12.5% come from different pipelines and are not an "improvement") | Pooled OOD RMSE 0.314 vs 0.359 (12.5% lower). In normalised RMSE the paired difference is −0.0072 [−0.0094, −0.0049] on a level of about 0.94 (< 1%). GRU matches the MLP (0.314). ID accuracy is similar (supported). |
| §3.1 | Recurrent architectures amplify shift; LSTM OOD error 4.48× its ID value; naive beats LSTM and GRU OOD | **contradicted** (naive), **weakened** (ratio) | Naive OOD 0.406 is worse than every learned model (0.29–0.36). Pooled OOD/ID ratios: LSTM 4.1, MLP 4.0, GRU 4.5, linear 4.5. The degradation is common to all learned models, not specific to recurrence. |
| §3.1 | Linear model "fails to capture nonlinear dynamics" (RMSE 1.791) | **contradicted** | Linear SGD 0.068 and ridge 0.060 pooled ID RMSE; normalised RMSE 0.336 [0.29, 0.384] and 0.328 [0.281, 0.375] vs networks 0.314–0.328 (no predeclared paired test against ridge). The failure did not reproduce under v3; scaling, windowing, early stopping and the feature set changed together, so the cause is not isolated. |
| §3.1, `fig:hidden-state` | Hidden-state displacement and collapse explain the LSTM's OOD failure | **untested** | The generator fed predictions back into the decoder and used a different checkpoint; it was not rerun under v3. The claim should be removed or re-derived. |
| Abstract, §3.2, conclusions | Conformal achieves "exact in-distribution validity" (90.1–90.5%) | **contradicted as stated** (empirical over-coverage; this does not refute conformal validity, whose exchangeability assumptions are not established for overlapping windows from 9 calibration recordings) | All calibrated methods over-cover in distribution (≈ 92.6–93.5% at nominal 90%; 60.5–65.5% at nominal 50%, calibrated methods, per repeat), driven by surge (94–96%) and sway (98.5–99%). p, r and φ are ≈ 90%. No finite-sample guarantee applies. |
| Abstract, contribution 2, §3.2 | Only the calibrated methods (conformal, recalibrated Gaussian) reach near-nominal ID coverage; raw ensemble and MC dropout are severely under-covered | **partly supported** | Raw ensembles (60–61%) and raw LSTM MC dropout (26%) under-cover (supported). Raw MLP MC dropout reaches 71%. The **raw Gaussian already reaches ≈ 93% ID without recalibration**, so the claim that it needs recalibration is contradicted. |
| §3.3, abstract, discussion | Conformal (MLP) has the best OOD coverage (68.0%) and "degrades the least"; OOD ranking conformal > ensembles > Gaussian > MC dropout | **contradicted** | All four are conformal or not depending on construction. At 90% with common calibration: EN-SCP 76–78% and DN-SCP 70–74% OOD coverage vs AR-SCP 52% and GN-SCP 48–50%, with wider intervals but lower (better) OOD interval scores (5.2–5.7 vs 7.4–7.8). The predeclared pair (EN-SCP on ensemble mean − AR-SCP on one member) +24.7 [23.6, 25.8] pp mixes score and predictor. Matched post hoc (same ensemble mean): score effect +25.6 pp; predictor effect −0.9 pp (seed ranges only). |
| Contribution 3, §3.4, discussion | Backbone OOD generalisation drives OOD uncertainty quality; backbone effect comparable to or larger than the method effect | **contradicted for these two backbones and constructions** | Backbone differences (MLP − LSTM, same construction, OOD coverage): AR +0.8 [0.6, 1.0], EN +1.5 [1.3, 1.8], DN +4.0 [3.6, 4.4], GN −1.7 [−2.1, −1.4] pp. The score choice on a fixed predictor changes OOD coverage by ≈ 25 pp (post hoc). This does not support a general "calibration matters more than backbone" statement: raw→calibrated, score and backbone effects are distinct. |
| §3.5, discussion | OOD coverage loss concentrated in surge and sway; roll rate p best covered | **partly supported** | p is the best-covered OOD channel for every calibrated method (63–89%). Losses are largest in u and yaw rate r (r 34–66%); v is intermediate. |
| §3.2 | MC dropout's narrow spread indicates the network "routes around" dropout | **untested** | Raw MC dropout under-covers for the LSTM (26%) far more than for the MLP (71%). LSTM dropout acts only between stacked layers, but no mechanism was tested. |
| §2.2/§2.6 | 1 Hz sampling, H = T = 30 | **supported** | Data audit. |
| §2.3 | ≈70/15/15 recording split | **contradicted** | Deposit split 57/9/30/29. v3 uses 48 fit / 9 calibration recordings from train. |
| Limitations | Coverage levels not harmonised; single level; no horizon analysis | **resolved in v3** | Levels 0.5/0.8/0.9/0.95; horizon-resolved figure; common calibration. |
| Limitations | Gaussian convergence uncertain | **resolved (as a protocol issue)** | All runs early-stopped (Gaussian best epochs 2–10); none hit the budget. |
| §2.4, `tab:baseline-rmse` | GRU 330K parameters | **contradicted** | 305,157. |

## New observations, not previously claimed (need interpretation before use)
1. **In-distribution over-coverage for every calibrated method.**
   - Coverage is ≈ 3 points above nominal at 90% and larger at lower levels, concentrated in u and v.
   - The calibration recordings were descriptively slower than the fit recordings (SMD −0.52 on mean u).
   - Whether calibration–test differences between recordings cause this has **not** been tested.
2. **Early validation optimum.** Recurrent models and TCN reach their best validation loss at epochs 2–5, after which validation loss rises. The historical fixed 20-epoch runs selected the best-validation checkpoint, so they were probably selecting early epochs too, though this is unverified.
3. **Floor (P10), verified over 18 evaluations × 5 channels × 2 variants × 4 levels × 2 splits.** Maximum floored calibration fraction 6.3e-6 (eval 3.8e-6). floor_rel 1e-4: |Δcoverage| ≤ 7e-7. floor_rel 1e-2: |Δcoverage| ≤ 0.0025, |Δnorm. width| ≤ 0.049, |Δnorm. interval score| ≤ 0.076. **Resolved for this benchmark only.**
4. **Simultaneous vs marginal coverage.** Whole-trajectory coverage (all 30 steps × 5 channels) on ID test is 0.16–0.40 depending on construction; every construction targets marginal coverage only.
