# Post hoc analyses 2.1–2.3: results (2026-10-30)

Protocol: `PROTOCOL.md` (fixed before computation; two presentation deviations for 2.2 recorded there). Script: `scripts/posthoc_v5.py`.
Inputs: 168 files of `{lstm,mlp}_{single,ens,mcd,gauss}_rep{0,1,2}` copied from the cluster, SHA-256 identical to the cluster copies.
Consistency: the rebuilt intervals reproduce the stored metrics of all 84 checked evaluations (max difference 3.2e-7).
All results are **post hoc and descriptive**; OOD results are exploratory. Means over three repeats; brackets are 95% recording-level bootstrap intervals.

## 2.1 Width-matched comparison (nominal 90%)

Δ = coverage of the spread-normalised construction minus coverage of absolute-residual intervals on the same mean, uniformly widened per channel to the same mean width (`width_matched.csv`; per step and channel gives the same picture).

| Split | Backbone | Construction | Coverage (%) | Width-matched AR (%) | Δ (pp) [95% CI] | IS construction / matched AR |
|---|---|---|---|---|---|---|
| OOD | LSTM | EN-SCP | 76.2 | 81.2 | −5.0 [−5.6, −4.4] | 5.57 / 5.03 |
| OOD | LSTM | DN-SCP | 70.4 | 72.7 | −2.4 [−2.9, −1.9] | 5.69 / 5.50 |
| OOD | LSTM | GN-SCP | 49.9 | 51.7 | −1.8 [−3.1, −0.4] | 7.68 / 7.69 |
| OOD | MLP | EN-SCP | 77.7 | 81.7 | −3.9 [−4.5, −3.4] | 5.22 / 4.85 |
| OOD | MLP | DN-SCP | 74.3 | 75.7 | −1.4 [−2.1, −0.7] | 5.31 / 5.29 |
| OOD | MLP | GN-SCP | 48.2 | 50.2 | −2.0 [−3.4, −0.6] | 7.78 / 7.68 |
| ID | LSTM | EN-SCP | 93.4 | 95.3 | −1.9 [−2.7, −0.9] | 1.63 / 1.64 |
| ID | LSTM | DN-SCP | 93.5 | 93.4 | 0.1 [−0.5, 1.0] | 1.54 / 1.62 |
| ID | MLP | EN-SCP | 93.4 | 95.0 | −1.6 [−2.4, −0.7] | 1.61 / 1.63 |
| ID | MLP | DN-SCP | 93.2 | 93.1 | 0.1 [−0.8, 1.1] | 1.49 / 1.61 |

**Reading (rule fixed in advance):** no construction allocates width better than a uniform per-channel widening of the same mean width. Under the shift, uniform widening covers 1.4–5.0 points more (differences computed before rounding), with equal or lower interval scores. The OOD coverage gain of EN- and DN-SCP over AR on the same mean (+19 to +26 points) therefore comes from how much wider the intervals become overall, not from where the extra width is placed. The width-matched reference uses the mean width observed on the evaluated split, which a deployed forecaster does not know; the ensemble and dropout spreads supplied this overall widening without that knowledge.

## 2.2 Coverage by operating point (nominal 90%, OOD test)

`operating_point.csv`. Out-of-range fraction = share of the 330 input values of the 30-s history outside the fitting-recording range of their channel.

| Construction | 0 | (0, 0.05] | (0.05, 0.2] | > 0.2 |
|---|---|---|---|---|
| Windows (recordings) | 29,535 (29) | 21,715 (29) | 48,053 (29) | 3,415 (29) |
| AR-SCP, LSTM | 59.3 [57.4, 61.2] | 52.2 [50.8, 54.0] | 47.5 [46.0, 48.9] | 36.6 [34.9, 38.7] |
| AR-SCP, MLP | 58.5 | 52.2 | 49.4 | 39.8 |
| EN-SCP, LSTM (width) | 73.8 (1.74) | 75.5 (2.20) | 77.8 (2.82) | 79.1 (3.94) |
| EN-SCP, MLP (width) | 74.8 (1.74) | 77.0 (2.15) | 79.6 (2.55) | 81.2 (3.29) |
| DN-SCP, LSTM | 71.7 | 70.8 | 69.7 | 65.2 |
| DN-SCP, MLP | 70.7 | 72.3 | 77.3 | 76.7 |
| GN-SCP, LSTM | 56.3 | 50.3 | 46.4 | 41.6 |
| GN-SCP, MLP | 56.0 | 49.9 | 43.6 | 35.1 |

- AR- and GN-SCP coverage falls as more inputs leave the training range (AR-SCP LSTM 59% → 37%), and also with mean speed (56.6% → 46.8% from the second to the fifth quintile) and steering activity (63.4% without steering → 48.3% in the top quintile).
- EN-SCP coverage rises with the out-of-range fraction because its width grows (1.7 → 3.9); DN-SCP behaves differently on the two backbones.
- Even when every input lies within the training range, AR-SCP covers only about 59% of OOD windows: the shift is not only an extrapolation of the inputs.
- In distribution, 98% of windows have no out-of-range input; there AR-SCP covers 93.2%.

## 2.3 Calibration-draw sensitivity (nominal 90%)

`calibration_draws.csv`. Coverage (%) with the original nine calibration recordings, the range over leave-one-recording-out subsets, and the 2.5–97.5 percentiles over 1000 bootstrap resamples of the nine recordings.

| Split | Construction | Original | Leave-one-out | Bootstrap 95% | Test-recording bootstrap (Tables 6, 8) |
|---|---|---|---|---|---|
| ID | AR-SCP, LSTM | 92.8 | 91.6–93.2 | 89.3–95.0 | 90.9–94.4 |
| ID | EN-SCP, LSTM | 93.4 | 93.0–93.7 | 91.4–94.6 | 92.7–94.2 |
| ID | DN-SCP, LSTM | 93.5 | 92.7–94.0 | 90.6–95.3 | 92.3–94.7 |
| ID | GN-SCP, LSTM | 92.7 | 92.5–92.8 | 92.2–93.3 | 92.4–93.1 |
| OOD | AR-SCP, LSTM | 51.5 | 49.2–52.4 | 45.6–56.4 | — |
| OOD | EN-SCP, LSTM | 76.2 | 75.6–76.7 | 73.5–77.8 | — |

(MLP values are similar; see the CSV. The AR-SCP width ranges 0.81–1.25 over bootstrap draws.)

- For AR-SCP, the choice of the nine calibration recordings moves ID coverage over a wider range (89.3–95.0%) than the finite set of test recordings does (90.9–94.4%), and the nominal 90% lies inside that range. For EN- and DN-SCP the range stays above 90% (lower ends 91.4% and 90.6%).
- The AR-SCP over-coverage (92.8%) is therefore within what a different draw of nine calibration recordings could produce, which supports the calibration draw as a partial explanation of O1 without establishing it. It does not explain the over-coverage of GN-SCP, whose coverage stays at 92.2–93.3% over all draws, nor that of the uncalibrated Gaussian bands (93.2–93.7%), which use no calibration data; for these, easier ID test windows or an overestimated σ remain the candidate explanations.
- GN-SCP is the least sensitive to the draw (its thresholds are rescalings of an already fitted σ).
