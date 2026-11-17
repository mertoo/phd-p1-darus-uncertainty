# Post hoc: non-learned input-novelty normaliser (internal review comment #17), protocol

Written 2026-11-16, **before any novelty-normalised interval was computed**. Requested by Mert ("novelty normaliseri de yap").
Post hoc and descriptive, like the other analyses marked post hoc in the manuscript.

## Question
EN- and DN-SCP gain 19-29 OOD coverage points over AR-SCP. Does a spread that only measures how unfamiliar the
input window is, with no learned uncertainty, give the same gain? If so, the ensemble and dropout spreads act mainly
as shift detectors.

## Construction (NV-SCP)
Same as the spread-normalised constructions of the benchmark, with the spread replaced by a novelty score of the
input window that is the same for all steps and channels:
- input x = the standardised 30 x 11 input window, flattened (330 values), as seen by every forecaster;
- mean m and covariance C of x over the 170,016 fitting windows; C regularised by 1e-6 * trace(C)/330 on the diagonal;
- d^2(x) = (x - m)' C^-1 (x - m) (squared Mahalanobis distance).
Two spreads, both fixed here:
- (L) leverage: s = sqrt(1 + 1/n + d^2/n), n = 170,016, i.e. the OLS prediction-standard-error factor of a linear map
  of x with intercept (exact for ridge with the selected zero penalty);
- (M) Mahalanobis: s = d.
Intervals: mu +/- q max(s, floor), q per step and channel from |y - mu| / max(s, floor) on the nine calibration
recordings (src.uncertainty.intervals.scaled_spread_intervals, horizon_channel, floor_rel 1e-3, level 0.9).
mu: the stored point forecasts of the AR-SCP forecasters (ridge; LSTM and MLP member 0; GRU, TCN, linear SGD single
runs), so NV-SCP and AR-SCP differ only in the score.

## Reported
ID and OOD coverage, normalised width and interval score at 0.9, means over repeats; for the LSTM and MLP also EN-SCP and
DN-SCP rebuilt from the stored predictions. Recording-level 95% intervals (2000 draws, seed 20261116, repeats averaged
within each draw) for NV - AR and EN - NV(M) OOD coverage. Check: rebuilt AR-SCP and EN-SCP coverage must reproduce
the manuscript values (tolerance 1e-4), and the dataset targets must equal the stored targets.

## Reading rule (fixed in advance)
If NV-SCP (M or L) reaches OOD coverage within the recording-level interval of EN-SCP at no larger width, the gain of
EN-SCP is attributed to shift detection, and the text says so. Otherwise the text reports how much of the EN-SCP gain a
novelty normaliser recovers. No variant is chosen after seeing the results; both are reported.
