# Post hoc analyses for manuscript v5 (2.1–2.3): protocol

Written 2026-10-30, **before any of these quantities was computed**. Approved in principle by Mert on 2026-10-30 ("analizleri yapabilirsin, HPC gerektirmeyenleri başlat").
All three analyses are **post hoc** (not in the frozen protocol v3.0) and are reported as descriptive.
They use only stored predictions; no model is trained, recalibrated for reporting, or tuned, and nothing in the
benchmark (splits, thresholds of the reported results, floors, tables 4–11) changes.

## Common settings

| Item | Value |
|---|---|
| Evaluations | `{lstm,mlp}_{single,ens,mcd,gauss}_rep{0,1,2}` in `experiments/eval/v3` (copies of the cluster outputs of commit `83585da`, checked by SHA-256 against the cluster) |
| Files read | `metrics.json`, `predictions_{cal,test,ood_test}.npz`, `windows_{cal,test,ood_test}.csv`, the DaRUS deposit files, `experiments/manifests/calibration_selection.json`, `split_manifest.csv` |
| Constructions | EN-, DN-, GN-SCP (spread-normalised score on the ensemble, dropout or Gaussian mean); AR on the same mean (absolute-residual score; the post hoc matched predictor of Table 10); AR-SCP on the single model (member 0) |
| Threshold rule | as in the benchmark: `src.uncertainty.intervals` (order statistic k = ⌈(n+1)(1−α)⌉ per step and channel, `horizon_channel`), floor 10⁻³ as in the benchmark |
| Levels | 0.9 primary; 0.5, 0.8, 0.95 secondary |
| Splits | ID test (30 recordings) and OOD test (29; exploratory) |
| Repeats | computed per repeat, then averaged over the three repeats (as in the tables) |
| Recording-level intervals | cluster bootstrap over test recordings, 2000 draws, seed 20261030, repeats averaged within each draw (as in the benchmark); quantities fixed on the full split (e.g. matching factors) are held fixed within draws |
| Consistency check | before any analysis, the rebuilt coverage and width of every construction must reproduce the stored metrics (`EvalDir.consistency`, tolerance 1e-4); otherwise the analysis stops |

## 2.1 Width-matched comparison (supervisor's ML review M1)

**Question:** do spread-normalised intervals cover more than absolute-residual intervals with the same mean width, i.e. does the spread put the extra width on the windows that need it?

For each spread construction X ∈ {EN, DN, GN}, backbone, repeat, split and level:
1. Intervals of X: μ ± q^X_{t,d} max(s, f).
2. AR on the same mean μ: thresholds q^AR_{t,d} from |y − μ| on the calibration recordings.
3. Width-matched AR (primary, per channel): half-width c_d q^AR_{t,d}, with c_d chosen so that the mean width over all windows and steps of channel d on that split equals that of X. Secondary (per step and channel): c_{t,d} matching each (t, d) separately.
4. Report coverage (all values and per channel) and normalised interval score of X and of width-matched AR, and Δ = coverage(X) − coverage(width-matched AR), with recording-level 95% intervals.

**Reading rule (fixed in advance):** Δ > 0 with an interval excluding 0 means the spread allocates width better than a uniform per-channel widening of the same mean width; Δ ≤ 0 or an interval including 0 means no evidence of better allocation. Because c_d uses the widths on the evaluated split, width-matched AR is a diagnostic reference, not a deployable method.

## 2.2 Coverage by operating point

**Question:** where, in terms of quantities known at forecast time, does coverage fail?

Window statistics from the 30-s input history (inputs only, known when the forecast is made):
- (a) mean surge velocity u; (b) mean shaft speed n; (c) standard deviation of the left azimuth angle δ_l (steering activity);
- (d) out-of-range fraction: share of the 30 × 11 input values outside the [min, max] range of the same channel in the 48 fitting recordings.

Bins: (a)–(c) quintiles of the pooled ID and OOD test windows (the same edges for every construction); (d) {0}, (0, 0.05], (0.05, 0.2], > 0.2.
Report, at 0.9 and per bin: coverage and mean normalised width of AR-SCP (single model), EN-, DN- and GN-SCP for LSTM and MLP, with window counts, number of recordings per bin and recording-level 95% intervals. Descriptive; no threshold or monitor is fitted.

## 2.3 Calibration-draw sensitivity

**Question:** how much does coverage depend on which nine recordings were used for calibration?

Using the stored predictions on the nine calibration recordings, thresholds are recomputed from:
- (i) each leave-one-recording-out subset (9 subsets of 8 recordings);
- (ii) 1000 bootstrap resamples of the 9 recordings with replacement (seed 20261031; a recording drawn twice contributes its windows twice).

The floor is held at its full-calibration value (it was essentially inactive, Appendix A.3). For each draw: coverage and mean normalised width on the ID and OOD test sets at 0.9 for AR-SCP (single model), EN-, DN- and GN-SCP, LSTM and MLP, averaged over repeats.
Report the original value, the leave-one-out range and the 2.5–97.5 percentiles over bootstrap draws, next to the recording-level bootstrap intervals of the test sets. Descriptive.

## Outputs
`experiments/analysis/v3/posthoc_v5/`: `consistency.csv`, `width_matched.csv`, `operating_point.csv`, `calibration_draws.csv`, `run_record.json` (input hashes, protocol hash, code version, seeds).

## Deviations (recorded after the first full run, 2026-10-30)

Both concern only the presentation of 2.2; no computed coverage changed.
1. **Steering quintiles.** In about 40% of the pooled windows (77% of the ID test and 1% of the OOD windows) the left azimuth angle is constant over the 30-s history (SD = 0), so the lower quintile edges of the SD of δ_l coincide at 0 and the first quintile is empty. The windows with SD = 0 are reported as one bin ("no steering"); the remaining bins keep the pre-specified edges.
2. **Sparse bins.** In the first run, bootstrap draws in which a bin was empty were counted as zero coverage, which produced spurious lower bounds of 0. Recording-level intervals are now computed over the draws in which the bin is non-empty, and bins with fewer than 5 recordings are flagged as not estimable (no interval).
