"""Old-vs-new comparison for RESULTS_CHANGELOG.md.

    python -m scripts.results_changelog --eval_root experiments/eval/v3 --out RESULTS_CHANGELOG_values.csv

Historical values are copied verbatim from main v2.2.tex (source table given per
row) and are NOT corrected here. New values are read from the v3 evaluation
files only (mean over the available repeats; ranges in brackets). Where the
historical construction differs from any v3 construction, the closest v3
quantity is named explicitly and the row is flagged `not_like_for_like`.
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

# (quantity, model/method, split, historical value, v2.2 source, new-value spec, like_for_like note)
HIST = [
    # Table tab:baseline-rmse: RMSE averaged over horizon and all five channels = pooled RMSE, physical units
    *[("pooled RMSE", m, sp, v, "tab:baseline-rmse", ("point", tag), "same definition (pooled over channels), new windows/scaling/splits")
      for m, tag, vals in [("LSTM seq2seq", "lstm_single_rep*", (0.120, 0.537)), ("MLP", "mlp_single_rep*", (0.121, 0.330)),
                           ("GRU seq2seq", "gru_rep*", (0.132, 0.506)), ("TCN", "tcn_rep*", (0.155, 0.347)),
                           ("Naive", "naive", (0.173, 0.411)), ("Linear", "linear_rep*", (1.791, 2.297))]
      for sp, v in zip(("test", "ood_test"), vals)],
    ("pooled RMSE", "Linear (ridge, new)", "test", None, "not in v2.2", ("point", "ridge"), "new sanity baseline"),
    ("pooled RMSE", "Linear (ridge, new)", "ood_test", None, "not in v2.2", ("point", "ridge"), "new sanity baseline"),
    # ensembles / dropout: +/-2 sigma band with ddof=0 (95.4% Gaussian reference) vs v3 raw band at 95% with ddof=1
    *[("coverage", lab, sp, v, "tab:ensemble-metrics / tab:uq-summary",
       ("interval", tag, meth, "raw_gaussian", 0.95), "NOT like-for-like: historical +/-2 sigma (ddof=0, ~95.4%) vs v3 raw 95% band (ddof=1)")
      for lab, tag, meth, vals in [("Deep ensemble LSTM (raw)", "lstm_ens_rep*", "ensemble", (0.758, 0.490)),
                                   ("Deep ensemble MLP (raw)", "mlp_ens_rep*", "ensemble", (0.813, 0.574)),
                                   ("MC dropout LSTM (raw)", "lstm_mcd_rep*", "mc_dropout", (0.357, 0.165))]
      for sp, v in zip(("test", "ood_test"), vals)],
    *[("pooled RMSE", lab, sp, v, "tab:ensemble-metrics / tab:uq-summary", ("point", tag), "same definition")
      for lab, tag, vals in [("Deep ensemble LSTM mean", "lstm_ens_rep*", (0.111, 0.492)),
                             ("Deep ensemble MLP mean", "mlp_ens_rep*", (0.123, 0.324)),
                             ("MC dropout LSTM mean", "lstm_mcd_rep*", (0.123, 0.520)),
                             ("Gaussian LSTM mean", "lstm_gauss_rep*", (0.133, 0.550))]
      for sp, v in zip(("test", "ood_test"), vals)],
    # Gaussian: per-channel grid temperatures at 90% vs v3 spread-normalised `channel` calibration
    *[("coverage", "Gaussian LSTM (recalibrated)", sp, v, "tab:gaussian-perdof",
       ("interval", "lstm_gauss_rep*", "gaussian", "channel", 0.9), "closest: per-channel calibration; v3 uses a held-out calibration split")
      for sp, v in zip(("test", "ood_test"), (0.899, 0.446))],
    # Conformal: v2.2 'overall' used a pooled threshold (and Tab. 5 mixed runs); v3 per-channel `channel` variant
    *[("coverage", f"Conformal {bb}", sp, v, "tab:conformal-backbone",
       ("interval", tag, "conformal", "channel", 0.9), "NOT like-for-like: v2.2 pooled-threshold 'overall' (validation calibration); v3 per-channel thresholds, held-out calibration")
      for bb, tag, vals in [("LSTM", "lstm_single_rep*", (0.905, 0.662)), ("MLP", "mlp_single_rep*", (0.901, 0.680))]
      for sp, v in zip(("test", "ood_test"), vals)],
]


def new_value(eval_root, spec, split):
    kind, pattern = spec[0], spec[1]
    dirs = sorted(glob.glob(os.path.join(eval_root, pattern)))
    if not dirs:
        raise FileNotFoundError(pattern)
    vals = []
    for d in dirs:
        m = json.load(open(os.path.join(d, "metrics.json")))
        if kind == "point":
            vals.append(m["point"][split]["rmse_pooled"])
        else:
            _, _, meth, variant, level = spec
            iv = [i for i in m["intervals"] if i["split"] == split and i["method"] == meth
                  and i["variant"] == variant and abs(i["level"] - level) < 1e-9]
            if len(iv) != 1:
                raise KeyError((d, meth, variant, level, split))
            vals.append(iv[0]["coverage"])
    return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals)), len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", default="experiments/eval/v3")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    for q, name, sp, hist, src, spec, note in HIST:
        mean, lo, hi, n = new_value(args.eval_root, spec, sp)
        rows.append({"quantity": q, "model/method": name, "split": sp, "v2.2 value": hist, "v2.2 source": src,
                     "v3 value (mean over repeats)": mean, "v3 min": lo, "v3 max": hi, "v3 repeats": n,
                     "comparability": note})
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
