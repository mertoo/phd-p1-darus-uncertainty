"""Post hoc descriptive analyses (NOT predeclared) from stored per-evaluation metrics.

    python -m scripts.posthoc_analyses --metrics experiments/manifests/stageC_2026-09-28/metrics \
        --out experiments/analysis/v3/posthoc

Writes:
  o1_coverage_breakdown.csv       coverage by level/channel/horizon, width, interval score and
                                  whole-trajectory coverage per construction (means and seed ranges)
  matched_score_vs_predictor.csv  same predictor with two conformal scores, and the same score with
                                  two predictors, at nominal 90%
Only stored metrics are read: no intervals are recomputed and no recording-level intervals are produced.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

CH = ["u", "v", "p", "r", "phi"]
O1 = {"Abs-residual conformal, LSTM": ("lstm_single_rep{}", "conformal", "horizon_channel"),
      "Abs-residual conformal, MLP": ("mlp_single_rep{}", "conformal", "horizon_channel"),
      "Ensemble-normalised conformal, LSTM": ("lstm_ens_rep{}", "ensemble", "horizon_channel"),
      "Ensemble-normalised conformal, MLP": ("mlp_ens_rep{}", "ensemble", "horizon_channel"),
      "Dropout-normalised conformal, LSTM": ("lstm_mcd_rep{}", "mc_dropout", "horizon_channel"),
      "Dropout-normalised conformal, MLP": ("mlp_mcd_rep{}", "mc_dropout", "horizon_channel"),
      "Sigma-normalised conformal, LSTM": ("lstm_gauss_rep{}", "gaussian", "horizon_channel"),
      "Sigma-normalised conformal, MLP": ("mlp_gauss_rep{}", "gaussian", "horizon_channel"),
      "Gaussian raw (no calibration), LSTM": ("lstm_gauss_rep{}", "gaussian", "raw_gaussian"),
      "Gaussian raw (no calibration), MLP": ("mlp_gauss_rep{}", "gaussian", "raw_gaussian")}


def get(mdir, tag, meth, var, sp, lvl):
    m = json.load(open(os.path.join(mdir, f"{tag}.json")))
    hits = [i for i in m["intervals"] if i["split"] == sp and i["method"] == meth and i["variant"] == var
            and abs(i["level"] - lvl) < 1e-9]
    if len(hits) != 1:
        raise KeyError((tag, meth, var, sp, lvl))
    return hits[0]


def o1(mdir):
    rows = []
    for name, (pat, meth, var) in O1.items():
        for lvl in (0.5, 0.8, 0.9, 0.95):
            for sp in ("test", "ood_test"):
                v = [get(mdir, pat.format(r), meth, var, sp, lvl) for r in range(3)]
                cov = np.array([x["coverage"] for x in v])
                ch = np.array([x["coverage_channel"] for x in v]).mean(0)
                hz = np.array([x["coverage_horizon"] for x in v]).mean(0)
                tr = np.array([x["traj_coverage_channel"] for x in v]).mean(0)
                rows.append({"construction": name, "split": sp, "level": lvl, "coverage_mean": cov.mean(),
                             "coverage_seed_min": cov.min(), "coverage_seed_max": cov.max(),
                             **{f"cov_{c}": ch[i] for i, c in enumerate(CH)},
                             "cov_horizon_min": hz.min(), "cov_horizon_max": hz.max(), "cov_step1": hz[0], "cov_step30": hz[-1],
                             "width_norm": np.mean([x["width_norm_mean"] for x in v]),
                             "iscore_norm": np.mean([x["interval_score_norm_mean"] for x in v]),
                             "traj_all": np.mean([x["traj_coverage_all"] for x in v]),
                             **{f"traj_{c}": tr[i] for i, c in enumerate(CH)}})
    return pd.DataFrame(rows)


def matched(mdir):
    rows = []

    def add(bb, fam, label, pat, meth, var, sp):
        v = [get(mdir, pat.format(r), meth, var, sp, 0.9) for r in range(3)]
        c = np.array([x["coverage"] for x in v])
        w = np.array([x["width_norm_mean"] for x in v])
        s = np.array([x["interval_score_norm_mean"] for x in v])
        rows.append(dict(backbone=bb.upper(), family=fam, construction=label, split=sp, coverage=c.mean(),
                         cov_seed_range=f"{c.min():.3f}-{c.max():.3f}", width_norm=w.mean(), iscore_norm=s.mean(),
                         iscore_seed_range=f"{s.min():.2f}-{s.max():.2f}"))

    for bb in ("lstm", "mlp"):
        for fam, meth in (("ens", "ensemble"), ("mcd", "mc_dropout"), ("gauss", "gaussian")):
            for sp in ("test", "ood_test"):
                pat = f"{bb}_{fam}_rep{{}}"
                add(bb, fam, f"{fam}: spread-normalised conformal on {fam} mean", pat, meth, "horizon_channel", sp)
                add(bb, fam, f"{fam}: abs-residual conformal on {fam} mean", pat, f"{meth}+conformal", "horizon_channel", sp)
                add(bb, fam, f"{fam}: raw mean±z·s (no calibration)", pat, meth, "raw_gaussian", sp)
        for sp in ("test", "ood_test"):
            add(bb, "single", "single: abs-residual conformal on single member (model_0)", f"{bb}_single_rep{{}}",
                "conformal", "horizon_channel", sp)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="experiments/manifests/stageC_2026-09-28/metrics")
    ap.add_argument("--out", default="experiments/analysis/v3/posthoc")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    o1(args.metrics).to_csv(os.path.join(args.out, "o1_coverage_breakdown.csv"), index=False)
    matched(args.metrics).to_csv(os.path.join(args.out, "matched_score_vs_predictor.csv"), index=False)
    print(f"wrote post hoc CSVs to {args.out}")


if __name__ == "__main__":
    main()
