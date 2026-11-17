"""Floor sensitivity for spread-normalised intervals (CPU only; protocol v3.0 §5).

    python -m src.analysis.floor_sensitivity --evals experiments/eval/v3/*_{ens,mcd,gauss}_rep* \
        --out experiments/analysis/v3/floor_sensitivity.csv

Frozen rules (analysis_rules.yaml):
  * floor_rel in {1e-4, 1e-3 (primary), 1e-2}; variants channel and horizon_channel;
    all predeclared levels; splits = those the evaluation scored.
  * Computed for EVERY spread-method evaluation. A row is marked `triggered` when
    the primary floor clamps > 1% of calibration spreads in any channel; other rows
    are supplementary. The primary result is never replaced by another floor.
  * Before use, the primary (1e-3) rebuild must match the stored metrics within
    tolerance (coverage 1e-4, normalised width 1e-4 relative to its value);
    otherwise the script fails.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

from src.analysis.rebuild import SPREAD_METHODS, EvalDir
from src.evaluation.uq_metrics import interval_metrics

FLOORS = (1e-4, 1e-3, 1e-2)
PRIMARY = 1e-3
VARIANTS = ("channel", "horizon_channel")
TOL_COV = 1e-4
TOL_WIDTH_REL = 1e-4
TRIGGER = 0.01


def analyse(eval_dir):
    ed = EvalDir(eval_dir)
    if ed.method not in SPREAD_METHODS:
        return []
    levels = sorted({iv["level"] for iv in ed.metrics["intervals"]})
    rows = []
    for split in ed.eval_splits:
        y = ed.arrays(split)["y"]
        for variant in VARIANTS:
            for level in levels:
                base = None
                for fr in FLOORS:
                    iv = ed.interval(split, "spread", variant, level, floor_rel=fr)
                    m = interval_metrics(y, iv, scale=ed.scale)
                    if fr == PRIMARY:
                        st = ed.stored(split, ed.method, variant, level)
                        dc = abs(m["coverage"] - st["coverage"])
                        dw = abs(m["width_norm_mean"] - st["width_norm_mean"]) / max(abs(st["width_norm_mean"]), 1e-12)
                        if dc > TOL_COV or dw > TOL_WIDTH_REL:
                            raise RuntimeError(f"{eval_dir} {split}/{variant}/{level}: rebuild differs from stored "
                                               f"metrics (coverage {dc:.2e}, width rel {dw:.2e})")
                        base = m
                    fl_cal = np.asarray(iv.info["frac_floored_cal"])
                    rows.append({
                        "eval": os.path.basename(eval_dir), "method": ed.method, "split": split,
                        "variant": variant, "level": level, "floor_rel": fr, "primary": fr == PRIMARY,
                        "coverage": m["coverage"], "width_norm_mean": m["width_norm_mean"],
                        "interval_score_norm_mean": m["interval_score_norm_mean"],
                        "max_frac_floored_cal": float(fl_cal.max()),
                        "max_frac_floored_eval": float(np.asarray(iv.info["frac_floored_eval"]).max()),
                        **{f"frac_floored_cal_{i}": float(v) for i, v in enumerate(fl_cal)},
                    })
                for r in rows[-len(FLOORS):]:
                    r["d_coverage_vs_primary"] = r["coverage"] - base["coverage"]
                    r["d_width_norm_vs_primary"] = r["width_norm_mean"] - base["width_norm_mean"]
                    r["d_iscore_norm_vs_primary"] = r["interval_score_norm_mean"] - base["interval_score_norm_mean"]
    trig = any(r["primary"] and r["max_frac_floored_cal"] > TRIGGER for r in rows)
    for r in rows:
        r["triggered"] = trig
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dirs = sorted({d for pat in args.evals for d in glob.glob(pat)})
    if not dirs:
        raise FileNotFoundError(args.evals)
    rows = [r for d in dirs for r in analyse(d)]
    if not rows:
        raise ValueError("no spread-method evaluations among the given directories")
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    trig = df[df.triggered]["eval"].unique().tolist()
    print(f"{len(dirs)} evaluations, {len(df)} rows -> {args.out}; triggered: {trig or 'none'}")


if __name__ == "__main__":
    main()
