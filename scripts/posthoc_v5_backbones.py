"""Post hoc analysis 2.4 (protocol: experiments/analysis/v3/posthoc_v5/PROTOCOL_2.4.md).

    python -m scripts.posthoc_v5_backbones --out experiments/analysis/v3/posthoc_v5/backbones_2.4

For every backbone with stored evaluations: AR-SCP on member 0, EN-SCP, DN-SCP (where available) and the raw
ensemble/dropout bands; differences EN - AR and DN - AR with recording-level bootstrap intervals; width-matched
comparison as in 2.1. Reads stored predictions only. Descriptive.
"""

import argparse
import json
import os
import subprocess

import numpy as np
import pandas as pd

from scripts.posthoc_v5 import Agg, boot_weights, consistency, cov_stat, norm_stat, rec_index, sha
from src.analysis.rebuild import EvalDir
from src.uncertainty.intervals import conformal_threshold, normal_z, spread_floor

EVAL = "experiments/eval/v3"
LEVELS = (0.5, 0.8, 0.9, 0.95)
PRIMARY = 0.9
B, SEED = 2000, 20261030
NAMES = {"single": "AR-SCP", "ens": "EN-SCP", "mcd": "DN-SCP"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/analysis/v3/posthoc_v5/backbones_2.4")
    ap.add_argument("--backbones", nargs="+", default=["lstm", "mlp", "gru", "tcn", "linear"])
    ap.add_argument("--splits", nargs="+", default=["test", "ood_test"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    store, cons_rows, inputs, info = {}, [], {}, {}
    for bb in args.backbones:
        for fam in ("single", "ens", "mcd"):
            for rep in range(3):
                tag = f"{bb}_{fam}_rep{rep}"
                path = os.path.join(EVAL, tag)
                if not os.path.exists(os.path.join(path, "metrics.json")):
                    if fam == "mcd" and bb == "linear":
                        continue
                    raise FileNotFoundError(path)
                ed = EvalDir(path)
                for f in ["metrics.json"] + [f"predictions_{s}.npz" for s in ["cal"] + args.splits]:
                    inputs[f"{tag}/{f}"] = sha(os.path.join(path, f))
                cons_rows += consistency(ed, fam, args.splits)
                if max(max(r["coverage"], r["width_norm_mean"]) for r in cons_rows) > 1e-4:
                    raise RuntimeError(f"{tag}: rebuilt intervals do not reproduce stored metrics")
                cal = ed.arrays("cal")
                err_cal = np.abs(cal["y"] - cal["mean"])
                if fam != "single":
                    floor = spread_floor(cal["spread"], cal["y"], ed.floor_rel)
                    s_cal = np.maximum(cal["spread"], floor)
                for sp in args.splits:
                    ev = ed.arrays(sp)
                    w = ed.windows(sp)
                    ridx, recs = rec_index(w)
                    agg = Agg(ridx, len(recs))
                    N, T, D = ev["y"].shape
                    info.setdefault(sp, (agg.count, len(recs), T, D, ed.scale))
                    err = np.abs(ev["y"] - ev["mean"])

                    def put(kind, half, lv):
                        cov = err <= half
                        width = np.broadcast_to(2 * half, err.shape)
                        isc = width + 2.0 / (1 - lv) * np.maximum(err - half, 0.0)
                        store.setdefault((sp, bb, NAMES[fam], kind, lv), []).append(
                            (agg.sums(cov), agg.sums(width), agg.sums(isc)))
                        return width

                    for lv in LEVELS:
                        qa = conformal_threshold(err_cal, lv, axis=0)
                        if fam == "single":
                            put("X", np.broadcast_to(qa, err.shape), lv)
                            continue
                        s_ev = np.maximum(ev["spread"], floor)
                        qx = conformal_threshold(err_cal / s_cal, lv, axis=0)
                        width = put("X", qx * s_ev, lv)
                        put("raw", normal_z(lv) * ev["spread"], lv)
                        c_ch = width.mean(axis=(0, 1)) / (2 * qa).mean(axis=0)
                        put("AR_wm_channel", np.broadcast_to(c_ch * qa, err.shape), lv)
                    del ev, err
                print(f"done {tag}", flush=True)
    pd.DataFrame(cons_rows).to_csv(f"{args.out}/consistency.csv", index=False)

    rows = []
    for (sp, bb, name, kind, lv), lst in store.items():
        count, R, T, D, scale = info[sp]
        one = np.ones((1, R))
        rows.append(dict(split=sp, backbone=bb.upper(), construction=name, kind=kind, level=lv, n_repeats=len(lst),
                         coverage=np.mean([cov_stat(one, c, count, T, D)[0] for c, _, _ in lst]),
                         width_norm=np.mean([norm_stat(one, wsum, count, T, scale)[0] for _, wsum, _ in lst]),
                         iscore_norm=np.mean([norm_stat(one, isum, count, T, scale)[0] for _, _, isum in lst])))
    summ = pd.DataFrame(rows)

    # recording-level intervals at the primary level for the differences of the protocol
    diffs = []
    for sp in args.splits:
        count, R, T, D, scale = info[sp]
        W = boot_weights(R, B, SEED)
        cov = lambda key: np.mean([cov_stat(W, c, count, T, D) for c, _, _ in store[key]], axis=0)
        for bb in args.backbones:
            ar = cov((sp, bb, "AR-SCP", "X", PRIMARY))
            for name in ("EN-SCP", "DN-SCP"):
                if (sp, bb, name, "X", PRIMARY) not in store:
                    continue
                x = cov((sp, bb, name, "X", PRIMARY))
                for ref, refv in (("AR-SCP (member 0)", ar), ("AR width-matched", cov((sp, bb, name, "AR_wm_channel", PRIMARY)))):
                    d = 100 * (x - refv)
                    diffs.append(dict(split=sp, backbone=bb.upper(), construction=name, reference=ref,
                                      lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5)))
                raw = 100 * cov((sp, bb, name, "raw", PRIMARY))
                diffs.append(dict(split=sp, backbone=bb.upper(), construction=name, reference="raw band coverage",
                                  lo=np.percentile(raw, 2.5), hi=np.percentile(raw, 97.5)))
    summ.to_csv(f"{args.out}/summary.csv", index=False)
    pd.DataFrame(diffs).to_csv(f"{args.out}/differences_ci.csv", index=False)
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    json.dump({"protocol": "experiments/analysis/v3/posthoc_v5/PROTOCOL_2.4.md",
               "protocol_sha256": sha("experiments/analysis/v3/posthoc_v5/PROTOCOL_2.4.md"),
               "script_sha256": sha(__file__), "git_head": rev, "args": vars(args), "seed": SEED, "inputs": inputs},
              open(f"{args.out}/run_record.json", "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
