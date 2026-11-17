"""Post hoc: split-conformal calibration on the clean validation recordings instead of the noisy calibration set.

Background (verified 2026-11-16 with scripts/check_measurement_noise.py): all 57 training recordings, including the
9 calibration recordings, carry white noise of about 0.10 m/s on u and v, whereas the 9 validation and 30 ID-test
recordings are clean, and 17 of the 29 OOD recordings carry about 0.31 m/s. This script recalibrates every stored
construction on the 9 validation recordings and compares with the primary calibration.

The validation recordings were used for early stopping (checkpoint selection), not for fitting the weights. Their
residuals are therefore slightly optimistic, which narrows the recalibrated intervals and biases coverage downwards.

For each row of scripts/slurm/v3/eval_list.tsv the script
  1. loads the checkpoint(s) and predicts the validation windows (same inference code, 200 MC passes, seed 0),
  2. loads the stored calibration, ID-test and OOD predictions of experiments/eval/v3/<tag>,
  3. builds the 90% per-step-and-channel intervals of the manuscript with each calibration set,
  4. reports coverage (overall, per channel, and for the OOD noise groups), normalised width and interval score.
Step 3 with the stored calibration predictions reproduces the manuscript values (a check of the pipeline).

    PYTHONPATH=$REPO python scripts/posthoc_clean_calibration.py --tasks 1-59 --out posthoc_clean_cal
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from src.data_loading.darus_dataset import Standardizer, build_datasets, make_loader
from src.evaluation.benchmark_eval import check_scalers, instantiate, jsonable, load_run, predict, to_physical
from src.evaluation.uq_metrics import interval_metrics
from src.uncertainty.intervals import conformal_intervals, scaled_spread_intervals

LEVEL, VARIANT = 0.9, "horizon_channel"


def rows(tsv, tasks):
    df = pd.read_csv(tsv, sep="\t", dtype=str, keep_default_na=False)
    lo, hi = (int(t) for t in tasks.split("-"))
    return df[df.task.astype(int).between(lo, hi)]


def load_npz(path):
    z = np.load(path)
    return {k: z[k].astype(np.float64) for k in z.files}


def intervals(method, cal, p):
    if method == "point":
        return conformal_intervals(cal["mean"], cal["y"], p["mean"], LEVEL, VARIANT)
    return scaled_spread_intervals(cal["mean"], cal["spread"], cal["y"], p["mean"], p["spread"], LEVEL, VARIANT,
                                   method, 1e-3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.expanduser("~/phd-p1-darus-uncertainty"))
    ap.add_argument("--tasks", default="1-59")
    ap.add_argument("--noise", default="noise_check.csv")
    ap.add_argument("--out", default="posthoc_clean_cal")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nz = pd.read_csv(a.noise)
    noisy = set(nz[(nz.role == "OOD test") & (nz.u_sigma > 0.05)].file)
    os.chdir(a.repo)
    out_dir = a.out if os.path.isabs(a.out) else os.path.join(os.path.dirname(os.path.abspath(a.noise)), a.out)
    os.makedirs(out_dir, exist_ok=True)

    for _, r in rows("scripts/slurm/v3/eval_list.tsv", a.tasks).iterrows():
        t0 = time.time()
        run_dirs = sorted(os.path.dirname(f) for f in glob.glob(f"experiments/runs/v3/{r.runs}/best_model.pt"))
        if len(run_dirs) != int(r.n_members or 1):
            print(f"SKIP {r.tag}: {len(run_dirs)} checkpoints", flush=True)
            continue
        loaded = [load_run(rd, None, device) for rd in run_dirs]
        data_cfg = loaded[0][2]
        datasets, state = build_datasets(data_cfg)
        for _, _, _, st, _ in loaded:
            check_scalers(state, st)
        y_scaler = Standardizer.from_dict(state["y_scaler"])
        H, T = state["history"], state["horizon"]
        models = [instantiate(ck, cfg, datasets["train"], H, T, device) for ck, cfg, _, _, _ in loaded]
        mode = {"point": "point", "ensemble": "point", "mc_dropout": "mc_dropout", "gaussian": "gaussian"}[r.method]

        loader = make_loader(datasets["val"], 512)
        y_val = datasets["val"].targets_physical()
        if r.method == "ensemble":
            stack = np.stack([to_physical(*predict(m, loader, device), y_scaler)[0] for m in models])
            val = {"y": y_val, "mean": stack.mean(0), "spread": stack.std(0, ddof=1)}
        else:
            mean, spread = to_physical(*predict(models[0], loader, device, mode, 200, 0), y_scaler)
            val = {"y": y_val, "mean": mean, "spread": spread}
        ev = f"experiments/eval/v3/{r.tag}"
        np.savez_compressed(os.path.join(out_dir, f"predictions_val_{r.tag}.npz"),
                            **{k: v.astype(np.float32) for k, v in val.items() if v is not None})
        stored = {sp: load_npz(f"{ev}/predictions_{sp}.npz") for sp in ("cal", "test", "ood_test")}
        rec_ood = pd.read_csv(f"{ev}/windows_ood_test.csv").recording.to_numpy()
        is_noisy = np.isin(rec_ood, list(noisy))

        res = {"tag": r.tag, "method": r.method, "n_val_windows": len(y_val), "results": []}
        for cal_name, cal in (("cal", stored["cal"]), ("val", val)):
            for sp in ("test", "ood_test"):
                p = stored[sp]
                iv = intervals(r.method, cal, p)
                m = interval_metrics(p["y"], iv, scale=y_scaler.std)
                row = {"calibration": cal_name, "split": sp, "coverage": m["coverage"],
                       "coverage_channel": m["coverage_channel"], "width_norm_mean": m["width_norm_mean"],
                       "interval_score_norm_mean": m["interval_score_norm_mean"]}
                if sp == "ood_test":
                    inside = (p["y"] >= iv.lower) & (p["y"] <= iv.upper)
                    row["coverage_noisy"] = float(inside[is_noisy].mean())
                    row["coverage_clean"] = float(inside[~is_noisy].mean())
                res["results"].append(row)
        with open(os.path.join(out_dir, f"{r.tag}.json"), "w") as f:
            json.dump(jsonable(res), f, indent=1)
        cov = {(x["calibration"], x["split"]): x["coverage"] for x in res["results"]}
        print(f"{r.tag:22s} ID {cov['cal', 'test']:.3f}->{cov['val', 'test']:.3f}  "
              f"OOD {cov['cal', 'ood_test']:.3f}->{cov['val', 'ood_test']:.3f}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
