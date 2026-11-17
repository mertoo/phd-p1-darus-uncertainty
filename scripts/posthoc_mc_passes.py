"""Post hoc: is the MC-dropout spread stable at 200 passes?

For the LSTM and MLP dropout models (3 repeats each) and every 10th window of the calibration,
ID-test and OOD-test splits, a fresh set of 400 passes with seed 1 is drawn. The spread from the
first N in {25, 50, 100, 200, 400} passes is compared with the stored spread (200 passes, seed 0),
and DN-SCP (horizon_channel, 90%) is recalibrated on the same calibration subset with each spread.

    PYTHONPATH=$REPO python scripts/posthoc_mc_passes.py --out mc_passes
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.data_loading.darus_dataset import Standardizer, build_datasets
from src.evaluation.benchmark_eval import instantiate, load_run
from src.evaluation.uq_metrics import interval_metrics
from src.uncertainty.intervals import scaled_spread_intervals

STRIDE, NS, NMAX, SEED = 10, (25, 50, 100, 200, 400), 400, 1
LEVEL, VARIANT, FLOOR = 0.9, "horizon_channel", 1e-3


@torch.no_grad()
def passes(model, ds, idx, device, n, seed):
    model.train()
    torch.manual_seed(seed)
    out = []
    for X, _ in DataLoader(Subset(ds, idx), batch_size=512):
        X = X.to(device)
        out.append(torch.stack([model(X) for _ in range(n)]).cpu().numpy())
    return np.concatenate(out, axis=1).astype(np.float64)   # (n, N, T, D), scaled units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mc_passes")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for bb in ("lstm", "mlp"):
        for rep in range(3):
            ckpt, cfg, data_cfg, _, _ = load_run(f"experiments/runs/v3/{bb}_dropout/rep{rep}", device=device)
            ds, state = build_datasets(data_cfg)
            ys = Standardizer.from_dict(state["y_scaler"])
            model = instantiate(ckpt, cfg, ds["train"], state["history"], state["horizon"], device)
            stored, new = {}, {}
            for sp in ("cal", "test", "ood_test"):
                idx = np.arange(0, len(ds[sp]), STRIDE)
                z = np.load(f"experiments/eval/v3/{bb}_mcd_rep{rep}/predictions_{sp}.npz")
                stored[sp] = {k: z[k][idx].astype(np.float64) for k in ("y", "mean", "spread")}
                S = passes(model, ds[sp], idx, device, NMAX, SEED)
                new[sp] = {n: (ys.inverse_transform(S[:n].mean(0)), ys.inverse_scale(S[:n].std(0, ddof=1)))
                           for n in NS}
                # sanity: fresh 200-pass mean must be close to the stored one
                d_mean = np.abs(new[sp][200][0] - stored[sp]["mean"]).mean() / np.abs(stored[sp]["y"]).mean()
                print(f"{bb} rep{rep} {sp}: n={len(idx)}, rel. mean diff vs stored {d_mean:.4f}", flush=True)
            for sp in ("test", "ood_test"):
                def cov(mu_c, s_c, mu, s):
                    iv = scaled_spread_intervals(mu_c, s_c, stored["cal"]["y"], mu, s, LEVEL, VARIANT, "mcd", FLOOR)
                    m = interval_metrics(stored[sp]["y"], iv, scale=ys.std)
                    return 100 * m["coverage"], m["width_norm_mean"]
                c0, w0 = cov(stored["cal"]["mean"], stored["cal"]["spread"], stored[sp]["mean"], stored[sp]["spread"])
                rows.append({"backbone": bb.upper(), "rep": rep, "split": sp, "passes": "200 (stored, seed 0)",
                             "coverage": c0, "width_norm": w0, "median_rel_spread_diff": 0.0})
                for n in NS:
                    mu_c, s_c = new["cal"][n]
                    mu, s = new[sp][n]
                    c, w = cov(mu_c, s_c, mu, s)
                    rel = np.median(np.abs(s - stored[sp]["spread"]) / np.maximum(stored[sp]["spread"], 1e-12))
                    rows.append({"backbone": bb.upper(), "rep": rep, "split": sp, "passes": f"{n} (seed 1)",
                                 "coverage": c, "width_norm": w, "median_rel_spread_diff": rel})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "mc_passes_raw.csv"), index=False)
    summ = df.groupby(["split", "backbone", "passes"], sort=False)[["coverage", "width_norm", "median_rel_spread_diff"]].mean().round(3)
    summ.to_csv(os.path.join(a.out, "mc_passes_summary.csv"))
    print(summ.to_string())
    json.dump({"stride": STRIDE, "passes": NS, "seed_new": SEED, "seed_stored": 0, "level": LEVEL,
               "variant": VARIANT, "floor_rel": FLOOR}, open(os.path.join(a.out, "run_record.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
