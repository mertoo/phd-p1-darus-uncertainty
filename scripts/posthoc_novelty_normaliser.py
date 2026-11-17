"""Post hoc: split-conformal intervals normalised by a non-learned input-novelty score (NV-SCP).

Protocol: experiments/analysis/v3/posthoc_novelty/PROTOCOL.md (written before any result).
Spreads: (L) OLS leverage factor sqrt(1 + 1/n + d^2/n), (M) Mahalanobis distance d of the standardised, flattened
30 x 11 input window to the fitting windows. Same mu as AR-SCP, same thresholds per step and channel.

    PYTHONPATH=$REPO python scripts/posthoc_novelty_normaliser.py --out novelty
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml

from src.data_loading.darus_dataset import Standardizer, build_datasets
from src.evaluation.uq_metrics import interval_metrics
from src.uncertainty.intervals import conformal_intervals, scaled_spread_intervals

LEVEL, VARIANT, FLOOR = 0.9, "horizon_channel", 1e-3
N_BOOT, SEED = 2000, 20261116
AR = {"Ridge": ["ridge"], "LSTM": [f"lstm_single_rep{i}" for i in range(3)],
      "MLP": [f"mlp_single_rep{i}" for i in range(3)], "GRU": [f"gru_rep{i}" for i in range(3)],
      "TCN": [f"tcn_rep{i}" for i in range(3)], "Linear (SGD)": [f"linear_rep{i}" for i in range(3)]}
SPREAD = {"EN-SCP": "ens", "DN-SCP": "mcd"}


def windows_flat(ds):
    """Standardised input windows (N, H*F) in dataset order."""
    out = np.empty((len(ds), ds.history * ds.input_dim))
    for i, (sid, s) in enumerate(ds.index):
        out[i] = ds._segXs[sid][s: s + ds.history].ravel()
    return out


def load(tag, split):
    z = np.load(f"experiments/eval/v3/{tag}/predictions_{split}.npz")
    return {k: z[k].astype(np.float64) for k in z.files}


def per_recording(inside, rec, recs):
    """Per-recording sums of inside counts and scalar counts."""
    w = inside.reshape(len(inside), -1).sum(1)
    n = np.full(len(inside), inside[0].size)
    df = pd.DataFrame({"r": rec, "w": w, "n": n}).groupby("r")[["w", "n"]].sum().reindex(recs)
    return df.to_numpy(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="novelty")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = yaml.safe_load(open("experiments/runs/v3/gru/rep0/config.yaml"))["data"]
    ds, state = build_datasets(cfg)
    y_scaler = Standardizer.from_dict(state["y_scaler"])

    Xf = windows_flat(ds["train"])
    n, p = Xf.shape
    m = Xf.mean(0)
    C = np.cov(Xf, rowvar=False, bias=True)
    C[np.diag_indices(p)] += 1e-6 * np.trace(C) / p
    Ci = np.linalg.inv(C)
    del Xf
    nov = {}
    for sp in ("cal", "test", "ood_test"):
        Z = windows_flat(ds[sp]) - m
        d2 = ((Z @ Ci) * Z).sum(1)
        nov[sp] = {"L": np.sqrt(1 + 1 / n + d2 / n), "M": np.sqrt(d2)}
        print(f"{sp}: d median {np.median(np.sqrt(d2)):.1f}, 95% {np.percentile(np.sqrt(d2), 95):.1f}, "
              f"leverage factor median {np.median(nov[sp]['L']):.4f}, max {nov[sp]['L'].max():.3f}", flush=True)
    meta = {sp: pd.read_csv(f"experiments/eval/v3/ridge/windows_{sp}.csv") for sp in ("test", "ood_test")}
    recs = {sp: sorted(meta[sp].recording.unique()) for sp in meta}

    rows, inside_store = [], {}
    def record(cons, backbone, rep, sp, y, iv):
        mt = interval_metrics(y, iv, scale=y_scaler.std)
        rows.append({"construction": cons, "backbone": backbone, "rep": rep, "split": sp,
                     "coverage": 100 * mt["coverage"], "width_norm": mt["width_norm_mean"],
                     "iscore_norm": mt["interval_score_norm_mean"],
                     **{f"cov_{c}": 100 * v for c, v in zip("u v p r phi".split(), mt["coverage_channel"])}})
        if sp != "cal":
            inside = (y >= iv.lower) & (y <= iv.upper)
            inside_store[(cons, backbone, rep, sp)] = per_recording(inside, meta[sp].recording.to_numpy(), recs[sp])

    for backbone, tags in AR.items():
        for rep, tag in enumerate(tags):
            P = {sp: load(tag, sp) for sp in ("cal", "test", "ood_test")}
            for sp in ("test", "ood_test"):
                assert np.allclose(P[sp]["y"], ds[sp].targets_physical(), atol=1e-4), (tag, sp)
                record("AR-SCP", backbone, rep, sp, P[sp]["y"],
                       conformal_intervals(P["cal"]["mean"], P["cal"]["y"], P[sp]["mean"], LEVEL, VARIANT))
                for v in ("L", "M"):
                    sc = np.broadcast_to(nov["cal"][v][:, None, None], P["cal"]["y"].shape)
                    se = np.broadcast_to(nov[sp][v][:, None, None], P[sp]["y"].shape)
                    record(f"NV-SCP ({v})", backbone, rep, sp, P[sp]["y"],
                           scaled_spread_intervals(P["cal"]["mean"], sc, P["cal"]["y"], P[sp]["mean"], se,
                                                   LEVEL, VARIANT, "novelty", FLOOR))
            print(f"{tag} done", flush=True)
    for cons, k in SPREAD.items():
        for backbone in ("LSTM", "MLP"):
            for rep in range(3):
                tag = f"{backbone.lower()}_{k}_rep{rep}"
                P = {sp: load(tag, sp) for sp in ("cal", "test", "ood_test")}
                for sp in ("test", "ood_test"):
                    record(cons, backbone, rep, sp, P[sp]["y"],
                           scaled_spread_intervals(P["cal"]["mean"], P["cal"]["spread"], P["cal"]["y"],
                                                   P[sp]["mean"], P[sp]["spread"], LEVEL, VARIANT, cons, FLOOR))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "novelty_raw.csv"), index=False)
    mean = df.groupby(["split", "construction", "backbone"]).mean(numeric_only=True).drop(columns="rep").round(3)
    mean.to_csv(os.path.join(a.out, "novelty_summary.csv"))
    pd.set_option("display.width", 200)
    print(mean.to_string())

    # Recording-level bootstrap of coverage differences (repeats averaged within each draw).
    rng = np.random.default_rng(SEED)
    out = []
    for sp in ("test", "ood_test"):
        R = len(recs[sp])
        draws = rng.integers(0, R, size=(N_BOOT, R))
        def cov(cons, backbone):
            reps = [v for k, v in inside_store.items() if k[0] == cons and k[1] == backbone and k[3] == sp]
            full = np.mean([s[:, 0].sum() / s[:, 1].sum() for s in reps])
            boot = np.mean([s[draws, 0].sum(1) / s[draws, 1].sum(1) for s in reps], axis=0)
            return full, boot
        pairs = [(f"NV-SCP ({v})", "AR-SCP", b) for v in ("L", "M") for b in AR] + \
                [(c, f"NV-SCP ({v})", b) for c in SPREAD for v in ("L", "M") for b in ("LSTM", "MLP")]
        for A, B, b in pairs:
            fa, ba = cov(A, b)
            fb, bb = cov(B, b)
            lo, hi = np.percentile(100 * (ba - bb), [2.5, 97.5])
            out.append({"split": sp, "backbone": b, "a": A, "b": B, "delta_cov_pp": 100 * (fa - fb),
                        "ci_low": lo, "ci_high": hi})
    pd.DataFrame(out).round(3).to_csv(os.path.join(a.out, "novelty_paired.csv"), index=False)
    print(pd.DataFrame(out).round(2).to_string())
    with open(os.path.join(a.out, "run_record.json"), "w") as f:
        json.dump({"n_fit_windows": int(n), "p": int(p), "n_boot": N_BOOT, "seed": SEED, "level": LEVEL,
                   "variant": VARIANT, "floor_rel": FLOOR}, f, indent=1)


if __name__ == "__main__":
    main()
