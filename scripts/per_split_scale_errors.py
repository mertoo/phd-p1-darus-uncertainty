"""Point errors on each split's own scale (post hoc, internal review comments #10 and #13).

Normalising OOD errors by the fitting-set standard deviation inflates them, because the OOD targets vary more
(r, v and phi). For each point forecaster of Table 3 (mean over training repeats) this script reports, per channel
and split:
  - RMSE in physical units (check against Table 3 of the manuscript),
  - the standard deviation of the targets over all windows and steps of that split (sigma_split),
  - NRMSE_own = RMSE / sigma_split,
  - skill against climatology = 1 - RMSE / sigma_split (climatology = the split's own mean, an oracle reference),
  - skill against persistence = 1 - RMSE / RMSE_persistence.

    python scripts/per_split_scale_errors.py --eval experiments/eval/v3 --out per_split_scale_errors.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np

CH = ["u", "v", "p", "r", "phi"]
# Table 3 sources: single-model runs of the main benchmark.
MODELS = {"LSTM": [f"lstm_single_rep{i}" for i in range(3)], "GRU": [f"gru_rep{i}" for i in range(3)],
          "TCN": [f"tcn_rep{i}" for i in range(3)], "MLP": [f"mlp_single_rep{i}" for i in range(3)],
          "Linear (SGD)": [f"linear_rep{i}" for i in range(3)], "Ridge": ["ridge"], "Persistence": ["naive"]}


def rmse(tag, split, ev):
    z = np.load(ev / tag / f"predictions_{split}.npz")
    y, mu = z["y"].astype(np.float64), z["mean"].astype(np.float64)
    return np.sqrt(((mu - y) ** 2).reshape(-1, 5).mean(0)), y.reshape(-1, 5).std(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=Path("experiments/eval/v3"))
    ap.add_argument("--out", type=Path, default=Path("per_split_scale_errors.csv"))
    a = ap.parse_args()
    rows = []
    for split in ("test", "ood_test"):
        res = {}
        for name, tags in MODELS.items():
            r = [rmse(t, split, a.eval) for t in tags]
            res[name] = np.mean([x[0] for x in r], 0)
            sd = r[0][1]
        print(f"\n=== {split}: sigma_split", " ".join(f"{c}={s:.4g}" for c, s in zip(CH, sd)))
        for name, e in res.items():
            own = e / sd
            skp = 1 - e / res["Persistence"]
            print(f"{name:14s} RMSE " + " ".join(f"{x:.4f}" for x in e) + " | NRMSE_own " +
                  " ".join(f"{x:.3f}" for x in own) + f" mean {own.mean():.3f} | skill_pers " +
                  " ".join(f"{x:+.2f}" for x in skp))
            for i, c in enumerate(CH):
                rows.append({"split": split, "model": name, "channel": c, "rmse": e[i], "sigma_split": sd[i],
                             "nrmse_own": own[i], "skill_climatology": 1 - own[i], "skill_persistence": skp[i]})
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
