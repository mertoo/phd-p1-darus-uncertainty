"""OOD coverage of the stored split-conformal intervals, split by measurement-noise group.

Reads the per-window outputs of experiments/eval/v3 (one row per window: per-channel count of the 30 steps inside
the interval) and the per-recording noise estimates of scripts/check_measurement_noise.py. A recording is "noisy"
if its estimated white-noise level on u exceeds 0.05 m/s (17 of the 29 OOD recordings, about 0.31 m/s; the clean
ones are below 0.005 m/s). Coverage is pooled over windows, steps and channels, as in the manuscript, and averaged
over training repeats. Also reports ID-test coverage per channel, to show where the over-coverage sits.

    python scripts/noise_group_coverage.py --eval experiments/eval/v3 --noise noise_check.csv
"""
import argparse
import re
from pathlib import Path

import pandas as pd

CH = ["u", "v", "p", "r", "phi"]
# File suffix -> construction. The manuscript uses thresholds per step and channel ("horizon_channel").
# "<method>_horizon_channel_90" holds the spread-normalised conformal intervals; "<method>_conformal_..." the
# absolute-residual intervals around that method's mean (not used here). Checked against metrics.json.
KIND = {"conformal_horizon_channel_90": "AR-SCP", "ensemble_horizon_channel_90": "EN-SCP",
        "mc_dropout_horizon_channel_90": "DN-SCP", "gaussian_horizon_channel_90": "GN-SCP"}


def coverage(df):
    steps = df["n_steps"].sum()
    per = {c: df[f"inside_{i}"].sum() / steps * 100 for i, c in enumerate(CH)}
    per["all"] = sum(df[f"inside_{i}"].sum() for i in range(5)) / (5 * steps) * 100
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=Path("experiments/eval/v3"))
    ap.add_argument("--noise", type=Path, default=Path("experiments/analysis/v3/noise_check.csv"))
    ap.add_argument("--out", type=Path, default=Path("noise_group_coverage.csv"))
    a = ap.parse_args()

    nz = pd.read_csv(a.noise)
    noisy = set(nz[(nz.role == "OOD test") & (nz.u_sigma > 0.05)].file)
    rows = []
    for d in sorted(a.eval.iterdir()):
        m = re.fullmatch(r"(ridge|naive)|([a-z]+)_(single|ens|mcd|gauss)_rep(\d)", d.name)
        if not m:
            continue
        backbone = m.group(1) or m.group(2)
        rep = int(m.group(4)) if m.group(4) else 0
        for split in ("test", "ood_test"):
            for f in d.glob(f"per_window_{split}_*_90.csv"):
                kind = KIND.get(f.name[len(f"per_window_{split}_"):-4])
                if kind is None or (kind == "AR-SCP" and m.group(3) not in (None, "single")):
                    continue
                df = pd.read_csv(f)
                groups = {"all": df}
                if split == "ood_test":
                    groups |= {"noisy": df[df.recording.isin(noisy)], "clean": df[~df.recording.isin(noisy)]}
                for g, sub in groups.items():
                    rows.append({"backbone": backbone, "construction": kind, "rep": rep, "split": split,
                                 "group": g, "n_rec": sub.recording.nunique(), **coverage(sub)})
    res = pd.DataFrame(rows)
    res.to_csv(a.out, index=False)
    mean = (res.groupby(["split", "construction", "backbone", "group"])[CH + ["all", "n_rec"]].mean()
            .round(1).reset_index())
    pd.set_option("display.width", 200)
    for split in ("test", "ood_test"):
        print(f"\n=== {split} (coverage %, mean over repeats)")
        t = mean[mean.split == split].drop(columns="split")
        print(t.to_string(index=False))


if __name__ == "__main__":
    main()
