"""Per-recording estimate of additive white measurement noise in the target channels.

Checks the claim (internal review comments of 5-6 Oct 2026) that the training and calibration recordings
carry about 0.10 m/s of additive noise on u and v, that the validation and ID-test recordings are clean, and that
17 of the 29 OOD recordings are noisy.

For white noise of standard deviation s added to a smooth 1-Hz signal, the second difference
x[t+1] - 2 x[t] + x[t-1] has variance 6 s^2 plus a small signal term, and the first differences have a lag-1
autocorrelation near -0.5. A smooth signal without noise gives a much smaller second difference and a lag-1
autocorrelation of the first differences near +1. Both statistics are reported per recording and channel.

Stdlib only, so it runs without the project environment:
    /usr/bin/python3 -I scripts/check_measurement_noise.py [--data data/processed/darus]
"""
import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

CHANNELS = ["u", "v", "p", "r", "phi"]


def noise_stats(x):
    d1 = [b - a for a, b in zip(x, x[1:])]
    d2 = [b - a for a, b in zip(d1, d1[1:])]
    sigma = math.sqrt(sum(e * e for e in d2) / len(d2) / 6)
    m = st.fmean(d1)
    num = sum((a - m) * (b - m) for a, b in zip(d1, d1[1:]))
    den = sum((a - m) ** 2 for a in d1)
    return sigma, num / den if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed/darus"))
    ap.add_argument("--manifest", type=Path, default=Path("experiments/manifests/split_manifest.csv"))
    ap.add_argument("--calibration", type=Path, default=Path("experiments/manifests/calibration_selection.json"))
    ap.add_argument("--out", type=Path, default=Path("experiments/analysis/v3/noise_check.csv"))
    a = ap.parse_args()

    split = {r["file"]: r["split"] for r in csv.DictReader(open(a.manifest))}
    cal = set(json.load(open(a.calibration))["random"]["files"])
    files = {p.name: p for p in a.data.rglob("*.csv")}

    rows = []
    for name, s in sorted(split.items()):
        role = "calibration" if name in cal else {"train": "fitting", "val": "validation", "test": "ID test",
                                                  "ood_test": "OOD test"}[s]
        cols = {c: [] for c in CHANNELS}
        with open(files[name]) as f:
            for r in csv.DictReader(f):
                for c in CHANNELS:
                    cols[c].append(float(r[c]))
        row = {"file": name, "role": role}
        for c in CHANNELS:
            sigma, rho = noise_stats(cols[c])
            row[f"{c}_sigma"], row[f"{c}_rho1"] = sigma, rho
        rows.append(row)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'role':12s} {'n':>3s}  " + "  ".join(f"{c + ' sigma (min/med/max)':>30s}" for c in ("u", "v")) +
          "  " + "  ".join(f"{c + ' rho1 med':>10s}" for c in CHANNELS))
    for role in ("fitting", "calibration", "validation", "ID test", "OOD test"):
        g = [r for r in rows if r["role"] == role]
        def mmm(k):
            v = sorted(r[k] for r in g)
            return f"{v[0]:.3f}/{st.median(v):.3f}/{v[-1]:.3f}"
        print(f"{role:12s} {len(g):3d}  " + "  ".join(f"{mmm(c + '_sigma'):>30s}" for c in ("u", "v")) + "  " +
              "  ".join(f"{st.median(r[c + '_rho1'] for r in g):10.2f}" for c in CHANNELS))
    print(f"\nper-recording values: {a.out}")


if __name__ == "__main__":
    main()
