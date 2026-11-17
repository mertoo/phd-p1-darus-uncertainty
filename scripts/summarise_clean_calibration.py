"""Summarise the outputs of scripts/posthoc_clean_calibration.py (one JSON per stored evaluation).

Means over training repeats, by construction and backbone, for the primary calibration set ("cal", 9 noisy
training recordings) and the clean validation recordings ("val").

    python scripts/summarise_clean_calibration.py experiments/analysis/v3/posthoc_clean_cal
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

CH = ["u", "v", "p", "r", "phi"]


def label(tag):
    if tag in ("ridge", "naive"):
        return "AR-SCP", tag, 0
    m = re.fullmatch(r"([a-z]+)_(single|ens|mcd|gauss)?_?rep(\d)", tag)
    b, k, rep = m.group(1), m.group(2), int(m.group(3))
    kind = {None: "AR-SCP", "single": "AR-SCP", "ens": "EN-SCP", "mcd": "DN-SCP", "gauss": "GN-SCP"}[k]
    # gru/tcn/linear have two AR-SCP sources: the single runs of the main benchmark (gru_rep*) and member 0 of the
    # post hoc ensembles (gru_single_rep*); keep both apart.
    if k == "single" and b in ("gru", "tcn", "linear"):
        b += " (ens member 0)"
    return kind, b, rep


def main():
    d = Path(sys.argv[1])
    rows = []
    for f in sorted(d.glob("*.json")):
        j = json.loads(f.read_text())
        kind, b, rep = label(j["tag"])
        for r in j["results"]:
            row = {"construction": kind, "backbone": b, "rep": rep, "calibration": r["calibration"],
                   "split": r["split"], "coverage": 100 * r["coverage"], "width": r["width_norm_mean"],
                   "iscore": r["interval_score_norm_mean"],
                   **{f"cov_{c}": 100 * v for c, v in zip(CH, r["coverage_channel"])}}
            if r["split"] == "ood_test":
                row |= {"cov_clean": 100 * r["coverage_clean"], "cov_noisy": 100 * r["coverage_noisy"]}
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(d / "summary_long.csv", index=False)
    g = df.groupby(["split", "construction", "backbone", "calibration"]).mean(numeric_only=True).drop(columns="rep")
    pd.set_option("display.width", 220)
    for sp, cols in (("test", ["coverage", "width", "iscore"] + [f"cov_{c}" for c in CH]),
                     ("ood_test", ["coverage", "cov_clean", "cov_noisy", "width", "iscore"])):
        t = g.loc[sp][cols].unstack("calibration").round(2)
        t.columns = [f"{a}|{b}" for a, b in t.columns]
        print(f"\n=== {sp}")
        print(t.to_string())
    g.round(4).to_csv(d / "summary.csv")


if __name__ == "__main__":
    main()
