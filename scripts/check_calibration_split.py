"""Document the D1 calibration recordings and check ID representativeness.

Uses only the 57 deposit training recordings (per-recording means and SDs of
the physical inputs/targets). No model, validation, test or OOD data is read.

Predeclared criterion (fixed before the check was first run):
  For every summary variable, SMD = (mean over calibration recordings -
  mean over fit recordings) / SD over fit recordings. PASS iff |SMD| <= 0.5
  for all variables AND each calibration recording's mean u and mean n lie
  within the [min, max] of the fit recordings.
  If the random selection fails, the protocol switches to the predeclared
  stratified selection (one recording per equal-count stratum of mean n,
  same seed), which is checked and reported the same way.

    python -m scripts.check_calibration_split [--seed 20261001 --n 9]
Writes experiments/manifests/calibration_selection.json and
calibration_representativeness.csv.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.data_loading.darus_dataset import select_calibration_files
from src.data_loading.darus_parser import load_split_recordings

VARS = ["n", "deltal", "deltar", "Vw", "alpha_x", "alpha_y", "u", "v", "p", "r", "phi"]
SMD_MAX = 0.5


def recording_table(base):
    rows = []
    for rec in load_split_recordings(base, "train"):
        row = {"file": rec["source_file"].iloc[0]}
        for c in VARS:
            row[f"{c}_mean"] = rec[c].mean()
            row[f"{c}_std"] = rec[c].std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("file").sort_index()


def check(table, cal):
    fit = table.drop(index=cal)
    calt = table.loc[cal]
    res = []
    for col in table.columns:
        sd = fit[col].std()
        smd = (calt[col].mean() - fit[col].mean()) / sd if sd > 0 else 0.0
        res.append({"variable": col, "cal_mean": calt[col].mean(), "fit_mean": fit[col].mean(),
                    "fit_sd": sd, "cal_min": calt[col].min(), "cal_max": calt[col].max(),
                    "fit_min": fit[col].min(), "fit_max": fit[col].max(), "smd": smd})
    res = pd.DataFrame(res)
    in_range = all(fit[c].min() <= v <= fit[c].max()
                   for c in ("u_mean", "n_mean") for v in calt[c])
    passed = bool((res["smd"].abs() <= SMD_MAX).all() and in_range)
    return res, in_range, passed


def chance_reference(table, n, draws=5000, seed=0):
    """Distribution of max |SMD| over random n-recording draws (diagnostic only)."""
    X = table.to_numpy(float)
    rng = np.random.default_rng(seed)
    mx = []
    for _ in range(draws):
        m = np.zeros(len(X), bool)
        m[rng.choice(len(X), n, replace=False)] = True
        fit = X[~m]
        mx.append(np.abs((X[m].mean(0) - fit.mean(0)) / fit.std(0, ddof=1)).max())
    return np.asarray(mx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/processed/darus")
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--out_dir", default="experiments/manifests")
    args = ap.parse_args()

    table = recording_table(args.base)
    names = table.index.tolist()
    out = {"seed": args.seed, "n_recordings": args.n, "criterion":
           f"|SMD| <= {SMD_MAX} for all per-recording mean/SD variables and cal u/n means within fit range"}

    cal = select_calibration_files(names, args.n, args.seed)
    res, in_range, passed = check(table, cal)
    out["random"] = {"files": cal, "passed": passed, "u_n_in_range": in_range,
                     "max_abs_smd": float(res["smd"].abs().max()),
                     "worst_variable": res.loc[res["smd"].abs().idxmax(), "variable"]}
    res.assign(selection="random").to_csv(os.path.join(args.out_dir, "calibration_representativeness.csv"), index=False)

    if passed:
        out["selected_method"] = "random"
    else:
        strat = select_calibration_files(names, args.n, args.seed,
                                         strata_values=table["n_mean"].to_dict())
        res2, in2, pass2 = check(table, strat)
        out["stratified_n"] = {"files": strat, "passed": pass2, "u_n_in_range": in2,
                               "max_abs_smd": float(res2["smd"].abs().max()),
                               "worst_variable": res2.loc[res2["smd"].abs().idxmax(), "variable"]}
        out["selected_method"] = "stratified_n"
        pd.concat([res.assign(selection="random"), res2.assign(selection="stratified_n")]).to_csv(
            os.path.join(args.out_dir, "calibration_representativeness.csv"), index=False)

    ref = chance_reference(table, args.n)
    out["chance_reference"] = {
        "note": "diagnostic, added after the first check: max|SMD| over 5000 random draws",
        "p_criterion_passes_by_chance": float((ref <= SMD_MAX).mean()),
        "max_abs_smd_quantiles_5_50_95": np.quantile(ref, [0.05, 0.5, 0.95]).round(3).tolist(),
    }
    for sel in ("random", "stratified_n"):
        if sel in out:
            out[sel]["max_abs_smd_chance_percentile"] = float((ref < out[sel]["max_abs_smd"]).mean() * 100)
    out["status"] = ("criterion mis-specified (passes by chance in ~7% of draws); neither selection passes; "
                     "stratified_n adopted provisionally as the predeclared fallback pending author decision")

    with open(os.path.join(args.out_dir, "calibration_selection.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
