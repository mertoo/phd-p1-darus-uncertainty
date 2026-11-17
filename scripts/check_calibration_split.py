"""Document the D1 calibration recordings and check ID representativeness.

Uses only the 57 deposit training recordings (per-recording means and SDs of
the physical inputs/targets). No model, validation, test or OOD data is read.

Predeclared criterion (fixed before the check was first run):
  For every summary variable, SMD = (mean over calibration recordings -
  mean over fit recordings) / SD over fit recordings. PASS iff |SMD| <= 0.5
  for all variables AND each calibration recording's mean u and mean n lie
  within the [min, max] of the fit recordings.
  The stratified alternative (one recording per equal-count stratum of mean
  n, same seed) is always computed and reported as well.

Decision (author, 2026-09-28): the fixed-seed random selection is the PRIMARY
calibration split; the criterion's failure is reported descriptively and is
NOT replaced by a post-hoc threshold. The stratified selection is retained only
as an optional sensitivity analysis. Selection history is written to the JSON.

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

    strat = select_calibration_files(names, args.n, args.seed,
                                     strata_values=table["n_mean"].to_dict())
    res2, in2, pass2 = check(table, strat)
    out["stratified_n"] = {"files": strat, "passed": pass2, "u_n_in_range": in2,
                           "max_abs_smd": float(res2["smd"].abs().max()),
                           "worst_variable": res2.loc[res2["smd"].abs().idxmax(), "variable"]}
    pd.concat([res.assign(selection="random"), res2.assign(selection="stratified_n")]).to_csv(
        os.path.join(args.out_dir, "calibration_representativeness.csv"), index=False)
    out["primary"] = "random"
    out["sensitivity"] = ["stratified_n"]

    ref = chance_reference(table, args.n)
    out["chance_reference"] = {
        "note": ("descriptive only (added after the first check): distribution of max|SMD| over 5000 "
                 "random draws. Not used as a replacement threshold."),
        "p_criterion_passes_by_chance": float((ref <= SMD_MAX).mean()),
        "max_abs_smd_quantiles_5_50_95": np.quantile(ref, [0.05, 0.5, 0.95]).round(3).tolist(),
    }
    for sel in ("random", "stratified_n"):
        if sel in out:
            out[sel]["max_abs_smd_chance_percentile"] = float((ref < out[sel]["max_abs_smd"]).mean() * 100)
    out["history"] = [
        "2026-09-28 draft protocol: 9 of 57 training recordings, numpy default_rng(20261001), random draw",
        "2026-09-28 predeclared check (|SMD|<=0.5 on 22 variables, u/n in range): random draw FAILED "
        f"(max |SMD| {out['random']['max_abs_smd']:.2f}); predeclared stratified fallback also FAILED "
        f"(max |SMD| {out['stratified_n']['max_abs_smd']:.2f})",
        "2026-09-28 diagnostic: criterion passes in ~7% of random draws; both draws typical (~67th percentile)",
        "2026-09-28 stratified_n briefly set as provisional default in configs (commit 1970dd9)",
        "2026-09-28 author decision: random draw is PRIMARY; no eligibility problem found (all 9 unique deposit "
        "training files, complete, no content duplicates); failure reported descriptively; stratified_n kept "
        "as optional sensitivity analysis",
    ]

    with open(os.path.join(args.out_dir, "calibration_selection.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
