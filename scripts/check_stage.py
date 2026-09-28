"""Gate check after a benchmark stage (correctness only; never reads test/OOD metrics
for decisions — evaluation checks look at validity fields, not at coverage values).

    python -m scripts.check_stage --stages 0            # after stage A (ablation)
    python -m scripts.check_stage --stages 1a 1b 1c 1d 1e   # after stage B (trainings)
    python -m scripts.check_stage --evals               # after stage C (evaluations)

Exit 1 if any correctness check fails. Floor activation (P10) is collected as a
diagnostic table, never as a pass.
"""

import argparse
import csv
import json
import math
import os

ROOT, EVAL_ROOT = "experiments/runs/v3", "experiments/eval/v3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", default=[])
    ap.add_argument("--evals", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cal = json.load(open("experiments/manifests/calibration_selection.json"))
    primary = cal[cal["primary"]]["files"]
    fails, diags, hashes = [], [], {}

    for row in csv.DictReader(open("scripts/slurm/v3/run_list.tsv"), delimiter="\t"):
        if row["stage"] not in args.stages:
            continue
        p = os.path.join(ROOT, row["run_name"], "train_log.json")
        if not os.path.exists(p):
            fails.append(f"{row['run_name']}: missing train_log.json")
            continue
        log = json.load(open(p))
        ep = log["epochs"]
        losses = [e["train_loss"] for e in ep] + [e["val_loss"] for e in ep]
        exp_steps = math.ceil(log["n_train_windows"] / log["batch_size"])
        checks = {
            "finite losses": all(map(math.isfinite, losses)),
            "seed matches run list": log["seed"] == int(row["seed"]),
            "primary calibration files": log["cal_files"] == primary,
            "standardisation on": log["normalize"] is True,
            "170016 training windows": log["n_train_windows"] == 170016,
            "optimiser steps per epoch": all(e["optimizer_steps"] == exp_steps for e in ep),
        }
        if row["features_override"]:
            checks["feature override applied"] = (("time" in log["features"]) == (row["features_override"] == "legacy"))
        for k, ok in checks.items():
            if not ok:
                fails.append(f"{row['run_name']}: {k}")
        hashes.setdefault("split_hash", set()).add(log["split_hash"])
        if not row["features_override"]:
            hashes.setdefault("scaler_hash (common features)", set()).add(log["scaler_hash"])
        if log["best_epoch"] == len(ep) and log["stopped_by"] == "epoch_budget":
            diags.append(f"{row['run_name']}: best epoch = last epoch ({len(ep)}); stopped by budget")
    for k, v in hashes.items():
        if len(v) > 1:
            fails.append(f"{k} differs across runs: {sorted(v)}")

    floor_rows = []
    if args.evals:
        for row in csv.DictReader(open("scripts/slurm/v3/eval_list.tsv"), delimiter="\t"):
            p = os.path.join(EVAL_ROOT, row["tag"], "metrics.json")
            if not os.path.exists(p):
                fails.append(f"eval {row['tag']}: missing metrics.json (skipped or failed)")
                continue
            m = json.load(open(p))
            prov = m["provenance"]
            if prov["data_state"]["split_files"].get("cal") != primary:
                fails.append(f"eval {row['tag']}: calibration files differ from primary")
            if prov["member_validation"]["n_members"] != int(row["n_members"] or 1):
                fails.append(f"eval {row['tag']}: member count")
            if prov["eval_splits"] != ["test", "ood_test"]:
                fails.append(f"eval {row['tag']}: eval splits {prov['eval_splits']}")
            bad = [iv for iv in m["intervals"] for k in ("coverage", "width_norm_mean", "interval_score_norm_mean")
                   if isinstance(iv.get(k), float) and math.isnan(iv[k])]
            if bad:
                fails.append(f"eval {row['tag']}: NaN interval metrics")
            seen = set()
            for iv in m["intervals"]:
                fl = iv.get("info", {}).get("frac_floored_cal")
                key = (iv["method"], iv["variant"])
                if fl is not None and key not in seen:
                    seen.add(key)
                    floor_rows.append({"tag": row["tag"], "method": iv["method"], "variant": iv["variant"],
                                       "frac_floored_cal": fl, "exceeds_1pct": max(fl) > 0.01})
        n_exceed = sum(r["exceeds_1pct"] for r in floor_rows)
        diags.append(f"P10 floor activation: {n_exceed} of {len(floor_rows)} method/variant evaluations exceed 1% "
                     "-> predeclared floor sensitivity required" if n_exceed else
                     "P10 floor activation: no evaluation exceeds 1% (still reported)")

    report = {"stages": args.stages, "evals": args.evals, "status": "FAIL" if fails else "PASS",
              "failures": fails, "diagnostics": diags, "floor_activation": floor_rows}
    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)
    for f in fails:
        print("FAIL", f)
    for d in diags:
        print("DIAG", d)
    print("STAGE CHECK", report["status"])
    raise SystemExit(0 if not fails else 1)


if __name__ == "__main__":
    main()
