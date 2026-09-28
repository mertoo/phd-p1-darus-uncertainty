"""Summarise the timing pilot, apply its pass/fail criteria, project full cost.

    python -m scripts.pilot_report --runs experiments/runs/v3/pilot \
        --evals experiments/eval/v3/pilot --out experiments/eval/v3/pilot/pilot_report.json

Pass/fail criteria (EXPERIMENT_PROTOCOL.md §7a) are evaluated mechanically.
Only validation-split outputs exist for the pilot; no test/OOD numbers are read.
"""

import argparse
import csv
import glob
import json
import math
import os

GPU_MEM_MAX_GB, RSS_MAX_GB = 40.0, 14.0
TYPICAL_EPOCHS, MAX_EPOCHS = 40, 100
BUDGET_TYPICAL_H, BUDGET_WORST_H = 40.0, 100.0
EVAL_BYTES_MAX = 1e9
# planned config -> pilot run used as its timing proxy
PROXY = {"lstm": "lstm", "lstm_dropout": "lstm_dropout", "gru": "lstm", "tcn": "tcn",
         "mlp": "mlp", "mlp_dropout": "mlp", "linear": "mlp",
         "lstm_gaussian": "lstm_gaussian", "mlp_gaussian": "mlp_gaussian"}


def dir_bytes(path):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--evals", required=True)
    ap.add_argument("--run_list", default="scripts/slurm/v3/run_list.tsv")
    ap.add_argument("--cal_manifest", default="experiments/manifests/calibration_selection.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    checks, runs = [], {}
    for log_path in sorted(glob.glob(os.path.join(args.runs, "*", "train_log.json"))):
        name = os.path.basename(os.path.dirname(log_path))
        with open(log_path) as f:
            log = json.load(f)
        ep = log["epochs"]
        kind = name.rsplit("_s", 1)[0]
        finite = all(math.isfinite(e["train_loss"]) and math.isfinite(e["val_loss"]) for e in ep)
        runs[name] = {"kind": kind, "sec_per_epoch": log["sec_per_epoch"], "epochs": len(ep),
                      "gpu_mem_gb": max((e["gpu_max_mem_gb"] or 0) for e in ep),
                      "rss_gb": max(e["peak_rss_gb"] for e in ep), "split_hash": log["split_hash"]}
        checks.append((f"{name}: 3 epochs, finite losses", len(ep) == 3 and finite))
        checks.append((f"{name}: train loss decreased epoch 1->3", ep[-1]["train_loss"] < ep[0]["train_loss"]))
        checks.append((f"{name}: GPU memory < {GPU_MEM_MAX_GB} GB", runs[name]["gpu_mem_gb"] < GPU_MEM_MAX_GB))
        checks.append((f"{name}: host RSS < {RSS_MAX_GB} GB", runs[name]["rss_gb"] < RSS_MAX_GB))
    if not runs:
        raise FileNotFoundError(f"no train_log.json under {args.runs}")
    checks.append(("identical split hash across pilot runs", len({r["split_hash"] for r in runs.values()}) == 1))

    with open(args.cal_manifest) as f:
        cal_doc = json.load(f)
    expected_cal = cal_doc[cal_doc["selected_method"]]["files"]

    evals = {}
    for mpath in sorted(glob.glob(os.path.join(args.evals, "*", "metrics.json"))):
        name = os.path.basename(os.path.dirname(mpath))
        with open(mpath) as f:
            m = json.load(f)
        prov = m["provenance"]
        nan_found = any(
            isinstance(v, float) and math.isnan(v)
            for iv in m["intervals"] for k, v in iv.items() if k.startswith(("coverage", "width_norm", "interval_score_norm")))
        size = dir_bytes(os.path.dirname(mpath))
        evals[name] = {"timing": prov["timing"], "bytes": size, "method": prov["method"]}
        checks.append((f"eval {name}: no NaN in coverage/width/score", not nan_found))
        checks.append((f"eval {name}: only validation scored", prov["eval_splits"] == ["val"]))
        checks.append((f"eval {name}: calibration recordings match manifest",
                       prov["data_state"]["split_files"].get("cal") == expected_cal))
        checks.append((f"eval {name}: output < 1 GB", size < EVAL_BYTES_MAX))
    boot = os.path.join(args.evals, "bootstrap", "summary_val.csv")
    checks.append(("bootstrap summary produced", os.path.exists(boot)))

    # ---- projection for the full plan ----
    per_kind = {}
    for r in runs.values():
        per_kind.setdefault(r["kind"], []).append(r["sec_per_epoch"])
    sec = {k: sum(v) / len(v) for k, v in per_kind.items()}
    planned = {}
    with open(args.run_list) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            planned[row["config"]] = planned.get(row["config"], 0) + 1
    train_h = {"typical": 0.0, "worst": 0.0}
    missing = []
    for cfg, n in planned.items():
        proxy = PROXY.get(cfg)
        if proxy not in sec:
            missing.append(cfg)
            continue
        train_h["typical"] += n * TYPICAL_EPOCHS * sec[proxy] / 3600
        train_h["worst"] += n * MAX_EPOCHS * sec[proxy] / 3600
    # evaluation: pilot scored cal+val; full evaluation scores cal+test+ood
    ref = next(iter(evals.values()))["timing"] if evals else {}
    n_pilot = sum(v["n_windows"] for v in ref.values()) if ref else 1
    scale = (31878 + 106260 + 102718) / n_pilot
    eval_h_per_method = {k: sum(t["inference_s"] for t in v["timing"].values()) * scale / 3600
                         for k, v in evals.items()}
    # full plan: 3 repeats of each evaluated method family
    eval_total_h = 3 * sum(eval_h_per_method.values())
    proj = {"sec_per_epoch": sec, "planned_trainings": planned, "unprojected_configs": missing,
            "train_gpu_h": train_h, "eval_gpu_h_per_method_full": eval_h_per_method,
            "eval_gpu_h_total_est": eval_total_h,
            "total_gpu_h": {k: v + eval_total_h for k, v in train_h.items()},
            "assumptions": f"{TYPICAL_EPOCHS} typical / {MAX_EPOCHS} max epochs; GRU and linear timed by LSTM and MLP proxies"}
    checks.append((f"projected typical total <= {BUDGET_TYPICAL_H} GPU-h", proj["total_gpu_h"]["typical"] <= BUDGET_TYPICAL_H))
    checks.append((f"projected worst-case total <= {BUDGET_WORST_H} GPU-h", proj["total_gpu_h"]["worst"] <= BUDGET_WORST_H))
    checks.append(("every planned config has a timing proxy", not missing))

    report = {"passed": all(ok for _, ok in checks),
              "checks": [{"check": c, "ok": bool(ok)} for c, ok in checks],
              "runs": runs, "evals": evals, "projection": proj}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    for c, ok in checks:
        print(("PASS " if ok else "FAIL ") + c)
    print(json.dumps(proj["total_gpu_h"]), "->", "PILOT PASSED" if report["passed"] else "PILOT FAILED")


if __name__ == "__main__":
    main()
