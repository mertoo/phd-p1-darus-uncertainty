"""Timing/correctness pilot report (EXPERIMENT_PROTOCOL.md §7a).

    python -m scripts.pilot_report --runs experiments/runs/v3/pilot \
        --evals experiments/eval/v3/pilot --out experiments/eval/v3/pilot/pilot_report.json

Only validation-split outputs exist for the pilot; no test/OOD numbers are read,
and no coverage value is used to decide anything.

Three separate groups:
  correctness  failure if violated: non-finite losses, missing/extra epochs,
               split / calibration / feature / scaling mismatch, invalid or NaN
               interval metrics, provenance problems, missing outputs
  resources    failure if violated: GPU / host memory limits, artifact size
  diagnostics  never automatic failures; listed for inspection (e.g. non-monotonic
               losses over three epochs, floored spreads, infinite intervals)
A separate planning section projects the full run list; exceeding its budget
blocks the full benchmark until the plan is revised, but is not a pilot failure.
"""

import argparse
import csv
import glob
import json
import math
import os

GPU_MEM_MAX_GB, RSS_MAX_GB = 40.0, 14.0
EVAL_BYTES_MAX = 1e9
TYPICAL_EPOCHS, MAX_EPOCHS = 40, 100
PLAN_TYPICAL_H, PLAN_WORST_H = 40.0, 100.0
FULL_EVAL_WINDOWS = 31878 + 106260 + 102718        # cal + ID test + OOD
PROXY = {"lstm": "lstm", "lstm_dropout": "lstm_dropout", "gru": "lstm", "tcn": "tcn",
         "mlp": "mlp", "mlp_dropout": "mlp", "linear": "mlp",
         "lstm_gaussian": "lstm_gaussian", "mlp_gaussian": "mlp_gaussian"}


def dir_bytes(path):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs)


def monotone(xs):
    return all(b < a for a, b in zip(xs, xs[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--evals", required=True)
    ap.add_argument("--run_list", default="scripts/slurm/v3/run_list.tsv")
    ap.add_argument("--pilot_list", default="scripts/slurm/v3/pilot_runs.tsv")
    ap.add_argument("--cal_manifest", default="experiments/manifests/calibration_selection.json")
    ap.add_argument("--expected_evals", nargs="+",
                    default=["point_lstm", "point_mlp", "ens_lstm", "mcd_lstm", "gauss_lstm", "gauss_mlp"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    correctness, resources, diagnostics = [], [], []
    with open(args.cal_manifest) as f:
        cal_doc = json.load(f)
    expected_cal = cal_doc[cal_doc["primary"]]["files"]

    # ---------------------------------------------------------- training runs
    with open(args.pilot_list) as f:
        expected_runs = [os.path.basename(r["run_name"]) for r in csv.DictReader(f, delimiter="\t")]
    runs = {}
    for name in expected_runs:
        p = os.path.join(args.runs, name, "train_log.json")
        if not os.path.exists(p):
            correctness.append((f"{name}: train_log.json exists", False))
            continue
        with open(p) as f:
            log = json.load(f)
        ep = log["epochs"]
        tr, va = [e["train_loss"] for e in ep], [e["val_loss"] for e in ep]
        runs[name] = {"kind": name.rsplit("_s", 1)[0], "sec_per_epoch": log["sec_per_epoch"],
                      "train_windows_per_sec": log["train_windows_per_sec"],
                      "train_s": ep[-1]["elapsed_s"] if ep else None,
                      "gpu_mem_gb": max((e["gpu_max_mem_gb"] or 0) for e in ep) if ep else None,
                      "rss_gb": max(e["peak_rss_gb"] for e in ep) if ep else None,
                      "train_loss": tr, "val_loss": va,
                      "split_hash": log["split_hash"], "scaler_hash": log["scaler_hash"],
                      "features": log["features"]}
        correctness.append((f"{name}: exactly 3 epochs", len(ep) == 3))
        correctness.append((f"{name}: all losses finite", all(map(math.isfinite, tr + va))))
        correctness.append((f"{name}: calibration recordings == primary manifest list", log["cal_files"] == expected_cal))
        correctness.append((f"{name}: training-only standardisation on", log["normalize"] is True))
        resources.append((f"{name}: peak GPU memory < {GPU_MEM_MAX_GB} GB", (runs[name]["gpu_mem_gb"] or 0) < GPU_MEM_MAX_GB))
        resources.append((f"{name}: peak host RSS < {RSS_MAX_GB} GB", runs[name]["rss_gb"] < RSS_MAX_GB))
        if not monotone(tr):
            diagnostics.append(f"{name}: train loss not monotonically decreasing over 3 epochs: {tr}")
        if not monotone(va):
            diagnostics.append(f"{name}: val loss not monotonically decreasing over 3 epochs: {va}")
    if runs:
        for key in ("split_hash", "scaler_hash"):
            correctness.append((f"identical {key} across all pilot runs", len({r[key] for r in runs.values()}) == 1))
        correctness.append(("identical feature list across all pilot runs",
                            len({tuple(r["features"]) for r in runs.values()}) == 1))

    # -------------------------------------------------------------- evaluations
    evals = {}
    for name in args.expected_evals:
        mpath = os.path.join(args.evals, name, "metrics.json")
        if not os.path.exists(mpath):
            correctness.append((f"eval {name}: metrics.json exists", False))
            continue
        with open(mpath) as f:
            m = json.load(f)
        prov = m["provenance"]
        size = dir_bytes(os.path.dirname(mpath))
        evals[name] = {"method": prov["method"], "timing": prov["timing"], "bytes": size,
                       "gpu_mem_gb": prov.get("gpu_max_mem_gb")}
        bad = [(iv["variant"], iv["level"]) for iv in m["intervals"]
               if any(isinstance(iv.get(k), float) and math.isnan(iv[k])
                      for k in ("coverage", "width_norm_mean", "interval_score_norm_mean"))]
        correctness.append((f"eval {name}: no NaN coverage/width/score", not bad))
        correctness.append((f"eval {name}: scored validation split only", prov["eval_splits"] == ["val"]))
        correctness.append((f"eval {name}: calibration split = primary manifest list",
                            prov["data_state"]["split_files"].get("cal") == expected_cal))
        correctness.append((f"eval {name}: member provenance validated", "member_validation" in prov))
        resources.append((f"eval {name}: artifacts < 1 GB", size < EVAL_BYTES_MAX))
        if prov.get("gpu_max_mem_gb") is not None:
            resources.append((f"eval {name}: peak GPU memory < {GPU_MEM_MAX_GB} GB", prov["gpu_max_mem_gb"] < GPU_MEM_MAX_GB))
        seen = set()
        for iv in m["intervals"]:
            if iv.get("frac_infinite", 0) > 0:
                diagnostics.append(f"eval {name}: {iv['method']}/{iv['variant']}@{iv['level']} has "
                                   f"{iv['frac_infinite']:.3%} infinite bounds")
            fl = iv.get("info", {}).get("frac_floored_cal")
            key = (iv["method"], iv["variant"])
            if fl is not None and max(fl) > 0.01 and key not in seen:   # same spread at every level
                seen.add(key)
                diagnostics.append(f"eval {name}: {iv['method']}/{iv['variant']} floors >1% of calibration "
                                   f"spreads per channel {[round(x, 4) for x in fl]}")
    correctness.append(("bootstrap summary produced",
                        os.path.exists(os.path.join(args.evals, "bootstrap", "summary_val.csv"))))

    # ----------------------------------------------------------------- planning
    per_kind = {}
    for r in runs.values():
        per_kind.setdefault(r["kind"], []).append(r["sec_per_epoch"])
    sec = {k: sum(v) / len(v) for k, v in per_kind.items()}
    planned = {}
    with open(args.run_list) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            planned[row["config"]] = planned.get(row["config"], 0) + 1
    train_h, missing = {"typical": 0.0, "worst": 0.0}, []
    for cfg, n in planned.items():
        proxy = PROXY.get(cfg)
        if proxy not in sec:
            missing.append(cfg)
            continue
        train_h["typical"] += n * TYPICAL_EPOCHS * sec[proxy] / 3600
        train_h["worst"] += n * MAX_EPOCHS * sec[proxy] / 3600
    eval_full_h = {}
    for k, v in evals.items():
        n_win = sum(t["n_windows"] for t in v["timing"].values())
        eval_full_h[k] = sum(t["inference_s"] for t in v["timing"].values()) * FULL_EVAL_WINDOWS / n_win / 3600
    eval_total = 3 * sum(eval_full_h.values())   # 3 repeats; 2-member pilot ensemble under-estimates 5 members
    total = {k: v + eval_total for k, v in train_h.items()}
    planning = {"sec_per_epoch": sec, "planned_trainings": planned, "unprojected_configs": missing,
                "train_gpu_h": train_h, "eval_gpu_h_per_method_full": eval_full_h,
                "eval_gpu_h_total": eval_total, "total_gpu_h": total,
                "within_plan_budget": (total["typical"] <= PLAN_TYPICAL_H and total["worst"] <= PLAN_WORST_H
                                       and not missing),
                "assumptions": (f"{TYPICAL_EPOCHS} typical / {MAX_EPOCHS} max epochs; GRU/linear timed by LSTM/MLP; "
                                "includes all 63 run-list rows (incl. conditional stage 0b); excludes SLURM "
                                "queue/startup overhead; ensemble eval scaled from a 2-member pilot")}

    measured = {"train_compute_s": sum(r["train_s"] or 0 for r in runs.values()),
                "eval_inference_s": sum(sum(t["inference_s"] for t in v["timing"].values()) for v in evals.values()),
                "note": "in-process compute only; billed GPU-hours come from sacct (see protocol §7a)"}

    failed = [c for c, ok in correctness + resources if not ok]
    report = {"status": "FAIL" if failed else "PASS",
              "correctness": [{"check": c, "ok": bool(ok)} for c, ok in correctness],
              "resources": [{"check": c, "ok": bool(ok)} for c, ok in resources],
              "diagnostics": diagnostics, "failed": failed,
              "runs": runs, "evals": evals, "measured": measured, "planning": planning}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    for title, group in (("CORRECTNESS", correctness), ("RESOURCES", resources)):
        print(f"== {title}")
        for c, ok in group:
            print(("  PASS " if ok else "  FAIL ") + c)
    print("== DIAGNOSTICS (inspect; not automatic failures)")
    for d in diagnostics or ["none"]:
        print("  " + d)
    print("== PLANNING", json.dumps(total), "within plan budget:", planning["within_plan_budget"])
    print("PILOT", report["status"])
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
