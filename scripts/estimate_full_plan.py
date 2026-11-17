"""Full-plan resource estimate from measured pilot data (no guessing by hand).

    python -m scripts.estimate_full_plan [--contingency 1.5]

Inputs: pilot_report.json and sacct.txt in experiments/manifests/pilot_2026-09-28/,
scripts/slurm/v3/run_list.tsv (63 training rows incl. conditional stage 0b) and
scripts/slurm/v3/eval_list.tsv (35 evaluations). Writes
experiments/manifests/full_plan_estimate.json.

GPU-hours = allocated elapsed time x GPUs (what the scheduler bills), estimated as
measured compute + measured per-job start-up overhead, then x contingency.
"""

import argparse
import csv
import json
import re

PILOT = "experiments/manifests/pilot_2026-09-28"
MAX_EPOCHS = 100
FULL_EVAL_WINDOWS = 31878 + 106260 + 102718        # cal + ID test + OOD
PILOT_EVAL_WINDOWS = 31878 + 31878                 # cal + val
# pilot run used as timing source for each planned config; `measured` = timed directly
TRAIN_PROXY = {"lstm": ("lstm", True), "mlp": ("mlp", True), "lstm_dropout": ("lstm_dropout", True),
               "lstm_gaussian": ("lstm_gaussian", True), "mlp_gaussian": ("mlp_gaussian", True),
               "tcn": ("tcn", True), "gru": ("lstm", False), "mlp_dropout": ("mlp", False),
               "linear": ("mlp", False)}
# pilot eval wall-time source (process wall incl. metrics) per (method, backbone)
EVAL_SOURCE = {"point": "point_lstm", "gaussian": "gauss_lstm", "mc_dropout": "mcd_lstm", "ensemble": "ens_lstm"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contingency", type=float, default=1.5)
    args = ap.parse_args()

    rep = json.load(open(f"{PILOT}/pilot_report.json"))
    sacct = {}
    for line in open(f"{PILOT}/sacct.txt"):
        f = line.split()
        if f and re.match(r"98424[34](_\d)?$", f[0]):
            sacct[f[0]] = int(f[4])
    order = ["lstm_s9001", "lstm_s9002", "mlp_s9001", "lstm_dropout_s9001",
             "lstm_gaussian_s9001", "mlp_gaussian_s9001", "tcn_s9001"]
    overhead = [sacct[f"984243_{i}"] - rep["runs"][r]["train_s"] for i, r in enumerate(order, 1)]
    ovh_train = max(overhead)                                    # conservative: worst measured start-up
    sec_epoch = {}
    for r, v in rep["runs"].items():
        sec_epoch.setdefault(v["kind"], []).append(v["sec_per_epoch"])
    sec_epoch = {k: max(v) for k, v in sec_epoch.items()}

    train_rows, stage_h = [], {}
    for row in csv.DictReader(open("scripts/slurm/v3/run_list.tsv"), delimiter="\t"):
        proxy, measured = TRAIN_PROXY[row["config"]]
        s = MAX_EPOCHS * sec_epoch[proxy] + ovh_train
        train_rows.append({"stage": row["stage"], "config": row["config"], "bound_s": s, "timed_directly": measured})
        stage_h[row["stage"]] = stage_h.get(row["stage"], 0) + s / 3600
    train_h = sum(stage_h.values())

    # evaluation: pilot process wall (log lines) scaled to full windows; ensemble x5/2 members
    wall = {"point_lstm": 10, "point_mlp": 9, "ens_lstm": 18, "mcd_lstm": 60, "gauss_lstm": 16, "gauss_mlp": 16}
    scale = FULL_EVAL_WINDOWS / PILOT_EVAL_WINDOWS
    ovh_eval = 15.7                                              # per separate eval job start-up (= worst train start-up)
    eval_s = 0.0
    n_eval = 0
    for row in csv.DictReader(open("scripts/slurm/v3/eval_list.tsv"), delimiter="\t"):
        w = wall[EVAL_SOURCE[row["method"]]] * scale
        if row["method"] == "ensemble":
            w *= 5 / 2
        eval_s += w + ovh_eval
        n_eval += 1
    eval_h = eval_s / 3600

    sizes_mb = {"point": 59, "spread": 124}                      # pilot eval dirs at cal+val scale
    n_point = sum(1 for r in csv.DictReader(open("scripts/slurm/v3/eval_list.tsv"), delimiter="\t") if r["method"] == "point")
    storage_gb = (n_point * sizes_mb["point"] + (n_eval - n_point) * sizes_mb["spread"]) * scale / 1000 \
        + (len(train_rows) + 2) * 1.6 / 1000

    out = {
        "basis": "pilot jobs 984243/984244 (L40, ada1), commit 54a69d5; max epoch budget, no early stopping assumed",
        "sec_per_epoch_used": sec_epoch,
        "train_startup_overhead_s_used": ovh_train,
        "training_rows": len(train_rows),
        "training_gpu_h_bound_by_stage": {k: round(v, 3) for k, v in stage_h.items()},
        "training_gpu_h_bound": round(train_h, 3),
        "training_rows_with_proxy_timing": sorted({r["config"] for r in train_rows if not r["timed_directly"]}),
        "evaluations": n_eval,
        "eval_gpu_h": round(eval_h, 3),
        "gpu_h_without_contingency": round(train_h + eval_h, 3),
        "contingency_factor": args.contingency,
        "gpu_h_with_contingency": round((train_h + eval_h) * args.contingency, 3),
        "cpu": "ridge fit + naive baseline + bootstrap + table/figure generation: < 1 CPU-h on partition `common`",
        "storage_gb": round(storage_gb, 1),
        "storage_note": "eval artifacts scale with window count (x%.2f from pilot); checkpoints ~1.6 MB each" % scale,
    }
    json.dump(out, open("experiments/manifests/full_plan_estimate.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
