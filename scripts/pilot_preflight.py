"""Pre-submission checks for the timing pilot (run on the login node, no GPU).

    python -m scripts.pilot_preflight

Fails (exit 1) unless:
  * every pilot config has an identical data block using the PRIMARY
    calibration split (random, seed from the manifest), training-only
    standardisation, and one feature set;
  * the loader's calibration selection equals the manifest's primary list;
  * the SLURM hard limits of ALL pilot jobs (every array task + evaluation)
    sum to <= the authorised GPU-hour budget;
  * no automatic requeue is possible and no earlier pilot outputs exist;
  * the working tree is clean at a commit that exists on the remote branch.
"""

import csv
import json
import os
import re
import subprocess
import sys

import yaml

from src.data_loading.darus_dataset import list_split_files, select_calibration_files

BUDGET_GPU_H = 5.5
TRAIN_SH, EVAL_SH = "scripts/slurm/v3/pilot_train.sh", "scripts/slurm/v3/pilot_eval.sh"
RUNS_TSV = "scripts/slurm/v3/pilot_runs.tsv"


def sbatch_opts(path):
    opts = {}
    for line in open(path):
        m = re.match(r"#SBATCH\s+--([\w-]+)(?:=(\S+))?", line)
        if m:
            opts[m.group(1)] = m.group(2)
    return opts


def hours(t):
    parts = [int(x) for x in t.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] + parts[1] / 60 + parts[2] / 3600


def gpus(opts):
    return int(opts["gres"].rsplit(":", 1)[-1])


def main():
    ok, lines = True, []

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        lines.append(f"{'PASS' if cond else 'FAIL'} {name}{(': ' + detail) if detail else ''}")

    with open("experiments/manifests/calibration_selection.json") as f:
        man = json.load(f)
    primary = man[man["primary"]]["files"]
    check("manifest primary selection is 'random'", man["primary"] == "random")

    rows = list(csv.DictReader(open(RUNS_TSV), delimiter="\t"))
    data_blocks = {}
    for r in rows:
        cfg = yaml.safe_load(open(f"experiments/configs/v3/{r['config']}.yaml"))
        data_blocks[r["run_name"]] = cfg["data"]
    blocks = list(data_blocks.values())
    check("identical data block in every pilot config", all(b == blocks[0] for b in blocks))
    d = blocks[0]
    cal = d.get("calibration", {})
    check("calibration method random", cal.get("method") == "random", str(cal))
    check("calibration seed matches manifest", cal.get("seed") == man["seed"], f"{cal.get('seed')} vs {man['seed']}")
    check("training-only standardisation on", d.get("normalize") is True)
    names = [os.path.basename(p) for p in list_split_files(d["base_path"], "train")]
    sel = select_calibration_files(names, int(cal["n_recordings"]), int(cal["seed"]))
    check("loader calibration selection == manifest primary list", sel == primary)

    tr, ev = sbatch_opts(TRAIN_SH), sbatch_opts(EVAL_SH)
    n_tasks = len(rows)
    train_h = n_tasks * hours(tr["time"]) * gpus(tr)
    eval_h = hours(ev["time"]) * gpus(ev)
    total = train_h + eval_h
    check(f"summed SLURM limits <= {BUDGET_GPU_H} GPU-h", total <= BUDGET_GPU_H + 1e-9,
          f"{n_tasks} array tasks x {tr['time']} x {gpus(tr)} GPU = {train_h:.2f} h; eval {ev['time']} x {gpus(ev)} GPU = {eval_h:.2f} h; total {total:.2f} h")
    check("no-requeue set on both scripts", "no-requeue" in tr and "no-requeue" in ev)
    check("no earlier pilot outputs", not os.path.exists("experiments/runs/v3/pilot")
          and not os.path.exists("experiments/eval/v3/pilot"))

    try:
        dirty = "\n".join(l for l in subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True).splitlines()
            if not l.endswith(".DS_Store"))
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        remote = subprocess.check_output(["git", "branch", "-r", "--contains", head], text=True).strip()
        check("tracked files clean", not dirty, dirty[:200])
        check("HEAD is on a remote branch", bool(remote), f"{head[:7]} {remote}")
    except Exception as e:
        check("git state readable", False, str(e))

    print("\n".join(lines))
    print(f"submit with: TRAIN=$(sbatch --parsable --array=1-{n_tasks} {TRAIN_SH}); "
          f"sbatch --dependency=afterany:$TRAIN {EVAL_SH}")
    print("PREFLIGHT OK" if ok else "PREFLIGHT FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
