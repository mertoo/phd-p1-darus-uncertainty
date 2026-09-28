"""Predeclared time-feature decision (EXPERIMENT_PROTOCOL.md §3).

Reads ONLY best validation losses from train_log.json of the ablation runs
    <root>/<backbone>_<features>/seed<S>/train_log.json
for backbones {lstm, mlp} and features {legacy, no_time}. Test and OOD data
are never read.

Rule (fixed before any ablation run):
  L_b,f   = mean over available seeds of the best validation loss (standardised
            targets, so the loss is the channel-averaged normalised MSE and is
            comparable across feature sets).
  gain_b  = (L_b,no_time - L_b,legacy) / L_b,no_time   (> 0: `time` helps)
  borderline if any gain_b lies in [1%, 3%], or the backbones disagree about
            whether gain_b > 2%.
  Round 1 (seed 1): not borderline -> "legacy" if gain_b > 2% for BOTH
            backbones, else "no_time". Borderline -> "repeat" (train seeds 2, 3
            for all four configurations).
  Round 2 (seeds 1-3): "legacy" if gain_b > 2% for both backbones, otherwise
            "no_time" (the parsimonious set, since `time` encodes position
            within a recording rather than a physical state). A remaining
            disagreement is reported as a finding; no further seeds are run.
  One common feature set is used for every model in the primary comparison.
"""

import argparse
import glob
import json
import os

THRESH, BAND = 0.02, (0.01, 0.03)
BACKBONES, FEATS = ("lstm", "mlp"), ("legacy", "no_time")


def collect(root):
    losses = {}
    for bb in BACKBONES:
        for ft in FEATS:
            vals = {}
            for p in sorted(glob.glob(os.path.join(root, f"{bb}_{ft}", "seed*", "train_log.json"))):
                with open(p) as f:
                    log = json.load(f)
                vals[int(p.split("seed")[-1].split(os.sep)[0])] = float(log["best_val_loss"])
            losses[(bb, ft)] = vals
    return losses


def decide(losses):
    seed_sets = {k: set(v) for k, v in losses.items()}
    common = set.intersection(*seed_sets.values()) if seed_sets else set()
    if not common:
        raise ValueError(f"no seed present for all four configurations: {seed_sets}")
    if any(s != common for s in seed_sets.values()):
        raise ValueError(f"unequal seeds across configurations: {seed_sets}")
    seeds = sorted(common)
    rnd = 1 if seeds == [1] else 2
    if rnd == 2 and seeds != [1, 2, 3]:
        raise ValueError(f"round 2 requires seeds 1-3, found {seeds}")

    mean = {k: sum(v[s] for s in seeds) / len(seeds) for k, v in losses.items()}
    gain = {bb: (mean[(bb, "no_time")] - mean[(bb, "legacy")]) / mean[(bb, "no_time")] for bb in BACKBONES}
    helps = {bb: gain[bb] > THRESH for bb in BACKBONES}
    disagree = len(set(helps.values())) > 1
    borderline = disagree or any(BAND[0] <= g <= BAND[1] for g in gain.values())

    if rnd == 1 and borderline:
        decision = "repeat"
    else:
        decision = "legacy" if all(helps.values()) else "no_time"
    return {
        "round": rnd, "seeds": seeds, "mean_val_loss": {f"{b}_{f}": v for (b, f), v in mean.items()},
        "gain": gain, "time_helps": helps, "backbones_disagree": disagree,
        "borderline": borderline, "decision": decision,
        "rule": "legacy iff gain > 2% for both LSTM and MLP; borderline (1-3% or disagreement) -> one repeat round with seeds 2,3",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/runs/v3/ablation")
    ap.add_argument("--out", default="experiments/manifests/feature_decision.json")
    args = ap.parse_args()
    result = decide(collect(args.root))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
