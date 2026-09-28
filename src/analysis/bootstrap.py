"""Recording-level bootstrap confidence intervals and paired method differences.

    python -m src.analysis.bootstrap --spec SPEC.yaml --out OUT_DIR

SPEC.yaml:
    level: 90                       # per-window files are written at this level
    n_boot: 2000
    seed: 0
    ci: 0.95
    splits: [test, ood_test]
    methods:                        # name -> interval tag + evaluation dirs (one per repeat)
      conformal_lstm: {tag: conformal_horizon_channel, evals: [experiments/eval/v3/lstm_single_rep*]}
      ensemble_lstm:  {tag: ensemble_horizon_channel,  evals: [experiments/eval/v3/lstm_ens_rep*]}
    pairs:                          # paired differences A - B on the same resamples
      - [ensemble_lstm, conformal_lstm]

Method:
  * Windows overlap heavily, so the resampling unit is the recording. Per-window
    sums (squared error, inside counts, width, interval score) are first summed
    per recording; each bootstrap draw resamples recordings with replacement
    and recomputes every statistic from the resampled sums.
  * All methods and repeats share the same draws, so differences are paired on
    identical recordings. Methods must have been evaluated on identical windows.
  * With several repeats (independent training seeds), each draw averages the
    statistic over repeats. Between-seed variability is reported separately as
    the SD of the full-data statistic across repeats; it is not folded into the
    recording-level CI.
  * Output: summary_<split>.csv (method x statistic: estimate, CI, seed SD) and
    paired_<split>.csv (pair x statistic: difference, CI, share of draws > 0).
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import yaml

TARGETS = ["u", "v", "p", "r", "phi"]
KINDS = ("sse", "inside", "width", "iscore")


def load_eval(eval_dir, split, tag, level):
    path = os.path.join(eval_dir, f"per_window_{split}_{tag}_{level}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    with open(os.path.join(eval_dir, "metrics.json")) as f:
        scale = np.asarray(json.load(f)["provenance"]["scale_for_normalised_metrics"], float)
    return frame, scale


def recording_sums(frame, n_ch):
    """Per-recording sums; returns (recordings, sums (R, K), window key)."""
    cols = [f"{k}_{d}" for k in KINDS for d in range(n_ch)]
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(f"per-window file lacks {missing}")
    frame = frame.sort_values(["recording", "segment", "start_row"])
    key = pd.util.hash_pandas_object(frame[["recording", "segment", "start_row"]], index=False).sum()
    g = frame.groupby("recording", sort=True)
    sums = g[cols].sum()
    n_scalar = g["n_steps"].sum().to_numpy(float)          # scalar predictions per channel
    return sums.index.tolist(), np.column_stack([sums.to_numpy(float), n_scalar]), int(key)


def statistics(S, scale):
    """
    S: (..., 4*D + 1) summed [sse_d, inside_d, width_d, iscore_d, n_scalar_per_channel].
    Returns dict of arrays with leading shape S.shape[:-1].
    """
    D = len(scale)
    n = S[..., -1:]
    blocks = {k: S[..., i * D:(i + 1) * D] / n for i, k in enumerate(KINDS)}
    out = {}
    rmse = np.sqrt(blocks["sse"])
    for d, name in enumerate(TARGETS[:D]):
        out[f"rmse_{name}"] = rmse[..., d]
        out[f"coverage_{name}"] = blocks["inside"][..., d]
        out[f"width_{name}"] = blocks["width"][..., d]
        out[f"iscore_{name}"] = blocks["iscore"][..., d]
    out["coverage"] = blocks["inside"].mean(-1)            # equal counts per channel
    out["rmse_norm_mean"] = (rmse / scale).mean(-1)
    out["width_norm_mean"] = (blocks["width"] / scale).mean(-1)
    out["iscore_norm_mean"] = (blocks["iscore"] / scale).mean(-1)
    return out


def analyse(spec, split, method_inputs, rng_seed, n_boot, ci):
    """
    method_inputs: {method: [(recordings, sums (R,K), key, scale), ...one per repeat]}
    """
    ref_recs, ref_key, ref_scale = None, None, None
    for m, reps in method_inputs.items():
        for recs, _, key, scale in reps:
            if ref_recs is None:
                ref_recs, ref_key, ref_scale = recs, key, scale
            if recs != ref_recs or key != ref_key:
                raise ValueError(f"{m}: evaluated windows differ from the other methods on {split}; cannot pair")
            if not np.allclose(scale, ref_scale, rtol=1e-9):
                raise ValueError(f"{m}: normalisation scale differs; runs used different training data")

    R = len(ref_recs)
    rng = np.random.default_rng(rng_seed)
    W = np.stack([np.bincount(rng.integers(0, R, R), minlength=R) for _ in range(n_boot)]).astype(float)
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2

    est, boot, seed_sd, nrep = {}, {}, {}, {}
    for m, reps in method_inputs.items():
        full = [statistics(A.sum(0), ref_scale) for _, A, _, _ in reps]
        drawn = [statistics(W @ A, ref_scale) for _, A, _, _ in reps]
        est[m] = {k: float(np.mean([f[k] for f in full])) for k in full[0]}
        seed_sd[m] = {k: (float(np.std([f[k] for f in full], ddof=1)) if len(full) > 1 else np.nan) for k in full[0]}
        boot[m] = {k: np.mean([d[k] for d in drawn], axis=0) for k in drawn[0]}
        nrep[m] = len(reps)

    rows = []
    for m in method_inputs:
        for k, v in est[m].items():
            lo, hi = np.quantile(boot[m][k], [lo_q, hi_q])
            rows.append({"split": split, "method": m, "statistic": k, "estimate": v,
                         "ci_low": lo, "ci_high": hi, "seed_sd": seed_sd[m][k],
                         "n_repeats": nrep[m], "n_recordings": R})
    prow = []
    for a, b in spec.get("pairs", []):
        for k in est[a]:
            diff = boot[a][k] - boot[b][k]
            lo, hi = np.quantile(diff, [lo_q, hi_q])
            prow.append({"split": split, "method_a": a, "method_b": b, "statistic": k,
                         "difference": est[a][k] - est[b][k], "ci_low": lo, "ci_high": hi,
                         "share_draws_positive": float((diff > 0).mean()), "n_recordings": R})
    pair_cols = ["split", "method_a", "method_b", "statistic", "difference", "ci_low", "ci_high",
                 "share_draws_positive", "n_recordings"]
    return pd.DataFrame(rows), pd.DataFrame(prow, columns=pair_cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.spec) as f:
        spec = yaml.safe_load(f)
    level = int(spec.get("level", 90))
    os.makedirs(args.out, exist_ok=True)

    for split in spec.get("splits", ["test", "ood_test"]):
        inputs = {}
        for m, mspec in spec["methods"].items():
            dirs = sorted({d for pat in mspec["evals"] for d in glob.glob(pat)})
            if not dirs:
                raise FileNotFoundError(f"{m}: no eval dirs match {mspec['evals']}")
            reps = []
            for d in dirs:
                frame, scale = load_eval(d, split, mspec["tag"], level)
                recs, sums, key = recording_sums(frame, len(scale))
                reps.append((recs, sums, key, scale))
            inputs[m] = reps
        summary, paired = analyse(spec, split, inputs, int(spec.get("seed", 0)),
                                  int(spec.get("n_boot", 2000)), float(spec.get("ci", 0.95)))
        summary.to_csv(os.path.join(args.out, f"summary_{split}.csv"), index=False)
        paired.to_csv(os.path.join(args.out, f"paired_{split}.csv"), index=False)
        print(f"{split}: {len(inputs)} methods, {summary['n_recordings'].iloc[0]} recordings")
    with open(os.path.join(args.out, "spec_used.yaml"), "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False)


if __name__ == "__main__":
    main()
