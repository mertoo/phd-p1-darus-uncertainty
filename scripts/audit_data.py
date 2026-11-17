"""Audit the processed DaRUS recordings and write a split manifest.

Reads every recording under data/processed/darus, checks time-axis integrity
(monotonicity, sampling interval, gaps, duplicates, missing values), hashes
each file, counts valid within-recording windows, and counts the windows the
legacy concatenating loader produced that straddle a recording join.

Usage:
    python -m scripts.audit_data [--base data/processed/darus] [--history 30 --horizon 30]

Outputs (tracked, small):
    experiments/manifests/split_manifest.csv     one row per recording
    experiments/manifests/data_audit.json        per-split summary
"""

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

from src.data_loading.darus_parser import SPLIT_DIRS, list_split_files

EXPECTED_COLUMNS = ["time", "n", "deltal", "deltar", "Vw", "alpha_x", "alpha_y",
                    "u", "v", "p", "r", "phi"]
TARGETS = ["u", "v", "p", "r", "phi"]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def audit_file(path, history, horizon):
    df = pd.read_csv(path)
    t = df["time"].to_numpy(dtype=float)
    dt = np.diff(t)
    L = len(df)
    row = {
        "file": os.path.basename(path),
        "sha256": sha256(path),
        "n_rows": L,
        "columns_ok": list(df.columns) == EXPECTED_COLUMNS,
        "n_missing": int(df.isna().sum().sum()),
        "time_start": float(t[0]),
        "time_end": float(t[-1]),
        "dt_median": float(np.median(dt)) if L > 1 else float("nan"),
        "dt_min": float(dt.min()) if L > 1 else float("nan"),
        "dt_max": float(dt.max()) if L > 1 else float("nan"),
        "n_nonmonotone": int((dt <= 0).sum()),
        "n_irregular_dt": int((np.abs(dt - np.median(dt)) > 1e-6).sum()) if L > 1 else 0,
        "n_duplicate_rows": int(df.duplicated().sum()),
        "valid_windows": max(0, L - history - horizon + 1),
    }
    for c in TARGETS + ["n", "Vw"]:
        row[f"{c}_min"] = float(df[c].min())
        row[f"{c}_max"] = float(df[c].max())
        row[f"{c}_mean"] = float(df[c].mean())
        row[f"{c}_std"] = float(df[c].std())
    # content fingerprint independent of file name, for cross-split duplicate checks
    row["content_hash"] = hashlib.sha256(
        np.ascontiguousarray(df[TARGETS].to_numpy()).tobytes()).hexdigest()
    return row


def legacy_window_counts(lengths, history, horizon):
    """Reproduce the legacy SequenceDataset count on the concatenation and
    count windows whose span [i, i+H+T) crosses a recording join."""
    total = int(sum(lengths))
    n_legacy = total - history - horizon            # legacy __len__ (off by one)
    starts = np.cumsum([0] + list(lengths[:-1]))
    ends = starts + np.asarray(lengths)             # exclusive
    idx = np.arange(max(n_legacy, 0))
    rec = np.searchsorted(ends, idx, side="right")  # recording containing window start
    crosses = idx + history + horizon > ends[rec]
    return n_legacy, int(crosses.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/processed/darus")
    ap.add_argument("--history", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--out_dir", default="experiments/manifests")
    args = ap.parse_args()

    rows, summary = [], {}
    for split in SPLIT_DIRS:
        files = list_split_files(args.base, split)
        split_rows = [audit_file(f, args.history, args.horizon) for f in files]
        for r in split_rows:
            r["split"] = split
        rows.extend(split_rows)

        lengths = [r["n_rows"] for r in split_rows]
        n_legacy, n_cross = legacy_window_counts(lengths, args.history, args.horizon)
        n_valid = sum(r["valid_windows"] for r in split_rows)
        summary[split] = {
            "n_recordings": len(split_rows),
            "n_rows": int(sum(lengths)),
            "duration_s": float(sum(r["time_end"] - r["time_start"] for r in split_rows)),
            "valid_windows": int(n_valid),
            "legacy_windows": int(n_legacy),
            "legacy_boundary_crossing_windows": n_cross,
            "legacy_boundary_crossing_fraction": n_cross / n_legacy if n_legacy > 0 else float("nan"),
            "dt_values": sorted({round(r["dt_median"], 6) for r in split_rows}),
            "all_regular": all(r["n_irregular_dt"] == 0 and r["n_nonmonotone"] == 0 for r in split_rows),
            "total_missing": int(sum(r["n_missing"] for r in split_rows)),
            "time_starts_at_zero": all(r["time_start"] == 0 for r in split_rows),
        }
        for c in TARGETS:
            summary[split][f"{c}_range"] = [min(r[f"{c}_min"] for r in split_rows),
                                            max(r[f"{c}_max"] for r in split_rows)]

    manifest = pd.DataFrame(rows)
    dup_names = manifest[manifest.duplicated("file", keep=False)]
    dup_content = manifest[manifest.duplicated("content_hash", keep=False)]
    summary["cross_checks"] = {
        "duplicate_file_names": dup_names[["split", "file"]].values.tolist(),
        "duplicate_contents": dup_content[["split", "file"]].values.tolist(),
        "all_columns_ok": bool(manifest["columns_ok"].all()),
    }
    summary["params"] = {"history": args.history, "horizon": args.horizon, "base": args.base}

    os.makedirs(args.out_dir, exist_ok=True)
    manifest.to_csv(os.path.join(args.out_dir, "split_manifest.csv"), index=False)
    with open(os.path.join(args.out_dir, "data_audit.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
