"""Evaluate one UQ method on calibration / ID-test / OOD-test and save
structured artifacts. Every coverage/width number comes from an `Intervals`
object built once per (variant, level).

    python -m src.evaluation.benchmark_eval --method point    --runs RUN_DIR               --out OUT
    python -m src.evaluation.benchmark_eval --method ensemble --runs RUN_DIR/model_*       --out OUT
    python -m src.evaluation.benchmark_eval --method mc_dropout --runs RUN_DIR --passes 200 --out OUT
    python -m src.evaluation.benchmark_eval --method gaussian --runs RUN_DIR               --out OUT

Legacy checkpoints without provenance need --legacy_config CONFIG; they are
evaluated with recording-aware windows on the historical 12-feature,
unscaled inputs, calibrated on the validation split (flagged in the output).

Outputs in OUT/:
    metrics.json          point + interval metrics per split/variant/level
    predictions_<split>.npz   y, mean, spread (physical units)
    windows_<split>.csv   window metadata (recording, segment, start_row, start_time)
    per_window_<split>_<variant>_<level>.csv  per-window sums for recording bootstrap
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
import yaml

from src.data_loading.darus_dataset import Standardizer, build_datasets, make_loader
from src.evaluation.uq_metrics import interval_metrics, per_window_frame, point_metrics
from src.models.factory import build_model
from src.uncertainty.intervals import (VARIANTS, conformal_intervals,
                                       raw_gaussian_intervals, scaled_spread_intervals)

LEVELS = (0.5, 0.8, 0.9, 0.95)          # predeclared nominal levels
EVAL_SPLITS = ("test", "ood_test")


# ---------------------------------------------------------------- loading ---
def load_run(run_dir, legacy_config=None, device="cpu"):
    ckpt = torch.load(os.path.join(run_dir, "best_model.pt"), map_location=device, weights_only=False)
    if "data_state" in ckpt:
        cfg, ds_state = ckpt["config"], ckpt["data_state"]
        data_cfg = {"base_path": ds_state["base_path"], "history": ds_state["history"],
                    "horizon": ds_state["horizon"], "stride": ds_state["stride"],
                    "features": ds_state["features"], "normalize": ds_state["x_scaler"] is not None}
        if "cal" in ds_state["split_files"]:
            data_cfg["calibration"] = {"files": ds_state["split_files"]["cal"]}
        legacy = False
    else:
        if legacy_config is None:
            raise ValueError(f"{run_dir}: checkpoint has no provenance; pass --legacy_config")
        with open(legacy_config) as f:
            cfg = yaml.safe_load(f)
        data_cfg = {"base_path": cfg["data"].get("base_path", "data/processed/darus"),
                    "history": cfg["data"]["history"], "horizon": cfg["data"]["horizon"]}
        ds_state = None
        legacy = True
    return ckpt, cfg, data_cfg, ds_state, legacy


def check_scalers(new_state, saved_state):
    for key in ("x_scaler", "y_scaler"):
        a, b = new_state[key], saved_state[key]
        if (a is None) != (b is None):
            raise RuntimeError(f"{key} presence differs from checkpoint")
        if a is not None:
            np.testing.assert_allclose(a["mean"], b["mean"], rtol=1e-9)
            np.testing.assert_allclose(a["std"], b["std"], rtol=1e-9)
    if new_state["split_files"] != saved_state["split_files"]:
        raise RuntimeError("Split membership differs from the checkpoint's recorded split")


def instantiate(ckpt, cfg, ds, history, horizon, device):
    model = build_model(cfg["model"], ds.input_dim, ds.target_dim, history, horizon)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device)


# ------------------------------------------------------------- inference ---
@torch.no_grad()
def predict(model, loader, device, mode="point", passes=1, seed=0):
    """Returns (mean, spread) in scaled units; spread is None for point models."""
    means, spreads = [], []
    if mode == "mc_dropout":
        model.train()                       # dropout active (only nn.LSTM inter-layer dropout here)
        gen_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
    else:
        model.eval()
    for X, _ in loader:
        X = X.to(device)
        if mode == "gaussian":
            mu, logvar = model(X)
            means.append(mu.cpu().numpy())
            spreads.append(torch.exp(0.5 * logvar).cpu().numpy())
        elif mode == "mc_dropout":
            s = torch.stack([model(X) for _ in range(passes)]).cpu().numpy()
            means.append(s.mean(0))
            spreads.append(s.std(0, ddof=1))
        else:
            means.append(model(X).cpu().numpy())
    if mode == "mc_dropout":
        torch.random.set_rng_state(gen_state)
    mean = np.concatenate(means).astype(np.float64)
    spread = np.concatenate(spreads).astype(np.float64) if spreads else None
    return mean, spread


def to_physical(mean, spread, y_scaler):
    if y_scaler is None:
        return mean, spread
    return y_scaler.inverse_transform(mean), (None if spread is None else y_scaler.inverse_scale(spread))


# ------------------------------------------------------------------ main ---
def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["point", "ensemble", "mc_dropout", "gaussian"])
    ap.add_argument("--runs", nargs="+", required=True, help="run dir(s); globs allowed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--legacy_config", default=None)
    ap.add_argument("--passes", type=int, default=200)
    ap.add_argument("--mc_seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--floor_rel", type=float, default=1e-3)
    ap.add_argument("--levels", type=float, nargs="+", default=list(LEVELS))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if os.path.exists(os.path.join(args.out, "metrics.json")) and not args.overwrite:
        raise FileExistsError(f"{args.out}/metrics.json exists; use a new --out or --overwrite")
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dirs = sorted({d for pat in args.runs for d in glob.glob(pat)})
    if not run_dirs:
        raise FileNotFoundError(args.runs)
    if args.method == "ensemble" and len(run_dirs) < 2:
        raise ValueError("ensemble needs >= 2 member runs")
    if args.method != "ensemble" and len(run_dirs) != 1:
        raise ValueError(f"{args.method} takes exactly one run dir")

    first = load_run(run_dirs[0], args.legacy_config, device)
    _, cfg0, data_cfg, ds_state0, legacy = first
    datasets, new_state = build_datasets(data_cfg)
    if ds_state0 is not None:
        check_scalers(new_state, ds_state0)
    y_scaler = Standardizer.from_dict(new_state["y_scaler"]) if new_state["y_scaler"] else None
    cal_split = "cal" if "cal" in datasets else "val"
    splits = (cal_split,) + EVAL_SPLITS
    H, T = new_state["history"], new_state["horizon"]
    mode = {"point": "point", "ensemble": "point", "mc_dropout": "mc_dropout", "gaussian": "gaussian"}[args.method]

    members = []
    for rd in run_dirs:
        ckpt, cfg, dcfg, st, _ = load_run(rd, args.legacy_config, device)
        if dcfg != data_cfg:
            raise RuntimeError(f"{rd}: data config differs from {run_dirs[0]}")
        members.append({"dir": rd, "model": instantiate(ckpt, cfg, datasets["train"], H, T, device),
                        "seed": ckpt.get("seed"), "best_epoch": ckpt.get("best_epoch"),
                        "git": ckpt.get("git")})

    preds = {}
    for sp in splits:
        loader = make_loader(datasets[sp], args.batch_size)
        y = datasets[sp].targets_physical()
        if args.method == "ensemble":
            outs = [to_physical(*predict(m["model"], loader, device), y_scaler)[0] for m in members]
            stack = np.stack(outs)
            mean, spread = stack.mean(0), stack.std(0, ddof=1)
        else:
            mean, spread = to_physical(*predict(members[0]["model"], loader, device, mode,
                                                args.passes, args.mc_seed), y_scaler)
        preds[sp] = {"y": y, "mean": mean, "spread": spread}
        # float32 on disk (metrics below use the float64 arrays in memory)
        arrays = {"y": y, "mean": mean, **({"spread": spread} if spread is not None else {})}
        np.savez_compressed(os.path.join(args.out, f"predictions_{sp}.npz"),
                            **{k: v.astype(np.float32) for k, v in arrays.items()})
        datasets[sp].meta.to_csv(os.path.join(args.out, f"windows_{sp}.csv"))

    scale = y_scaler.std if y_scaler is not None else None
    if scale is None:  # dimensionless reference for legacy runs: train-target std
        tr = datasets["train"].targets_physical().reshape(-1, datasets["train"].target_dim)
        scale = tr.std(axis=0)

    cal = preds[cal_split]
    results = {"point": {}, "intervals": []}
    for sp in EVAL_SPLITS:
        results["point"][sp] = point_metrics(preds[sp]["y"], preds[sp]["mean"], scale=scale)

    for level in args.levels:
        for sp in EVAL_SPLITS:
            p = preds[sp]
            ivs = []
            if args.method in ("point",):
                for v in VARIANTS:
                    ivs.append(conformal_intervals(cal["mean"], cal["y"], p["mean"], level, v))
            else:
                ivs.append(raw_gaussian_intervals(p["mean"], p["spread"], level, args.method))
                for v in VARIANTS:
                    ivs.append(scaled_spread_intervals(cal["mean"], cal["spread"], cal["y"],
                                                       p["mean"], p["spread"], level, v,
                                                       args.method, args.floor_rel))
                    # plain residual conformal on the method's mean, for comparison
                    ivs.append(conformal_intervals(cal["mean"], cal["y"], p["mean"], level, v,
                                                   method=f"{args.method}+conformal"))
            for iv in ivs:
                m = interval_metrics(p["y"], iv, scale=scale)
                m.update({"split": sp, "calibration_split": cal_split, "info": iv.info})
                results["intervals"].append(m)
                if level == 0.9:
                    tag = f"{iv.method}_{iv.variant}".replace("+", "_")
                    per_window_frame(p["y"], p["mean"], iv, datasets[sp].meta).to_csv(
                        os.path.join(args.out, f"per_window_{sp}_{tag}_{int(level*100)}.csv"), index=False)

    results["provenance"] = {
        "method": args.method,
        "runs": [{k: v for k, v in m.items() if k != "model"} for m in members],
        "legacy_checkpoint": legacy,
        "calibration_split": cal_split,
        "calibration_note": ("validation split reused for model selection and calibration"
                             if cal_split == "val" else "held-out calibration recordings"),
        "data_state": {k: v for k, v in new_state.items() if k not in ("x_scaler", "y_scaler")},
        "levels": args.levels,
        "mc_passes": args.passes if args.method == "mc_dropout" else None,
        "mc_seed": args.mc_seed if args.method == "mc_dropout" else None,
        "spread_convention": "sample std, ddof=1" if args.method in ("ensemble", "mc_dropout") else None,
        "floor_rel": args.floor_rel,
        "scale_for_normalised_metrics": np.asarray(scale).tolist(),
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(jsonable(results), f, indent=1)
    print(f"Wrote {args.out}/metrics.json")


if __name__ == "__main__":
    main()
