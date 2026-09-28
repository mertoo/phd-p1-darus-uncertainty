"""Unified training entry point for deterministic and Gaussian models.

    python -m src.training.train --config CONFIG [--seed S] [--run_name NAME] [--overwrite]

Writes to <logging.save_dir>/<run_name>/ :
    best_model.pt    model_state + provenance (config, seed, data state incl.
                     split files/hash and scalers, best epoch, git commit)
    config.yaml      resolved config actually used
    train_log.json   per-epoch losses
An existing run directory is never overwritten unless --overwrite is given.
"""

import argparse
import copy
import json
import math
import os
import platform
import random
import resource
import sys
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.data_loading.darus_dataset import build_datasets, make_loader
from src.models.factory import build_model, count_parameters, is_gaussian

DEFAULT_TRAINING = {"epochs": 20, "lr": 1e-3, "batch_size": 256, "weight_decay": 0.0,
                    "patience": None, "device": "auto", "deterministic": False}


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def git_state():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                             text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def peak_rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r / 1e6   # bytes on macOS, KiB on Linux


def gaussian_nll(mu, logvar, y):
    """Mean elementwise Gaussian NLL including the 0.5*log(2*pi) constant."""
    return 0.5 * (math.log(2 * math.pi) + logvar + (y - mu) ** 2 * torch.exp(-logvar)).mean()


def loss_fn(model_cfg):
    if is_gaussian(model_cfg):
        return lambda out, y: gaussian_nll(out[0], out[1], y)
    mse = nn.MSELoss()
    return lambda out, y: mse(out, y)


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, count = 0.0, 0
    with torch.set_grad_enabled(train):
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            loss = criterion(model(X), Y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * X.size(0)   # all losses are per-element means
            count += X.size(0)
    return total / count


def resolve_config(config, seed=None, run_name=None, epochs=None, patience="keep"):
    cfg = copy.deepcopy(config)
    tr = {**DEFAULT_TRAINING, **cfg.get("training", {})}
    # batch_size historically lived under data: or training:; accept either, not both
    if "batch_size" in cfg.get("data", {}) and "batch_size" in cfg.get("training", {}):
        if cfg["data"]["batch_size"] != cfg["training"]["batch_size"]:
            raise ValueError("Conflicting batch_size in data: and training:")
    tr["batch_size"] = cfg.get("training", {}).get("batch_size", cfg.get("data", {}).get("batch_size", 256))
    cfg["data"].pop("batch_size", None)
    if seed is not None:
        tr["seed"] = int(seed)
    if epochs is not None:
        tr["epochs"] = int(epochs)
    if patience != "keep":
        tr["patience"] = patience
    if "seed" not in tr:
        raise ValueError("training.seed must be set in the config or via --seed")
    unknown = set(tr) - set(DEFAULT_TRAINING) - {"seed"}
    if unknown:
        raise KeyError(f"Unknown training config keys: {sorted(unknown)}")
    cfg["training"] = tr
    if run_name is not None:
        cfg["logging"]["run_name"] = run_name
    return cfg


def train(config, overwrite=False):
    tr = config["training"]
    out_dir = os.path.join(config["logging"]["save_dir"], config["logging"]["run_name"])
    ckpt_path = os.path.join(out_dir, "best_model.pt")
    if os.path.exists(ckpt_path) and not overwrite:
        raise FileExistsError(f"{ckpt_path} exists; choose a new run_name or pass --overwrite")
    os.makedirs(out_dir, exist_ok=True)

    set_seed(tr["seed"], tr["deterministic"])
    device = ("cuda" if torch.cuda.is_available() else "cpu") if tr["device"] == "auto" else tr["device"]

    datasets, data_state = build_datasets(config["data"])
    train_ds = datasets["train"]
    history, horizon = data_state["history"], data_state["horizon"]
    model = build_model(config["model"], train_ds.input_dim, train_ds.target_dim, history, horizon)
    model_type = config["model"]["type"].lower()
    if model_type == "naive" and data_state["x_scaler"]:
        xs, ys = data_state["x_scaler"], data_state["y_scaler"]
        model.set_scaling(xs["mean"][-5:], xs["std"][-5:], ys["mean"], ys["std"])
    model = model.to(device)

    meta = {
        "config": config,
        "seed": tr["seed"],
        "data_state": data_state,
        "git": git_state(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__},
        "n_params": count_parameters(model),
        "device": device,
    }
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    if model_type == "naive":
        torch.save({"model_state": model.state_dict(), **meta, "best_epoch": 0}, ckpt_path)
        print(f"Naive baseline saved to {ckpt_path}")
        return ckpt_path

    g = tr["seed"]
    train_loader = make_loader(train_ds, tr["batch_size"], shuffle=True, seed=g)
    val_loader = make_loader(datasets["val"], tr["batch_size"])
    criterion = loss_fn(config["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tr["lr"]),
                                 weight_decay=float(tr["weight_decay"]))

    log, best_val, best_epoch, since_best = [], float("inf"), 0, 0
    t0 = time.time()
    for epoch in range(1, int(tr["epochs"]) + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)
        log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                    "elapsed_s": time.time() - t0,
                    "gpu_max_mem_gb": (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else None,
                    "peak_rss_gb": peak_rss_gb()})
        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            raise FloatingPointError(f"non-finite loss at epoch {epoch}: train {train_loss}, val {val_loss}")
        print(f"Epoch {epoch}/{tr['epochs']}  train {train_loss:.6f}  val {val_loss:.6f}", flush=True)
        if val_loss < best_val:
            best_val, best_epoch, since_best = val_loss, epoch, 0
            torch.save({"model_state": model.state_dict(), **meta,
                        "best_epoch": best_epoch, "best_val_loss": best_val}, ckpt_path)
        else:
            since_best += 1
            if tr["patience"] and since_best >= int(tr["patience"]):
                print(f"Early stopping at epoch {epoch} (patience {tr['patience']})")
                break

    stopped = "early_stopping" if (tr["patience"] and since_best >= int(tr["patience"])) else "epoch_budget"
    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        n_ep = len(log)
        json.dump({"epochs": log, "best_epoch": best_epoch, "best_val_loss": best_val,
                   "n_train_windows": len(train_ds), "n_val_windows": len(datasets["val"]),
                   "sec_per_epoch": log[-1]["elapsed_s"] / n_ep if n_ep else None,
                   "train_windows_per_sec": len(train_ds) * n_ep / log[-1]["elapsed_s"] if n_ep else None,
                   "stopped_by": stopped, "seed": tr["seed"],
                   "split_hash": data_state["split_hash"], "git": meta["git"]}, f, indent=2)
    print(f"Best val loss {best_val:.6f} at epoch {best_epoch} ({stopped}); saved {ckpt_path}")
    return ckpt_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--epochs", type=int, default=None, help="override training.epochs (timing pilot)")
    ap.add_argument("--no_patience", action="store_true", help="disable early stopping (timing pilot)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    cfg = resolve_config(config, args.seed, args.run_name, args.epochs,
                         None if args.no_patience else "keep")
    train(cfg, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
