"""Closed-form multi-output ridge regression on the flattened history window.

Sanity check for the SGD-trained linear baseline: same inputs, same targets,
exact least-squares solution. The penalty is chosen on the validation split
only (normalised MSE), never on test/OOD data.

    python -m src.training.fit_ridge --config experiments/configs/v3/linear.yaml --run_name ridge

Writes a LinearBaseline-compatible checkpoint (with provenance) so the model
can be scored by src.evaluation.benchmark_eval --method point.
"""

import argparse
import json
import os

import numpy as np
import torch
import yaml

from src.data_loading.darus_dataset import build_datasets, make_loader
from src.models.linear import LinearBaseline
from src.training.train import git_state

ALPHAS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


def gram(ds, batch_size=4096):
    """Accumulate [X 1]^T [X 1] and [X 1]^T Y over all windows."""
    A = B = None
    for X, Y in make_loader(ds, batch_size):
        Xf = X.reshape(X.shape[0], -1).double()
        Xf = torch.cat([Xf, torch.ones(Xf.shape[0], 1, dtype=Xf.dtype)], dim=1)
        Yf = Y.reshape(Y.shape[0], -1).double()
        A = Xf.T @ Xf if A is None else A + Xf.T @ Xf
        B = Xf.T @ Yf if B is None else B + Xf.T @ Yf
    return A.numpy(), B.numpy()


def mse_of(W, ds, batch_size=4096):
    sse, n = 0.0, 0
    for X, Y in make_loader(ds, batch_size):
        Xf = np.concatenate([X.reshape(X.shape[0], -1).numpy(), np.ones((X.shape[0], 1))], 1)
        sse += ((Xf @ W - Y.reshape(Y.shape[0], -1).numpy()) ** 2).sum()
        n += Y.numel()
    return sse / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_name", default="ridge")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config["model"] = {"type": "linear"}
    config["logging"]["run_name"] = args.run_name
    out_dir = os.path.join(config["logging"]["save_dir"], args.run_name)
    if os.path.exists(os.path.join(out_dir, "best_model.pt")) and not args.overwrite:
        raise FileExistsError(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    datasets, state = build_datasets(config["data"])
    A, B = gram(datasets["train"])
    p = A.shape[0]
    reg = np.eye(p)
    reg[-1, -1] = 0.0                       # do not penalise the intercept
    scores = {}
    for a in ALPHAS:
        W = np.linalg.solve(A + a * A.trace() / p * reg, B)
        scores[a] = mse_of(W, datasets["val"])
        print(f"alpha={a:g}  val MSE={scores[a]:.6f}")
    best = min(scores, key=scores.get)
    W = np.linalg.solve(A + best * A.trace() / p * reg, B)

    ds = datasets["train"]
    model = LinearBaseline(ds.input_dim, state["history"], ds.target_dim, state["horizon"])
    model.net.weight.data = torch.tensor(W[:-1].T, dtype=torch.float32)
    model.net.bias.data = torch.tensor(W[-1], dtype=torch.float32)
    torch.save({"model_state": model.state_dict(), "config": config, "seed": None,
                "data_state": state, "git": git_state(), "best_epoch": None,
                "ridge_alpha_rel": best, "val_mse_by_alpha": scores}, os.path.join(out_dir, "best_model.pt"))
    with open(os.path.join(out_dir, "ridge_selection.json"), "w") as f:
        json.dump({"alpha_rel_selected": best, "val_mse": {str(k): v for k, v in scores.items()},
                   "note": "penalty = alpha_rel * trace(X^T X)/p; selected on validation MSE"}, f, indent=2)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"Selected alpha_rel={best:g}; saved {out_dir}")


if __name__ == "__main__":
    main()
