import argparse
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import yaml

from src.uncertainty.deep_ensemble import predict_ensemble
from src.data_loading.darus_dataset import create_dataloaders
from src.models.lstm import LSTMSeq2Seq
from src.models.gru import GRUSeq2Seq
from src.models.mlp import MLP
from src.models.tcn import TCN
from src.models.linear import LinearBaseline
from src.models.naive import NaiveBaseline

# Must match P1 ordering exactly
TARGET_NAMES = ["u", "v", "p", "r", "phi"]


def build_member(config, input_dim, target_dim, history, horizon, device):
    """Instantiate a single ensemble member from config."""
    mcfg = config["model"]
    mtype = mcfg["type"].lower()

    if mtype == "lstm":
        return LSTMSeq2Seq(
            input_dim=input_dim,
            hidden_dim=mcfg["hidden_dim"],
            num_layers=mcfg["num_layers"],
            dropout=mcfg.get("dropout", 0.0),
            horizon=horizon,
            target_dim=target_dim,
        ).to(device)
    elif mtype == "gru":
        return GRUSeq2Seq(
            input_dim=input_dim,
            hidden_dim=mcfg["hidden_dim"],
            num_layers=mcfg["num_layers"],
            dropout=mcfg.get("dropout", 0.0),
            horizon=horizon,
            target_dim=target_dim,
        ).to(device)
    elif mtype == "mlp":
        return MLP(
            input_dim=input_dim,
            history=history,
            horizon=horizon,
            output_dim=target_dim,
            hidden_dim=mcfg.get("hidden_dim", 256),
        ).to(device)
    elif mtype == "tcn":
        return TCN(
            input_dim=input_dim,
            target_dim=target_dim,
            history=history,
            horizon=horizon,
            num_channels=mcfg.get("num_channels", [64, 64, 64]),
            kernel_size=mcfg.get("kernel_size", 3),
            dropout=mcfg.get("dropout", 0.1),
        ).to(device)
    elif mtype == "linear":
        return LinearBaseline(
            input_dim=input_dim, history=history,
            output_dim=target_dim, horizon=horizon,
        ).to(device)
    elif mtype == "naive":
        return NaiveBaseline(output_dim=target_dim, horizon=horizon).to(device)
    else:
        raise ValueError(f"Unknown model type: {mtype}")


def load_ensemble(model_dir, config, device, input_dim, target_dim):
    models = []

    history = config["data"]["history"]
    horizon = config["data"]["horizon"]

    for sub in sorted(os.listdir(model_dir)):
        ckpt_path = os.path.join(model_dir, sub, "best_model.pt")

        if not os.path.exists(ckpt_path):
            print(f"⚠️ Skipping missing: {ckpt_path}")
            continue

        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get("model_state") or ckpt.get("model_state_dict") or ckpt

        model = build_member(config, input_dim, target_dim, history, horizon, device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)

    print(f"Loaded {len(models)} ensemble members")
    return models


def evaluate_ensemble(models, loader, device):
    """Run all ensemble members over the full loader.

    Returns:
        means: (N, H, D)  – mean prediction across members
        stds:  (N, H, D)  – std across members (epistemic uncertainty)
        ys:    (N, H, D)  – ground truth
    """
    all_means, all_stds, all_ys = [], [], []

    for x, y in loader:
        x = x.to(device)
        mean, std = predict_ensemble(models, x, device)  # (B, H, D) each
        all_means.append(mean)
        all_stds.append(std)
        all_ys.append(y.numpy())

    means = np.concatenate(all_means, axis=0)  # (N, H, D)
    stds  = np.concatenate(all_stds,  axis=0)  # (N, H, D)
    ys    = np.concatenate(all_ys,    axis=0)  # (N, H, D)
    return means, stds, ys


def print_metrics(means, stds, ys, split_label):
    """Print aggregate RMSE and ±2σ coverage to stdout."""
    rmse_overall = float(np.sqrt(((means - ys) ** 2).mean()))
    rmse_per_dof = np.sqrt(((means - ys) ** 2).mean(axis=(0, 1)))  # (D,)

    in_interval = np.abs(ys - means) <= 2.0 * stds
    coverage_overall = float(in_interval.mean())
    coverage_per_dof = in_interval.mean(axis=(0, 1))  # (D,)

    print(f"\n{'='*50}")
    print(f"Deep Ensemble metrics – {split_label.upper()}")
    print(f"{'='*50}")
    print(f"RMSE (overall):    {rmse_overall:.6f}")
    print(f"Coverage ±2σ:      {coverage_overall:.3f}  (nominal 0.954)")
    print(f"{'─'*50}")
    print(f"{'DoF':<6}  {'RMSE':>10}  {'Coverage ±2σ':>14}")
    for i, name in enumerate(TARGET_NAMES):
        print(f"{name:<6}  {rmse_per_dof[i]:>10.6f}  {coverage_per_dof[i]:>14.3f}")
    print(f"{'='*50}\n")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["test", "ood"])
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    history = config["data"]["history"]
    horizon = config["data"]["horizon"]
    batch_size = config["data"]["batch_size"]

    train_loader, val_loader, test_loader, ood_loader = create_dataloaders(
        history=history,
        horizon=horizon,
        batch_size=batch_size,
    )

    input_dim  = train_loader.dataset.X.shape[-1]
    target_dim = train_loader.dataset.Y.shape[-1]

    loader = test_loader if args.split == "test" else ood_loader

    models = load_ensemble(args.ensemble_dir, config, device, input_dim, target_dim)

    # ── Full-dataset evaluation ─────────────────────────────────────────────
    print(f"Evaluating ensemble over full {args.split} set…")
    means, stds, ys = evaluate_ensemble(models, loader, device)
    print_metrics(means, stds, ys, args.split)

    # ── Plots: single example from first batch ───────────────────────────────
    out_dir = os.path.join(args.ensemble_dir, "plots", args.split)
    os.makedirs(out_dir, exist_ok=True)

    t = np.arange(horizon)
    for i, name in enumerate(TARGET_NAMES):
        plt.figure(figsize=(10, 5))
        plt.plot(means[0, :, i], label="Ensemble Mean", color="black")
        plt.fill_between(
            t,
            means[0, :, i] - 2 * stds[0, :, i],
            means[0, :, i] + 2 * stds[0, :, i],
            alpha=0.3,
            label="±2σ interval",
        )
        plt.plot(ys[0, :, i], "--", label="Ground Truth", linewidth=2)
        plt.xlabel("Prediction Step")
        plt.ylabel(name)
        plt.title(f"Deep Ensemble – {name} ({args.split.upper()})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, f"ensemble_{name}_{args.split}.png"),
            dpi=300,
        )
        plt.close()

    print(f"Ensemble plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
