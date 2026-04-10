import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from src.uncertainty.mc_dropout import mc_dropout_predict
from src.data_loading.darus_dataset import create_dataloaders
from src.models.lstm import LSTMSeq2Seq


TARGET_NAMES = ["u", "v", "p", "r", "phi"]


def infer_config_from_model_dir(model_dir):
    name = os.path.basename(model_dir)
    if "lstm" in name:
        return "p1_lstm_seq2seq.yaml"
    if "gru" in name:
        return "p1_gru_seq2seq.yaml"
    if "mlp" in name:
        return "p1_mlp.yaml"
    if "tcn" in name:
        return "p1_tcn.yaml"
    if "linear" in name:
        return "p1_linear.yaml"
    if "naive" in name:
        return "p1_naive.yaml"
    raise ValueError(f"Cannot infer config from model_dir: {model_dir}")


def evaluate_mc_dropout(model, loader, device, n_samples):
    """Run MC Dropout over the full loader.

    For each batch, performs n_samples stochastic forward passes (dropout
    active) and accumulates the per-sample mean and std predictions.

    Returns:
        means: (N, H, D)  – mean prediction across MC samples
        stds:  (N, H, D)  – std across MC samples (epistemic uncertainty)
        ys:    (N, H, D)  – ground truth
    """
    model.train()  # keep dropout active throughout

    all_means, all_stds, all_ys = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            # n_samples forward passes for this batch
            batch_preds = []
            for _ in range(n_samples):
                out = model(x)
                batch_preds.append(out.cpu().numpy())

            batch_preds = np.stack(batch_preds, axis=0)   # (S, B, H, D)
            all_means.append(batch_preds.mean(axis=0))    # (B, H, D)
            all_stds.append(batch_preds.std(axis=0))      # (B, H, D)
            all_ys.append(y.numpy())                       # (B, H, D)

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
    print(f"MC Dropout metrics – {split_label.upper()}")
    print(f"{'='*50}")
    print(f"RMSE (overall):    {rmse_overall:.6f}")
    print(f"Coverage ±2σ:      {coverage_overall:.3f}  (nominal 0.954)")
    print(f"{'─'*50}")
    print(f"{'DoF':<6}  {'RMSE':>10}  {'Coverage ±2σ':>14}")
    for i, name in enumerate(TARGET_NAMES):
        print(f"{name:<6}  {rmse_per_dof[i]:>10.6f}  {coverage_per_dof[i]:>14.3f}")
    print(f"{'='*50}\n")


def plot_mc_with_uncertainty(mean, std, truth, save_dir):
    """Plot a single example: mean prediction ± 2σ band vs ground truth."""
    T, D = mean.shape
    t = np.arange(T)

    for i, name in enumerate(TARGET_NAMES):
        lower = mean[:, i] - 2 * std[:, i]
        upper = mean[:, i] + 2 * std[:, i]

        plt.figure(figsize=(7, 4))
        plt.fill_between(t, lower, upper, alpha=0.3, label="±2σ interval")
        plt.plot(t, mean[:, i], linewidth=2, label="MC Mean")
        plt.plot(t, truth[:, i], linestyle="--", linewidth=2, label="Ground Truth")
        plt.xlabel("Prediction step")
        plt.ylabel(name)
        plt.title(f"MC Dropout – {name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"mc_{name}.png"), dpi=300)
        plt.close()


def plot_epistemic_growth(std, save_dir):
    """Plot std (epistemic uncertainty) growth over the forecast horizon."""
    T, D = std.shape
    t = np.arange(T)

    for i, name in enumerate(TARGET_NAMES):
        plt.figure(figsize=(6, 4))
        plt.plot(t, std[:, i], linewidth=2)
        plt.xlabel("Prediction step")
        plt.ylabel("Standard deviation")
        plt.yscale("log")
        plt.title(f"Epistemic Uncertainty Growth – {name}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"mc_{name}_std.png"), dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--split", type=str, default="test", choices=["test", "ood"])
    args = parser.parse_args()

    ckpt_path = os.path.join(args.model_dir, "best_model.pt")
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    config_name = infer_config_from_model_dir(args.model_dir)
    config_path = f"experiments/configs/{config_name}"
    print(f"Using config: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    history = config["data"]["history"]
    horizon = config["data"]["horizon"]
    batch_size = config["data"]["batch_size"]

    train_loader, val_loader, test_loader, ood_loader = create_dataloaders(
        history, horizon, batch_size
    )

    input_dim  = train_loader.dataset.X.shape[-1]
    target_dim = train_loader.dataset.Y.shape[-1]

    model = LSTMSeq2Seq(
        input_dim=input_dim,
        target_dim=target_dim,
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        horizon=horizon,
    )
    model.load_state_dict(ckpt["model_state"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Using device: {device}")

    loader = ood_loader if args.split == "ood" else test_loader

    # ── Full-dataset evaluation ─────────────────────────────────────────────
    print(f"Running MC Dropout ({args.n_samples} samples) over full {args.split} set…")
    means, stds, ys = evaluate_mc_dropout(model, loader, device, args.n_samples)
    print_metrics(means, stds, ys, args.split)

    # ── Plots: single example from first batch ───────────────────────────────
    save_dir = os.path.join(args.model_dir, "mc_dropout_plots", args.split)
    os.makedirs(save_dir, exist_ok=True)

    plot_mc_with_uncertainty(means[0], stds[0], ys[0], save_dir)
    plot_epistemic_growth(stds[0], save_dir)

    print("✅ MC Dropout evaluation complete.")
    print(f"Plots saved to: {save_dir}")


if __name__ == "__main__":
    main()
