import argparse
import os
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt

from src.models.lstm import LSTMSeq2Seq
from src.data_loading.darus_dataset import create_dataloaders

TARGET_NAMES = ["phi", "p", "r", "u", "v"]


def conformal_scores(model, loader, device):
    """
    Compute absolute residuals on the calibration set.
    Returns tensor of shape (N, H, D).
    """
    model.eval()
    residuals = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            res = torch.abs(pred - y)  # (B, H, D)
            residuals.append(res.cpu())

    return torch.cat(residuals, dim=0)


def compute_quantiles(residuals, alpha):
    """
    Per-step, per-dimension quantile:
      residuals: (N, H, D)
      returns q: (H, D)
    """
    return torch.quantile(residuals, 1 - alpha, dim=0)


def plot_intervals(mean_pred, y_true, q, out_dir, split):
    """
    mean_pred, y_true, q: (H, D)
    Save one plot per target dimension.
    """
    H, D = mean_pred.shape
    os.makedirs(out_dir, exist_ok=True)

    for i, name in enumerate(TARGET_NAMES):
        plt.figure(figsize=(9, 5))

        lower = mean_pred[:, i] - q[:, i]
        upper = mean_pred[:, i] + q[:, i]

        plt.plot(mean_pred[:, i], label="Prediction")
        plt.plot(y_true[:, i], "--", label="Ground truth")
        plt.fill_between(
            range(H),
            lower,
            upper,
            alpha=0.3,
            label="Conformal interval",
        )

        plt.title(f"Conformal interval – {name} ({split})")
        plt.xlabel("Prediction step")
        plt.ylabel(name)
        plt.legend()
        plt.grid(True)

        fname = f"conformal_{name}_{split}.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing best_model.pt for the baseline LSTM.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config used to train this model.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Miscoverage level; for example alpha=0.1 gives 90 percent intervals.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "ood"],
        help="Which split to evaluate conformal coverage on.",
    )
    args = parser.parse_args()

    # ------------------------------
    # 1. Load config
    # ------------------------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    history = config["data"]["history"]
    horizon = config["data"]["horizon"]
    batch_size = config["data"]["batch_size"]

    hidden_dim = config["model"].get("hidden_dim", 128)
    num_layers = config["model"].get("num_layers", 2)
    dropout = config["model"].get("dropout", 0.0)

    # ------------------------------
    # 2. Build dataloaders
    # ------------------------------
    print("🔧 create_dataloaders called!")
    train_loader, val_loader, test_loader, ood_loader = create_dataloaders(
        history=history,
        horizon=horizon,
        batch_size=batch_size,
    )

    # use validation as calibration set
    cal_loader = val_loader
    eval_loader = test_loader if args.split == "test" else ood_loader

    # infer input/output dims from data
    input_dim = train_loader.dataset.X.shape[-1]
    target_dim = train_loader.dataset.Y.shape[-1]

    # ------------------------------
    # 3. Build model and load weights
    # ------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LSTMSeq2Seq(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        horizon=horizon,
        target_dim=target_dim,
    ).to(device)

    ckpt_path = os.path.join(args.model_dir, "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    # robustly fetch state dict
    if isinstance(ckpt, dict):
        state_dict = (
            ckpt.get("model_state_dict")
            or ckpt.get("model_state")
            or ckpt  # if it was saved as raw state_dict
        )
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.eval()

    # ------------------------------
    # 4. Calibration: compute residual quantiles
    # ------------------------------
    print("📏 Computing calibration residuals...")
    residuals = conformal_scores(model, cal_loader, device)
    q = compute_quantiles(residuals, alpha=args.alpha)  # (H, D)

    # ------------------------------
    # 5. Evaluate on one batch from chosen split
    # ------------------------------
    x, y = next(iter(eval_loader))
    x = x.to(device)
    y = y.to(device)

    with torch.no_grad():
        pred = model(x)  # (B, H, D)
    pred = pred[0].cpu()  # (H, D)
    y_true = y[0].cpu()   # (H, D)

    # ------------------------------
    # 6. Plot intervals
    # ------------------------------
    out_dir = os.path.join(args.model_dir, "conformal", args.split)
    plot_intervals(pred, y_true, q, out_dir, args.split)

    print("\n✅ Conformal prediction intervals saved to:")
    print(f"   {out_dir}\n")


if __name__ == "__main__":
    main()
