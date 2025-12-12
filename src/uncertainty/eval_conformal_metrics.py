import argparse
import os

import numpy as np
import torch
import yaml

from src.data_loading.darus_dataset import create_dataloaders
from src.models.lstm import LSTMSeq2Seq


# Use the same DoF ordering you use elsewhere (adjust if needed)
DOF_NAMES = ["u", "v", "p", "r", "phi"]


def build_model_from_config(config, input_dim, target_dim, horizon, device):
    """Build an LSTMSeq2Seq model from the given config and data dims."""
    model_cfg = config["model"]

    model = LSTMSeq2Seq(
        input_dim=input_dim,
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg.get("dropout", 0.0),
        horizon=horizon,
        target_dim=target_dim,
    ).to(device)

    return model


def collect_residuals(loader, model, device):
    """Collect absolute residuals |y - y_hat| over a loader.

    Returns:
        residuals: (N_flat, D) numpy array, where D is number of DoFs.
    """
    model.eval()
    all_res = []

    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)

            Yhat = model(X)  # (B, H, D)
            abs_err = (Yhat - Y).abs().cpu().numpy()  # (B, H, D)

            B, H, D = abs_err.shape
            all_res.append(abs_err.reshape(-1, D))  # (B*H, D)

    residuals = np.concatenate(all_res, axis=0)  # (N_flat, D)
    return residuals


def evaluate_split(loader, model, q_per_dim, q_all, device):
    """Compute coverage and average width for a given split.

    Args:
        loader: DataLoader for test or ood.
        model: trained model.
        q_per_dim: (D,) quantiles per DoF.
        q_all: scalar quantile across all DoFs.
        device: "cpu" or "cuda".

    Returns:
        coverage_per_dim: (D,) array
        width_per_dim:   (D,) array
        coverage_all:    scalar
        width_all:       scalar
    """
    model.eval()

    D = len(q_per_dim)
    cover_count_dim = np.zeros(D, dtype=np.float64)
    total_count_dim = 0.0

    cover_count_all = 0.0
    total_count_all = 0.0

    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)

            Yhat = model(X)  # (B, H, D)

            Y_np = Y.cpu().numpy()
            Yhat_np = Yhat.cpu().numpy()

            # Per-dim intervals
            lower = Yhat_np - q_per_dim.reshape(1, 1, -1)
            upper = Yhat_np + q_per_dim.reshape(1, 1, -1)
            inside = (Y_np >= lower) & (Y_np <= upper)  # (B, H, D)

            B, H, D_check = inside.shape
            assert D_check == D

            flat_inside = inside.reshape(-1, D)  # (B*H, D)
            n_points = flat_inside.shape[0]

            cover_count_dim += flat_inside.sum(axis=0).astype(np.float64)
            total_count_dim += float(n_points)

            # Overall scalar interval
            lower_all = Yhat_np - q_all
            upper_all = Yhat_np + q_all
            inside_all = (Y_np >= lower_all) & (Y_np <= upper_all)  # (B, H, D)

            cover_count_all += inside_all.sum().item()
            total_count_all += float(inside_all.size)

    coverage_per_dim = cover_count_dim / total_count_dim
    width_per_dim = 2.0 * q_per_dim

    coverage_all = cover_count_all / total_count_all
    width_all = 2.0 * q_all

    return coverage_per_dim, width_per_dim, coverage_all, width_all


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate conformal prediction metrics (coverage and interval width) "
            "for a trained LSTM baseline using validation-based calibration."
        )
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "ood"],
    )

    args = parser.parse_args()

    # ----------------------------
    # Load config and data loaders
    # ----------------------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    history = config["data"]["history"]
    horizon = config["data"]["horizon"]
    batch_size = config["data"].get("batch_size", 256)

    print("🔧 create_dataloaders called!")
    train_loader, val_loader, test_loader, ood_loader = create_dataloaders(
        history=history,
        horizon=horizon,
        batch_size=batch_size,
    )

    # Infer input/output dims from dataset
    input_dim = train_loader.dataset.X.shape[-1]
    target_dim = train_loader.dataset.Y.shape[-1]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------------
    # Build and load model
    # ----------------------------
    model = build_model_from_config(
        config=config,
        input_dim=input_dim,
        target_dim=target_dim,
        horizon=horizon,
        device=device,
    )

    ckpt_path = os.path.join(args.model_dir, "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.to(device)

    # ----------------------------
    # 1) Calibration residuals on validation set
    # ----------------------------
    print("📏 Collecting calibration residuals from validation set...")
    cal_residuals = collect_residuals(val_loader, model, device)  # (N_flat, D)
    D = cal_residuals.shape[1]

    if len(DOF_NAMES) != D:
        # Fallback if DoF ordering differs
        dof_names = [f"DoF_{i}" for i in range(D)]
    else:
        dof_names = DOF_NAMES

    alpha = args.alpha
    nominal_coverage = 1.0 - alpha

    # Per-dimension quantiles
    q_per_dim = np.quantile(cal_residuals, 1.0 - alpha, axis=0)  # (D,)
    # Overall quantile
    q_all = np.quantile(cal_residuals.reshape(-1), 1.0 - alpha)

    # ----------------------------
    # 2) Evaluate on requested split
    # ----------------------------
    if args.split == "test":
        eval_loader = test_loader
    else:
        eval_loader = ood_loader

    (
        coverage_per_dim,
        width_per_dim,
        coverage_all,
        width_all,
    ) = evaluate_split(eval_loader, model, q_per_dim, q_all, device)

    # ----------------------------
    # 3) Print metrics
    # ----------------------------
    print("\n==========================================")
    print(f"Conformal metrics for split: {args.split.upper()}")
    print("------------------------------------------")
    print(f"Alpha:            {alpha:.3f}")
    print(f"Nominal coverage: {nominal_coverage:.3f}")
    print()
    print(f"Overall coverage: {coverage_all:.3f}")
    print(f"Overall width:    {width_all:.3f}")
    print("------------------------------------------")
    print("Per-DoF coverage and width:")
    for name, cov, width in zip(dof_names, coverage_per_dim, width_per_dim):
        print(f"  {name:4s}  coverage: {cov:.3f}   width: {width:.3f}")
    print("==========================================\n")


if __name__ == "__main__":
    main()
