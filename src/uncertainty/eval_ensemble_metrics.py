import argparse
import os
import yaml
import numpy as np
import torch

from src.data_loading.darus_dataset import create_dataloaders
from src.models.lstm import LSTMSeq2Seq
from src.uncertainty.deep_ensemble import predict_ensemble


TARGET_NAMES = ["u", "v", "p", "r", "phi"]


def load_ensemble(ensemble_dir, config, device, input_dim, output_dim, horizon):
    """
    Load a deep ensemble of LSTM models from subdirectories:
      ensemble_dir/model_0/best_model.pt, model_1/..., etc.
    Handles different checkpoint formats:
      - plain state_dict
      - {"model_state_dict": ...}
      - {"model_state": ...}
    """
    models = []

    hidden_dim = config["model"]["hidden_dim"]
    num_layers = config["model"]["num_layers"]
    dropout = config["model"].get("dropout", 0.0)

    for sub in sorted(os.listdir(ensemble_dir)):
        subdir = os.path.join(ensemble_dir, sub)

        # skip non-directories (e.g. plots folder)
        if not os.path.isdir(subdir):
            continue

        ckpt_path = os.path.join(subdir, "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"⚠️  Skipping missing: {ckpt_path}")
            continue

        ckpt = torch.load(ckpt_path, map_location=device)

        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif "model_state" in ckpt:
                state_dict = ckpt["model_state"]
            else:
                # assume the dict itself is a state_dict
                state_dict = ckpt
        else:
            # assume ckpt is already a state_dict
            state_dict = ckpt

        model = LSTMSeq2Seq(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon,
            target_dim=output_dim,
        ).to(device)

        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)

    print(f"✅ Loaded {len(models)} ensemble members from: {ensemble_dir}")
    return models


def evaluate_ensemble(models, loader, device):
    """
    Compute RMSE and error–uncertainty correlation for a deep ensemble.
    Returns:
      overall_rmse, rmse_per_dof (array length D),
      corr_error_std (Pearson r between |error| and ensemble std).
    """
    sum_sq_per_dof = None
    total_count = 0

    all_abs_err = []
    all_std = []

    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)  # (B, T, D)

            # predict_ensemble returns numpy arrays -> convert to tensors
            mean_np, std_np = predict_ensemble(models, X, device)  # (B, T, D) each

            if isinstance(mean_np, torch.Tensor):
                mean_pred = mean_np.to(device)
                std_pred = std_np.to(device)
            else:
                mean_pred = torch.from_numpy(mean_np).to(device=device, dtype=Y.dtype)
                std_pred = torch.from_numpy(std_np).to(device=device, dtype=Y.dtype)

            err = mean_pred - Y                          # (B, T, D)
            err_sq = err ** 2
            err_abs = err.abs()

            B, T, D = err.shape

            if sum_sq_per_dof is None:
                sum_sq_per_dof = torch.zeros(D, device=device)

            # sum over batch and time
            sum_sq_per_dof += err_sq.sum(dim=(0, 1))
            total_count += B * T

            # store flattened arrays for correlation
            all_abs_err.append(err_abs.cpu().numpy().reshape(-1))
            all_std.append(std_pred.cpu().numpy().reshape(-1))

    rmse_per_dof = torch.sqrt(sum_sq_per_dof / total_count).cpu().numpy()
    overall_rmse = float(rmse_per_dof.mean())

    all_abs_err = np.concatenate(all_abs_err, axis=0)
    all_std = np.concatenate(all_std, axis=0)

    # Filter out any degenerate cases
    mask = np.isfinite(all_abs_err) & np.isfinite(all_std)
    if mask.sum() > 0:
        corr_error_std = float(np.corrcoef(all_abs_err[mask], all_std[mask])[0, 1])
    else:
        corr_error_std = float("nan")

    return overall_rmse, rmse_per_dof, corr_error_std


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate deep ensemble RMSE and uncertainty-quality metrics."
    )
    parser.add_argument(
        "--ensemble_dir",
        type=str,
        required=True,
        help="Directory containing ensemble subfolders (model_0, model_1, ...).",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML config file used for training (e.g. p2_lstm_ensemble.yaml).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "ood"],
        help="Which split to evaluate: 'test' (ID) or 'ood'.",
    )
    args = parser.parse_args()

    # --------------------------------------------------
    # Load config
    # --------------------------------------------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    history = config["data"]["history"]
    horizon = config["data"]["horizon"]
    batch_size = config["data"]["batch_size"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --------------------------------------------------
    # Data loaders
    # --------------------------------------------------
    print("🔧 create_dataloaders called!")
    train_loader, val_loader, test_loader, ood_loader = create_dataloaders(
        history=history,
        horizon=horizon,
        batch_size=batch_size,
    )

    loader = test_loader if args.split == "test" else ood_loader

    # get dimensionalities from dataset
    input_dim = train_loader.dataset.X.shape[-1]
    output_dim = train_loader.dataset.Y.shape[-1]

    # --------------------------------------------------
    # Load ensemble
    # --------------------------------------------------
    models = load_ensemble(
        ensemble_dir=args.ensemble_dir,
        config=config,
        device=device,
        input_dim=input_dim,
        output_dim=output_dim,
        horizon=horizon,
    )

    if len(models) == 0:
        raise RuntimeError("No ensemble members loaded; check ensemble_dir path.")

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------
    print(f"📏 Evaluating ensemble on split: {args.split.upper()}")

    overall_rmse, rmse_per_dof, corr_error_std = evaluate_ensemble(
        models, loader, device
    )

    print("\n==========================================")
    print(f"Deep Ensemble Metrics ({args.split.upper()} split)")
    print("------------------------------------------")
    print(f"Overall RMSE: {overall_rmse:.6f}")
    print("Per-DoF RMSE:")
    for name, val in zip(TARGET_NAMES, rmse_per_dof):
        print(f"  {name:<5} {val:.6f}")
    print("------------------------------------------")
    print(
        "Correlation(|error|, ensemble std): "
        f"{corr_error_std:.3f} (higher = better uncertainty ranking)"
    )
    print("==========================================\n")


if __name__ == "__main__":
    main()
