import argparse
import os
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.data_loading.darus_dataset import create_dataloaders
# IMPORTANT: adjust this import to your gaussian model class
# If your gaussian model lives elsewhere, update accordingly.
from src.uncertainty.models.lstm_gaussian import LSTMGaussian

TARGET_NAMES = ["u", "v", "p", "r", "phi"]


def gaussian_nll(y, mu, sigma):
    # y, mu, sigma: (..., D)
    # NLL per element, averaged outside
    var = sigma ** 2
    return 0.5 * (torch.log(2 * torch.pi * var) + ((y - mu) ** 2) / var)


@torch.no_grad()
def collect_preds(model, loader, device, sigma_floor, sigma_cap):
    mus, sigmas, ys = [], [], []

    model.eval()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        out = model(x)  # expected shape: (B, T, 2*D) or a dict; adapt below
        # ---- ADAPT HERE if your model returns (mu, logvar) separately ----
        if isinstance(out, (tuple, list)) and len(out) == 2:
            mu, raw_sigma = out
            sigma = raw_sigma
        else:
            # assume last dim = 2*D => [mu, raw_scale] split
            D = y.shape[-1]
            mu = out[..., :D]
            raw = out[..., D:]
            sigma = F.softplus(raw)

        sigma = torch.clamp(sigma, min=sigma_floor, max=sigma_cap)

        mus.append(mu.cpu())
        sigmas.append(sigma.cpu())
        ys.append(y.cpu())

    mu = torch.cat(mus, dim=0)
    sigma = torch.cat(sigmas, dim=0)
    y = torch.cat(ys, dim=0)
    return mu, sigma, y


def fit_sigma_temperature(mu, sigma, y, t_min=0.5, t_max=5.0, n=60):
    # grid search for scalar temperature T
    ts = np.linspace(t_min, t_max, n)
    best_t = None
    best_nll = float("inf")

    y_t = y
    mu_t = mu

    for t in ts:
        s = sigma * float(t)
        nll = gaussian_nll(y_t, mu_t, s).mean().item()
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)

    return best_t, best_nll


def plot_example_phi(mu, sigma, y, out_path, title):
    # first sample, phi index = last
    phi_idx = 4
    mu0 = mu[0, :, phi_idx].numpy()
    s0 = sigma[0, :, phi_idx].numpy()
    y0 = y[0, :, phi_idx].numpy()
    t = np.arange(len(mu0))

    plt.figure(figsize=(9, 4))
    plt.plot(t, y0, "--", label="Ground Truth")
    plt.plot(t, mu0, label="Mean prediction")
    plt.fill_between(t, mu0 - 2 * s0, mu0 + 2 * s0, alpha=0.25, label="±2σ")
    plt.xlabel("Prediction step")
    plt.ylabel("phi")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["test", "ood"])
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    horizon = config["data"]["horizon"]
    batch_size = config.get("training", {}).get("batch_size", config.get("training", {}).get("batch", 256))

    uq = config.get("uq", {})
    sigma_floor = float(uq.get("sigma_floor", 1e-3))
    sigma_cap = float(uq.get("sigma_cap", 5.0))
    do_cal = bool(uq.get("calibrate_sigma", True))

    grid = uq.get("sigma_temp_grid", {})
    t_min = float(grid.get("t_min", 0.5))
    t_max = float(grid.get("t_max", 5.0))
    n_grid = int(grid.get("n", 60))

    train_loader, val_loader, test_loader, ood_loader = create_dataloaders(config, horizon, batch_size)
    loader = test_loader if args.split == "test" else ood_loader

    ckpt_path = os.path.join(args.model_dir, "best_model.pt")
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # infer dimensions from one batch
    x_sample, y_sample = next(iter(loader))
    input_dim = x_sample.shape[-1]
    target_dim = y_sample.shape[-1]

    model = LSTMGaussian(
        input_dim=input_dim,
        hidden_dim=config["model"]["hidden_dim"],
        target_dim=target_dim,
        num_layers=config["model"]["num_layers"],
        dropout=config["model"].get("dropout", 0.0),
        horizon=y_sample.shape[1],
    ).to(device)


    # your training saved either raw state_dict or dict wrapper
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    # collect val predictions for calibration
    mu_val, sigma_val, y_val = collect_preds(model, val_loader, device, sigma_floor, sigma_cap)

    T_sigma = 1.0
    if do_cal:
        T_sigma, best_val_nll = fit_sigma_temperature(
            mu_val, sigma_val, y_val, t_min=t_min, t_max=t_max, n=n_grid
        )
        print(f"✅ Sigma temperature fitted on VAL: T_sigma={T_sigma:.3f} (val NLL={best_val_nll:.4f})")

    # collect split predictions
    mu, sigma, y = collect_preds(model, loader, device, sigma_floor, sigma_cap)
    sigma = sigma * float(T_sigma)

    # metrics
    rmse = torch.sqrt(((mu - y) ** 2).mean()).item()
    nll = gaussian_nll(y, mu, sigma).mean().item()

    print("\n==========================================")
    print(f"Gaussian LSTM metrics ({args.split.upper()} split)")
    print("------------------------------------------")
    print(f"RMSE:       {rmse:.6f}")
    print(f"Avg NLL:    {nll:.6f}")
    print(f"T_sigma:    {T_sigma:.3f}")
    print(f"sigma_floor:{sigma_floor:g}   sigma_cap:{sigma_cap:g}")
    print("==========================================\n")

    out_dir = os.path.join(args.model_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, f"gaussian_phi_{args.split}.png")
    plot_example_phi(mu, sigma, y, plot_path, f"Gaussian LSTM φ ({args.split.upper()})")
    print(f"✅ Saved Gaussian φ plot to: {plot_path}")


if __name__ == "__main__":
    main()
