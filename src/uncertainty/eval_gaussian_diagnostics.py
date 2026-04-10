import argparse
import os
import math
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data_loading.darus_dataset import create_dataloaders
# IMPORTANT: adjust this import if your Gaussian model lives elsewhere
from src.models.lstm_gaussian import LSTMGaussianSeq2Seq


TARGET_NAMES = ["u", "v", "p", "r", "phi"]


def _normal_z(alpha: float, device: str) -> float:
    """Two-sided normal quantile for (1-alpha) coverage: mu +/- z*sigma."""
    dist = torch.distributions.Normal(
        torch.tensor(0.0, device=device),
        torch.tensor(1.0, device=device),
    )
    q = dist.icdf(torch.tensor(1.0 - alpha / 2.0, device=device))
    return float(q.item())


@torch.no_grad()
def _forward_mu_logvar(model, x):
    """
    Supports:
      - model(x) -> (mu, logvar)
      - model(x) -> tensor with last dim 2*D: [mu || logvar]
    Returns:
      mu: (B,T,D)
      logvar: (B,T,D)
    """
    out = model(x)
    if isinstance(out, (tuple, list)) and len(out) == 2:
        mu, logvar = out
        return mu, logvar

    if torch.is_tensor(out) and out.ndim == 3 and out.shape[-1] % 2 == 0:
        D2 = out.shape[-1]
        D = D2 // 2
        mu = out[..., :D]
        logvar = out[..., D:]
        return mu, logvar

    raise RuntimeError(
        "Unsupported Gaussian model output. Expected (mu, logvar) or (B,T,2*D) tensor."
    )


@torch.no_grad()
def evaluate_gaussian(
    model,
    loader,
    device: str,
    alpha: float,
    sigma_temp: float = 1.0,
):
    z = _normal_z(alpha, device)

    all_cover = []
    all_width = []
    all_err2 = []
    all_nll = []

    # per-horizon aggregates
    # we will accumulate sums and counts
    sum_cover_h = None
    sum_width_h = None
    count_h = None

    sum_cover_dof = None
    sum_width_dof = None
    count_dof = None

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)  # (B,T,D)

        mu, logvar = _forward_mu_logvar(model, x)
        # sigma: (B,T,D)
        sigma = torch.exp(0.5 * logvar) * float(sigma_temp)

        lower = mu - z * sigma
        upper = mu + z * sigma

        cover = ((y >= lower) & (y <= upper)).float()  # (B,T,D)
        width = (upper - lower)  # (B,T,D)

        err2 = (mu - y) ** 2

        # Gaussian NLL per element:
        # 0.5 * [log(2π) + log(sigma^2) + (y-mu)^2 / sigma^2]
        var = sigma ** 2
        nll = 0.5 * (math.log(2.0 * math.pi) + torch.log(var) + ((y - mu) ** 2) / var)

        # overall scalars
        all_cover.append(cover.mean().item())
        all_width.append(width.mean().item())
        all_err2.append(err2.mean().item())
        all_nll.append(nll.mean().item())

        B, T, D = y.shape

        if sum_cover_h is None:
            sum_cover_h = torch.zeros((T, D), device="cpu")
            sum_width_h = torch.zeros((T, D), device="cpu")
            count_h = torch.zeros((T, D), device="cpu")

            sum_cover_dof = torch.zeros((D,), device="cpu")
            sum_width_dof = torch.zeros((D,), device="cpu")
            count_dof = torch.zeros((D,), device="cpu")

        # aggregate per horizon step
        cover_cpu = cover.detach().cpu()
        width_cpu = width.detach().cpu()

        sum_cover_h += cover_cpu.mean(dim=0)  # (T,D)
        sum_width_h += width_cpu.mean(dim=0)  # (T,D)
        count_h += 1.0

        # aggregate per dof (over B,T)
        sum_cover_dof += cover_cpu.mean(dim=(0, 1))  # (D,)
        sum_width_dof += width_cpu.mean(dim=(0, 1))  # (D,)
        count_dof += 1.0

    overall_coverage = float(np.mean(all_cover)) if all_cover else float("nan")
    overall_width = float(np.mean(all_width)) if all_width else float("nan")
    rmse = float(math.sqrt(np.mean(all_err2))) if all_err2 else float("nan")
    avg_nll = float(np.mean(all_nll)) if all_nll else float("nan")

    cov_per_h = (sum_cover_h / count_h).numpy()  # (T,D)
    wid_per_h = (sum_width_h / count_h).numpy()  # (T,D)
    cov_per_dof = (sum_cover_dof / count_dof).numpy()  # (D,)
    wid_per_dof = (sum_width_dof / count_dof).numpy()  # (D,)

    return {
        "rmse": rmse,
        "avg_nll": avg_nll,
        "overall_coverage": overall_coverage,
        "overall_width": overall_width,
        "cov_per_h": cov_per_h,
        "wid_per_h": wid_per_h,
        "cov_per_dof": cov_per_dof,
        "wid_per_dof": wid_per_dof,
    }


@torch.no_grad()
def calibrate_sigma_temp(model, val_loader, device: str, alpha: float):
    """
    Find sigma_temp that makes empirical coverage on validation close to nominal (1-alpha).
    Uses a simple grid search over temperatures.
    """
    nominal = 1.0 - alpha
    temps = np.logspace(-1, 1, 41)  # 0.1 ... 10 (41 points)
    best_t = 1.0
    best_gap = float("inf")

    for t in temps:
        metrics = evaluate_gaussian(model, val_loader, device, alpha, sigma_temp=float(t))
        gap = abs(metrics["overall_coverage"] - nominal)
        if gap < best_gap:
            best_gap = gap
            best_t = float(t)

    return best_t, best_gap


def save_plots(metrics, out_dir: str, split: str):
    os.makedirs(out_dir, exist_ok=True)

    cov_per_h = metrics["cov_per_h"]  # (T,D)
    wid_per_h = metrics["wid_per_h"]  # (T,D)
    T, D = cov_per_h.shape

    # Coverage vs horizon, per DoF
    for i in range(D):
        plt.figure(figsize=(9, 4))
        plt.plot(np.arange(T), cov_per_h[:, i])
        plt.ylim(0.0, 1.0)
        plt.xlabel("Prediction step")
        plt.ylabel("Empirical coverage")
        plt.title(f"Gaussian interval coverage vs horizon: {TARGET_NAMES[i]} ({split})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"gaussian_coverage_{TARGET_NAMES[i]}_{split}.png"), dpi=300)
        plt.close()

    # Width vs horizon, per DoF
    for i in range(D):
        plt.figure(figsize=(9, 4))
        plt.plot(np.arange(T), wid_per_h[:, i])
        plt.xlabel("Prediction step")
        plt.ylabel("Average interval width")
        plt.title(f"Gaussian interval width vs horizon: {TARGET_NAMES[i]} ({split})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"gaussian_width_{TARGET_NAMES[i]}_{split}.png"), dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Gaussian LSTM: RMSE, NLL, Gaussian interval coverage/width.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["test", "ood"])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--sigma_temp", type=float, default=1.0, help="Multiply sigma by this temperature (post-hoc).")
    parser.add_argument("--calibrate_sigma_temp", action="store_true", help="Grid-search sigma_temp on validation set.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    horizon = int(config["data"]["horizon"])
    batch_size = int(config.get("training", {}).get("batch_size", config.get("training", {}).get("batch", 256)))

    # create_dataloaders returns (train, val, test, ood) in your repo
    history = int(config["data"]["history"])
    train_loader, val_loader, test_loader, ood_loader = create_dataloaders(history, horizon, batch_size)

    loader = test_loader if args.split == "test" else ood_loader

    ckpt_path = os.path.join(args.model_dir, "best_model.pt")
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # infer dims from a batch
    x0, y0 = next(iter(loader))
    input_dim = x0.shape[-1]
    target_dim = y0.shape[-1]

    model = LSTMGaussianSeq2Seq(
        input_dim=input_dim,
        hidden_dim=int(config["model"]["hidden_dim"]),
        num_layers=int(config["model"]["num_layers"]),
        dropout=float(config["model"].get("dropout", 0.0)),
        horizon=horizon,
        target_dim=target_dim,
    ).to(device)
    model.eval()

    # support multiple checkpoint formats
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    elif isinstance(ckpt, dict) and all(k.startswith("encoder.") or k.startswith("decoder.") or k.startswith("fc.") for k in ckpt.keys()):
        model.load_state_dict(ckpt)
    else:
        # many of your scripts save pure state_dict directly
        try:
            model.load_state_dict(ckpt)
        except Exception:
            raise RuntimeError("Unrecognized checkpoint format for Gaussian model.")

    sigma_temp = float(args.sigma_temp)
    if args.calibrate_sigma_temp:
        best_t, best_gap = calibrate_sigma_temp(model, val_loader, device, args.alpha)
        sigma_temp = best_t
        print(f"✅ Calibrated sigma_temp on VAL: {sigma_temp:.3f} (gap to nominal={best_gap:.4f})")

    metrics = evaluate_gaussian(model, loader, device, args.alpha, sigma_temp=sigma_temp)

    nominal = 1.0 - args.alpha
    print("\n==========================================")
    print(f"Gaussian diagnostics ({args.split.upper()} split)")
    print("------------------------------------------")
    print(f"alpha:               {args.alpha:.3f}")
    print(f"nominal coverage:    {nominal:.3f}")
    print(f"sigma_temp used:     {sigma_temp:.3f}")
    print("------------------------------------------")
    print(f"RMSE:                {metrics['rmse']:.6f}")
    print(f"Avg NLL:             {metrics['avg_nll']:.6f}")
    print(f"Overall coverage:    {metrics['overall_coverage']:.3f}")
    print(f"Overall width:       {metrics['overall_width']:.3f}")
    print("------------------------------------------")
    print("Per-DoF coverage and width:")
    for i, name in enumerate(TARGET_NAMES[: len(metrics["cov_per_dof"])]):
        print(f"  {name:<4} coverage: {metrics['cov_per_dof'][i]:.3f}   width: {metrics['wid_per_dof'][i]:.3f}")
    print("==========================================\n")

    out_dir = os.path.join(args.model_dir, "gaussian_diag", args.split)
    save_plots(metrics, out_dir, args.split)
    print(f"✅ Saved Gaussian diagnostic plots to: {out_dir}")


if __name__ == "__main__":
    main()
