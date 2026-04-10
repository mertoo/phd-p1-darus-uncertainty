import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml

from src.data_loading.darus_dataset import create_dataloaders
from src.models.lstm_gaussian import LSTMGaussianSeq2Seq


TARGET_NAMES = ["u", "v", "p", "r", "phi"]


# -------------------------
# Utilities
# -------------------------

def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _z_for_alpha(alpha: float) -> float:
    """
    Two-sided Normal interval: mean ± z * sigma
    For 90% interval -> alpha=0.1 -> z ~ 1.64485
    """
    # Hardcode common values to avoid scipy dependency.
    # If you want arbitrary alpha, you can approximate, but this is enough for paper runs.
    if abs(alpha - 0.10) < 1e-9:
        return 1.6448536269514722
    if abs(alpha - 0.05) < 1e-9:
        return 1.959963984540054
    if abs(alpha - 0.20) < 1e-9:
        return 1.2815515655446004

    # Fallback: approximate inverse CDF using numpy polynomial approximation.
    # Good enough for alpha not too extreme.
    p = 1.0 - alpha / 2.0
    # Abramowitz-Stegun approximation for inverse erf / normal quantile
    # Ref: https://web.archive.org/web/20150910044729/http://home.online.no/~pjacklam/notes/invnorm/
    # We'll implement a simple rational approximation (Peter John Acklam).
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        x = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    elif p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        x = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
             ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    else:
        q = p - 0.5
        r = q*q
        x = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
            (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    return float(x)


def _load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _infer_dims_from_loader(loader) -> Tuple[int, int]:
    x0, y0 = next(iter(loader))
    return int(x0.shape[-1]), int(y0.shape[-1])


def _build_model_from_data(config: Dict, input_dim: int, target_dim: int, device: str) -> torch.nn.Module:
    hidden_dim = int(config.get("model", {}).get("hidden_dim", 128))
    num_layers = int(config.get("model", {}).get("num_layers", 2))
    dropout = float(config.get("model", {}).get("dropout", 0.0))
    horizon = int(config.get("data", {}).get("horizon", 30))
    model = LSTMGaussianSeq2Seq(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        horizon=horizon,
        target_dim=target_dim,
    ).to(device)
    return model


def _load_checkpoint(model: torch.nn.Module, model_dir: str, device: str) -> None:
    ckpt_path = os.path.join(model_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Support multiple save formats
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        # could be raw state_dict
        state = ckpt

    model.load_state_dict(state, strict=True)


@torch.no_grad()
def _predict_mu_sigma(model: torch.nn.Module, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Expect model(X) -> (mu, log_var) or (mu, log_sigma2)
    Return: mu, sigma (positive)
    """
    out = model(X)
    if isinstance(out, (tuple, list)) and len(out) == 2:
        mu, log_var = out
    else:
        raise RuntimeError("Expected LSTMGaussian forward() to return (mu, log_var).")

    # sigma = sqrt(exp(log_var))
    sigma = torch.sqrt(torch.exp(log_var).clamp(min=1e-12))
    return mu, sigma


def _gaussian_nll(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Elementwise NLL for Gaussian with diagonal covariance:
    0.5*log(2*pi) + log(sigma) + 0.5*((y-mu)/sigma)^2
    """
    eps = 1e-12
    sigma = sigma.clamp(min=eps)
    return 0.5 * np.log(2.0 * np.pi) + torch.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2


@dataclass
class Metrics:
    rmse: float
    avg_nll: float
    overall_coverage: float
    overall_width: float
    per_dof: Dict[str, Dict[str, float]]


@torch.no_grad()
def evaluate_gaussian_intervals(
    model: torch.nn.Module,
    loader,
    device: str,
    alpha: float,
    temps_per_dof: np.ndarray,
) -> Metrics:
    z = _z_for_alpha(alpha)

    all_err2 = []
    all_nll = []
    all_cover = []
    all_width = []

    # per dof accumulators
    cover_d = {name: [] for name in TARGET_NAMES}
    width_d = {name: [] for name in TARGET_NAMES}
    rmse_d_err2 = {name: [] for name in TARGET_NAMES}

    temps_t = torch.tensor(temps_per_dof.reshape(1, 1, -1), device=device, dtype=torch.float32)

    for X, Y in loader:
        X = X.to(device)
        Y = Y.to(device)

        mu, sigma = _predict_mu_sigma(model, X)

        # apply per-dof temperature scaling
        sigma_cal = sigma * temps_t

        # intervals
        lo = mu - z * sigma_cal
        hi = mu + z * sigma_cal

        # coverage
        inside = (Y >= lo) & (Y <= hi)  # [B,T,D]
        cover = inside.float().mean().item()
        width = (hi - lo).mean().item()

        # rmse
        err2 = (mu - Y) ** 2
        rmse = torch.sqrt(err2.mean()).item()

        # nll
        nll = _gaussian_nll(Y, mu, sigma_cal).mean().item()

        all_cover.append(cover)
        all_width.append(width)
        all_err2.append(rmse ** 2)  # store squared to average properly-ish per batch
        all_nll.append(nll)

        # per dof
        for j, name in enumerate(TARGET_NAMES):
            cover_d[name].append(inside[..., j].float().mean().item())
            width_d[name].append((hi[..., j] - lo[..., j]).mean().item())
            rmse_d_err2[name].append(err2[..., j].mean().item())

    overall_rmse = float(np.sqrt(np.mean(all_err2)))
    avg_nll = float(np.mean(all_nll))
    overall_cov = float(np.mean(all_cover))
    overall_w = float(np.mean(all_width))

    per_dof = {}
    for j, name in enumerate(TARGET_NAMES):
        per_dof[name] = {
            "coverage": float(np.mean(cover_d[name])),
            "width": float(np.mean(width_d[name])),
            "rmse": float(np.sqrt(np.mean(rmse_d_err2[name]))),
            "sigma_temp": float(temps_per_dof[j]),
        }

    return Metrics(
        rmse=overall_rmse,
        avg_nll=avg_nll,
        overall_coverage=overall_cov,
        overall_width=overall_w,
        per_dof=per_dof,
    )


@torch.no_grad()
def calibrate_temps_per_dof(
    model: torch.nn.Module,
    val_loader,
    device: str,
    alpha: float,
    search_grid: List[float],
) -> np.ndarray:
    """
    Choose per-DoF sigma_temp to make coverage close to nominal (1-alpha)
    on the validation split.
    """
    z = _z_for_alpha(alpha)
    nominal = 1.0 - alpha

    # Collect mu, sigma, y for entire val split (vectorized calibration)
    mus = []
    sigmas = []
    ys = []
    for X, Y in val_loader:
        X = X.to(device)
        Y = Y.to(device)
        mu, sigma = _predict_mu_sigma(model, X)
        mus.append(mu.cpu())
        sigmas.append(sigma.cpu())
        ys.append(Y.cpu())

    mu_all = torch.cat(mus, dim=0)       # [N,T,D]
    sigma_all = torch.cat(sigmas, dim=0) # [N,T,D]
    y_all = torch.cat(ys, dim=0)         # [N,T,D]

    temps = np.ones((len(TARGET_NAMES),), dtype=np.float32)

    for j in range(len(TARGET_NAMES)):
        best_t = None
        best_gap = 1e9

        mu_j = mu_all[..., j]
        sig_j = sigma_all[..., j]
        y_j = y_all[..., j]

        for t in search_grid:
            lo = mu_j - z * (sig_j * t)
            hi = mu_j + z * (sig_j * t)
            cov = ((y_j >= lo) & (y_j <= hi)).float().mean().item()
            gap = abs(cov - nominal)
            if gap < best_gap:
                best_gap = gap
                best_t = t

        temps[j] = float(best_t)

    return temps


def save_example_plots(
    model: torch.nn.Module,
    loader,
    device: str,
    alpha: float,
    temps_per_dof: np.ndarray,
    out_dir: str,
    split_name: str,
    n_examples: int = 1,
) -> None:
    """
    Save a few trajectory plots per DoF: mean, ±z*sigma (scaled), and ground truth.
    """
    z = _z_for_alpha(alpha)
    temps_t = torch.tensor(temps_per_dof.reshape(1, 1, -1), device=device, dtype=torch.float32)

    X, Y = next(iter(loader))
    X = X.to(device)
    Y = Y.to(device)

    with torch.no_grad():
        mu, sigma = _predict_mu_sigma(model, X)
        sigma = sigma * temps_t

    mu = mu.cpu().numpy()
    sigma = sigma.cpu().numpy()
    Y = Y.cpu().numpy()

    horizon = mu.shape[1]
    xs = np.arange(horizon)

    for ex in range(min(n_examples, mu.shape[0])):
        for j, name in enumerate(TARGET_NAMES):
            plt.figure(figsize=(9, 4))
            plt.plot(xs, mu[ex, :, j], label="Gaussian mean")
            plt.fill_between(
                xs,
                mu[ex, :, j] - z * sigma[ex, :, j],
                mu[ex, :, j] + z * sigma[ex, :, j],
                alpha=0.3,
                label=f"{int((1-alpha)*100)}% interval",
            )
            plt.plot(xs, Y[ex, :, j], "--", label="Ground truth")
            plt.xlabel("Prediction step")
            plt.ylabel(name)
            plt.title(f"Gaussian LSTM – {name} ({split_name}) | temps={temps_per_dof[j]:.3f}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"gaussian_{name}_{split_name}_ex{ex}.png"), dpi=300)
            plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["test", "ood"])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--calibrate_per_dof", action="store_true",
                        help="Calibrate per-DoF sigma_temp on VAL (grid search).")
    parser.add_argument("--temps", type=float, nargs=5, default=None,
                        help="Provide 5 temps for [u v p r phi]. If set, overrides calibration.")
    parser.add_argument("--grid", type=float, nargs="*", default=None,
                        help="Optional custom grid for calibration, e.g. --grid 0.05 0.08 0.1 0.12 0.15 0.2")
    args = parser.parse_args()

    config = _load_config(args.config)
    device = _get_device()
    print(f"Using device: {device}")

    # Dataloaders (expects your create_dataloaders signature create_dataloaders(config, horizon, batch_size))
    horizon = config["data"]["horizon"]
    batch_size = config.get("training", {}).get("batch_size", config.get("data", {}).get("batch_size", 256))

    history = int(config["data"]["history"])
    train_loader, val_loader, test_loader, ood_loader = create_dataloaders(history, horizon, batch_size)

    # infer dims from VAL loader (most stable)
    input_dim, target_dim = _infer_dims_from_loader(val_loader)
    print(f"✅ Inferred input_dim={input_dim}, target_dim={target_dim}")

    model = _build_model_from_data(config, input_dim, target_dim, device)
    print(f"Loading checkpoint: {os.path.join(args.model_dir, 'best_model.pt')}")
    _load_checkpoint(model, args.model_dir, device)
    model.eval()

    # pick evaluation loader
    eval_loader = test_loader if args.split == "test" else ood_loader
    split_upper = args.split.upper()

    # temps: either given, calibrated, or default ones
    if args.temps is not None:
        temps = np.array(args.temps, dtype=np.float32)
        print(f"✅ Using provided per-DoF temps: {temps.tolist()}")
    elif args.calibrate_per_dof:
        if args.grid is not None and len(args.grid) > 0:
            grid = list(map(float, args.grid))
        else:
            # A sensible grid around what you observed (0.05–0.5),
            # but still allows wider if needed.
            grid = [0.03, 0.05, 0.08, 0.10, 0.12, 0.126, 0.14, 0.16, 0.18,
                    0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.00]
        temps = calibrate_temps_per_dof(model, val_loader, device, args.alpha, grid)
        nominal = 1.0 - args.alpha
        print(f"✅ Calibrated per-DoF temps on VAL for nominal coverage={nominal:.3f}:")
        for name, t in zip(TARGET_NAMES, temps.tolist()):
            print(f"  {name}: {t:.3f}")
    else:
        temps = np.ones((5,), dtype=np.float32)
        print("ℹ️ Using default temps = [1,1,1,1,1]")

    # evaluate
    metrics = evaluate_gaussian_intervals(model, eval_loader, device, args.alpha, temps)

    print("\n==========================================")
    print(f"Gaussian per-DoF diagnostics ({split_upper} split)")
    print("------------------------------------------")
    print(f"alpha:               {args.alpha:0.3f}")
    print(f"nominal coverage:    {1.0-args.alpha:0.3f}")
    print("------------------------------------------")
    print(f"RMSE:                {metrics.rmse:0.6f}")
    print(f"Avg NLL:             {metrics.avg_nll:0.6f}")
    print(f"Overall coverage:    {metrics.overall_coverage:0.3f}")
    print(f"Overall width:       {metrics.overall_width:0.3f}")
    print("------------------------------------------")
    print("Per-DoF coverage / width / rmse / temp:")
    for name in TARGET_NAMES:
        d = metrics.per_dof[name]
        print(f"  {name:4s}  cov: {d['coverage']:0.3f}   width: {d['width']:0.3f}   rmse: {d['rmse']:0.6f}   temp: {d['sigma_temp']:0.3f}")
    print("==========================================\n")

    # save outputs
    out_dir = os.path.join(args.model_dir, "gaussian_diag_per_dof", args.split)
    _ensure_dir(out_dir)

    # save temps
    with open(os.path.join(out_dir, "temps_per_dof.txt"), "w") as f:
        f.write("u v p r phi\n")
        f.write(" ".join([f"{t:.6f}" for t in temps.tolist()]) + "\n")

    # save plots
    save_example_plots(model, eval_loader, device, args.alpha, temps, out_dir, args.split, n_examples=1)

    print(f"✅ Saved per-DoF Gaussian diagnostics to: {out_dir}")


if __name__ == "__main__":
    main()
