import matplotlib.pyplot as plt
import numpy as np


DOF_NAMES_DEFAULT = ["u", "v", "p", "r", "phi"]


def plot_prediction_vs_truth(Y, Yhat, savepath, dof_names=None, sample_idx: int = 0):
    """
    Plot prediction vs truth for a single sample across all DoFs.

    Parameters
    ----------
    Y : np.ndarray, shape (N, H, D)
        Ground truth sequences.
    Yhat : np.ndarray, shape (N, H, D)
        Predicted sequences.
    savepath : str
        Where to save the figure.
    dof_names : list[str], optional
        Names of the DoFs, length D. Defaults to ["u","v","p","r","phi"].
    sample_idx : int
        Which sample in the batch to visualize.
    """
    if dof_names is None:
        dof_names = DOF_NAMES_DEFAULT

    # Ensure arrays
    Y = np.asarray(Y)
    Yhat = np.asarray(Yhat)

    if Y.ndim != 3 or Yhat.ndim != 3:
        raise ValueError(f"Expected Y, Yhat with shape (N,H,D), got {Y.shape}, {Yhat.shape}")

    N, H, D = Y.shape
    sample_idx = max(0, min(sample_idx, N - 1))

    y = Y[sample_idx]       # (H, D)
    yhat = Yhat[sample_idx] # (H, D)

    if len(dof_names) != D:
        dof_names = [f"DoF {i}" for i in range(D)]

    timesteps = np.arange(H)

    fig, axes = plt.subplots(D, 1, figsize=(10, 2.5 * D), sharex=True)
    if D == 1:
        axes = [axes]

    for d in range(D):
        axes[d].plot(timesteps, y[:, d], label="Truth", linewidth=1.5)
        axes[d].plot(timesteps, yhat[:, d], "--", label="Pred", linewidth=1.2)
        axes[d].set_ylabel(dof_names[d])
        axes[d].grid(True, alpha=0.3)

        if d == 0:
            axes[d].set_title("Prediction vs Truth (single trajectory)")

    axes[-1].set_xlabel("Forecast step")
    axes[0].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


def plot_multi_horizon_rmse(Y, Yhat, savepath, dof_names=None):
    """
    Plot RMSE as a function of forecast horizon, averaged over samples and DoFs.

    Parameters
    ----------
    Y : np.ndarray, shape (N, H, D)
    Yhat : np.ndarray, shape (N, H, D)
    savepath : str
    dof_names : unused for now (reserved for per-DoF curves later)
    """
    Y = np.asarray(Y)
    Yhat = np.asarray(Yhat)

    if Y.ndim != 3 or Yhat.ndim != 3:
        raise ValueError(f"Expected Y, Yhat with shape (N,H,D), got {Y.shape}, {Yhat.shape}")

    # Squared errors: (N, H, D)
    sq_err = (Yhat - Y) ** 2
    # Mean over N and D → per-horizon error (H,)
    rmse_h = np.sqrt(sq_err.mean(axis=(0, 2)))

    H = rmse_h.shape[0]
    horizons = np.arange(1, H + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(horizons, rmse_h, marker="o", linewidth=1.5)
    ax.set_xlabel("Forecast step")
    ax.set_ylabel("RMSE (averaged over DoFs)")
    ax.set_title("RMSE vs Forecast Horizon")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
