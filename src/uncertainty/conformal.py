import numpy as np


def compute_absolute_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute absolute errors |y_true - y_pred|.

    Parameters
    ----------
    y_true : np.ndarray
        Shape (N, H, D).
    y_pred : np.ndarray
        Shape (N, H, D).

    Returns
    -------
    np.ndarray
        Absolute errors, shape (N, H, D).
    """
    assert y_true.shape == y_pred.shape, "Shapes must match for conformal errors."
    return np.abs(y_true - y_pred)


def fit_split_conformal(
    y_cal: np.ndarray,
    yhat_cal: np.ndarray,
    alpha: float = 0.1,
) -> np.ndarray:
    """
    Fit split conformal quantiles on a calibration set.

    We compute a separate quantile for each (horizon, DoF) pair.

    Parameters
    ----------
    y_cal : np.ndarray
        Calibration targets, shape (N_cal, H, D).
    yhat_cal : np.ndarray
        Calibration predictions, shape (N_cal, H, D).
    alpha : float
        Miscoverage rate (e.g., 0.1 for 90% intervals).

    Returns
    -------
    q : np.ndarray
        Quantiles per horizon and DoF, shape (H, D).
        Intervals are y_hat ± q.
    """
    errs = compute_absolute_errors(y_cal, yhat_cal)  # (N, H, D)
    N, H, D = errs.shape

    # Flatten over N, keep H*D
    errs_flat = errs.reshape(N, H * D)
    q_flat = np.quantile(errs_flat, 1.0 - alpha, axis=0)
    q = q_flat.reshape(H, D)
    return q


def conformal_predict(
    yhat: np.ndarray,
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply split conformal intervals to new predictions.

    Parameters
    ----------
    yhat : np.ndarray
        Point predictions, shape (N, H, D).
    q : np.ndarray
        Quantiles per (H, D), shape (H, D).

    Returns
    -------
    lower, upper : np.ndarray, np.ndarray
        Lower and upper bounds of intervals, both shape (N, H, D).
    """
    assert yhat.shape[1:] == q.shape, "q must have shape (H, D) matching yhat."
    lower = yhat - q[None, ...]
    upper = yhat + q[None, ...]
    return lower, upper


def compute_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict:
    """
    Compute empirical coverage of conformal intervals.

    Parameters
    ----------
    y_true : np.ndarray
        True targets, shape (N, H, D).
    lower : np.ndarray
        Lower bounds, shape (N, H, D).
    upper : np.ndarray
        Upper bounds, shape (N, H, D).

    Returns
    -------
    dict with:
        - "overall": scalar coverage
        - "per_dof": shape (D,) coverage
        - "per_horizon": shape (H,) coverage
        - "per_horizon_dof": shape (H, D) coverage
    """
    assert y_true.shape == lower.shape == upper.shape

    inside = (y_true >= lower) & (y_true <= upper)  # (N, H, D)
    N, H, D = y_true.shape

    overall = inside.mean()
    per_dof = inside.mean(axis=(0, 1))          # (D,)
    per_horizon = inside.mean(axis=(0, 2))      # (H,)
    per_horizon_dof = inside.mean(axis=0)       # (H, D)

    return {
        "overall": float(overall),
        "per_dof": per_dof,
        "per_horizon": per_horizon,
        "per_horizon_dof": per_horizon_dof,
    }
