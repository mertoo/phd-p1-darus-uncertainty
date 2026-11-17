"""Point and interval metrics computed from full prediction arrays.

All functions take arrays of shape (N, T, D) in physical units and an
optional boolean `mask` of the same shape (True = counted). Aggregates are
count-weighted over the scalar predictions that are counted, so batch size
never affects a result.

RMSE conventions (reported separately, never interchanged):
  rmse_channel[d]   sqrt(mean over N, T of err^2) for channel d, physical units
  rmse_pooled       sqrt(mean over N, T, D of err^2) -- dominated by u, v
  rmse_norm_mean    mean over d of rmse_channel[d] / scale[d]  (dimensionless)
"""

import numpy as np

from src.uncertainty.intervals import Intervals


def _mask(shape, mask):
    if mask is None:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != tuple(shape):
        raise ValueError("mask shape mismatch")
    return mask


def _wmean(values, mask, axis):
    num = np.where(mask, values, 0.0).sum(axis=axis)
    den = mask.sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def point_metrics(y, yhat, scale=None, mask=None):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = _mask(y.shape, mask)
    se = (yhat - y) ** 2
    out = {
        "n_scalar": int(m.sum()),
        "rmse_channel": np.sqrt(_wmean(se, m, (0, 1))),
        "rmse_horizon_channel": np.sqrt(_wmean(se, m, 0)),
        "rmse_pooled": float(np.sqrt(_wmean(se, m, None))),
        "mae_channel": _wmean(np.abs(yhat - y), m, (0, 1)),
    }
    if scale is not None:
        out["rmse_norm_mean"] = float(np.mean(out["rmse_channel"] / np.asarray(scale)))
    return out


def interval_score(y, lower, upper, level):
    """Winkler / interval score (Gneiting & Raftery 2007), elementwise; lower is better."""
    alpha = 1.0 - level
    width = upper - lower
    below = (2.0 / alpha) * np.clip(lower - y, 0, None)
    above = (2.0 / alpha) * np.clip(y - upper, 0, None)
    return width + below + above


def interval_metrics(y, iv: Intervals, scale=None, mask=None):
    """
    Coverage / width / interval score for one Intervals object.

    Marginal coverage counts individual scalar predictions. Trajectory
    coverage is simultaneous: a window counts as covered for a channel only if
    all T horizon steps are inside ("traj_channel"), or for all channels and
    steps ("traj_all"). Windows with any masked element are excluded from the
    trajectory metrics.
    """
    y = np.asarray(y, float)
    m = _mask(y.shape, mask)
    inside = (y >= iv.lower) & (y <= iv.upper)
    width = iv.upper - iv.lower
    isc = interval_score(y, iv.lower, iv.upper, iv.level)
    full = m.all(axis=(1, 2))

    out = {
        "method": iv.method,
        "variant": iv.variant,
        "level": iv.level,
        "n_scalar": int(m.sum()),
        "n_windows_full": int(full.sum()),
        "coverage": float(_wmean(inside, m, None)),
        "coverage_channel": _wmean(inside, m, (0, 1)),
        "coverage_horizon": _wmean(inside, m, (0, 2)),
        "coverage_horizon_channel": _wmean(inside, m, 0),
        "count_channel": m.sum(axis=(0, 1)),
        "width_channel": _wmean(width, m, (0, 1)),
        "width_horizon_channel": _wmean(width, m, 0),
        "interval_score_channel": _wmean(isc, m, (0, 1)),
        "traj_coverage_channel": inside[full].all(axis=1).mean(axis=0) if full.any() else np.full(y.shape[2], np.nan),
        "traj_coverage_all": float(inside[full].all(axis=(1, 2)).mean()) if full.any() else float("nan"),
        "frac_infinite": float(np.isinf(width)[m].mean()),
    }
    if scale is not None:
        scale = np.asarray(scale, float)
        out["width_norm_channel"] = out["width_channel"] / scale
        out["width_norm_mean"] = float(np.mean(out["width_norm_channel"]))
        out["interval_score_norm_mean"] = float(np.mean(out["interval_score_channel"] / scale))
    return out


def per_window_frame(y, yhat, iv: Intervals | None, meta):
    """
    Per-window summaries (for recording-level resampling): squared error and
    coverage counts per channel, one row per window, joined to window metadata.
    """
    import pandas as pd
    se = ((yhat - y) ** 2).sum(axis=1)                   # (N, D)
    cols = {f"sse_{d}": se[:, d] for d in range(y.shape[2])}
    if iv is not None:
        inside = ((y >= iv.lower) & (y <= iv.upper)).sum(axis=1)
        width = (iv.upper - iv.lower).sum(axis=1)
        isc = interval_score(y, iv.lower, iv.upper, iv.level).sum(axis=1)
        for d in range(y.shape[2]):
            cols[f"inside_{d}"] = inside[:, d]
            cols[f"width_{d}"] = width[:, d]
            cols[f"iscore_{d}"] = isc[:, d]
    df = pd.DataFrame(cols)
    df["n_steps"] = y.shape[1]
    return pd.concat([meta.reset_index(drop=True), df], axis=1)
