"""Interval construction shared by every UQ method.

Each reported result is one `Intervals` object (lower/upper arrays of shape
(N, T, D) in physical units). Coverage, width and interval score are always
computed from those arrays by `src.evaluation.uq_metrics`, never from a
separate quantile, so a table row cannot mix interval definitions.

Calibration variants (all marginal, i.e. they target coverage of individual
scalar predictions, not simultaneous coverage of whole trajectories):

  "channel"          one threshold per output channel d, from the scores of
                     every calibration window and horizon step of that channel.
  "horizon_channel"  one threshold per (horizon step t, channel d), from the
                     N calibration windows at that (t, d).

Scores are either absolute residuals |y - yhat| (conformal on a point
forecaster) or spread-normalised residuals |y - mu| / max(s, floor) (used to
calibrate ensemble / MC-dropout / Gaussian spreads on the same calibration
recordings).

Thresholds use the split-conformal order statistic k = ceil((n + 1)(1 - alpha))
of the n calibration scores for each group. If k > n the threshold is +inf
(an unbounded interval) rather than being clipped. Calibration windows overlap
in time and come from a limited number of recordings, so they are not
exchangeable draws; the order statistic is the standard finite-sample
construction, but no formal coverage guarantee is claimed for this data.
"""

from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np

VARIANTS = ("channel", "horizon_channel")


def normal_z(level):
    """Two-sided standard normal quantile for central coverage `level`."""
    return NormalDist().inv_cdf(0.5 + level / 2.0)


def conformal_threshold(scores, level, axis=0):
    """
    Finite-sample split-conformal threshold along `axis`.

    scores: array of non-negative scores; the calibration sample runs along `axis`.
    Returns the k-th smallest score with k = ceil((n+1) * level), or +inf when k > n.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[axis]
    k = int(np.ceil((n + 1) * level))
    if k > n:
        shape = list(scores.shape)
        del shape[axis]
        return np.full(shape, np.inf)
    return np.partition(scores, k - 1, axis=axis).take(k - 1, axis=axis)


def calibrate_scores(scores, level, variant):
    """
    scores: (N, T, D) calibration scores.
    Returns a threshold array broadcastable to (N, T, D).
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 3:
        raise ValueError("scores must have shape (N, T, D)")
    N, T, D = scores.shape
    if variant == "channel":
        q = conformal_threshold(scores.reshape(N * T, D), level, axis=0)   # (D,)
        return q.reshape(1, 1, D)
    if variant == "horizon_channel":
        return conformal_threshold(scores, level, axis=0)[None]            # (1, T, D)
    raise ValueError(f"Unknown calibration variant '{variant}'. Options: {VARIANTS}")


@dataclass
class Intervals:
    lower: np.ndarray                 # (N, T, D), physical units
    upper: np.ndarray
    method: str                       # e.g. "conformal", "ensemble"
    variant: str                      # "raw_gaussian", "channel", "horizon_channel", ...
    level: float                      # nominal central coverage
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower/upper shape mismatch")
        # NaN bounds are always an error. +/-inf is legitimate (finite-sample
        # conformal threshold with k > n) and is reported via frac_infinite.
        if np.isnan(self.lower).any() or np.isnan(self.upper).any():
            raise ValueError(f"{self.method}/{self.variant}: NaN interval bounds")
        if np.any(self.upper < self.lower):
            raise ValueError("upper < lower")


class DegenerateSpreadError(ValueError):
    """A channel has zero calibration spread and zero calibration-target variation."""


def _check_finite(name, *arrays):
    for a in arrays:
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{name}: non-finite values in inputs (NaN/inf predictions, spreads or targets)")


def spread_floor(spread_cal, y_cal, rel=1e-3):
    """
    Predeclared, strictly positive floor for spread-normalised scores, per channel:

        floor_d = rel * max(mean calibration spread_d, std of calibration targets_d)

    The second term keeps the floor positive and in the channel's physical
    scale when a method reports (near-)zero spread for a whole channel. If
    both are zero, the channel is degenerate (constant target, no spread) and
    no meaningful normalised score exists, so DegenerateSpreadError is raised
    rather than silently returning NaN/inf intervals. Never tuned on test data.
    Returns shape (1, 1, D).
    """
    spread_cal = np.asarray(spread_cal, dtype=np.float64)
    y_cal = np.asarray(y_cal, dtype=np.float64)
    if np.any(spread_cal < 0):
        raise ValueError("negative spread")
    mean_spread = spread_cal.mean(axis=(0, 1))
    y_std = y_cal.reshape(-1, y_cal.shape[-1]).std(axis=0)
    ref = np.maximum(mean_spread, y_std)
    if np.any(ref <= 0):
        bad = np.where(ref <= 0)[0].tolist()
        raise DegenerateSpreadError(f"channels {bad}: zero calibration spread and constant targets")
    return (rel * ref).reshape(1, 1, -1)


def conformal_intervals(yhat_cal, y_cal, yhat, level, variant, method="conformal"):
    """Absolute-residual split conformal around point predictions."""
    _check_finite("conformal", yhat_cal, y_cal, yhat)
    q = calibrate_scores(np.abs(np.asarray(y_cal) - np.asarray(yhat_cal)), level, variant)
    q_full = np.broadcast_to(q, yhat.shape)
    return Intervals(yhat - q_full, yhat + q_full, method, variant, level,
                     info={"threshold": np.asarray(q).squeeze(0), "score": "abs_residual",
                           "n_cal_windows": int(np.asarray(y_cal).shape[0])})


def raw_gaussian_intervals(mu, sigma, level, method):
    """Uncalibrated mean +/- z * sigma band (Gaussian-reference nominal level).
    Zero spread gives a zero-width band (reported as such, not floored)."""
    _check_finite(method, mu, sigma)
    z = normal_z(level)
    return Intervals(mu - z * sigma, mu + z * sigma, method, "raw_gaussian", level,
                     info={"z": z})


def scaled_spread_intervals(mu_cal, s_cal, y_cal, mu, s, level, variant, method, floor_rel=1e-3):
    """
    Calibrate a spread s (ensemble std, dropout std, Gaussian sigma) with the
    spread-normalised score |y - mu| / max(s, floor). The resulting interval is
    mu +/- q * max(s, floor); for a Gaussian this is an empirically fitted
    per-group temperature T = q / z.
    """
    _check_finite(method, mu_cal, s_cal, y_cal, mu, s)
    floor = spread_floor(s_cal, y_cal, floor_rel)
    s_cal_f = np.maximum(s_cal, floor)
    s_f = np.maximum(s, floor)
    q = calibrate_scores(np.abs(y_cal - mu_cal) / s_cal_f, level, variant)
    half = q * s_f                      # s_f > 0, so q = inf gives inf, never NaN
    return Intervals(mu - half, mu + half, method, variant, level,
                     info={"threshold": np.asarray(q).squeeze(0), "score": "spread_normalised",
                           "floor": floor.squeeze((0, 1)), "floor_rel": floor_rel,
                           "frac_floored_cal": (s_cal < floor).mean(axis=(0, 1)),
                           "frac_floored_eval": (s < floor).mean(axis=(0, 1)),
                           "n_cal_windows": int(np.asarray(y_cal).shape[0])})
