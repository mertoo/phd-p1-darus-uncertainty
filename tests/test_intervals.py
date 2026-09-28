"""Gate A: conformal edge cases, aggregation identities, variance convention."""

import numpy as np
import pytest

from src.evaluation.uq_metrics import interval_metrics, point_metrics
from src.uncertainty.intervals import (
    Intervals, conformal_intervals, conformal_threshold, normal_z,
    raw_gaussian_intervals, scaled_spread_intervals,
)


def test_order_statistic():
    s = np.arange(1, 10, dtype=float)                     # n = 9
    assert conformal_threshold(s, 0.9) == 9.0             # k = ceil(10*0.9) = 9
    assert conformal_threshold(s, 0.5) == 5.0             # k = 5
    assert np.isinf(conformal_threshold(s, 0.95))         # k = 10 > n -> unbounded, not clipped


def test_finite_sample_coverage_exchangeable():
    # Exchangeable continuous scores: P(test <= k-th of n) = k/(n+1) >= level.
    rng = np.random.default_rng(0)
    s = np.abs(rng.normal(size=(20000, 51)))
    q = conformal_threshold(s[:, :-1], 0.9, axis=1)          # n = 50, k = 46
    cov = np.mean(s[:, -1] <= q)
    assert cov == pytest.approx(46 / 51, abs=0.006)
    assert cov >= 0.9 - 0.006


def test_variants_and_identities():
    rng = np.random.default_rng(1)
    N, T, D = 200, 6, 3
    scale = np.array([1.0, 10.0, 0.1])
    y_cal = rng.normal(size=(N, T, D)) * scale
    y = rng.normal(size=(N, T, D)) * scale
    zeros = np.zeros_like(y)
    for variant in ("channel", "horizon_channel"):
        iv = conformal_intervals(np.zeros_like(y_cal), y_cal, zeros, 0.9, variant)
        m = interval_metrics(y, iv)
        # equal counts, no mask: overall == mean of channel == mean of horizon coverage
        assert m["coverage"] == pytest.approx(m["coverage_channel"].mean())
        assert m["coverage"] == pytest.approx(m["coverage_horizon"].mean())
        assert m["coverage"] == pytest.approx(m["coverage_horizon_channel"].mean())
        # widths are per channel and in that channel's units
        assert m["width_channel"][1] > 50 * m["width_channel"][2]
    iv_c = conformal_intervals(np.zeros_like(y_cal), y_cal, zeros, 0.9, "channel")
    assert iv_c.info["threshold"].shape == (1, D)


def test_masked_weighting():
    y = np.zeros((4, 2, 1))
    iv = Intervals(y - 1, y + 1, "x", "v", 0.9)
    y2 = y.copy()
    y2[0, 0, 0] = 5.0                                     # one miss
    mask = np.ones_like(y, dtype=bool)
    mask[1:, 1, 0] = False                                # horizon 1 has 1 obs, horizon 0 has 4
    m = interval_metrics(y2, iv, mask=mask)
    assert m["n_scalar"] == 5
    assert m["coverage"] == pytest.approx(4 / 5)
    # weighted identity: overall == count-weighted mean of horizon coverage
    counts = mask.sum(axis=(0, 2))
    assert m["coverage"] == pytest.approx((m["coverage_horizon"] * counts).sum() / counts.sum())
    assert m["n_windows_full"] == 1


def test_metrics_are_batch_invariant():
    rng = np.random.default_rng(2)
    y, yhat = rng.normal(size=(101, 5, 2)), rng.normal(size=(101, 5, 2))
    full = point_metrics(y, yhat)["rmse_channel"]
    # emulate the old per-batch averaging bug: unequal last batch changes the result
    batches = [slice(0, 64), slice(64, 101)]
    naive = np.sqrt(np.mean([((yhat[b] - y[b]) ** 2).mean(axis=(0, 1)) for b in batches], axis=0))
    assert not np.allclose(naive, full)
    # sums/counts reproduce the full-array result
    sse = sum(((yhat[b] - y[b]) ** 2).sum(axis=(0, 1)) for b in batches)
    np.testing.assert_allclose(np.sqrt(sse / (101 * 5)), full)


def test_raw_gaussian_band_and_calibrated_spread():
    rng = np.random.default_rng(3)
    N, T, D = 3000, 4, 2
    true_sd = np.array([2.0, 0.5])
    mu_cal = np.zeros((N, T, D)); mu = np.zeros((N, T, D))
    s_cal = np.ones((N, T, D)); s = np.ones((N, T, D))          # badly scaled spread
    y_cal = rng.normal(size=(N, T, D)) * true_sd
    y = rng.normal(size=(N, T, D)) * true_sd
    raw = raw_gaussian_intervals(mu, s, 0.9, "ens")
    assert np.allclose(raw.upper, normal_z(0.9))
    raw_cov = interval_metrics(y, raw)["coverage_channel"]
    assert raw_cov[0] < 0.7 and raw_cov[1] > 0.99
    cal = scaled_spread_intervals(mu_cal, s_cal, y_cal, mu, s, 0.9, "channel", "ens")
    cov = interval_metrics(y, cal)["coverage_channel"]
    assert np.all(np.abs(cov - 0.9) < 0.02)


def test_ensemble_std_uses_n_minus_1():
    torch = pytest.importorskip("torch")
    from src.uncertainty.deep_ensemble import predict_ensemble

    class Const(torch.nn.Module):
        def __init__(self, c):
            super().__init__()
            self.c = c

        def forward(self, x):
            return torch.full((x.shape[0], 1, 1), float(self.c))

    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    mean, std = predict_ensemble([Const(v) for v in vals], torch.zeros(2, 1), "cpu")
    assert mean[0, 0, 0] == pytest.approx(3.0)
    assert std[0, 0, 0] == pytest.approx(np.std(vals, ddof=1))


# ---- zero / near-zero spread, NaN inputs, legitimate infinite intervals ----
from src.uncertainty.intervals import DegenerateSpreadError, spread_floor


def _cal_setup(seed=4, N=500, T=3, D=2):
    rng = np.random.default_rng(seed)
    y_cal = rng.normal(size=(N, T, D))
    y = rng.normal(size=(N, T, D))
    return y_cal, y, np.zeros((N, T, D))


def test_zero_spread_channel_gets_positive_scale_aware_floor():
    y_cal, y, mu = _cal_setup()
    s = np.ones_like(y_cal)
    s[..., 1] = 0.0                                        # a method reporting no spread on channel 1
    floor = spread_floor(s, y_cal)
    assert np.all(floor > 0)
    assert floor[0, 0, 1] == pytest.approx(1e-3 * y_cal[..., 1].std())
    s_eval = np.ones_like(y)
    s_eval[..., 1] = 0.0                                   # same behaviour at evaluation time
    iv = scaled_spread_intervals(mu, s, y_cal, mu, s_eval, 0.9, "channel", "ens")
    assert np.all(np.isfinite(iv.lower)) and np.all(np.isfinite(iv.upper))
    cov = interval_metrics(y, iv)["coverage_channel"]
    assert np.all(np.abs(cov - 0.9) < 0.04)                # floored score still calibrates
    assert iv.info["frac_floored_cal"][1] == 1.0 and iv.info["frac_floored_cal"][0] == 0.0


def test_near_zero_spread_is_floored_not_exploded():
    y_cal, y, mu = _cal_setup()
    s = np.full_like(y_cal, 1e-300)
    iv = scaled_spread_intervals(mu, s, y_cal, mu, s[: len(y)], 0.9, "horizon_channel", "mcd")
    assert np.all(np.isfinite(iv.upper - iv.lower))
    assert np.all(iv.info["frac_floored_cal"] == 1.0)


def test_degenerate_channel_raises():
    y_cal, y, mu = _cal_setup()
    y_cal[..., 0] = 3.0                                    # constant target ...
    s = np.ones_like(y_cal)
    s[..., 0] = 0.0                                        # ... and zero spread
    with pytest.raises(DegenerateSpreadError):
        scaled_spread_intervals(mu, s, y_cal, mu, s[: len(y)], 0.9, "channel", "ens")


@pytest.mark.parametrize("which", ["mu", "s", "y_cal"])
def test_nan_inputs_raise(which):
    y_cal, y, mu = _cal_setup()
    args = {"mu": mu.copy(), "s": np.ones_like(y_cal), "y_cal": y_cal.copy()}
    args[which][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        scaled_spread_intervals(args["mu"], args["s"], args["y_cal"], mu, np.ones_like(y), 0.9, "channel", "x")
    if which != "s":                                       # conformal takes no spread
        with pytest.raises(ValueError, match="non-finite"):
            conformal_intervals(args["mu"], args["y_cal"], mu, 0.9, "channel")


def test_nan_bounds_rejected():
    z = np.zeros((2, 2, 1))
    with pytest.raises(ValueError, match="NaN"):
        Intervals(z, np.full_like(z, np.nan), "x", "v", 0.9)


def test_infinite_conformal_interval_is_legitimate_and_reported():
    # n = 5 calibration windows, level 0.95 -> k = ceil(6*0.95) = 6 > 5 -> unbounded
    y_cal, y, _ = _cal_setup(N=5)
    for variant in ("channel", "horizon_channel"):
        n_cal = 5 if variant == "horizon_channel" else 5 * 3
        iv = conformal_intervals(np.zeros_like(y_cal), y_cal, np.zeros((7, 3, 2)), 0.95, variant)
        m = interval_metrics(np.zeros((7, 3, 2)), iv)
        if variant == "horizon_channel":
            assert np.all(np.isinf(iv.upper)) and m["frac_infinite"] == 1.0
            assert m["coverage"] == 1.0 and np.all(np.isinf(m["width_channel"]))
            assert not np.isnan(m["interval_score_channel"]).any()
        else:                                              # 15 pooled scores: k = 16 > 15 too
            assert n_cal == 15 and m["frac_infinite"] == 1.0
    # spread-normalised: inf threshold times a positive floored spread stays inf, never NaN
    s = np.zeros_like(y_cal)
    iv = scaled_spread_intervals(np.zeros_like(y_cal), s, y_cal, np.zeros((7, 3, 2)),
                                 np.zeros((7, 3, 2)), 0.95, "horizon_channel", "ens")
    assert np.all(np.isinf(iv.upper)) and not np.isnan(iv.upper).any()
