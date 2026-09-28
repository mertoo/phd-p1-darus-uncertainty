import json

import numpy as np
import pandas as pd
import pytest

from src.analysis import floor_sensitivity as fs
from src.evaluation.uq_metrics import interval_metrics
from src.uncertainty.intervals import scaled_spread_intervals


def make_eval(path, zero_frac=0.0, tamper=False, seed=0):
    rng = np.random.default_rng(seed)
    N, T, D = 400, 3, 2
    arrs = {}
    for sp in ("cal", "test"):
        y = rng.normal(size=(N, T, D))
        s = np.abs(rng.normal(1.0, 0.2, size=(N, T, D)))
        s[: int(zero_frac * N), :, 0] = 0.0                  # a block of zero spreads in channel 0
        arrs[sp] = {"y": y, "mean": np.zeros_like(y), "spread": s}
    path.mkdir(parents=True)
    for sp, a in arrs.items():
        np.savez(path / f"predictions_{sp}.npz", **{k: v.astype(np.float32) for k, v in a.items()})
    a32 = {sp: {k: v.astype(np.float32).astype(np.float64) for k, v in a.items()} for sp, a in arrs.items()}
    ivs = []
    for level in (0.9,):
        for variant in ("channel", "horizon_channel"):
            iv = scaled_spread_intervals(a32["cal"]["mean"], a32["cal"]["spread"], a32["cal"]["y"],
                                         a32["test"]["mean"], a32["test"]["spread"], level, variant, "ensemble")
            m = interval_metrics(a32["test"]["y"], iv, scale=np.ones(D))
            m.update(split="test")
            if tamper:
                m["coverage"] += 0.01
            ivs.append({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in m.items()})
    (path / "metrics.json").write_text(json.dumps({"intervals": ivs, "provenance": {
        "method": "ensemble", "calibration_split": "cal", "eval_splits": ["test"],
        "scale_for_normalised_metrics": [1.0] * D, "floor_rel": 1e-3}}))
    return path


def test_rows_and_trigger(tmp_path):
    rows = fs.analyse(str(make_eval(tmp_path / "e", zero_frac=0.05)))
    df = pd.DataFrame(rows)
    assert set(df.floor_rel) == {1e-4, 1e-3, 1e-2}
    prim = df[df.primary]
    assert np.allclose(prim.d_coverage_vs_primary, 0) and df.triggered.all()   # 5% of channel 0 floored > 1%
    assert not pd.DataFrame(fs.analyse(str(make_eval(tmp_path / "f")))).triggered.any()


def test_rebuild_mismatch_fails(tmp_path):
    with pytest.raises(RuntimeError, match="rebuild differs"):
        fs.analyse(str(make_eval(tmp_path / "e", tamper=True)))
