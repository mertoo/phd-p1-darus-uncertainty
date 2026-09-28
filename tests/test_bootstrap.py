"""Gate A/C: bootstrap entry point pairs methods on identical recordings."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src.analysis import bootstrap as bs

D, T = 5, 30


def fake_eval(path, recs=("a.csv", "b.csv", "c.csv", "d.csv"), per_rec=6, err=0.1, cover=25, seed=0,
              scale=(1, 1, 0.1, 0.1, 0.1), tag="m_channel"):
    rng = np.random.default_rng(seed)
    rows = []
    for r in recs:
        for w in range(per_rec):
            row = {"recording": r, "segment": 0, "start_row": w, "start_time": float(w), "n_steps": T}
            for d in range(D):
                row[f"sse_{d}"] = T * (err * (1 + d)) ** 2 * rng.uniform(0.5, 1.5)
                row[f"inside_{d}"] = int(rng.integers(cover - 5, cover + 5))
                row[f"width_{d}"] = T * 0.2
                row[f"iscore_{d}"] = T * 0.3
            rows.append(row)
    path.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(path / f"per_window_test_{tag}_90.csv", index=False)
    (path / "metrics.json").write_text(json.dumps({"provenance": {"scale_for_normalised_metrics": list(scale)}}))
    return pd.DataFrame(rows)


def run(tmp_path, spec):
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(spec))
    import sys
    argv = sys.argv
    sys.argv = ["x", "--spec", str(tmp_path / "spec.yaml"), "--out", str(tmp_path / "out")]
    try:
        bs.main()
    finally:
        sys.argv = argv
    return (pd.read_csv(tmp_path / "out" / "summary_test.csv"),
            pd.read_csv(tmp_path / "out" / "paired_test.csv"))


def test_estimates_match_direct_computation_and_self_difference_is_zero(tmp_path):
    frame = fake_eval(tmp_path / "e1")
    fake_eval(tmp_path / "e2", seed=1, err=0.2)
    spec = {"splits": ["test"], "n_boot": 300, "methods": {
        "A": {"tag": "m_channel", "evals": [str(tmp_path / "e1")]},
        "A2": {"tag": "m_channel", "evals": [str(tmp_path / "e1")]},
        "B": {"tag": "m_channel", "evals": [str(tmp_path / "e2")]}},
        "pairs": [["A", "A2"], ["B", "A"]]}
    summary, paired = run(tmp_path, spec)
    a = summary[summary.method == "A"].set_index("statistic")
    n = len(frame) * T
    assert a.loc["coverage_u", "estimate"] == pytest.approx(frame["inside_0"].sum() / n)
    assert a.loc["rmse_v", "estimate"] == pytest.approx(np.sqrt(frame["sse_1"].sum() / n))
    assert a.loc["coverage", "estimate"] == pytest.approx(
        np.mean([frame[f"inside_{d}"].sum() / n for d in range(D)]))
    assert (a["ci_low"] <= a["estimate"] + 1e-12).all() and (a["estimate"] <= a["ci_high"] + 1e-12).all()
    assert (a["n_recordings"] == 4).all()
    same = paired[(paired.method_a == "A") & (paired.method_b == "A2")]
    assert np.allclose(same[["difference", "ci_low", "ci_high"]].to_numpy(), 0.0)
    ba = paired[(paired.method_a == "B") & (paired.statistic == "rmse_norm_mean")].iloc[0]
    assert ba.difference > 0 and ba.ci_low > 0 and ba.share_draws_positive == 1.0


def test_repeats_give_seed_sd(tmp_path):
    fake_eval(tmp_path / "r0", seed=0)
    fake_eval(tmp_path / "r1", seed=5, err=0.15)
    summary, _ = run(tmp_path, {"splits": ["test"], "n_boot": 100, "methods": {
        "A": {"tag": "m_channel", "evals": [str(tmp_path / "r*")]}}})
    row = summary[summary.statistic == "rmse_u"].iloc[0]
    assert row.n_repeats == 2 and row.seed_sd > 0


def test_unpaired_windows_rejected(tmp_path):
    fake_eval(tmp_path / "e1")
    fake_eval(tmp_path / "e2", per_rec=5)
    with pytest.raises(ValueError, match="cannot pair"):
        run(tmp_path, {"splits": ["test"], "n_boot": 10, "methods": {
            "A": {"tag": "m_channel", "evals": [str(tmp_path / "e1")]},
            "B": {"tag": "m_channel", "evals": [str(tmp_path / "e2")]}}})


def test_different_scale_rejected(tmp_path):
    fake_eval(tmp_path / "e1")
    fake_eval(tmp_path / "e2", scale=(2, 1, 0.1, 0.1, 0.1))
    with pytest.raises(ValueError, match="scale"):
        run(tmp_path, {"splits": ["test"], "n_boot": 10, "methods": {
            "A": {"tag": "m_channel", "evals": [str(tmp_path / "e1")]},
            "B": {"tag": "m_channel", "evals": [str(tmp_path / "e2")]}}})
