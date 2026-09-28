"""Pilot report: correctness/resource failures vs. diagnostics."""

import json
import sys

import pytest

from scripts import pilot_report

CAL = ["a.csv", "b.csv"]


def write_run(root, name, train=(3.0, 2.0, 1.0), val=(3.0, 2.0, 1.0), cal=CAL, scaler="s1", rss=2.0):
    d = root / "runs" / name
    d.mkdir(parents=True)
    eps = [{"epoch": i + 1, "train_loss": t, "val_loss": v, "elapsed_s": 10.0 * (i + 1),
            "gpu_max_mem_gb": 1.0, "peak_rss_gb": rss} for i, (t, v) in enumerate(zip(train, val))]
    (d / "train_log.json").write_text(json.dumps({
        "epochs": eps, "sec_per_epoch": 10.0, "train_windows_per_sec": 1000.0, "split_hash": "h",
        "scaler_hash": scaler, "features": ["n", "u"], "cal_files": cal, "normalize": True}))


def write_eval(root, name):
    d = root / "evals" / name
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({
        "intervals": [{"method": "m", "variant": "channel", "level": 0.9, "coverage": 0.9,
                       "width_norm_mean": 1.0, "interval_score_norm_mean": 2.0, "frac_infinite": 0.0}],
        "provenance": {"method": "point", "eval_splits": ["val"], "gpu_max_mem_gb": 1.0,
                       "member_validation": {}, "data_state": {"split_files": {"cal": CAL}},
                       "timing": {"val": {"n_windows": 10, "inference_s": 1.0}}}}))


def setup(tmp_path, **run_over):
    (tmp_path / "pilot.tsv").write_text("task\tconfig\tseed\trun_name\n1\tlstm\t1\tpilot/lstm_s1\n2\tmlp\t1\tpilot/mlp_s1\n")
    (tmp_path / "plan.tsv").write_text("stage\tconfig\tseed\trun_name\tfeatures_override\n1a\tlstm\t1\tx\t\n1a\tmlp\t1\ty\t\n")
    (tmp_path / "cal.json").write_text(json.dumps({"primary": "random", "random": {"files": CAL}}))
    write_run(tmp_path, "lstm_s1", **run_over)
    write_run(tmp_path, "mlp_s1")
    write_eval(tmp_path, "point_lstm")
    (tmp_path / "evals" / "bootstrap").mkdir()
    (tmp_path / "evals" / "bootstrap" / "summary_val.csv").write_text("x\n")


def run(tmp_path):
    argv = sys.argv
    sys.argv = ["x", "--runs", str(tmp_path / "runs"), "--evals", str(tmp_path / "evals"),
                "--run_list", str(tmp_path / "plan.tsv"), "--pilot_list", str(tmp_path / "pilot.tsv"),
                "--cal_manifest", str(tmp_path / "cal.json"), "--expected_evals", "point_lstm",
                "--out", str(tmp_path / "report.json")]
    try:
        with pytest.raises(SystemExit) as e:
            pilot_report.main()
    finally:
        sys.argv = argv
    return e.value.code, json.loads((tmp_path / "report.json").read_text())


def test_clean_pilot_passes(tmp_path):
    setup(tmp_path)
    code, rep = run(tmp_path)
    assert code == 0 and rep["status"] == "PASS" and rep["diagnostics"] == []


def test_non_monotonic_loss_is_diagnostic_not_failure(tmp_path):
    setup(tmp_path, val=(3.0, 3.5, 2.0))
    code, rep = run(tmp_path)
    assert code == 0 and rep["status"] == "PASS"
    assert any("val loss not monotonically" in d for d in rep["diagnostics"])


@pytest.mark.parametrize("over, fragment", [
    ({"train": (3.0, float("nan"), 1.0)}, "all losses finite"),
    ({"cal": ["z.csv", "b.csv"]}, "calibration recordings"),
    ({"scaler": "other"}, "identical scaler_hash"),
    ({"rss": 20.0}, "host RSS"),
])
def test_failures(tmp_path, over, fragment):
    setup(tmp_path, **over)
    code, rep = run(tmp_path)
    assert code == 1 and rep["status"] == "FAIL"
    assert any(fragment in f for f in rep["failed"])
