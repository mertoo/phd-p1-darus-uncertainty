"""Frozen analysis rules: example-window selection; missing inputs block output."""

import json

import numpy as np
import pytest
import yaml

from src.analysis import report


class FakeEval:
    def __init__(self, y, mean, scale):
        self._a = {"y": y, "mean": mean}
        self.scale = np.asarray(scale, float)

    def arrays(self, split):
        return self._a


def test_example_window_is_median_with_smallest_id_on_ties():
    y = np.zeros((5, 2, 2))
    err = np.array([0.1, 0.3, 0.2, 0.2, 0.9])                 # per-window errors; median 0.2 at ids 2 and 3
    mean = y + err[:, None, None]
    assert report.select_example_window(FakeEval(y, mean, [1.0, 1.0]), "test") == 2
    # scale enters the rule: channel 1 dominates when its scale is small
    mean2 = y.copy()
    mean2[:, :, 1] = err[:, None]
    assert report.select_example_window(FakeEval(y, mean2, [1.0, 0.5]), "test") == 2


def test_missing_required_evaluation_blocks_output(tmp_path):
    (tmp_path / "eval_list.tsv").write_text("task\tmethod\truns\tn_members\ttag\n1\tpoint\tx\t\tmissing_eval\n")
    rules = yaml.safe_load(open("experiments/configs/v3/analysis_rules.yaml"))
    rules.update(eval_root=str(tmp_path / "evals"), bootstrap_dir=str(tmp_path / "boot"),
                 out_dir=str(tmp_path / "out"), eval_list=str(tmp_path / "eval_list.tsv"))
    with pytest.raises(FileNotFoundError, match="missing_eval"):
        report.check_required(report.Inputs(rules), rules)
    assert not (tmp_path / "out").exists()


def test_missing_repeat_blocks_output(tmp_path):
    ev = tmp_path / "evals"
    for tag in ("lstm_single_rep0", "lstm_single_rep1"):          # rep2 missing
        (ev / tag).mkdir(parents=True)
        (ev / tag / "metrics.json").write_text(json.dumps({}))
    (tmp_path / "eval_list.tsv").write_text("task\tmethod\truns\tn_members\ttag\n")
    rules = yaml.safe_load(open("experiments/configs/v3/analysis_rules.yaml"))
    rules.update(eval_root=str(ev), bootstrap_dir=str(tmp_path / "boot"), out_dir=str(tmp_path / "out"),
                 eval_list=str(tmp_path / "eval_list.tsv"),
                 uq_methods={"conformal_lstm": {"label": "x", "evals": "lstm_single_rep*", "method": "point",
                                                "variant": "primary"}})
    with pytest.raises(FileNotFoundError, match="2 of 3 repeats"):
        report.check_required(report.Inputs(rules), rules)
