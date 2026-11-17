import json
import sys

import pytest

from scripts import check_stage

CAL = ["a.csv", "b.csv"]


def setup(tmp, steps=665, cal=CAL, n=170016):
    (tmp / "scripts/slurm/v3").mkdir(parents=True)
    (tmp / "experiments/manifests").mkdir(parents=True)
    (tmp / "experiments/manifests/calibration_selection.json").write_text(
        json.dumps({"primary": "random", "random": {"files": CAL}}))
    (tmp / "scripts/slurm/v3/run_list.tsv").write_text(
        "stage\tconfig\tseed\trun_name\tfeatures_override\n1a\tlstm\t1000\tlstm_ens/rep0/model_0\t\n")
    (tmp / "scripts/slurm/v3/eval_list.tsv").write_text(
        "task\tmethod\truns\tn_members\ttag\n1\tpoint\tlstm_ens/rep0/model_0\t\tlstm_single_rep0\n")
    d = tmp / "experiments/runs/v3/lstm_ens/rep0/model_0"
    d.mkdir(parents=True)
    ep = [{"train_loss": 1.0 - i / 10, "val_loss": 1.0 - i / 10, "optimizer_steps": steps} for i in range(3)]
    (d / "train_log.json").write_text(json.dumps({
        "epochs": ep, "n_train_windows": n, "batch_size": 256, "seed": 1000, "cal_files": cal,
        "normalize": True, "features": ["n"], "split_hash": "h", "scaler_hash": "s",
        "best_epoch": 2, "stopped_by": "early_stopping"}))


def run(tmp, monkeypatch, *argv):
    monkeypatch.chdir(tmp)
    monkeypatch.setattr(sys, "argv", ["x", *argv])
    with pytest.raises(SystemExit) as e:
        check_stage.main()
    return e.value.code


def test_clean_stage_passes(tmp_path, monkeypatch):
    setup(tmp_path)
    assert run(tmp_path, monkeypatch, "--stages", "1a") == 0


@pytest.mark.parametrize("over", [{"steps": 664}, {"cal": ["z.csv", "b.csv"]}, {"n": 1000}])
def test_stage_failures(tmp_path, monkeypatch, over):
    setup(tmp_path, **over)
    assert run(tmp_path, monkeypatch, "--stages", "1a") == 1


def test_missing_eval_is_failure(tmp_path, monkeypatch):
    setup(tmp_path)
    assert run(tmp_path, monkeypatch, "--evals") == 1
