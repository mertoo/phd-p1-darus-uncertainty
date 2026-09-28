"""Gate A: every ensemble member is checked, not just the first."""

import copy

import pytest

from src.evaluation.provenance import ProvenanceError, validate_members


def member(i, **over):
    ds = {"base_path": "d", "history": 30, "horizon": 30, "stride": 1,
          "features": ["n", "u"], "targets": ["u", "v", "p", "r", "phi"],
          "split_files": {"train": ["a.csv"], "cal": ["b.csv"]}, "split_hash": "h",
          "x_scaler": {"mean": [0, 0], "std": [1, 1]}, "y_scaler": {"mean": [0] * 5, "std": [1] * 5}}
    m = {"dir": f"run/model_{i}", "sha256": f"sha{i}", "seed": 100 + i,
         "config": {"model": {"type": "lstm", "hidden_dim": 128}},
         "data_state": ds, "state_shapes": {"fc.weight": (5, 128)},
         "best_epoch": 3, "git": None}
    for k, v in over.items():
        m[k] = v
    return m


def ens(n=5):
    return [member(i) for i in range(n)]


def test_valid_ensemble_passes():
    out = validate_members(ens(), expected_n=5)
    assert out["n_members"] == 5 and out["seeds"] == [100, 101, 102, 103, 104]


def test_wrong_size():
    with pytest.raises(ProvenanceError, match="expected 5"):
        validate_members(ens(4), expected_n=5)


def test_duplicate_checkpoint_and_seed():
    ms = ens()
    ms[3]["sha256"] = ms[1]["sha256"]
    with pytest.raises(ProvenanceError, match="duplicate checkpoint"):
        validate_members(ms, 5)
    ms = ens()
    ms[4]["seed"] = ms[0]["seed"]
    with pytest.raises(ProvenanceError, match="duplicate seeds"):
        validate_members(ms, 5)


@pytest.mark.parametrize("key, value", [
    ("y_scaler", {"mean": [1] * 5, "std": [1] * 5}),
    ("x_scaler", {"mean": [0, 0], "std": [2, 1]}),
    ("split_files", {"train": ["a.csv", "b.csv"]}),
    ("features", ["u", "n"]),
    ("horizon", 20),
])
def test_last_member_data_state_mismatch_detected(key, value):
    ms = ens()
    ms[-1]["data_state"] = copy.deepcopy(ms[-1]["data_state"])
    ms[-1]["data_state"][key] = value
    with pytest.raises(ProvenanceError, match=key):
        validate_members(ms, 5)


def test_target_order_checked():
    ms = ens()
    ms[2]["data_state"] = copy.deepcopy(ms[2]["data_state"])
    ms[2]["data_state"]["targets"] = ["v", "u", "p", "r", "phi"]
    with pytest.raises(ProvenanceError, match="target order"):
        validate_members(ms, 5)


def test_architecture_mismatch_detected():
    ms = ens()
    ms[4]["state_shapes"] = {"fc.weight": (5, 64)}
    with pytest.raises(ProvenanceError, match="shapes"):
        validate_members(ms, 5)
    ms = ens()
    ms[4]["config"] = {"model": {"type": "lstm", "hidden_dim": 64}}
    with pytest.raises(ProvenanceError, match="model config"):
        validate_members(ms, 5)


def test_mixed_legacy_rejected():
    ms = ens()
    ms[0]["data_state"] = None
    with pytest.raises(ProvenanceError, match="legacy"):
        validate_members(ms, 5)
