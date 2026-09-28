import pytest

from scripts.select_features import decide


def L(lstm_legacy, lstm_no, mlp_legacy, mlp_no, seeds=(1,)):
    vals = {("lstm", "legacy"): lstm_legacy, ("lstm", "no_time"): lstm_no,
            ("mlp", "legacy"): mlp_legacy, ("mlp", "no_time"): mlp_no}
    return {k: {s: v for s in seeds} for k, v in vals.items()}


def test_clear_no_time():
    assert decide(L(1.00, 1.00, 1.00, 0.99))["decision"] == "no_time"


def test_clear_legacy_requires_both_backbones():
    assert decide(L(0.90, 1.00, 0.90, 1.00))["decision"] == "legacy"


def test_disagreement_triggers_repeat_then_defaults_to_no_time():
    r1 = decide(L(0.90, 1.00, 1.00, 1.00))
    assert r1["backbones_disagree"] and r1["decision"] == "repeat"
    r2 = decide(L(0.90, 1.00, 1.00, 1.00, seeds=(1, 2, 3)))
    assert r2["round"] == 2 and r2["decision"] == "no_time" and r2["backbones_disagree"]


def test_borderline_band_triggers_repeat():
    assert decide(L(0.975, 1.00, 0.975, 1.00))["decision"] == "repeat"   # gains 2.5%


def test_round2_needs_three_seeds_everywhere():
    bad = L(1, 1, 1, 1, seeds=(1, 2, 3))
    bad[("mlp", "legacy")].pop(3)
    with pytest.raises(ValueError):
        decide(bad)
