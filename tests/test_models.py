"""Gate A: configs are consumed, historical architectures reproduce, MC dropout varies."""

import pytest

torch = pytest.importorskip("torch")

from src.models.factory import build_model, count_parameters


def test_unused_config_key_fails():
    # The historical p1_tcn.yaml set hidden_dim/num_layers that TCN never read.
    with pytest.raises(KeyError):
        build_model({"type": "tcn", "hidden_dim": 128, "num_layers": 4}, 12, 5, 30, 30)


@pytest.mark.parametrize("cfg, n_params", [
    # LSTM/MLP/TCN match main v2.2.tex Table 2; the GRU as implemented has
    # 305,157 parameters, not the 330K printed in the manuscript.
    ({"type": "lstm", "hidden_dim": 128, "num_layers": 2}, 407_000),
    ({"type": "mlp", "hidden_dim": 256, "num_layers": 3}, 197_000),
    ({"type": "tcn", "num_channels": [64, 64, 64], "kernel_size": 3}, 353_000),
    ({"type": "gru", "hidden_dim": 128, "num_layers": 2}, 305_157),
])
def test_historical_parameter_counts(cfg, n_params):
    m = build_model(cfg, input_dim=12, target_dim=5, history=30, horizon=30)
    assert abs(count_parameters(m) - n_params) < 1000


def test_output_shapes():
    x = torch.zeros(3, 30, 11)
    for cfg in ({"type": "lstm"}, {"type": "gru"}, {"type": "mlp"}, {"type": "tcn"},
                {"type": "linear"}, {"type": "naive"}):
        assert build_model(cfg, 11, 5, 30, 30)(x).shape == (3, 30, 5)
    mu, logvar = build_model({"type": "lstm_gaussian"}, 11, 5, 30, 30)(x)
    assert mu.shape == logvar.shape == (3, 30, 5)


def test_mc_dropout_varies_and_eval_is_deterministic():
    torch.manual_seed(0)
    m = build_model({"type": "lstm", "dropout": 0.2}, 12, 5, 30, 30)
    x = torch.randn(4, 30, 12)
    m.eval()
    with torch.no_grad():
        assert torch.equal(m(x), m(x))
    m.train()
    with torch.no_grad():
        assert not torch.allclose(m(x), m(x))


def test_lstm_decoder_input_is_zero_not_feedback():
    # The decoder never sees its own predictions: output depends on x only via encoder state.
    m = build_model({"type": "lstm"}, 12, 5, 30, 30).eval()
    x = torch.randn(2, 30, 12)
    with torch.no_grad():
        _, (h, c) = m.encoder(x)
        dec, _ = m.decoder(torch.zeros(2, 30, 5), (h, c))
        torch.testing.assert_close(m.fc(dec), m(x))
