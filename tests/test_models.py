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


def test_gaussian_nll_includes_constant():
    import math
    from src.training.train import gaussian_nll
    z = torch.zeros(1, 1, 1)
    assert gaussian_nll(z, z, z).item() == pytest.approx(0.5 * math.log(2 * math.pi))


def test_mlp_gaussian_learns_heteroscedastic_scale():
    # noise sd depends on the sign of the last input value; the head must learn it
    from src.training.train import gaussian_nll
    torch.manual_seed(0)
    m = build_model({"type": "mlp_gaussian", "hidden_dim": 32, "num_layers": 3}, 2, 1, 4, 3)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    x = torch.randn(2048, 4, 2)
    sd = torch.where(x[:, -1, :1] > 0, 1.0, 0.1).unsqueeze(1).expand(-1, 3, 1)
    y = sd * torch.randn(2048, 3, 1)
    first = None
    for _ in range(300):
        mu, logvar = m(x)
        loss = gaussian_nll(mu, logvar, y)
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first
    with torch.no_grad():
        _, logvar = m(x)
    s = torch.exp(0.5 * logvar)
    hi, lo = s[sd > 0.5].mean(), s[sd < 0.5].mean()
    assert hi / lo > 4                                     # true ratio is 10


def test_logvar_clamped():
    m = build_model({"type": "mlp_gaussian"}, 3, 5, 30, 30)
    with torch.no_grad():
        for p in m.parameters():
            p.fill_(10.0)
        _, logvar = m(torch.ones(2, 30, 3))
    assert logvar.max() <= 5.0 and logvar.min() >= -10.0
