"""Single place where model configs are turned into modules.

Every key under `model:` must be consumed by the chosen architecture; unknown
or unused keys raise instead of being silently ignored (the historical TCN
config set hidden_dim=128/num_layers=4 while the code ran [64, 64, 64]).
"""

from src.models.gru import GRUSeq2Seq
from src.models.linear import LinearBaseline
from src.models.lstm import LSTMSeq2Seq
from src.models.lstm_gaussian import LSTMGaussianSeq2Seq
from src.models.mlp import MLP
from src.models.naive import NaiveBaseline
from src.models.tcn import TCN

# model type -> {config key: default}
_SPECS = {
    "lstm": {"hidden_dim": 128, "num_layers": 2, "dropout": 0.0},
    "lstm_seq2seq": {"hidden_dim": 128, "num_layers": 2, "dropout": 0.0},
    "gru": {"hidden_dim": 128, "num_layers": 2, "dropout": 0.0},
    "lstm_gaussian": {"hidden_dim": 128, "num_layers": 2, "dropout": 0.0},
    "mlp": {"hidden_dim": 256, "num_layers": 3, "dropout": 0.0},
    "tcn": {"num_channels": [64, 64, 64], "kernel_size": 3, "dropout": 0.1},
    "linear": {},
    "naive": {},
}


def model_kwargs(model_cfg):
    mtype = model_cfg["type"].lower()
    if mtype not in _SPECS:
        raise ValueError(f"Unknown model type: {mtype}")
    spec = _SPECS[mtype]
    extra = set(model_cfg) - set(spec) - {"type"}
    if extra:
        raise KeyError(
            f"Model config keys {sorted(extra)} are not used by model type '{mtype}'. "
            f"Accepted keys: {sorted(spec)}"
        )
    return mtype, {k: model_cfg.get(k, v) for k, v in spec.items()}


def build_model(model_cfg, input_dim, target_dim, history, horizon):
    mtype, kw = model_kwargs(model_cfg)

    if mtype in ("lstm", "lstm_seq2seq"):
        return LSTMSeq2Seq(input_dim=input_dim, hidden_dim=kw["hidden_dim"],
                           num_layers=kw["num_layers"], dropout=float(kw["dropout"]),
                           horizon=horizon, target_dim=target_dim)
    if mtype == "gru":
        return GRUSeq2Seq(input_dim=input_dim, hidden_dim=kw["hidden_dim"],
                          num_layers=kw["num_layers"], dropout=float(kw["dropout"]),
                          horizon=horizon, target_dim=target_dim)
    if mtype == "lstm_gaussian":
        return LSTMGaussianSeq2Seq(input_dim=input_dim, hidden_dim=kw["hidden_dim"],
                                   num_layers=kw["num_layers"], dropout=float(kw["dropout"]),
                                   horizon=horizon, target_dim=target_dim)
    if mtype == "mlp":
        return MLP(input_dim=input_dim, history=history, horizon=horizon,
                   output_dim=target_dim, hidden_dim=kw["hidden_dim"],
                   num_layers=kw["num_layers"], dropout=float(kw["dropout"]))
    if mtype == "tcn":
        return TCN(input_dim=input_dim, target_dim=target_dim, history=history,
                   horizon=horizon, num_channels=list(kw["num_channels"]),
                   kernel_size=kw["kernel_size"], dropout=float(kw["dropout"]))
    if mtype == "linear":
        return LinearBaseline(input_dim=input_dim, history=history,
                              output_dim=target_dim, horizon=horizon)
    if mtype == "naive":
        return NaiveBaseline(output_dim=target_dim, horizon=horizon)
    raise AssertionError(mtype)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
