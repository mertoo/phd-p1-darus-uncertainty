import torch
import torch.nn as nn


class MLPGaussian(nn.Module):
    """
    Flattened-window MLP with a Gaussian output head, the feed-forward
    counterpart of LSTMGaussianSeq2Seq. Same trunk as MLP (num_layers counts
    Linear layers including the output layer); the last layer emits
    [mu, logvar] for every (horizon step, channel).
    """

    def __init__(self, input_dim, history, horizon, output_dim, hidden_dim=256,
                 num_layers=3, dropout=0.0):
        super().__init__()
        if num_layers < 2:
            raise ValueError("MLPGaussian needs num_layers >= 2")
        self.history, self.input_dim = history, input_dim
        self.horizon, self.output_dim = horizon, output_dim

        layers, width = [], history * input_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(width, hidden_dim), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width = hidden_dim
        layers.append(nn.Linear(width, 2 * horizon * output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        B = x.shape[0]
        out = self.net(x.reshape(B, self.history * self.input_dim))
        out = out.reshape(B, self.horizon, 2 * self.output_dim)
        mu, logvar = torch.split(out, self.output_dim, dim=-1)
        # same numerical clamp as LSTMGaussianSeq2Seq
        return mu, logvar.clamp(min=-10.0, max=5.0)
