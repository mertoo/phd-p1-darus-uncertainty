import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Flattened-window MLP. `num_layers` counts Linear layers (default 3:
    in -> hidden -> hidden -> out), matching the historical architecture and
    its state-dict keys when dropout == 0.
    """

    def __init__(self, input_dim, history, horizon, output_dim, hidden_dim=128,
                 num_layers=3, dropout=0.0):
        super().__init__()

        if num_layers < 2:
            raise ValueError("MLP needs num_layers >= 2 (at least one hidden layer)")

        self.history = history
        self.input_dim = input_dim
        self.horizon = horizon
        self.output_dim = output_dim

        # Flattened input = history × input_dim
        flat_in = history * input_dim
        flat_out = horizon * output_dim

        layers = []
        width = flat_in
        for _ in range(num_layers - 1):
            layers += [nn.Linear(width, hidden_dim), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width = hidden_dim
        layers.append(nn.Linear(width, flat_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, H, D)
        B = x.shape[0]
        x = x.reshape(B, self.history * self.input_dim)   # flatten to (B, H*D)
        y = self.net(x)                                   # (B, horizon*output_dim)
        return y.reshape(B, self.horizon, self.output_dim)
