import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, history, horizon, output_dim, hidden_dim=128):
        super().__init__()

        self.history = history
        self.input_dim = input_dim
        self.horizon = horizon
        self.output_dim = output_dim

        # Flattened input = history × input_dim
        flat_in = history * input_dim
        flat_out = horizon * output_dim

        self.net = nn.Sequential(
            nn.Linear(flat_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, flat_out)
        )

    def forward(self, x):
        # x: (B, H, D)
        B = x.shape[0]
        x = x.reshape(B, self.history * self.input_dim)   # flatten to (B, H*D)
        y = self.net(x)                                   # (B, horizon*output_dim)
        return y.reshape(B, self.horizon, self.output_dim)
