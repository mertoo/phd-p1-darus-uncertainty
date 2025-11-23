import torch
import torch.nn as nn

class LinearBaseline(nn.Module):
    """
    A simple linear autoregressive model:
    Flattens the last `history` timesteps and predicts `horizon * output_dim` future values.
    """
    def __init__(self, input_dim, history, output_dim, horizon):
        super().__init__()
        self.history = history
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon

        self.net = nn.Linear(history * input_dim, horizon * output_dim)

    def forward(self, x):
        B = x.shape[0]
        x_flat = x.reshape(B, self.history * self.input_dim)
        y = self.net(x_flat)
        return y.reshape(B, self.horizon, self.output_dim)
