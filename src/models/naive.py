import torch
import torch.nn as nn

class NaiveBaseline(nn.Module):
    """
    Predicts the last observed timestep, repeated for horizon steps.
    No training, no parameters.
    """
    def __init__(self, output_dim, horizon):
        super().__init__()
        self.horizon = horizon
        self.output_dim = output_dim

    def forward(self, x):
        last = x[:, -1, -self.output_dim:]  # last available DoF values
        return last.unsqueeze(1).repeat(1, self.horizon, 1)
