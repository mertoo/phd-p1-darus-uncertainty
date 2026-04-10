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
        # FEATURE_COLUMNS ends with the 5 output DoFs [u, v, p, r, phi],
        # so the last output_dim columns of x are exactly the targets.
        last = x[:, -1, -self.output_dim:]
        return last.unsqueeze(1).repeat(1, self.horizon, 1)
