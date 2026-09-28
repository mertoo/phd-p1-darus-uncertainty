import torch
import torch.nn as nn

class NaiveBaseline(nn.Module):
    """
    Predicts the last observed timestep, repeated for horizon steps.
    No training, no parameters.

    The targets must be the last `output_dim` input columns (true for both
    feature sets in darus_dataset). When inputs and targets are standardised
    with different scalers, call set_scaling() so the held value is mapped
    from input-scaled to target-scaled units.
    """
    def __init__(self, output_dim, horizon):
        super().__init__()
        self.horizon = horizon
        self.output_dim = output_dim
        # y_scaled = x_scaled * gain + bias  (identity unless set_scaling is called)
        self.register_buffer("gain", torch.ones(output_dim), persistent=False)
        self.register_buffer("bias", torch.zeros(output_dim), persistent=False)

    def set_scaling(self, x_mean, x_std, y_mean, y_std):
        """x_* are the scaler stats of the last output_dim input columns."""
        x_mean, x_std, y_mean, y_std = (torch.as_tensor(a, dtype=torch.float32)
                                        for a in (x_mean, x_std, y_mean, y_std))
        self.gain = x_std / y_std
        self.bias = (x_mean - y_mean) / y_std

    def forward(self, x):
        last = x[:, -1, -self.output_dim:] * self.gain + self.bias
        return last.unsqueeze(1).repeat(1, self.horizon, 1)
