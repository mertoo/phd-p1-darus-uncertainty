import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    """Remove extra padding introduced by dilated convolutions."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[..., :-self.chomp_size]


class TemporalBlock(nn.Module):
    """A single TCN residual block."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else None
        )

        self.final_relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)

        return self.final_relu(out + res)


class TCN(nn.Module):
    """
    TCN baseline for forecasting:
      Input:  (batch, history, input_dim)
      Output: (batch, horizon, target_dim)
    """

    def __init__(
        self,
        input_dim,
        target_dim=5,
        history=30,
        horizon=30,
        num_channels=[64, 64, 64],
        kernel_size=3,
        dropout=0.1,
    ):
        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_dim if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        self.tcn = nn.Sequential(*layers)
        self.history = history
        self.target_dim = target_dim
        self.horizon = horizon

        # Map TCN output → forecast
        self.fc = nn.Linear(num_channels[-1] * history, target_dim * horizon)

    def forward(self, x):
        # x: (batch, history, input_dim)
        x = x.transpose(1, 2)  # → (batch, input_dim, history)

        y = self.tcn(x)
        y = y.reshape(y.shape[0], -1)  # flatten

        y = self.fc(y)
        return y.reshape(-1, self.horizon, self.target_dim)
