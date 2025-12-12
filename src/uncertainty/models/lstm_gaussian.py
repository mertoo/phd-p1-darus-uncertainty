import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMGaussian(nn.Module):
    """
    Gaussian seq2seq LSTM:
      Input:  (B, H, input_dim)
      Output: mean, std with shape (B, T, target_dim)

    The network predicts (mu, log_sigma) per target per horizon step.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        target_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        horizon: int = 30,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.target_dim = target_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.horizon = horizon

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.decoder = nn.LSTM(
            input_size=target_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output predicts both mean and log_sigma for each target dimension
        self.fc = nn.Linear(hidden_dim, 2 * target_dim)

        # For stable sigma
        self.sigma_floor = 1e-4
        self.sigma_ceiling = 50.0

    def forward(self, x, y0=None):
        """
        x: (B, H, input_dim)
        y0: (B, target_dim) initial decoder input (optional); default zeros
        """
        B = x.shape[0]
        device = x.device

        _, (h, c) = self.encoder(x)

        if y0 is None:
            dec_in = torch.zeros(B, 1, self.target_dim, device=device)
        else:
            dec_in = y0.view(B, 1, self.target_dim)

        mus = []
        sigmas = []

        h_dec, c_dec = h, c
        for _ in range(self.horizon):
            out, (h_dec, c_dec) = self.decoder(dec_in, (h_dec, c_dec))  # out: (B,1,H)
            params = self.fc(out[:, 0, :])  # (B, 2*target_dim)
            mu = params[:, : self.target_dim]
            log_sigma = params[:, self.target_dim :]

            # softplus for positivity, with floor+ceiling for stability
            sigma = F.softplus(log_sigma) + self.sigma_floor
            sigma = torch.clamp(sigma, max=self.sigma_ceiling)

            mus.append(mu)
            sigmas.append(sigma)

            # autoregressive decoding using predicted mean
            dec_in = mu.unsqueeze(1)

        mu = torch.stack(mus, dim=1)       # (B,T,D)
        sigma = torch.stack(sigmas, dim=1) # (B,T,D)
        return mu, sigma
