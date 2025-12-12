import torch
import torch.nn as nn


class LSTMGaussianSeq2Seq(nn.Module):
    """
    LSTM encoder–decoder with Gaussian output:
      - Input:  (batch, history, input_dim)
      - Output: mean and log-variance
                mu, logvar: (batch, horizon, target_dim)

    We use:
      y_t ~ N(mu_t, diag(exp(logvar_t)))
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 30,
        target_dim: int = 5,
    ):
        super().__init__()

        self.horizon = horizon
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder over full feature vector
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Decoder gets zeros but uses encoder state
        self.decoder = nn.LSTM(
            input_size=target_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Map hidden state -> [mu, logvar] for each DOF
        self.fc = nn.Linear(hidden_dim, 2 * target_dim)

    def forward(self, x: torch.Tensor):
        """
        x: (batch, history, input_dim)
        Returns:
          mu:     (batch, horizon, target_dim)
          logvar: (batch, horizon, target_dim)
        """
        batch_size = x.size(0)
        device = x.device

        # ----- Encoder -----
        _, (h_n, c_n) = self.encoder(x)

        # ----- Decoder -----
        # Zero input, just to unroll horizon steps
        decoder_input = torch.zeros(
            batch_size, self.horizon, self.target_dim, device=device
        )
        decoder_outputs, _ = self.decoder(decoder_input, (h_n, c_n))
        # (B, T, hidden_dim)

        out = self.fc(decoder_outputs)  # (B, T, 2 * target_dim)
        mu, logvar = torch.split(out, self.target_dim, dim=-1)

        # Clamp log-variance for numerical stability
        logvar = logvar.clamp(min=-10.0, max=5.0)

        return mu, logvar
