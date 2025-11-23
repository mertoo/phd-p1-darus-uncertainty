import torch
import torch.nn as nn


class GRUSeq2Seq(nn.Module):
    """
    GRU encoder–decoder model:
      - Encoder GRU reads input sequence
      - Decoder GRU generates future sequence
      - FC projects hidden → 5 DOF at each future timestep
    """

    def __init__(self, input_dim, hidden_dim=128, num_layers=2,
                 dropout=0.1, horizon=30, target_dim=5):
        super().__init__()

        self.horizon = horizon
        self.target_dim = target_dim

        # GRU Encoder
        self.encoder = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # GRU Decoder
        self.decoder = nn.GRU(
            target_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Hidden → DOF projection
        self.fc = nn.Linear(hidden_dim, target_dim)

    def forward(self, x):
        """
        x: (batch, history, input_dim)
        return: (batch, horizon, target_dim)
        """
        batch = x.size(0)
        device = x.device

        # ----- Encoder -----
        _, h_n = self.encoder(x)  # h_n: (num_layers, B, hidden_dim)

        # ----- Decoder -----
        # zero decoder input
        dec_input = torch.zeros(
            batch, self.horizon, self.target_dim, device=device
        )

        dec_out, _ = self.decoder(dec_input, h_n)
        y_hat = self.fc(dec_out)

        return y_hat
