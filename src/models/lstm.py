import torch
import torch.nn as nn


class LSTMSeq2Seq(nn.Module):
    """
    Encoder–decoder LSTM for seq2seq:
      - Encoder reads past 'history' steps (full feature vector)
      - Decoder runs for 'horizon' steps starting from encoder state
      - Each decoder step is projected to 5-DOF outputs

    Input:  x (batch, history, input_dim)
    Output: y_hat (batch, horizon, target_dim)
    """

    def __init__(self, input_dim, hidden_dim=128, num_layers=2,
                 dropout=0.1, horizon=30, target_dim=5):
        super().__init__()

        self.horizon = horizon
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder reads the full input feature vector
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Decoder reads a dummy zero sequence, but uses the encoder's hidden state
        self.decoder = nn.LSTM(
            target_dim,           # decoder input size (we'll feed zeros of this size)
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Project decoder hidden -> target DOFs
        self.fc = nn.Linear(hidden_dim, target_dim)

    def forward(self, x):
        """
        x: (batch, history, input_dim)
        returns: (batch, horizon, target_dim)
        """
        batch_size = x.size(0)
        device = x.device

        # ----- Encoder -----
        # h_n, c_n: (num_layers, batch, hidden_dim)
        _, (h_n, c_n) = self.encoder(x)

        # ----- Decoder -----
        # We'll feed a zero tensor of shape (batch, horizon, target_dim)
        # and let the decoder evolve from the encoder's final state.
        decoder_input = torch.zeros(
            batch_size, self.horizon, self.target_dim, device=device
        )  # (B, T_out, target_dim)

        decoder_outputs, _ = self.decoder(decoder_input, (h_n, c_n))
        # decoder_outputs: (batch, horizon, hidden_dim)

        # Project each time step to target_dim
        y_hat = self.fc(decoder_outputs)  # (batch, horizon, target_dim)

        return y_hat
