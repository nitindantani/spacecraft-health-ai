import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, num_layers=1):
        super().__init__()

        # Encoder
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True
        )

        # Decoder
        self.decoder = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):

        # Encode
        _, (hidden, cell) = self.encoder(x)

        # Repeat hidden state for each timestep
        seq_len = x.size(1)
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)

        # Decode
        decoded, _ = self.decoder(decoder_input)

        # Map to sensor space
        output = self.output_layer(decoded)

        return output