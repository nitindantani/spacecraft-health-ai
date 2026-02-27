import torch
import torch.nn as nn


class TransformerAutoencoder(nn.Module):
    def __init__(self, n_features=9, d_model=64, nhead=4, num_layers=2):
        super().__init__()

        self.input_proj = nn.Linear(n_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.output_proj = nn.Linear(d_model, n_features)

    def forward(self, x):
        # x: (batch, window, features)
        x = self.input_proj(x)
        x = self.encoder(x)
        x = self.output_proj(x)
        return x
