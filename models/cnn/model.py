import torch
import torch.nn as nn


class CNNAutoencoder(nn.Module):
    def __init__(self, n_features=9):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, n_features, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x shape: (batch, window, features)
        x = x.permute(0, 2, 1)     # → (batch, features, window)

        z = self.encoder(x)
        out = self.decoder(z)

        out = out.permute(0, 2, 1) # → back to (batch, window, features)
        return out
