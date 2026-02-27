import torch
from models.lstm.model_lstm import LSTMAutoencoder

device = "cuda" if torch.cuda.is_available() else "cpu"

model = LSTMAutoencoder().to(device)

# Fake batch like your real data
x = torch.randn(64, 128, 9).to(device)

y = model(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)
print("Model OK on", device)
