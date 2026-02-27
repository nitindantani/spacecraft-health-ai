import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from core.build_loader import get_loader
from models.lstm.model import LSTMAutoencoder



# ---------- CONFIG ----------
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- DATA ----------
loader = get_loader(batch_size=BATCH_SIZE)


# ---------- MODEL ----------
model = LSTMAutoencoder().to(DEVICE)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=LR)


best_loss = float("inf")

print("Training on:", DEVICE)
print("Starting training...\n")


# ---------- TRAIN LOOP ----------
for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    progress = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for x, y in progress:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        # forward
        output = model(x)

        # loss
        loss = criterion(output, x)  # Autoencoder: compare output to input

        # backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)

    print(f"\nEpoch {epoch+1} loss: {avg_loss:.6f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "outputs/weights/best_lstm.pt")
        print("✔ Best model saved\n")
