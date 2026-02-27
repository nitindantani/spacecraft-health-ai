import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from core.build_loader import get_loader
from models.transformer.model import TransformerAutoencoder


BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

loader = get_loader(batch_size=BATCH_SIZE)

model = TransformerAutoencoder().to(DEVICE)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=LR)

best_loss = float("inf")

print("Training Transformer on:", DEVICE)

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    progress = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for x, _ in progress:

        x = x.to(DEVICE)

        optimizer.zero_grad()

        out = model(x)
        loss = criterion(out, x)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)

    print(f"\nEpoch {epoch+1} loss: {avg_loss:.6f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "outputs/weights/best_transformer.pt")
        print("✔ Transformer saved\n")

print("Training complete.")
