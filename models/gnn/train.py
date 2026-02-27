import torch
from tqdm import tqdm

from core.sensor_graph import edge_index
from models.gnn.model import SpacecraftGNN


print("Loading graph features...")
features = torch.load("outputs/metrics/graph_features.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = SpacecraftGNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Training GNN...")

model.train()

for epoch in range(5):

    total_loss = 0

    for f in tqdm(features):

        # reshape for node input
        x = f.unsqueeze(-1).to(DEVICE)      # [nodes, 1]
        ei = edge_index.to(DEVICE)

        optimizer.zero_grad()

        out = model(x, ei)

        # dummy self-reconstruction target
        loss = ((out.squeeze() - x.squeeze())**2).mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} loss: {total_loss/len(features):.6f}")

torch.save(model.state_dict(), "outputs/weights/best_gnn.pt")
print("Saved -> outputs/weights/best_gnn.pt")
