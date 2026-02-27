import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class SpacecraftGNN(nn.Module):
    def __init__(self, num_nodes=9, hidden=16):
        super().__init__()

        self.conv1 = GCNConv(1, hidden)
        self.conv2 = GCNConv(hidden, hidden)

        self.fc = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        # x shape: [num_nodes, 1]
        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        x = self.fc(x)

        # squash output to safe range
        x = torch.tanh(x)

        return x