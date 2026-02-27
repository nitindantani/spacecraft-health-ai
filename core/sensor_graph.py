import torch

# ---------------------------
# SENSOR NODES
# ---------------------------
SENSORS = [
    "Temp X", "Temp Y", "Temp Z",
    "Accel coarse X", "Accel coarse Y", "Accel coarse Z",
    "Accel fine X", "Accel fine Y", "Accel fine Z"
]

# map sensor → node index
sensor_to_idx = {s: i for i, s in enumerate(SENSORS)}

# ---------------------------
# EDGES (define relationships)
# ---------------------------

edges = []

# thermal subsystem (temps connected)
thermal = ["Temp X", "Temp Y", "Temp Z"]
for a in thermal:
    for b in thermal:
        if a != b:
            edges.append([sensor_to_idx[a], sensor_to_idx[b]])

# structure subsystem (accel coarse)
coarse = ["Accel coarse X", "Accel coarse Y", "Accel coarse Z"]
for a in coarse:
    for b in coarse:
        if a != b:
            edges.append([sensor_to_idx[a], sensor_to_idx[b]])

# structure subsystem (accel fine)
fine = ["Accel fine X", "Accel fine Y", "Accel fine Z"]
for a in fine:
    for b in fine:
        if a != b:
            edges.append([sensor_to_idx[a], sensor_to_idx[b]])

# cross-links (thermal affects vibration)
cross = [
    ("Temp X", "Accel coarse X"),
    ("Temp Y", "Accel coarse Y"),
    ("Temp Z", "Accel coarse Z"),
]

for a, b in cross:
    edges.append([sensor_to_idx[a], sensor_to_idx[b]])
    edges.append([sensor_to_idx[b], sensor_to_idx[a]])

# convert to tensor for PyG
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

print("Nodes:", len(SENSORS))
print("Edges:", edge_index.shape[1])
