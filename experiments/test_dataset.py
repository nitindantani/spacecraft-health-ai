from core.dataset import SatelliteDataset

dataset = SatelliteDataset(
    data_dir="normalized",
    window=128
)

print("Dataset length:", len(dataset))

sample = dataset[0]
print("Sample shape:", sample.shape)
