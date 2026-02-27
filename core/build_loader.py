from torch.utils.data import DataLoader
from core.dataset import SatelliteDataset


def get_loader(batch_size=64, shuffle=True):

    dataset = SatelliteDataset(data_dir="data/normalized")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,      # keep 0 on Windows
        pin_memory=True
    )

    return loader
