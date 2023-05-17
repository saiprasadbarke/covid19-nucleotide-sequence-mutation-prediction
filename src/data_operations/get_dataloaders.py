# Standard

# Local
from data_operations.sequences_dataset import SequencesDataset

# External
from torch.utils.data import DataLoader


def get_dataloader(inputs: list, targets: list, batch_size: int = 1, shuffle: bool = True):

    dataset = SequencesDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
