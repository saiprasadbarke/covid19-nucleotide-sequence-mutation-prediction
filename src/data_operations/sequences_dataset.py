# Standard
from pathlib import Path

# External
from torch.utils.data import Dataset


class SequencesDataset(Dataset):
    """
    This class extends the Dataset implementation from torch.utils.data.Dataset. The 3 methods below have to be overridden.
    """

    def __init__(self, input_sequences, target_sequences):

        self.input_sequences = input_sequences
        self.target_sequences = target_sequences

    def __len__(self):
        return len(self.target_sequences)

    def __getitem__(self, idx):
        input_sequence = self.input_sequences[idx]
        target_sequence = self.target_sequences[idx]
        return input_sequence, target_sequence
