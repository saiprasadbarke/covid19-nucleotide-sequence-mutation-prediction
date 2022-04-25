# Standard
from json import load
from pathlib import Path
from typing import List

# External
from torch.utils.data import Dataset, DataLoader


class SequencesDataset(Dataset):
    """
    This class extends the Dataset implementation from torch.utils.data.Dataset. The 3 methods below have to be overridden.
    """

    def __init__(self, dataset_file_path, x_transform=None, y_transform=None):
        self.dataset_file_path = dataset_file_path
        (
            self.x_values,
            self.y_values,
        ) = SequencesDataset.parse_sequences_json(self.dataset_file_path)
        self.x_transform = x_transform
        self.y_transform = y_transform

    def __len__(self):
        return len(self.y_values)

    def __getitem__(self, idx):
        x_value = self.x_values[idx]
        y_value = self.y_values[idx]
        if self.x_transform:
            x_value = self.x_transform(x_value)
        if self.y_transform:
            y_value = self.y_transform(y_value)
        return x_value, y_value

    @staticmethod
    def parse_sequences_json(dataset_file_path):
        with open(dataset_file_path) as json_file:
            data = load(json_file)
            x_sequences = [sequence_pair["x_sequence"] for sequence_pair in data]
            y_sequences = [sequence_pair["y_sequence"] for sequence_pair in data]
            assert len(x_sequences) == len(y_sequences)
            x_sequences = SequencesDataset.prepend_append_start_end(x_sequences, is_target=False)
            y_sequences = SequencesDataset.prepend_append_start_end(y_sequences)
        return x_sequences, y_sequences

    @staticmethod
    def prepend_append_start_end(sequences: List[List[int]], is_target: bool = True):
        """This function appends or prepends a EOS / BOS respectively depending on whether the sequence is an input sequence or a target sequence"""
        for sequence in sequences:
            if is_target:
                sequence.insert(0, 0)  # Prepend 0 as BOS at the start
            else:
                sequence.append(1)  # Append 1 as EOS at the end
        return sequences


if __name__ == "__main__":

    data_path = f"{Path.cwd().parents[0]}/data/21M_21L_test.json"
    sequences_data = SequencesDataset(dataset_file_path=data_path)
    data_loader = DataLoader(sequences_data, batch_size=2, shuffle=True)
    for idx, xy_values in enumerate(data_loader):
        print(f"XY at position {idx} is {xy_values}")
