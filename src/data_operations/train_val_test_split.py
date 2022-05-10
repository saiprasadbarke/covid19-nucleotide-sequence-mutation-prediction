# Standard
from pathlib import Path

# Local
from data_operations.encode_sequences_onehot import kmer_to_onehot
from data_operations.sequences_dataset import SequencesDataset
from settings.constants import train_remaining_fraction, validation_test_fraction

# External
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def get_split_data(dataset_file_path: str, batch_size: int):

    inputs, targets = kmer_to_onehot(dataset_file_path=dataset_file_path)
    inputs_train, inputs_remainder, targets_train, targets_remainder = train_test_split(
        inputs, targets, test_size=train_remaining_fraction, random_state=42
    )
    inputs_validation, inputs_test, targets_validation, targets_test = train_test_split(
        inputs_remainder, targets_remainder, test_size=validation_test_fraction, random_state=42
    )

    train_dataset = SequencesDataset(inputs_train, targets_train)
    val_dataset = SequencesDataset(inputs_validation, targets_validation)
    test_dataset = SequencesDataset(inputs_test, targets_test)

    print(f"Size of Training dataset: {train_dataset.__len__()}")
    print(f"Size of Validation dataset: {val_dataset.__len__()}")
    print(f"Size of Test dataset: {test_dataset.__len__()}")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    path = f"{Path.cwd().parents[0]}/data/encoded/21A_21J_test.json"
    print(path)
    train, val, test = get_split_data(path, 6)
    print()
