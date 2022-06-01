# Standard

from json import dump, load
from pathlib import Path
from typing import List, Tuple

from torch import float16, tensor

# Local

from settings.constants import CURRENT_RUN_DATA_DIR
from data_operations.vocabulary import Vocabulary


class Tokenize:
    def __init__(self, kmer_length: int = 3) -> None:
        self.kmer_length = kmer_length
        self.vocabulary = Vocabulary(kmer_length)

    def sliding_window(self, sequence: str) -> List[str]:
        kmerized_sequence = []
        for i in range(len(sequence) - self.kmer_length + 1):
            kmerized_sequence.append(sequence[i : i + self.kmer_length])
        return kmerized_sequence

    def tokenize_encode_sequence(self, sequence: str, is_target: bool = True) -> List[int]:
        kmer_sequence = self.sliding_window(sequence)
        kmer_sequence = self.add_special_characters(kmer_sequence, is_target)
        encoded_kmer_seq = [self.vocabulary.stoi[kmer] for kmer in kmer_sequence]
        return encoded_kmer_seq

    def kmerize_numericalize_pad_tensorize_sequences(self, dataset_type: str):
        assert (
            dataset_type == "train" or dataset_type == "val" or dataset_type == "test"
        ), "Valid values of dataset_type are [train, val ,test]"
        dataset_file_path = f"{CURRENT_RUN_DATA_DIR}/{dataset_type}.json"
        inputs = []
        targets = []
        data = load(open(dataset_file_path))
        for (input, target) in data:
            x_sequence = self.tokenize_encode_sequence(input, is_target=False)
            y_sequence = self.tokenize_encode_sequence(target)
            assert len(x_sequence) == len(y_sequence) - 1, "Incorrect input and target sequence lengths"
            x_sequence = tensor(x_sequence, dtype=float16)
            y_sequence = tensor(y_sequence, dtype=float16)
            inputs.append(x_sequence)
            targets.append(y_sequence)
        print(f"Kmerization and Numericalization complete for {dataset_type} data...")
        return inputs, targets

    def add_special_characters(self, sequence: List[str], is_target: bool = True) -> List[str]:
        """This function appends or prepends a EOS , BOS  depending on whether the sequence is an input sequence or a target sequence"""
        if is_target:
            sequence.insert(0, "<BOS>")
            sequence.append("<EOS>")
        else:
            sequence.append("<EOS>")
        return sequence
