# Standard
from pathlib import Path
from typing import List

# Local
from json import load
from settings.constants import num_special_characters, KMER_LENGTH

# External
from numpy import zeros


def kmer_to_onehot(dataset_file_path: str):
    with open(dataset_file_path) as json_file:
        data = load(json_file)
        x_sequences = [sequence_pair["x_sequence"] for sequence_pair in data]
        y_sequences = [sequence_pair["y_sequence"] for sequence_pair in data]
        assert len(x_sequences) == len(y_sequences)
        x_sequences = prepend_append_start_end(x_sequences, is_target=False)
        y_sequences = prepend_append_start_end(y_sequences)

        x_sequences = encode_onehot(x_sequences)
        y_sequences = encode_onehot(y_sequences)
    return x_sequences, y_sequences


def prepend_append_start_end(sequences: List[List[int]], is_target: bool = True) -> List[List[int]]:
    """This function appends or prepends a EOS / BOS respectively depending on whether the sequence is an input sequence or a target sequence"""
    for sequence in sequences:
        if is_target:
            sequence.insert(0, 0)  # Prepend 0 as BOS at the start
        else:
            sequence.append(1)  # Append 1 as EOS at the end
    return sequences


def encode_onehot(sequences: List[List[int]]) -> List[List[list[int]]]:
    all_sequences = []
    for sequence in sequences:
        all_sequences.append(encode_onehot_singleseq(sequence))
    return all_sequences


def encode_onehot_singleseq(sequence: List[int]) -> List[list[int]]:
    sequence_vector = zeros(
        [len(sequence), 1, 4**KMER_LENGTH + num_special_characters]
    )  # 4**KMER_LENGTH is the size of the vocabulary
    for index, kmer in enumerate(sequence):
        sequence_vector[index][0][kmer] = 1
    return sequence_vector


# We are not using this method instead line 131 inserts the "one" at the appropriate position as per the kmer value
def encode_onehot_kmer(kmer: int) -> List[int]:
    onehot_vector = zeros(
        [1, 4**KMER_LENGTH + num_special_characters]
    )  # 4**KMER_LENGTH is the size of the vocabulary
    onehot_vector[0][kmer] = 1
    return onehot_vector


if __name__ == "__main__":
    path = f"{Path.cwd().parents[0]}/data/encoded/21A_21J_test.json"
    print(path)
    x, y = kmer_to_onehot(path)
