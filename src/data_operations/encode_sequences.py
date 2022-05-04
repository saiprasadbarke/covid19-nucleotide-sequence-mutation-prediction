# Standard
from itertools import combinations_with_replacement, permutations
from json import dump, load
from os import listdir
from pathlib import Path
from typing import Dict, List
from torch.utils.data import Dataset, DataLoader
from numpy import zeros

NUCLEOTIDES = ["A", "C", "G", "T"]
KMER_LENGTH = 5
num_special_characters = 3  # 0, 1 and 2 are for BOS , EOS and PAD respectively


def generate_vocabulary(kmer_length: int) -> Dict[str, int]:
    # The number of possible k-combinations of these nucleotides taken with repetition is 4^kmer_length
    vocabulary_list = [list(x) for x in combinations_with_replacement(NUCLEOTIDES, kmer_length)]
    permuted_vocabulary = []
    for word in vocabulary_list:
        permuted_vocabulary += [list(x) for x in permutations(word, kmer_length)]

    for permuted_word in permuted_vocabulary:
        if permuted_word not in vocabulary_list:
            vocabulary_list.append(permuted_word)

    vocabulary_dict = {}
    for index, kmer in enumerate(vocabulary_list, 3):  # 0, 1 and 2 are for BOS , EOS and PAD respectively
        vocabulary_dict["".join(kmer)] = index
    return vocabulary_dict


VOCABULARY = generate_vocabulary(KMER_LENGTH)


def sliding_window(sequence: str, kmer_length: int) -> List[str]:
    kmerized_sequence = []
    for i in range(len(sequence) - kmer_length + 1):
        kmerized_sequence.append(sequence[i : i + kmer_length])
    return kmerized_sequence


def encode_sequence(sequence: str, kmer_length: int) -> List[int]:
    kmer_sequence = sliding_window(sequence, kmer_length)
    encoded_kmer_seq = [VOCABULARY[kmer] for kmer in kmer_sequence]
    return encoded_kmer_seq


def create_encoded_sequence_pairs_file(
    permuted_clade_pair_folder: str, encoded_permuted_clade_pair_folder: str, kmer_length: int
):
    files_list = listdir(permuted_clade_pair_folder)
    print(f"Found {len(files_list)} files with names {files_list}")
    for file in files_list:
        data = open(f"{permuted_clade_pair_folder}/{file}")
        encoded_sequences_list = []
        for line in data:
            line_data = line.split(",")
            x_sequence = encode_sequence(line_data[0], kmer_length)
            y_sequence = encode_sequence(line_data[1].split("\n")[0], kmer_length)
            assert len(x_sequence) == len(y_sequence)
            x_y_dict = {"x_sequence": x_sequence, "y_sequence": y_sequence}
            encoded_sequences_list.append(x_y_dict)
        with open(f"{encoded_permuted_clade_pair_folder}/{file.split('.')[0]}.json", "w") as fout:
            dump(encoded_sequences_list, fout)
        print(f"Encoded {file}. Wrote {file.split('.')[0]}.json to disk.")


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

            x_sequences = SequencesDataset.encode_onehot(x_sequences)
            y_sequences = SequencesDataset.encode_onehot(y_sequences)
        return x_sequences, y_sequences

    @staticmethod
    def prepend_append_start_end(sequences: List[List[int]], is_target: bool = True) -> List[List[int]]:
        """This function appends or prepends a EOS / BOS respectively depending on whether the sequence is an input sequence or a target sequence"""
        for sequence in sequences:
            if is_target:
                sequence.insert(0, 0)  # Prepend 0 as BOS at the start
            else:
                sequence.append(1)  # Append 1 as EOS at the end
        return sequences

    @staticmethod
    def encode_onehot(sequences: List[List[int]]) -> List[List[list[int]]]:
        all_sequences = []
        for sequence in sequences:
            all_sequences.append(SequencesDataset.encode_onehot_singleseq(sequence))
        return all_sequences

    @staticmethod
    def encode_onehot_singleseq(sequence: List[int]) -> List[list[int]]:
        sequence_vector = zeros([len(sequence), 1, len(VOCABULARY) + num_special_characters])
        for index, kmer in enumerate(sequence):
            sequence_vector[index] = SequencesDataset.encode_onehot_kmer(kmer)
        return sequence_vector[:-1]

    @staticmethod
    def encode_onehot_kmer(kmer: int) -> List[int]:
        onehot_vector = zeros([1, len(VOCABULARY) + num_special_characters])
        onehot_vector[0][kmer] = 1
        return onehot_vector


if __name__ == "__main__":
    # generate_vocabulary(5)
    # print(encode_sequence("AGCTAT", 3))
    # print(generate_vocabulary(4))
    # permuted_clade_pair_folder = f"{Path.cwd().parents[0]}/data/permuted"
    # encoded_permuted_clade_pair_folder = f"{Path.cwd().parents[0]}/data/encoded"
    # create_encoded_sequence_pairs_file(
    #    permuted_clade_pair_folder, encoded_permuted_clade_pair_folder, kmer_length=KMER_LENGTH
    # )

    data_path = f"{Path.cwd().parents[0]}/data/21M_21L_test.json"
    sequences_data = SequencesDataset(dataset_file_path=data_path)
    data_loader = DataLoader(sequences_data, batch_size=2, shuffle=True)
    for idx, xy_values in enumerate(data_loader):
        print(f"XY at position {idx} is {xy_values}")
