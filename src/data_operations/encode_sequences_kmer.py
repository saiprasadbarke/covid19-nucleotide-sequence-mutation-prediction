# Standard

from json import dump, load
from os import listdir
from pathlib import Path
from typing import Dict, List

# Local

from settings.constants import KMER_LENGTH
from data_operations.vocabulary import Vocabulary

vocabulary = Vocabulary(KMER_LENGTH)


def sliding_window(sequence: str, kmer_length: int) -> List[str]:
    kmerized_sequence = []
    for i in range(len(sequence) - kmer_length + 1):
        kmerized_sequence.append(sequence[i : i + kmer_length])
    return kmerized_sequence


def encode_sequence(sequence: str, kmer_length: int) -> List[int]:
    kmer_sequence = sliding_window(sequence, kmer_length)
    encoded_kmer_seq = [vocabulary.stoi[kmer] for kmer in kmer_sequence]
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


if __name__ == "__main__":
    permuted_clade_pair_folder = f"{Path.cwd().parents[0]}/data/permuted"
    encoded_permuted_clade_pair_folder = f"{Path.cwd().parents[0]}/data/encoded"
    create_encoded_sequence_pairs_file(
        permuted_clade_pair_folder, encoded_permuted_clade_pair_folder, kmer_length=KMER_LENGTH
    )
