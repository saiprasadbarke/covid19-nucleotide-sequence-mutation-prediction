# Standard
from pprint import pprint as pp

# Local
from settings.constants import CLADE_PAIRS_NEXTCLADE
from data_operations.generate_datasets import generate_datasets


def create_dataset():
    pp(CLADE_PAIRS_NEXTCLADE, indent=3)
    dataset_number = int(input("--------->Choose a dataset by entering the number :   "))
    clade_pair = CLADE_PAIRS_NEXTCLADE[dataset_number]
    number_of_sequence_pairs = int(input("--------->Choose total number of sequence pairs for train+val+test :   "))
    max_seq_length = int(
        input(
            "--------->Choose the sequence length.\nHint: 1. Should be less than 3700 for spike nucleotide sequences.\n2. Should be between less than 26000 for full nucleotide sequences. :   "
        )
    )
    sequence_start_postion = int(
        input(
            f"--------->Choose the sequence start position\n Hint: 1. Should be between 0 and 3700 - max_seq_length for spike nucleotide sequences.\n2. Should be between 0 and 26000 - max_seq_length for full nucleotide sequences. :   "
        )
    )
    sequence_end_postion = sequence_start_postion + max_seq_length
    minimum_levenshtein_distance = int(
        input(
            "--------->Choose a minimum levenshtein distance\n Hint: 1. Should be between [5, 15] for spike nucleotide sequences.\n2. Should be between [500,600] for full nucleotide sequences. :    "
        )
    )
    maximum_levenshtein_distance = int(
        input(
            "--------->Choose a maximum levenshtein distance\n Hint: 1. Should be between [15, 25] for spike nucleotide sequences.\n2. Should be between [600,700] for full nucleotide sequences. :    "
        )
    )
    generate_datasets(
        clade_pair=clade_pair,
        number_of_sequence_pairs=number_of_sequence_pairs,
        max_seq_length=max_seq_length,
        sequence_start_postion=sequence_start_postion,
        sequence_end_postion=sequence_end_postion,
        minimum_levenshtein_distance=minimum_levenshtein_distance,
        maximum_levenshtein_distance=maximum_levenshtein_distance,
    )
