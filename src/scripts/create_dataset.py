# Standard
from json import dump
from pprint import pprint as pp

# Local
from settings.constants import CLADE_PAIRS_NEXTCLADE, CURRENT_RUN_DIR
from data_operations.generate_datasets import generate_datasets


def create_dataset():
    pp(CLADE_PAIRS_NEXTCLADE, indent=3)
    dataset_number = int(input("Choose a dataset by entering the number: "))
    clade_pair = CLADE_PAIRS_NEXTCLADE[dataset_number]
    number_of_sequence_pairs = int(input("Choose total number of sequence pairs for train+val+test: "))
    max_seq_length = int(input("Choose the sequence length between [100,3700]:   "))
    sequence_start_postion = int(input(f"Choose the sequence start position between [0,3700-max_seq_length]: "))
    sequence_end_postion = sequence_start_postion + max_seq_length
    minimum_levenshtein_distance = int(input("Choose a minimum levenshtein distance between [5,15]: "))
    maximum_levenshtein_distance = int(input("Choose a maximum levenshtein distance between [15,25]: "))
    generate_datasets(
        clade_pair=clade_pair,
        number_of_sequence_pairs=number_of_sequence_pairs,
        max_seq_length=max_seq_length,
        sequence_start_postion=sequence_start_postion,
        sequence_end_postion=sequence_end_postion,
        minimum_levenshtein_distance=minimum_levenshtein_distance,
        maximum_levenshtein_distance=maximum_levenshtein_distance,
    )

    data_parameters_dict = {
        "clade_pair": clade_pair,
        "number_of_sequence_pairs": number_of_sequence_pairs,
        "max_seq_length": max_seq_length,
        "sequence_start_postion": sequence_start_postion,
        "sequence_end_postion": sequence_end_postion,
        "minimum_levenshtein_distance": minimum_levenshtein_distance,
        "maximum_levenshtein_distance": maximum_levenshtein_distance,
    }
    data_parameters_dict_path = f"{CURRENT_RUN_DIR}/data_parameters.json"
    with open(data_parameters_dict_path, "w") as f:
        dump(data_parameters_dict, f)
