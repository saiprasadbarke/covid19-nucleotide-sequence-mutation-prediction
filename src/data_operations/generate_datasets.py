# Standard
from itertools import product
from json import dump, load
from random import sample
from typing import Dict, Tuple

# Local
from helpers.check_dir_exists import check_dir_exists
from settings.constants import MERGED_DATA, CURRENT_RUN_DATA_DIR

# External
from Levenshtein import distance


def generate_datasets(
    clade_pair: Tuple[str],
    number_of_sequence_pairs: int = 30000,
    max_seq_length: int = 500,
    sequence_start_postion: int = 0,
    sequence_end_postion: int = 500,
    minimum_levenshtein_distance: int = 10,
    maximum_levenshtein_distance: int = 25,
):
    data = load(open(MERGED_DATA))
    clade1 = clade_pair[0]
    clade2 = clade_pair[1]

    clade1_sequences = list(data[clade1].keys())
    clade2_sequences = list(data[clade2].keys())
    count_sequences = 0
    cartesian_product_list = list(product(clade1_sequences, clade2_sequences))
    random_sampled_cartesian_product_list = sample(
        cartesian_product_list,
        len(cartesian_product_list),
    )
    input_target_list = []
    for seq_pair in random_sampled_cartesian_product_list:
        if count_sequences == number_of_sequence_pairs:
            break
        elif is_valid_sequence_pair(
            seq_pair[0],
            seq_pair[1],
            max_seq_length,
            minimum_levenshtein_distance,
            maximum_levenshtein_distance,
        ):
            input_target_list.append(
                (
                    seq_pair[0][sequence_start_postion:sequence_end_postion],
                    seq_pair[1][sequence_start_postion:sequence_end_postion],
                )
            )
            count_sequences += 1
            if count_sequences % 1000 == 0:
                print(f"Found {count_sequences} valid pairs .")
    # Split data into train, validation and test datasets
    split = {"train": 0.8, "val": 0.1, "test": 0.1}
    train_val_test_indices = {
        "train_upto": int(split["train"] * number_of_sequence_pairs),
        "val_upto": int(split["train"] * number_of_sequence_pairs + split["val"] * number_of_sequence_pairs),
    }
    train_list = []
    val_list = []
    test_list = []
    for i, input_target_pair in enumerate(input_target_list, 1):
        if i <= train_val_test_indices["train_upto"]:
            train_list.append(input_target_pair)
        if i > train_val_test_indices["train_upto"] and i <= train_val_test_indices["val_upto"]:
            val_list.append(input_target_pair)
        else:
            test_list.append(input_target_pair)

    # File operations
    check_dir_exists(CURRENT_RUN_DATA_DIR)
    with open(f"{CURRENT_RUN_DATA_DIR}/train.json", "w") as fout:
        dump(train_list, fout)
    with open(f"{CURRENT_RUN_DATA_DIR}/val.json", "w") as fout:
        dump(val_list, fout)
    with open(f"{CURRENT_RUN_DATA_DIR}/test.json", "w") as fout:
        dump(test_list, fout)
    print(
        f"Completed parsing merged data...\n Completed creation of [input, target] pairs...\nSaved train, validation and test files to path :\n{CURRENT_RUN_DATA_DIR}"
    )


def is_valid_sequence_pair(
    seq1: str,
    seq2: str,
    max_seq_length: int,
    minimum_levenshtein_distance: int,
    maximum_levenshtein_distance: int,
) -> bool:
    lev_distance = distance(seq1, seq2)
    if (
        len(seq1) < max_seq_length
        or len(seq2) < max_seq_length
        or lev_distance < minimum_levenshtein_distance
        or lev_distance > maximum_levenshtein_distance
    ):
        return False
    else:
        return True
