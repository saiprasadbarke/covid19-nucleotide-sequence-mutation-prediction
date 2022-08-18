# Standard
from itertools import product
from json import dump, load
from random import sample
from typing import Tuple

# Local
from settings.constants import MERGED_DATA, CURRENT_RUN_DATA_DIR

# External
from Levenshtein import distance


from visualization.plot_mutation_sites import get_mutations_and_plot


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
    # Get the relevant clade sequences
    clade1 = clade_pair[0]
    clade2 = clade_pair[1]
    clade1_sequences = list(data[clade1].keys())
    clade2_sequences = list(data[clade2].keys())

    count_sequences = 0
    cartesian_product_list = list(product(clade1_sequences, clade2_sequences))
    random_sampled_cartesian_product_list = sample(cartesian_product_list, len(cartesian_product_list),)
    input_target_list = []

    # Filter the clade sequences as per the dataset parameters
    for seq_pair in random_sampled_cartesian_product_list:
        if count_sequences == number_of_sequence_pairs:
            break
        else:
            clipped_seq_1 = seq_pair[0][sequence_start_postion:sequence_end_postion]
            clipped_seq_2 = seq_pair[1][sequence_start_postion:sequence_end_postion]
            # if is_valid_sequence_pair(
            #     seq1=clipped_seq_1,
            #     seq2=clipped_seq_2,
            #     minimum_levenshtein_distance=minimum_levenshtein_distance,
            #     maximum_levenshtein_distance=maximum_levenshtein_distance,
            # ):
            # Append the valid sequence to the list of valid sequences
            if distance(clipped_seq_1, clipped_seq_2) > 0:
                input_target_list.append((clipped_seq_1, clipped_seq_2))

                # Increment the counter
                count_sequences += 1
                # Print the progress
                if count_sequences % 1000 == 0:
                    print(f"Found {count_sequences} valid pairs.")

    get_mutations_and_plot(
        sequences=[sequence_pair[1] for sequence_pair in input_target_list],
        sequence_start_postion=sequence_start_postion,
        sequence_end_postion=sequence_end_postion,
        seq_len=max_seq_length,
        y_type="overall_data",
    )
    # Split data into train, validation and test datasets
    split = {"train": 0.6, "val": 0.2, "test": 0.2}
    train_val_test_indices = {
        "train_upto": int(split["train"] * len(input_target_list)),
        "val_upto": int(split["train"] * len(input_target_list) + split["val"] * len(input_target_list)),
    }
    train_list = []
    val_list = []
    test_list = []
    for i, input_target_pair in enumerate(input_target_list, 1):
        if i <= train_val_test_indices["train_upto"]:
            train_list.append(input_target_pair)
        elif i > train_val_test_indices["train_upto"] and i <= train_val_test_indices["val_upto"]:
            val_list.append(input_target_pair)
        else:
            test_list.append(input_target_pair)

    # File operations
    with open(f"{CURRENT_RUN_DATA_DIR}/train.json", "w") as fout:
        dump(train_list, fout)
    with open(f"{CURRENT_RUN_DATA_DIR}/val.json", "w") as fout:
        dump(val_list, fout)
    with open(f"{CURRENT_RUN_DATA_DIR}/test.json", "w") as fout:
        dump(test_list, fout)
    print(
        f"Completed parsing merged data...\n Completed creation of [input, target] pairs...\nSaved train, validation and test files to path :\n{CURRENT_RUN_DATA_DIR}"
    )

    get_mutations_and_plot(
        sequences=[sequence_pair[1] for sequence_pair in train_list],
        sequence_start_postion=sequence_start_postion,
        sequence_end_postion=sequence_end_postion,
        seq_len=max_seq_length,
        y_type="train_ground_truth",
    )
    get_mutations_and_plot(
        sequences=[sequence_pair[1] for sequence_pair in val_list],
        sequence_start_postion=sequence_start_postion,
        sequence_end_postion=sequence_end_postion,
        seq_len=max_seq_length,
        y_type="val_ground_truth",
    )
    get_mutations_and_plot(
        sequences=[sequence_pair[1] for sequence_pair in test_list],
        sequence_start_postion=sequence_start_postion,
        sequence_end_postion=sequence_end_postion,
        seq_len=max_seq_length,
        y_type="test_ground_truth",
    )


def is_valid_sequence_pair(
    seq1: str, seq2: str, minimum_levenshtein_distance: int, maximum_levenshtein_distance: int,
) -> bool:
    lev_distance = distance(seq1, seq2)
    if lev_distance < minimum_levenshtein_distance or lev_distance > maximum_levenshtein_distance:
        return False
    else:
        return True
