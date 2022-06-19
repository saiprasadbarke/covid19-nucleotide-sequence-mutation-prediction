# Standard
from itertools import product
from json import dump, load
from random import sample
from typing import List, Tuple

# Local
from settings.constants import MERGED_DATA, CURRENT_RUN_DATA_DIR, SAVED_PLOTS_PATH, SAVED_STATS_PATH

# External
from Levenshtein import distance
from settings.reference_sequence import REFERENCE_GENOME

from visualization.plot_mutation_sites import plot_mutations


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
    random_sampled_cartesian_product_list = sample(cartesian_product_list, len(cartesian_product_list),)
    input_target_list = []
    difference_indices_x_y = {}
    difference_indices_refgen_gt = {}
    for seq_pair in random_sampled_cartesian_product_list:
        if count_sequences == number_of_sequence_pairs:
            break
        else:
            clipped_seq_1 = seq_pair[0][sequence_start_postion:sequence_end_postion]
            clipped_seq_2 = seq_pair[1][sequence_start_postion:sequence_end_postion]
            if is_valid_sequence_pair(
                seq1=clipped_seq_1,
                seq2=clipped_seq_2,
                minimum_levenshtein_distance=minimum_levenshtein_distance,
                maximum_levenshtein_distance=maximum_levenshtein_distance,
            ):
                # Append the valid sequence to the list of valid sequences
                input_target_list.append((clipped_seq_1, clipped_seq_2))

                # Compute differences wrt x sequence and reference genome and add them to the respective index dictionaries
                difference_x_y = get_string_difference_indices(clipped_seq_1, clipped_seq_2, sequence_start_postion,)
                for idx_x_y in difference_x_y:
                    if idx_x_y in difference_indices_x_y.keys():
                        difference_indices_x_y[idx_x_y] += 1
                    else:
                        difference_indices_x_y[idx_x_y] = 1

                difference_reference_gt = get_string_difference_indices(
                    REFERENCE_GENOME[sequence_start_postion:sequence_end_postion],
                    clipped_seq_2,
                    sequence_start_postion,
                )
                for idx_rg_y in difference_reference_gt:
                    if idx_rg_y in difference_indices_refgen_gt.keys():
                        difference_indices_refgen_gt[idx_rg_y] += 1  # Added a 1 for every new instance of the index
                    else:
                        difference_indices_refgen_gt[idx_rg_y] = 1  # Capture the first instance of the index

                # Increment the counter
                count_sequences += 1
                # Print the progress
                if count_sequences % 1000 == 0:
                    print(f"Found {count_sequences} valid pairs.")

    # Generate reports for mutations between input and target sequences
    difference_indices_file_xy = f"{SAVED_STATS_PATH}/difference_indices_xy.json"
    with open(difference_indices_file_xy, "w") as fout:
        dump(difference_indices_x_y, fout)
    mutations_graph_path_xy = f"{SAVED_PLOTS_PATH}/mutation_sites_xy.png"
    data_dump_path_xy = f"{SAVED_STATS_PATH}/sorted_difference_indices_xy.json"
    plot_mutations(difference_indices_file_xy, mutations_graph_path_xy, data_dump_path_xy)

    # Generate reports for mutations between reference genome and target sequences
    difference_indices_file_ref = f"{SAVED_STATS_PATH}/difference_indices_ref_gt.json"
    with open(difference_indices_file_ref, "w") as fout:
        dump(difference_indices_refgen_gt, fout)
    mutations_graph_path_ref = f"{SAVED_PLOTS_PATH}/mutation_sites_ref_gt.png"
    data_dump_path_ref = f"{SAVED_STATS_PATH}/sorted_difference_indices_ref_gt.json"
    plot_mutations(difference_indices_file_ref, mutations_graph_path_ref, data_dump_path_ref)

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


def is_valid_sequence_pair(
    seq1: str, seq2: str, minimum_levenshtein_distance: int, maximum_levenshtein_distance: int,
) -> bool:
    lev_distance = distance(seq1, seq2)
    if lev_distance < minimum_levenshtein_distance or lev_distance > maximum_levenshtein_distance:
        return False
    else:
        return True


def get_string_difference_indices(str1: str, str2: str, start: int = 0) -> List[int]:
    return [i + start for i in range(len(str1)) if str1[i] != str2[i]]
