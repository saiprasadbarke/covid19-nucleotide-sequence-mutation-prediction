from json import load, dump
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
from settings.constants import SAVED_PLOTS_PATH, SAVED_STATS_PATH

from settings.reference_sequence import REFERENCE_GENOME


def get_mutations_and_plot(
    sequences: List[str], sequence_start_postion: int, sequence_end_postion: int, seq_len: int, y_type: str,
):
    difference_indices = {}
    similar_indices = {}
    for sequence in sequences:
        # Compute differences wrt reference genome and add them to difference_indices
        difference = get_string_difference_indices(
            REFERENCE_GENOME[sequence_start_postion:sequence_end_postion], sequence, sequence_start_postion, seq_len,
        )
        for idx in difference:
            if idx in difference_indices.keys():
                difference_indices[idx] += 1  # Added a 1 for every new instance of the index
            else:
                difference_indices[idx] = 1  # Capture the first instance of the index
        # Compute similarity wrt reference genome and add them to similar_indices
        similarity = get_string_similarities_indices(
            REFERENCE_GENOME[sequence_start_postion:sequence_end_postion], sequence, sequence_start_postion, seq_len,
        )
        for idy in similarity:
            if idy in similar_indices.keys():
                similar_indices[idy] += 1
            else:
                similar_indices[idy] = 1
    # Sort the difference indices data
    sorted_difference_indices = dict(sorted(difference_indices.items(), key=lambda x: x[0]))
    # Add 0 for indices with no difference
    complete_sequence_mutation_data = {}
    for i in range(sequence_start_postion, sequence_end_postion):
        if i in sorted_difference_indices.keys():
            complete_sequence_mutation_data[i] = sorted_difference_indices[i]
        else:
            complete_sequence_mutation_data[i] = 0
    # Generate reports for mutations between reference genome and target sequences
    difference_indices_file = f"{SAVED_STATS_PATH}/difference_indices_ref_{y_type}.json"
    with open(difference_indices_file, "w") as fout:
        dump(complete_sequence_mutation_data, fout)
    sorted_difference_indices_by_value = dict(sorted(complete_sequence_mutation_data.items(), key=lambda x: x[1]))
    difference_indices_file_sort_by_value = f"{SAVED_STATS_PATH}/difference_indices_ref_{y_type}_sort_by_value.json"
    with open(difference_indices_file_sort_by_value, "w") as fout:
        dump(sorted_difference_indices_by_value, fout)

    # Sort the similarity indices data
    sorted_similarity_indices = dict(sorted(similar_indices.items(), key=lambda x: x[0]))
    # Add 0 for indices with no difference
    complete_sequence_similarity_data = {}
    for i in range(sequence_start_postion, sequence_end_postion):
        if i in sorted_similarity_indices.keys():
            complete_sequence_similarity_data[i] = sorted_similarity_indices[i]
        else:
            complete_sequence_similarity_data[i] = 0
    # Generate reports for mutations between reference genome and target sequences
    similarity_indices_file = f"{SAVED_STATS_PATH}/similarity_indices_ref_{y_type}.json"
    with open(similarity_indices_file, "w") as fout:
        dump(complete_sequence_similarity_data, fout)
    sorted_similarity_indices_by_value = dict(sorted(complete_sequence_similarity_data.items(), key=lambda x: x[1]))
    similarity_indices_file_sort_by_value = f"{SAVED_STATS_PATH}/similarity_indices_ref_{y_type}_sort_by_value.json"
    with open(similarity_indices_file_sort_by_value, "w") as fout:
        dump(sorted_similarity_indices_by_value, fout)

    # Plotting the graph
    mutations_graph_path_ref = f"{SAVED_PLOTS_PATH}/mutation_sites_ref_{y_type}.png"
    x = list(complete_sequence_mutation_data.keys())
    y = list(complete_sequence_mutation_data.values())
    x1 = list(complete_sequence_similarity_data.keys())
    y1 = list(complete_sequence_similarity_data.values())
    assert x == x1, "Labels for both similarities and differences dont match"

    fig, ax = plt.subplots(figsize=(60, 20))
    ax.bar(x, y, label="Differences")
    ax.bar(x, y1, bottom=y, label="Similarities")
    # plt.bar(x, y)
    ax.set_xlabel("Indices")
    ax.set_ylabel("Frequencies")
    ax.set_title(
        f"Distribution of similarities and differences at different indices of {y_type} with respect to Reference genome"
    )
    ax.legend()
    # plt.xlabel("Indices")
    # plt.xticks(rotation=90)
    # plt.ylabel("Frequency of mutations")
    # plt.title(f"Distribution of mutations at different indices of {y_type} with respect to Reference genome")
    plt.savefig(mutations_graph_path_ref)
    plt.show()


def get_string_difference_indices(str1: str, str2: str, start: int = 0, seq_len: int = 500) -> List[int]:
    return [i + start for i in range(seq_len) if str1[i] != str2[i]]


def get_string_similarities_indices(str1: str, str2: str, start: int = 0, seq_len: int = 500) -> List[int]:
    return [i + start for i in range(seq_len) if str1[i] == str2[i]]

