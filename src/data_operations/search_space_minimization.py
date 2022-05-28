# Standard
from itertools import product
from json import dump, load
from pathlib import Path
from typing import List

# Local

# External
from Levenshtein import distance

CLADE_PAIRS = {("21A", "21J")}

def get_dataset_mutation_stats(paired_clades_file: str, levenshtein_file: str, difference_indices_file:str):
    data = load(open(paired_clades_file))
    for clade_pair, clades_lists_dict in data.items():
        if (clade_pair.split("_")[0], clade_pair.split("_")[1]) in CLADE_PAIRS:
            levenshtein_distances = {}
            difference_indices ={}
            clade1_sequences = list(clades_lists_dict.values())[0] # 2756
            clade2_sequences = list(clades_lists_dict.values())[1] # 2536
            cartesian_product_iterator = product(clade1_sequences, clade2_sequences)
            for i, current_clade_pair in enumerate(cartesian_product_iterator):
                lev_distance = distance(current_clade_pair[0], current_clade_pair[1])
                if lev_distance in levenshtein_distances.keys():
                    levenshtein_distances[lev_distance] += 1
                else:
                    levenshtein_distances[lev_distance] = 1
                difference = get_string_difference_indices(current_clade_pair[0], current_clade_pair[1])
                for index in difference:
                    if index in difference_indices.keys():
                        difference_indices[index]+=1
                    else:
                        difference_indices[index] = 1
                if i%10000 ==0:
                    print(f"Completed comparing {i} pairs")
    sorted_levenshtein_distances = dict(sorted(levenshtein_distances.items(), key=lambda item: item[1]))
    sorted_difference_indices = dict(sorted(difference_indices.items(), key=lambda item: item[1]))
    with open(levenshtein_file, "w")  as fout:
        dump(sorted_levenshtein_distances, fout)
    with open(difference_indices_file, "w") as fout:
        dump(sorted_difference_indices, fout)


def get_string_difference_indices(str1:str, str2:str)-> List[int]:
    return [i for i in range(min(len(str1), len(str2))) if str1[i]!=str2[i]]

if __name__ == "__main__":
    paired_clades_file = f"{Path.cwd().parents[0]}/data/paired.json"
    levenshtein_file = f"{Path.cwd()}/reports/stats/levenshtein_distances_distribution.json"
    difference_indices = f"{Path.cwd()}/reports/stats/most_mutated_indices.json"
    get_dataset_mutation_stats(paired_clades_file, levenshtein_file, difference_indices)
        