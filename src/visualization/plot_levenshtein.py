from itertools import product
from json import load
from random import sample
from settings.constants import MERGED_DATA, CLADE_PAIRS_NEXTCLADE, COMMON_REPORTS_PATH
from settings.reference_sequence import REFERENCE_GENOME
from Levenshtein import distance


def plot_lev_intraclade():
    data = load(open(MERGED_DATA))
    dump = {}
    for clade_pair in CLADE_PAIRS_NEXTCLADE.values():
        clade1 = clade_pair[0]
        clade2 = clade_pair[1]
        print(f"Plotting {clade1}-{clade2}")
        clade1_sequences = list(data[clade1].keys())
        clade2_sequences = list(data[clade2].keys())
        cartesian_product_list = list(product(clade1_sequences, clade2_sequences))
        lev_distance_dict = {}
        for pair in cartesian_product_list:
            lev_dist = distance(pair[0], pair[1])
            if lev_dist in lev_distance_dict.keys():
                lev_distance_dict[lev_dist] += 1
            else:
                lev_distance_dict[lev_dist] = 1
        sorted_lev_dist_dict = dict(sorted(lev_distance_dict.items(), key=lambda item: item[1]))
        dump[f"{clade1}-{clade2}"] = sorted_lev_dist_dict
    with open(f"{COMMON_REPORTS_PATH}/plot_lev_intraclade.json", "w") as fout:
        dump(dump, fout)


def plot_lev_refgen():

    raise NotImplementedError

