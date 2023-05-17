from json import dump, load
from Levenshtein import distance
from settings.constants import CLADE_PAIRS_NEXTCLADE, COMMON_REPORTS_PATH, MERGED_DATA
from settings.reference_sequence import REFERENCE_GENOME
from visualization.plot_mutation_sites import get_mutations_and_plot


def plot_mutations_all_clades():
    data = load(open(MERGED_DATA))
    for clade in data:
        list_of_seq = list(data[clade].keys())
        get_mutations_and_plot(
            sequences=list_of_seq, sequence_start_postion=0, sequence_end_postion=3700, seq_len=3700, y_type=clade,
        )


def plot_lev_refgen():
    data = load(open(MERGED_DATA))
    data_dump = {}
    for clade in data:
        list_of_seq = list(data[clade].keys())
        lev_distances = {}
        for seq in list_of_seq:
            lev_d = distance(seq, REFERENCE_GENOME)
            if lev_d in lev_distances:
                lev_distances[lev_d] += 1
            else:
                lev_distances[lev_d] = 1
        sorted_lev_dist_dict = dict(sorted(lev_distances.items(), key=lambda item: item[1]))
        data_dump[clade] = sorted_lev_dist_dict
    with open(f"{COMMON_REPORTS_PATH}/plot_lev_refgen.json", "w") as fout:
        dump(data_dump, fout)


def print_number_pairs():
    data = load(open(MERGED_DATA))
    data_dump = {}
    for clade_pair in CLADE_PAIRS_NEXTCLADE.values():
        clade1 = clade_pair[0]
        clade2 = clade_pair[1]
        clade1_sequences = list(data[clade1].keys())
        clade2_sequences = list(data[clade2].keys())
        prod = len(clade1_sequences) * len(clade2_sequences)
        data_dump[f"{clade1}-{clade2}"] = prod
    with open(f"{COMMON_REPORTS_PATH}/number_of_clade_pair_samples.json", "w") as fout:
        dump(data_dump, fout)


if __name__ == "__main__":
    # plot_mutations_all_clades()
    # plot_lev_refgen()
    print_number_pairs()
