from json import dump, load
from Levenshtein import distance
from settings.constants import COMMON_REPORTS_PATH, MERGED_DATA
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
            if lev_d in lev_distances.keys():
                lev_distances[lev_d] += 1
            else:
                lev_distances[lev_d] = 1
        sorted_lev_dist_dict = dict(sorted(lev_distances.items(), key=lambda item: item[1]))
        data_dump[clade] = sorted_lev_dist_dict
    with open(f"{COMMON_REPORTS_PATH}/plot_lev_refgen.json", "w") as fout:
        dump(data_dump, fout)


if __name__ == "__main__":
    plot_mutations_all_clades()
    #plot_lev_refgen()
