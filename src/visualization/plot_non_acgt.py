# Standard
from math import ceil
from pathlib import Path
from statistics import mean
from json import dump

# External
from Bio import SeqIO
from matplotlib import pyplot as plt


def stat_non_acgt(stats_file_path: str, inp_file: str = None):
    list_of_total_non_acgt_chars = []
    count = 0
    for seq_record in SeqIO.parse(inp_file, "fasta"):
        num_A = str(seq_record.seq).count("A")
        num_C = str(seq_record.seq).count("C")
        num_G = str(seq_record.seq).count("G")
        num_T = str(seq_record.seq).count("T")
        total_len = len(str(seq_record.seq))
        num_total_non_acgt = total_len - (num_A + num_C + num_G + num_T)
        if num_total_non_acgt > 0:
            count += 1
        list_of_total_non_acgt_chars.append(num_total_non_acgt)

    stats_dict = {
        "max_non_acgt": max(list_of_total_non_acgt_chars),
        "min_non_acgt": min(list_of_total_non_acgt_chars),
        "avg_non_acgt": ceil(mean(list_of_total_non_acgt_chars)),
        "Number of seq with non ACGT": count,
    }

    f = open(stats_file_path, "w")
    dump(stats_dict, f)

    return list_of_total_non_acgt_chars


def plot_non_acgt(list_of_total_non_acgt_chars: list[int], hist_path: str):
    unique_total_nonacgt_array = []
    for number in list_of_total_non_acgt_chars:
        if number not in unique_total_nonacgt_array:
            unique_total_nonacgt_array.append(number)

    number_of_nonacgt_and_counts_dict = {}
    for unique_len in unique_total_nonacgt_array:
        number_of_nonacgt_and_counts_dict[str(unique_len)] = list_of_total_non_acgt_chars.count(unique_len)
    sorted_dict = dict(sorted(number_of_nonacgt_and_counts_dict.items(), key=lambda item: item[1]))
    plt.figure(figsize=(20, 20))
    plt.bar(list(sorted_dict.keys()), list(sorted_dict.values()))
    plt.xlabel("Sequence length")
    plt.xticks(rotation=90, fontsize=5)
    plt.ylabel("Frequency of occurance")
    plt.savefig(hist_path)
    plt.show()


if __name__ == "__main__":
    sequences_file_path = f"{Path.cwd()}/data/01cleaned/sequences.fasta"
    stats_file = f"{Path.cwd()}/reports/stats/num_total_non_acgt.json"
    histogram_path = f"{Path.cwd()}/reports/plots/num_total_non_acgt.png"
    list_of_total_non_acgt_chars = stat_non_acgt(stats_file, sequences_file_path)
    plot_non_acgt(list_of_total_non_acgt_chars, histogram_path)
