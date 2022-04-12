# Standard
from pathlib import Path

# External
import matplotlib.pyplot as plt
from Bio import SeqIO


def plot_seq_len_hist(inp_file: str, hist_path: str):
    length_array = []
    for seq_record in SeqIO.parse(inp_file, "fasta"):
        length_array.append(len(seq_record.seq))

    unique_len_array = []
    for length in length_array:
        if length not in unique_len_array:
            unique_len_array.append(length)

    number_of_seq_and_counts_dict = {}
    for unique_len in unique_len_array:
        number_of_seq_and_counts_dict[str(unique_len)] = length_array.count(unique_len)
    sorted_dict = dict(sorted(number_of_seq_and_counts_dict.items(), key=lambda item: item[1]))
    plt.figure(figsize=(60, 20))
    plt.bar(list(sorted_dict.keys()), list(sorted_dict.values()))
    plt.xlabel("Sequence length")
    plt.xticks(rotation=90, fontsize=5)
    plt.ylabel("Frequency of occurance")
    plt.savefig(hist_path)
    plt.show()


if __name__ == "__main__":
    histogram_path = f"{Path.cwd()}/plots/sequence_length_plot.png"
    sequences_file_path = f"{Path.cwd()}/data/01cleaned/sequences.fasta"
    plot_seq_len_hist(inp_file=sequences_file_path, hist_path=histogram_path)
