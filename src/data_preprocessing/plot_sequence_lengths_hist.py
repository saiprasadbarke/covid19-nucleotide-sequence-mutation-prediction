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
            unique_len_array.append(str(length))

    counts_list = []
    for unique_len in unique_len_array:
        unique_len = int(unique_len)
        counts_list.append(length_array.count(unique_len))

    plt.figure(figsize=(20, 20))
    plt.bar(unique_len_array, counts_list)
    plt.xlabel("Sequence length")
    plt.xticks(rotation=90)
    plt.title("Frequency of occurance")
    plt.savefig(hist_path)
    plt.show()


if __name__ == "__main__":
    histogram_path = f"{Path.cwd()}/plots/sequence_length_plot.png"
    completed_sequences_file_path = f"{Path.cwd()}/data/complete_sequences/complete_sequences.fasta"
    plot_seq_len_hist(inp_file=completed_sequences_file_path, hist_path=histogram_path)
