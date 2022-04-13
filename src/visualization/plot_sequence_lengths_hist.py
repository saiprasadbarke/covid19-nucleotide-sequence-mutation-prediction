# Standard
from pathlib import Path
from json import load

# External
import matplotlib.pyplot as plt
from Bio import SeqIO


def plot_seq_len_hist(hist_path: str, inp_file: str = None, sequences_array: list[str] = None):
    length_array = []
    if inp_file and not sequences_array:
        for seq_record in SeqIO.parse(inp_file, "fasta"):
            length_array.append(len(seq_record.seq))
    elif sequences_array and not inp_file:
        for seq in sequences_array:
            length_array.append(len(seq))
    else:
        return "Warning: You can either choose the file input or the sequences array and not both!"

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


def plot_cladewise_sequence_lengths(merged_file: str, output_folder: str):
    data = load(open(merged_file))
    for clade, sequences_list in data.items():
        plot_seq_len_hist(hist_path=f"{output_folder}/{clade}.png", sequences_array=sequences_list)


if __name__ == "__main__":
    # histogram_path = f"{Path.cwd()}/plots/sequence_length_plot.png"
    # sequences_file_path = f"{Path.cwd()}/data/01cleaned/sequences.fasta"
    # plot_seq_len_hist(inp_file=sequences_file_path, hist_path=histogram_path)

    merged_file = f"{Path.cwd()}/data/02merged/clade_seq.json"
    output_folder = f"{Path.cwd()}/plots/cladewise_seqlen"
    plot_cladewise_sequence_lengths(merged_file=merged_file, output_folder=output_folder)
