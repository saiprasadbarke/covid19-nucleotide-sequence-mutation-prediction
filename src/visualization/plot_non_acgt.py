# Standard
from math import ceil
from pathlib import Path
from statistics import mean
from json import dump

# External
from Bio import SeqIO


def plot_non_acgt(stats_file_path: str, inp_file: str = None):
    list_of_total_non_acgt_chars = []
    for seq_record in SeqIO.parse(inp_file, "fasta"):
        num_A = str(seq_record.seq).count("A")
        num_C = str(seq_record.seq).count("C")
        num_G = str(seq_record.seq).count("G")
        num_T = str(seq_record.seq).count("T")
        total_len = len(str(seq_record.seq))
        num_total_non_acgt = total_len - (num_A + num_C + num_G + num_T)
        list_of_total_non_acgt_chars.append(num_total_non_acgt)

    stats_dict = {
        "max_non_acgt": max(list_of_total_non_acgt_chars),
        "min_non_acgt": min(list_of_total_non_acgt_chars),
        "avg_non_acgt": ceil(mean(list_of_total_non_acgt_chars)),
    }

    f = open(stats_file_path, "w")
    dump(stats_dict, f)


if __name__ == "__main__":
    sequences_file_path = f"{Path.cwd()}/data/01cleaned/sequences.fasta"
    stats_file = f"{Path.cwd()}/reports/stats/num_total_non_acgt.json"
    plot_non_acgt(stats_file, sequences_file_path)
