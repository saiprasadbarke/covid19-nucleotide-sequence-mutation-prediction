# Standard
from typing import Dict
from json import dump, load
from pathlib import Path
from os import chdir, listdir

# Local

# External

from Bio import SeqIO

LIST_OF_CLADES = [
    "20C",
    "19A",
    "21C (Epsilon)",
    "21F (Iota)",
    "20B",
    "20J (Gamma, V3)",
    "20A",
    "20E (EU1)",
    "19B",
    "21B (Kappa)",
    "21J (Delta)",
    "20F",
    "21A (Delta)",
]  # All clades with number of sequences >18000 were chosen and the rest were droppped


def create_merged_data(sequences_file: str, clades_file: str, output_file: str):
    description_sequence_dict = {}
    for sequence_record in SeqIO.parse(sequences_file, "fasta"):
        description_sequence_dict[sequence_record.description] = str(sequence_record.seq)

    seqname_clade_dict = {}
    for line in open(clades_file):
        line_data = line.split("\t")
        seqname_clade_dict[line_data[0].replace('"', "")] = line_data[1].replace('"', "")

    ds = [description_sequence_dict, seqname_clade_dict]
    merged_dict = {}
    for description in description_sequence_dict.keys():
        merged_dict[description] = tuple(d[description] for d in ds)

    clade_sequence_dict = {}
    for valid_clade in LIST_OF_CLADES:
        single_clade_sequences = {}
        for _seq_description, sequence_clade_tuple in merged_dict.items():
            if sequence_clade_tuple[1] == valid_clade:
                single_clade_sequences[sequence_clade_tuple[0]] = ""
        clade_sequence_dict[valid_clade] = single_clade_sequences

    for clade, sequences in clade_sequence_dict.items():
        print(f"Number of sequences for clade {clade} after removing duplicates = {len(sequences)}")
    with open(output_file, "w") as fout:
        dump(clade_sequence_dict, fout)


if __name__ == "__main__":

    print(Path.cwd())
    # Paths
    clade_filepath = f"{Path.cwd()}/data/complete_sequences/complete_clades.tabular"
    input_fasta_filepath = f"{Path.cwd()}/data/complete_sequences/complete_sequences.fasta"
    claded_sequences_filepath = f"{Path.cwd()}/data/claded_sequences/claded_sequences.json"

    # Function call
    claded_sequences = create_merged_data(
        sequences_file=input_fasta_filepath, clades_file=clade_filepath, output_file=claded_sequences_filepath
    )
    print("Completed data pre processing...")
