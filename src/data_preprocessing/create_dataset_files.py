# Standard
from typing import Dict
from json import dump, load
from pathlib import Path
from os import chdir, listdir

# Local

# External

from Bio import SeqIO

LIST_OF_CLADES = [
    "19A",
    "19B",
    "20A",
    "20B",
    "20C",
    "20E",
    "21A",
    "21B",
    "21D",
    "21H",
    "20D",
    "20F",
    "20I",
    "20J",
    "21M",
    "21E",
    "20G",
    "20H",
    "21C",
    "21F",
    "21I",
    "21J",
    "21G",
    "21K",
    "21L",
]


def create_merged_data(sequences_file: str, clades_file: str, output_file: str):
    description_sequence_dict = {}
    for sequence_record in SeqIO.parse(sequences_file, "fasta"):
        description_sequence_dict[sequence_record.description] = str(sequence_record.seq)

    description_clade_dict = {}
    for line in open(clades_file):
        line_data = line.split("\t")
        # The strings have an additional pair of quotes which needs to be removed for proper comparision.
        # The split and indexing operation on the right hand side splits the full clade name (Eg: 21I (Delta)) by space and selects only the alphanumeric clade name
        if line_data[3] == "good":  # Index is 4 for the clades.tabular
            description_clade_dict[line_data[0].replace('"', "")] = line_data[1].replace('"', "").split(" ")[0]

    ds = [description_sequence_dict, description_clade_dict]
    merged_dict = {}
    for description in description_sequence_dict.keys():
        if description in description_clade_dict.keys():
            merged_dict[description] = tuple(d[description] for d in ds)

    clade_sequence_dict = {}
    for valid_clade in LIST_OF_CLADES:
        single_clade_sequences_dict = {}
        for _seq_description, sequence_clade_tuple in merged_dict.items():
            # The sequence clade tuple has the sequence in index 0 and the clade in index 1
            if sequence_clade_tuple[1] == valid_clade:
                # Adding the sequences as keys in a dictionary ensures there is no duplication
                single_clade_sequences_dict[sequence_clade_tuple[0]] = ""
        if single_clade_sequences_dict:
            clade_sequence_dict[valid_clade] = single_clade_sequences_dict

    for clade, sequences in clade_sequence_dict.items():
        print(f"Number of sequences for clade {clade} after removing duplicates and filtering by qc = {len(sequences)}")
    with open(output_file, "w") as fout:
        dump(clade_sequence_dict, fout)


if __name__ == "__main__":

    print(f"Parent dir : {Path.cwd().parents[0]}")
    # Paths
    # clade_filepath = f"{Path.cwd().parents[0]}/data/clades.tabular"
    # input_fasta_filepath = f"{Path.cwd().parents[0]}/data/sequences.fasta"
    # claded_sequences_filepath = f"{Path.cwd().parents[0]}/data/clade_seq.json"

    clade_filepath = f"{Path.cwd()}/data/test/india.tabular"
    input_fasta_filepath = f"{Path.cwd()}/data/test/india.fasta"
    claded_sequences_filepath = f"{Path.cwd()}/data/merged/india.json"
    # Function call
    claded_sequences = create_merged_data(
        sequences_file=input_fasta_filepath, clades_file=clade_filepath, output_file=claded_sequences_filepath
    )
    print("Completed data pre processing...")
