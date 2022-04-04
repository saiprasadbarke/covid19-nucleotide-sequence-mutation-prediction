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
    "21C",
    "21F",
    "20B",
    "20J",
    "20A",
    "20E",
    "19B",
]  # All clades with number of sequences >18000 were chosen and the rest were droppped


def create_merged_data(sequences_file: str, clades_file: str):
    description_sequence_dict = {}
    for sequence_record in SeqIO.parse(sequences_file, "fasta"):
        description_sequence_dict[sequence_record.description] = str(
            sequence_record.seq
        )

    seqname_clade_dict = {}
    for line in open(clades_file):
        line_data = line.split("\t")
        seqname_clade_dict[line_data[0].replace('"', "")] = line_data[1].split(" ")[0]

    ds = [description_sequence_dict, seqname_clade_dict]
    merged_dict = {}
    for description in description_sequence_dict.keys():
        merged_dict[description] = tuple(d[description] for d in ds)
    claded_sequences = {}
    for _description, sequence_clade_tuple in merged_dict.items():
        # Don't need a seperate function to remove duplicates
        claded_sequences[sequence_clade_tuple[0]] = sequence_clade_tuple[1]
    return claded_sequences


def create_cladewise_datasets(claded_sequences: Dict[str, str], output_folder: str):
    for valid_clade in LIST_OF_CLADES:
        single_clade_sequences = []
        for sequence, clade in claded_sequences.items():
            if clade == valid_clade:
                single_clade_sequences.append(sequence)
        if len(single_clade_sequences) > 0:
            with open(f"{output_folder}/{valid_clade}.json", "w") as fout:
                dump(single_clade_sequences, fout)


def count_sequences(folder_path: str):
    chdir(folder_path)
    for file in listdir():
        file_path = f"{folder_path}/{file}"
        f = open(file_path)
        list_of_claded_sequences = load(f)
        print(f"Number of sequences for {file} is {len(list_of_claded_sequences)}")


if __name__ == "__main__":

    print(Path.cwd())
    # Paths
    clade_filepath = f"{Path.cwd()}/data/complete_sequences/complete_clades.tabular"
    input_fasta_filepath = (
        f"{Path.cwd()}/data/complete_sequences/complete_sequences.fasta"
    )
    individual_claded_sequences_folder = f"{Path.cwd()}/data/claded_sequences/clades"

    # Nigeria
    clade_filepath_nigeria = (
        f"{Path.cwd()}/data/countrywise_split/clades_tabular/nigeria_clades.tabular"
    )
    input_fasta_filepath_nigeria = (
        f"{Path.cwd()}/data/countrywise_split/fasta/Nigeria.fasta"
    )
    individual_claded_sequences_folder_nigeria = (
        f"{Path.cwd()}/data/claded_sequences/nigeria"
    )

    # Function call
    claded_sequences = create_merged_data(
        sequences_file=input_fasta_filepath,
        clades_file=clade_filepath,
    )
    create_cladewise_datasets(claded_sequences, individual_claded_sequences_folder)
    count_sequences(individual_claded_sequences_folder)
    print("Completed data pre processing...")
