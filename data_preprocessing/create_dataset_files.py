from typing import List
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
from pathlib import Path
from claded_seq import CladedSequence
from json import dump
from data_cleaning import remove_duplicate_sequences, remove_clades
from globals.constants import list_of_clades


def create_merged_data(sequences_file: str, clades_file: str, output_file: str = None):
    claded_sequences = []
    sequence_records = SeqIO.parse(sequences_file, "fasta")
    data = pd.read_table(clades_file, header=None)
    sequence_names = data.iloc[:, 0]
    clades = data.iloc[:, 1]
    for sequence_record in sequence_records:
        for name, clade in zip(sequence_names, clades):
            if sequence_record.description == name:
                claded_sequence = CladedSequence(str(sequence_record.seq), clade)
                claded_sequences.append(claded_sequence.__dict__)
    if output_file is not None:
        with open(output_file, "w") as fout:
            dump(claded_sequences, fout)
    return claded_sequences


def create_clade_datasets(claded_sequences: List[dict[str, str]], output_folder: str):
    for clade in list_of_clades:
        single_clade_sequences = []
        for claded_sequence in claded_sequences:
            if claded_sequence["clade"] == clade:
                single_clade_sequences.append(claded_sequence)
        with open(f"{output_folder}/{clade}.json", "w") as fout:
            dump(single_clade_sequences, fout)


if __name__ == "__main__":

    clade_filepath = f"{Path.cwd()}/data/complete_sequences/complete_sequences.tabular"
    input_fasta_filepath = (
        f"{Path.cwd()}/data/complete_sequences/complete_sequences.fasta"
    )
    # claded_sequences_filepath = (
    #    f"{Path.cwd()}/data/claded_sequences/claded_sequences_nigeria.json"
    # )
    individual_claded_sequences_folder = f"{Path.cwd()}/data/claded_sequences/clades"
    claded_sequences = create_merged_data(
        sequences_file=input_fasta_filepath,
        clades_file=clade_filepath,
        # output_file=claded_sequences_filepath,
    )
    print(f"Original Length : {len(claded_sequences)}")
    claded_sequences = remove_duplicate_sequences(claded_sequences)
    print(f"New Length after removing duplicates: {len(claded_sequences)}")
    claded_sequences = remove_clades(claded_sequences)
    print(f"New Length after removing clades: {len(claded_sequences)}")
    create_clade_datasets(claded_sequences, individual_claded_sequences_folder)
    print("Completed data pre processing...")
