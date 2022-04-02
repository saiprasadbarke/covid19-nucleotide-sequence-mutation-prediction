from typing import List
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
from pathlib import Path
from claded_seq import CladedSequence
from json import dump
from data_cleaning import remove_duplicate_sequences


def create_merged_data(sequences_file: str, clades_file: str, output_file: str):
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
    with open(output_file, "w") as fout:
        dump(claded_sequences, fout)
    return claded_sequences


if __name__ == "__main__":
    # clade_filepath = f"{Path.cwd()}/data/complete_sequences/complete_clades.tabular"
    # input_fasta_file_path = (
    #    f"{Path.cwd()}/data/complete_sequences/complete_sequences.fasta"
    # )
    clade_filepath = (
        f"{Path.cwd()}/data/countrywise_split/clades_tabular/nigeria_clades.tabular"
    )
    input_fasta_filepath = f"{Path.cwd()}/data/countrywise_split/fasta/Nigeria.fasta"
    claded_sequences_filepath = (
        f"{Path.cwd()}/data/claded_sequences/claded_sequences_nigeria.json"
    )
    claded_sequences = create_merged_data(
        sequences_file=input_fasta_filepath,
        clades_file=clade_filepath,
        output_file=claded_sequences_filepath,
    )
    print(f"Original Length : {len(claded_sequences)}")
    claded_sequences = remove_duplicate_sequences(claded_sequences)
    print(f"New Length : {len(claded_sequences)}")
