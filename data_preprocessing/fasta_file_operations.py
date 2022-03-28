# Standard

# Local

# External
from Bio import SeqIO


def read_fasta_file(import_file_path: str):  # provide return types here
    for seq_record in SeqIO.parse("data/spikenuc0312.fasta", "fasta"):
    return None


def create_subset_fasta_file(export_file_path: str):
    return None


def count_number_of_sequences(import_file_path: str):
    count = 0
    sequence_file = open(import_file_path)
    for sequence in sequence_file:
        if sequence.startswith(">"):
            count+=1
    sequence_file.close()
    return count