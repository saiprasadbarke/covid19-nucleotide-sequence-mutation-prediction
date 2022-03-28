# Standard
from pathlib import Path
import os
# Local
from data_preprocessing.helper import batch_iterator
# External
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

def read_fasta_file(import_file_path: str):  # provide return types here
    for seq_record in SeqIO.parse(import_file_path, "fasta"):
        date = get_seq_date(seq=seq_record)
        continue 

def create_subset_fasta_file(import_file_path: str,  number_of_sequences: int,export_file_path: str = None):
    record_iter = SeqIO.parse(open(import_file_path), "fasta")
    for i, batch in enumerate(batch_iterator(record_iter, number_of_sequences)):
        filename = f"{export_file_path}group_{i+1}.fasta"
        with open(filename, "w") as handle:
            count = SeqIO.write(batch, handle, "fasta")
        print("Wrote %i records to %s" % (count, filename))
    return None


def count_number_of_sequences(import_file_path: str):
    count = 0
    sequence_file = open(import_file_path)
    for sequence in sequence_file:
        if sequence.startswith(">"):
            count += 1
    sequence_file.close()
    return count

def get_seq_date(seq:SeqRecord):
    name_string = seq.name
    sequence_date = name_string.split("|")[2]
    return sequence_date

if __name__ == "__main__":
    #fasta_file_path = f"{Path.cwd()}/data/input/spikenuc0312.fasta"
    subset_file_path = f"{Path.cwd()}/data/subsets/"
    #create_subset_fasta_file(import_file_path=fasta_file_path, number_of_sequences=200000, export_file_path=subset_file_path)
    read_fasta_file(import_file_path=f"{subset_file_path}/group_1.fasta")
