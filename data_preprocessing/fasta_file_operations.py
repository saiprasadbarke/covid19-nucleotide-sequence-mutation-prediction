# Standard
from pathlib import Path
# Local
from data_preprocessing.helper import batch_iterator
from data_preprocessing.data_cleaning import check_sequence_completeness
# External
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

def create_complete_fasta_file(import_file_path: str, export_file_path: str):
    fasta_file_with_completed_sequences = open(export_file_path, "w")
    number_of_complete_sequences = 0
    for seq_record in SeqIO.parse(import_file_path, "fasta"):
        if check_sequence_completeness(sequence_record_object=seq_record):
            number_of_complete_sequences+=1
            fasta_file_with_completed_sequences.write(seq_record.format("fasta"))
        if number_of_complete_sequences%1000==0:
            print(f"Wrote {number_of_complete_sequences} to file")
    fasta_file_with_completed_sequences.close()
     

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
    name_string = seq.description
    sequence_date = name_string.split("|")[2]
    return sequence_date

if __name__ == "__main__":
    input_fasta_file_path = f"{Path.cwd()}/data/input/spikenuc0312.fasta"
    completed_sequences_file_path = f"{Path.cwd()}/data/complete_sequences/complete_sequences.fasta"
    create_complete_fasta_file(import_file_path=input_fasta_file_path, export_file_path=completed_sequences_file_path)
