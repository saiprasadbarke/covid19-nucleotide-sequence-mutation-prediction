# Standard
from pathlib import Path
from re import match

# Local

# External
from Bio import SeqIO


def create_cleaned_fasta_file(import_file_path: str, export_file_path: str):
    fasta_file_with_cleaned_sequences = open(export_file_path, "w")
    number_of_clean_sequences = 0
    for seq_record in SeqIO.parse(import_file_path, "fasta"):
        if check_sequence_completeness(sequence=str(seq_record.seq)):
            number_of_clean_sequences += 1
            fasta_file_with_cleaned_sequences.write(seq_record.format("fasta"))
            if number_of_clean_sequences % 100000 == 0:
                print(f"Wrote {number_of_clean_sequences} clean sequences to file")
    fasta_file_with_cleaned_sequences.close()


def check_sequence_completeness(sequence: str) -> bool:
    if match("^([ACTG])+$", sequence):
        return False
    else:
        return True


if __name__ == "__main__":
    input_fasta_file_path = f"{Path.cwd()}/data/00unfiltered/unfiltered.fasta"
    sequences_file_path = f"{Path.cwd()}/data/01cleaned/sequences.fasta"
    create_cleaned_fasta_file(
        import_file_path=input_fasta_file_path,
        export_file_path=sequences_file_path,
    )
