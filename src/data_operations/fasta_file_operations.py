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
        if is_valid_sequence(sequence=str(seq_record.seq)):
            number_of_clean_sequences += 1
            fasta_file_with_cleaned_sequences.write(seq_record.format("fasta"))
            if number_of_clean_sequences % 100000 == 0:
                print(f"Wrote {number_of_clean_sequences} clean sequences to file")
    print(f"Wrote {number_of_clean_sequences} clean sequences to file")
    fasta_file_with_cleaned_sequences.close()


def is_valid_sequence(sequence: str) -> bool:
    return match(r"^([ACTG])+$", sequence)


if __name__ == "__main__":
    input_fasta_file_path = f"{Path.cwd().parents[0]}/data/spikenuc0428.fasta"
    sequences_file_path = f"{Path.cwd().parents[0]}/data/sequences.fasta"
    create_cleaned_fasta_file(
        import_file_path=input_fasta_file_path,
        export_file_path=sequences_file_path,
    )
