# Standard
from pathlib import Path

# Local

# External
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def create_complete_fasta_file(import_file_path: str, export_file_path: str):
    fasta_file_with_completed_sequences = open(export_file_path, "w")
    number_of_complete_sequences = 0
    for seq_record in SeqIO.parse(import_file_path, "fasta"):
        if check_sequence_completeness(sequence_record_object=seq_record):
            number_of_complete_sequences += 1
            fasta_file_with_completed_sequences.write(seq_record.format("fasta"))
            if number_of_complete_sequences % 100000 == 0:
                print(f"Wrote {number_of_complete_sequences} to file")
    fasta_file_with_completed_sequences.close()


def check_sequence_completeness(sequence_record_object: SeqRecord) -> bool:
    if len(sequence_record_object) != 3822:
        return False
    elif "N" in sequence_record_object:
        return False
    else:
        return True


if __name__ == "__main__":
    input_fasta_file_path = f"{Path.cwd()}/data/input/spikenuc0312.fasta"
    completed_sequences_file_path = (
        f"{Path.cwd()}/data/complete_sequences/complete_sequences.fasta"
    )
    create_complete_fasta_file(
        import_file_path=input_fasta_file_path,
        export_file_path=completed_sequences_file_path,
    )
