from pathlib import Path
from data_preprocessing.fasta_file_operations import create_complete_fasta_file

print(Path.cwd()) 
input_fasta_file_path = f"{Path.cwd()}/data/input/spikenuc0312.fasta"
completed_sequences_file_path = f"{Path.cwd()}/data/complete_sequences/complete_sequences.fasta"
create_complete_fasta_file(import_file_path=input_fasta_file_path, export_file_path=completed_sequences_file_path)