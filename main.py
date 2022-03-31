from pathlib import Path
from data_preprocessing.fasta_file_operations import create_complete_fasta_file
from visualization.plot_by_country_year import (
    split_sequences_by_country,
    write_fasta_file_by_country,
    generate_histogram_by_country,
)

print(Path.cwd())
# input_fasta_file_path = f"{Path.cwd()}/data/input/spikenuc0312.fasta"
# completed_sequences_file_path = f"{Path.cwd()}/data/complete_sequences/complete_sequences.fasta"
# create_complete_fasta_file(import_file_path=input_fasta_file_path, export_file_path=completed_sequences_file_path)

input_fasta_file_path = f"{Path.cwd()}/data/complete_sequences/complete_sequences.fasta"
histogram_path = f"{Path.cwd()}/plots/countrywise_wo_year.png"
countrywise_fasta_path = f"{Path.cwd()}/data/countrywise_split"
countrywise_sequences_dictionary = split_sequences_by_country(
    import_file_path=input_fasta_file_path
)
write_fasta_file_by_country(
    countrywise_dictionary=countrywise_sequences_dictionary,
    output_path=countrywise_fasta_path,
)
generate_histogram_by_country(
    countrywise_dictionary=countrywise_sequences_dictionary, output_path=histogram_path
)
