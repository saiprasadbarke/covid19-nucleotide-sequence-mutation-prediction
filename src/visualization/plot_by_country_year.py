# Standard
from typing import Dict, List
from pathlib import Path
from os import chdir, listdir

# Local

# External
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import matplotlib.pyplot as plt


LIST_OF_COUNTRIES = [
    "USA",
    "UK",
    "Germany",
    "Denmark",
    "Canada",
    "France",
    "Japan",
    "India",
    "Sweden",
    "Switzerland",
    "Brazil",
    "Spain",
    "Italy",
    "Belgium",
    "Turkey",
    "Austria",
    "Australia",
    "Poland",
    "Ireland",
    "Slovenia",
    "Israel",
    "Mexico",
    "Norway",
    "Lithuania",
    "Portugal",
    "China",
    "Finland",
    "Croatia",
    "Chile",
    "Peru",
    "Argentina",
    "Thailand",
    "Greece",
    "Russia",
    "Colombia",
    "Singapore",
    "Bangladesh",
    "Nigeria",
    "South Africa",
    "Senegal",
]  # Ref: https://www.gisaid.org/submission-tracker-global/


def split_sequences_by_country(import_file_path: str) -> Dict[str, List[SeqRecord]]:
    countrywise_sequences_dictionary = dict.fromkeys(LIST_OF_COUNTRIES)
    for country in countrywise_sequences_dictionary.keys():
        sequences_list = []
        for seq_record in SeqIO.parse(import_file_path, "fasta"):
            if country in seq_record.description or country.lower() in seq_record.description:
                sequences_list.append(seq_record)
        countrywise_sequences_dictionary[country] = sequences_list
    return countrywise_sequences_dictionary


def write_fasta_file_by_country(countrywise_dictionary: Dict[str, List[SeqRecord]], output_path: str):

    for country, sequences in countrywise_dictionary.items():
        fasta_file_country = open(f"{output_path}/{country}", "w")
        for sequence_record in sequences:
            fasta_file_country.write(sequence_record.format("fasta"))
        fasta_file_country.close()


def generate_histogram_by_country(
    output_path: str,
    countrywise_dictionary: Dict[str, List[SeqRecord]] = None,
    from_fasta_file: bool = False,
):
    if not from_fasta_file:
        countries = list(countrywise_dictionary.keys())
        number_of_sequences = [len(x) for x in list(countrywise_dictionary.values())]
    else:
        country_dir_path = f"{Path.cwd()}/data/countrywise_split"
        countries = []
        number_of_sequences = []
        chdir(country_dir_path)
        for file in listdir():
            file_path = f"{country_dir_path}/{file}"
            countries.append(file)
            number_of_sequences.append(read_fasta_file_and_return_length(path=file_path))
    plt.figure(figsize=(10, 10))
    plt.bar(countries, number_of_sequences)
    plt.xlabel("Countries")
    plt.ylabel("# of completed seq")
    plt.xticks(rotation=90)
    plt.title("Countrywise distribution of completed sequences")
    plt.savefig(output_path)
    plt.show()


def read_fasta_file_and_return_length(path: str) -> int:

    return len([1 for line in open(path) if line.startswith(">")])


if __name__ == "__main__":
    input_fasta_file_path = f"{Path.cwd()}/data/01cleaned/sequences.fasta"
    histogram_path = f"{Path.cwd()}/plots/countrywise_wo_year.png"
    countrywise_fasta_path = f"{Path.cwd()}/data/countrywise"
    countrywise_sequences_dictionary = split_sequences_by_country(import_file_path=input_fasta_file_path)
    write_fasta_file_by_country(
        countrywise_dictionary=countrywise_sequences_dictionary,
        output_path=countrywise_fasta_path,
    )
    generate_histogram_by_country(
        countrywise_dictionary=countrywise_sequences_dictionary,
        output_path=histogram_path,
    )
