from typing import Dict, List
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from pathlib import Path
from globals.constants import list_of_countries
import matplotlib.pyplot as plt


def get_list_of_countries(import_file_path: str) -> List[str]:
    list_of_countries = []
    for seq_record in SeqIO.parse(import_file_path, "fasta"):
        country = get_country(seq_record)
        if country in list_of_countries:
            continue
        else:
            list_of_countries.append(country)
    return list_of_countries


def get_country(seq_record_obj: SeqRecord):

    return seq_record_obj.description.split("|")[-1]


def split_sequences_by_country(import_file_path: str) -> Dict[str, List[SeqRecord]]:
    # list_of_countries = get_list_of_countries(import_file_path=import_file_path)
    countrywise_sequences_dictionary = dict.fromkeys(list_of_countries)
    for country in countrywise_sequences_dictionary.keys():
        sequences_list = []
        for seq_record in SeqIO.parse(import_file_path, "fasta"):
            if (
                country in seq_record.description
                or country.lower() in seq_record.description
            ):
                sequences_list.append(seq_record)
        countrywise_sequences_dictionary[country] = sequences_list
    return countrywise_sequences_dictionary


def write_fasta_file_by_country(
    countrywise_dictionary: Dict[str, List[SeqRecord]], output_path: str
):

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
    countries = list(countrywise_dictionary.keys())
    number_of_sequences = [len(x) for x in list(countrywise_dictionary.values())]
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlabel("Countries")
    ax.set_ylabel("# of completed seq")
    ax.set_title("Countrywise distribution of completed sequences")
    ax.bar(countries, number_of_sequences)
    plt.show()
    plt.savefig(output_path)


if __name__ == "__main__":
    input_fasta_file_path = (
        f"{Path.cwd()}/data/complete_sequences/complete_sequences.fasta"
    )
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
        countrywise_dictionary=countrywise_sequences_dictionary,
        output_path=histogram_path,
    )
