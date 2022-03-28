# Standard
from pathlib import Path
# Local
from data_preprocessing.helper import batch_iterator
# External
from Bio import SeqIO

#DATA_FOLDER = Path("")
def read_fasta_file(import_file_path: str):  # provide return types here

    return None


def create_subset_fasta_file(import_file_path: str, export_file_path: str = None):
    record_iter = SeqIO.parse(open(import_file_path), "fasta")
    for i, batch in enumerate(batch_iterator(record_iter, 25000)):
        filename = "group_%i.fasta" % (i + 1)
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


if __name__ == "__main__":
    fasta_file_path = "spikenuc0312.fasta"
    create_subset_fasta_file(import_file_path=fasta_file_path)
