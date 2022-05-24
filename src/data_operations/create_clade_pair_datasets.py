# Standard
from json import load, dump
from pathlib import Path
from itertools import product

# External
from Levenshtein import distance

from Bio import SeqIO

# These variables control the various dataset properties
NUM_SEQ = 30000
LEVENSHTEIN_THRESHOLD = 10
MAX_SEQ_LENGTH = 3700
START_POSITION = 0
END_POSITION = 3700
CLADE_PAIRS = [
    # ("19A", "19B"),
    # ("19A", "20A"),
    # ("20A", "20B"),
    # ("20A", "20C"),
    # ("20A", "20E"),
    # ("20A", "21A"),
    # ("20A", "21B"),
    # ("20A", "21D"),
    # ("20A", "21H"),
    # ("20B", "20D"),
    # ("20B", "20F"),
    # ("20B", "20I"),
    # ("20B", "20J"),
    # ("20B", "21E"),
    # ("20B", "21M"),
    # ("20C", "20G"),
    # ("20C", "20H"),
    # ("20C", "21C"),
    # ("20C", "21F"),
    # ("21A", "21I"),
    ("21A", "21J"),
    # ("20D", "21G"),
    # ("21M", "21K"),
    # ("21M", "21L"),
    # ("21L", "22A"),
    # ("21L", "22B"),
    # ("21L", "22C"),
]

LIST_OF_CLADES = [
    "19A",
    "19B",
    "20A",
    "20B",
    "20C",
    "20E",
    "21A",
    "21B",
    "21D",
    "21H",
    "20D",
    "20F",
    "20I",
    "20J",
    "21M",
    "21E",
    "20G",
    "20H",
    "21C",
    "21F",
    "21I",
    "21J",
    "21G",
    "21K",
    "21L",
    "22A",
    "22B",
    "22C",
]


def create_merged_data(sequences_file: str, clades_file: str, sequence_clade_merged_file: str):
    description_sequence_dict = {}
    for sequence_record in SeqIO.parse(sequences_file, "fasta"):
        description_sequence_dict[sequence_record.description] = str(sequence_record.seq)

    description_clade_dict = {}
    for line in open(clades_file):
        line_data = line.split("\t")
        # The strings have an additional pair of quotes which needs to be removed for proper comparision.
        # The split and indexing operation on the right hand side splits the full clade name (Eg: 21I (Delta)) by space and selects only the alphanumeric clade name
        if line_data[4] == "good":  # Index is 4 for the clades.tabular file
            description_clade_dict[line_data[0].replace('"', "")] = line_data[1].replace('"', "").split(" ")[0]

    ds = [description_sequence_dict, description_clade_dict]
    merged_dict = {}
    for description in description_sequence_dict.keys():
        if description in description_clade_dict.keys():
            merged_dict[description] = tuple(d[description] for d in ds)

    clade_sequence_merged_dict = {}
    for valid_clade in LIST_OF_CLADES:
        single_clade_sequences_dict = {}
        for _seq_description, sequence_clade_tuple in merged_dict.items():
            # The sequence clade tuple has the sequence in index 0 and the clade in index 1
            if sequence_clade_tuple[1] == valid_clade:
                # Adding the sequences as keys in a dictionary ensures there is no duplication
                single_clade_sequences_dict[sequence_clade_tuple[0]] = ""
        if single_clade_sequences_dict:
            clade_sequence_merged_dict[valid_clade] = single_clade_sequences_dict

    for clade, sequences in clade_sequence_merged_dict.items():
        print(f"Number of sequences for clade {clade} after removing duplicates and filtering by qc = {len(sequences)}")
    with open(sequence_clade_merged_file, "w") as fout:
        dump(clade_sequence_merged_dict, fout)


def generate_clade_pair_files(sequence_clade_merged_file: str, paired_clades_file: str):
    cladepair_sequences_dict = {}
    data = load(open(sequence_clade_merged_file))
    for clade_pair in CLADE_PAIRS:
        # Check if both clades exist in the dictionary before pairing them.
        if clade_pair[0] in list(data.keys()) and clade_pair[1] in list(data.keys()):
            sequences_pair_dict = {}
            for clade, sequences_dict_list in data.items():
                if clade == clade_pair[0] or clade == clade_pair[1]:
                    sequences_pair_dict[clade] = list(sequences_dict_list.keys())
            cladepair_sequences_dict[f"{clade_pair[0]}_{clade_pair[1]}"] = sequences_pair_dict
    with open(paired_clades_file, "w") as fout:
        dump(cladepair_sequences_dict, fout)


def permute_clade_pairs(paired_clades_file: str, permuted_output_folder: str):
    data = load(open(paired_clades_file))
    for clade_pair, clades_lists_dict in data.items():
        if (clade_pair.split("_")[0], clade_pair.split("_")[1]) in CLADE_PAIRS:
            clade1_sequences = list(clades_lists_dict.values())[0]
            clade2_sequences = list(clades_lists_dict.values())[1]
            num_seq = 0
            cartesian_product_iterator = product(clade1_sequences, clade2_sequences)
            with open(f"{permuted_output_folder}/{clade_pair}.csv", "w") as fout:

                for current_clade_pair in cartesian_product_iterator:
                    if num_seq == NUM_SEQ:
                        break
                    elif is_valid_sequence_pair(current_clade_pair[0], current_clade_pair[1]):
                        fout.write(
                            f"{current_clade_pair[0][START_POSITION:END_POSITION]},{current_clade_pair[1][START_POSITION:END_POSITION]}"
                        )
                        fout.write("\n")
                        num_seq += 1
            print(f"Wrote {num_seq} clade pairs for {(clade_pair.split('_')[0], clade_pair.split('_')[1])}")


def is_valid_sequence_pair(seq1: str, seq2: str) -> bool:
    if len(seq1) < MAX_SEQ_LENGTH or len(seq2) < MAX_SEQ_LENGTH or distance(seq1, seq2) > LEVENSHTEIN_THRESHOLD:
        return False
    else:
        return True


if __name__ == "__main__":

    # clades_file = f"{Path.cwd().parents[0]}/data/clades.tabular"
    # sequences_file = f"{Path.cwd().parents[0]}/data/sequences.fasta"
    # sequence_clade_merged_file = f"{Path.cwd().parents[0]}/data/merged.json"
    paired_clades_file = f"{Path.cwd().parents[0]}/data/paired.json"
    permuted_sequences_folder = f"{Path.cwd().parents[0]}/data/permuted"

    # create_merged_data(
    #    clades_file=clades_file, sequences_file=sequences_file, sequence_clade_merged_file=sequence_clade_merged_file
    # )
    # generate_clade_pair_files(
    #    sequence_clade_merged_file=sequence_clade_merged_file, paired_clades_file=paired_clades_file
    # )
    # print("Clade pair file generated...Permuting pairs now...")
    permute_clade_pairs(paired_clades_file=paired_clades_file, permuted_output_folder=permuted_sequences_folder)
    print("Completed data pre processing...")
