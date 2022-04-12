from json import load, dump
from pathlib import Path

CLADE_PAIRS = [
    ("19A", "19B"),
    ("19A", "20A"),
    ("20A", "20B"),
    ("20A", "20B"),
    ("20A", "20C"),
    ("20A", "20E"),
    ("20A", "21A"),
    ("20A", "21B"),
    ("20A", "21D"),
    ("20A", "21H"),
    ("20B", "20D"),
    ("20B", "20F"),
    ("20B", "20I"),
    ("20B", "20J"),
    ("20B", "21E"),
    ("20B", "21M"),
    ("20C", "20G"),
    ("20C", "20H"),
    ("20C", "21C"),
    ("20C", "21F"),
    ("21A", "21I"),
    ("21A", "21J"),
    ("20D", "21G"),
    ("21M", "21K"),
    ("21M", "21L"),
]


def generate_clade_pair_dataset_files(claded_sequences_file: str, output_file: str):
    cladepair_sequences_dict = {}
    data = load(open(claded_sequences_file))
    for clade_pair in CLADE_PAIRS:
        # Check if both clades exist in the dictionary before pairing them.
        if clade_pair[0] in list(data.keys()) and clade_pair[1] in list(data.keys()):
            sequences_pair_dict = {}
            for clade, sequences_dict_list in data.items():
                if clade == clade_pair[0] or clade == clade_pair[1]:
                    sequences_pair_dict[clade] = list(sequences_dict_list.keys())
            cladepair_sequences_dict[f"{clade_pair[0]}_{clade_pair[1]}"] = sequences_pair_dict
    with open(output_file, "w") as fout:
        dump(cladepair_sequences_dict, fout)


def permute_clade_pairs(cladepair_sequences_file: str, output_folder: str):
    data = load(open(cladepair_sequences_file))
    return None


if __name__ == "__main__":
    # claded_sequences_filepath = f"{Path.cwd().parents[0]}/data/clade_seq.json"
    # paired_clades_path = f"{Path.cwd().parents[0]}/data/paired_clades.json"
    # permuted_sequences_folder = f"{Path.cwd()}/data/permuted/india/"

    claded_sequences_filepath = f"{Path.cwd()}/data/02merged/india.json"
    paired_clades_path = f"{Path.cwd()}/data/03paired/paired_clades_india.json"
    permuted_sequences_folder = f"{Path.cwd()}/data/04permuted/india"
    generate_clade_pair_dataset_files(claded_sequences_file=claded_sequences_filepath, output_file=paired_clades_path)
    permute_clade_pairs(cladepair_sequences_file=paired_clades_path, output_folder=permuted_sequences_folder)
