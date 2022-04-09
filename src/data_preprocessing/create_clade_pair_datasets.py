from json import load

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


def generate_clade_pair_dataset_files(claded_sequences_file: str, output_folder: str):
    cladepair_sequences_dict = {}
    for clade_pair in CLADE_PAIRS:
        sequences_pair_list = []
        for clade, sequences_list in load(open(claded_sequences_file)):
            if clade == clade_pair[0] or clade == clade_pair[1]:
                sequences_pair_list.append(sequences_list)
        cladepair_sequences_dict[f"{clade_pair[0]}_{clade_pair[1]}"] = sequences_pair_list
    return cladepair_sequences_dict


def permute_clade_pairs():
    return None
