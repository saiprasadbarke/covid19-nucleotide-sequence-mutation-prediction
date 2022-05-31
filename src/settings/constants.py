from pathlib import Path
from torch.cuda import is_available

#################### Vocabulary

NUCLEOTIDES = ["A", "C", "G", "T"]
NUM_SPECIAL_CHARS = 2  # 0, 1 are for BOS , EOS respectively
BOS_IDX = 0
EOS_IDX = 1

################### Dataset
RANDOM_SEED = 42
CLADE_PAIRS_NEXTCLADE = {
    1: ("19A", "19B"),
    2: ("19A", "20A"),
    3: ("20A", "20B"),
    4: ("20A", "20C"),
    5: ("20A", "20E"),
    6: ("20A", "21A"),
    7: ("20A", "21B"),
    8: ("20A", "21D"),
    9: ("20A", "21H"),
    10: ("20B", "20D"),
    11: ("20B", "20F"),
    12: ("20B", "20I"),
    13: ("20B", "20J"),
    14: ("20B", "21E"),
    # 15: ("20B", "21M"),
    16: ("20C", "20G"),
    17: ("20C", "20H"),
    18: ("20C", "21C"),
    19: ("20C", "21F"),
    20: ("21A", "21I"),
    21: ("21A", "21J"),
    22: ("20D", "21G"),
    # 23: ("21M", "21K"),
    # 24: ("21M", "21L"),
    # 25: ("21L", "22A"),
    # 26: ("21L", "22B"),
    # 27: ("21L", "22C"),
}


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
ROOT_DATA_DIR = f"{Path.cwd().parents[0]}/data"
MERGED_DATA = f"{ROOT_DATA_DIR}/merged.json"


################## RUN
RUN_NAME = "RUN_2"
ROOT_RUN_DIR = f"{Path.cwd().parents[0]}/runs"
CURRENT_RUN_DIR = f"{ROOT_RUN_DIR}/{RUN_NAME}"
CURRENT_RUN_DATA_DIR = f"{CURRENT_RUN_DIR}/data"

################### Training parameters

EARLY_STOPPING_THRESHOLD = 5
USE_CUDA = is_available()
SAVED_MODELS_PATH = f"{CURRENT_RUN_DIR}/saved_models"
SAVED_PLOTS_PATH = f"{CURRENT_RUN_DIR}/reports/plots"
SAVED_STATS_PATH = f"{CURRENT_RUN_DIR}/reports/stats"

# Entry point
ROUTINES_TO_EXECUTE = {
    1: "Create Datasets",
    2: "Train network",
    3: "Run Inference",
}
