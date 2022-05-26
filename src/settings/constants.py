# Vocabulary related

NUCLEOTIDES = ["A", "C", "G", "T"]
KMER_LENGTH = 3
NUM_SPECIAL_CHARS = 2  # 0, 1 are for BOS , EOS respectively
BOS_IDX = 0
EOS_IDX = 1

# Data set related parameters
# These variables control the various dataset properties
RANDOM_SEED = 42
NUM_SEQ = 30000
LEVENSHTEIN_THRESHOLD_MIN = 10
LEVENSHTEIN_THRESHOLD_MAX = 25
MAX_SEQ_LENGTH = 3700
START_POSITION = 0
END_POSITION = 500
CLADE_PAIRS = [
    ("19A", "19B"),
    ("19A", "20A"),
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
    ("21L", "22A"),
    ("21L", "22B"),
    ("21L", "22C"),
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
# Train-Val-Test split
TRAIN_REMAINDER_FRACTION = 0.2
VAL_TEST_FRACTION = 0.5

# Model parameters
EMBEDDING_SIZE = 4
LEN_VOCABULARY = 4**KMER_LENGTH + NUM_SPECIAL_CHARS
RNN_HIDDEN_SIZE = 16
RNN_NUM_LAYERS = 1
RNN_DROPOUT = 0

# Training related parameters
N_EPOCHS = 50
MINIBATCH_SIZE = 8
LEARNING_RATE_ALPHA = 1e-3
EARLY_STOPPING_THRESHOLD = 5
from torch.cuda import is_available

USE_CUDA = is_available()


# RUN
RUN_NAME = "RUN_1"
from pathlib import Path

SAVED_MODELS_PATH = f"{Path.cwd()}/saved_models"
