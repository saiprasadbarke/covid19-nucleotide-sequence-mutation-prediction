# Vocabulary related

NUCLEOTIDES = ["A", "C", "G", "T"]
KMER_LENGTH = 3
NUM_SPECIAL_CHARS = 2  # 0, 1 and 2 are for BOS , EOS and PAD respectively #TODO: Look at old commit. Changed 3 to 2 as not using PAD anymore
BOS_IDX = 0
EOS_IDX = 1
PAD_IDX = 2

# Data set related parameters
# These variables control the various dataset properties
NUM_SEQ = 30000
LEVENSHTEIN_THRESHOLD = 10
MAX_SEQ_LEN = 3700
START_POSITION = 0
END_POSITION = 3700
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
RNN_INPUT_FEATURE_SIZE = LEN_VOCABULARY = 4**KMER_LENGTH + NUM_SPECIAL_CHARS
RNN_INPUT_SEQUENCE_LENGTH = (MAX_SEQ_LEN - KMER_LENGTH + 1) + 1  # The additional one is for EOS
RNN_HIDDEN_SIZE = 64
RNN_NUM_LAYERS = 1
RNN_DROPOUT = 0

# Training related parameters
N_EPOCHS = 10
MINIBATCH_SIZE = 2
LEARNING_RATE_ALPHA = 0.0003
from torch.cuda import is_available

USE_CUDA = is_available()
