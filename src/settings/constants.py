NUCLEOTIDES = ["A", "C", "G", "T"]
KMER_LENGTH = 3
num_special_characters = 2  # 0, 1 and 2 are for BOS , EOS and PAD respectively #TODO: Look at old commit. Changed 3 to 2 as not using PAD anymore
BOS_IDX = 0
EOS_IDX = 1
PAD_IDX = 2

train_remaining_fraction = 0.2
validation_test_fraction = 0.5

from torch.cuda import is_available

USE_CUDA = is_available()
