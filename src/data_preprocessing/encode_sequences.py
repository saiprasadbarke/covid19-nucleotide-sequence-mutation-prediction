from typing import List, Dict
from itertools import combinations_with_replacement, permutations

NUCLEOTIDES = ["A", "C", "G", "T"]


def generate_vocabulary(kmer_length: int) -> Dict[str, int]:
    vocabulary_list = [list(x) for x in combinations_with_replacement(NUCLEOTIDES, kmer_length)]
    permuted_vocabulary = []
    for word in vocabulary_list:
        permuted_vocabulary += [list(x) for x in permutations(word, kmer_length)]

    for permuted_word in permuted_vocabulary:
        if permuted_word not in vocabulary_list:
            vocabulary_list.append(permuted_word)

    vocabulary_dict = {}
    for index, kmer in enumerate(vocabulary_list):
        vocabulary_dict["".join(kmer)] = index
    print(f"For a kmer of length {kmer_length}")
    print(f"The vocabulary is {vocabulary_dict}")
    print(f"Number of words in vocabulary is {len(vocabulary_dict)}")
    return vocabulary_dict


# The number of possible k-combinations of these nucleotides taken with repetition is 4^kmer_length
def encode_sequence(sequence: str, kmer_length: int) -> List[int]:
    vocabulary_encoding = generate_vocabulary(kmer_length=kmer_length)
    kmer_sequence = sliding_window(sequence, kmer_length)
    encoded_kmer_seq = [vocabulary_encoding[kmer] for kmer in kmer_sequence]
    return encoded_kmer_seq


def sliding_window(sequence: str, kmer_length: int) -> List[str]:
    kmerized_sequence = []
    for i in range(len(sequence) - kmer_length + 1):
        kmerized_sequence.append(sequence[i : i + kmer_length])
    return kmerized_sequence


if __name__ == "__main__":
    # generate_vocabulary(5)
    print(encode_sequence("AGCTAT", 3))
