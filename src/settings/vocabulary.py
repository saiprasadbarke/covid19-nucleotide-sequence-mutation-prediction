# Standard
from itertools import combinations_with_replacement, permutations
from typing import Dict

# Local
from settings.constants import NUCLEOTIDES, num_special_characters


def generate_vocabulary(kmer_length: int) -> Dict[str, int]:
    # The number of possible k-combinations of these nucleotides taken with repetition is 4^kmer_length
    vocabulary_list = [list(x) for x in combinations_with_replacement(NUCLEOTIDES, kmer_length)]
    permuted_vocabulary = []
    for word in vocabulary_list:
        permuted_vocabulary += [list(x) for x in permutations(word, kmer_length)]

    for permuted_word in permuted_vocabulary:
        if permuted_word not in vocabulary_list:
            vocabulary_list.append(permuted_word)

    vocabulary_dict = {}
    for index, kmer in enumerate(vocabulary_list, num_special_characters):
        vocabulary_dict["".join(kmer)] = index
    return vocabulary_dict


if __name__ == "__main__":
    print(generate_vocabulary(4))
