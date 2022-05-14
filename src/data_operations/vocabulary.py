# Standard
from json import dump
from collections import defaultdict
from itertools import combinations_with_replacement, permutations
from typing import Dict, List

# Local
from settings.constants import NUCLEOTIDES, num_special_characters


class Vocabulary:
    """Vocabulary represents mapping between tokens and indices."""

    def __init__(self, kmer_length: int) -> None:
        """
        Create vocabulary based on the size of the kmer.

        The number of possible k-combinations of [A, C, G, T] taken with repetition is 4^kmer_length
        eg: If kmer_length = 1
            vocabulary = [A, C, G, T]

            If kmer_length = 2
            vocabulary = [AA, AC, AG, AT, CA, CC, CG, CT, GA, GC, GG, GT, TA, TC, TG, TT]
            .
            .
            .
            and so on...
        """
        self.stoi = self._generate_vocabulary(kmer_length=kmer_length)
        self.itos = [s_v for s_v in self.stoi.values()]

    def _generate_vocabulary(self, kmer_length: int) -> Dict[str, int]:
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

    def __str__(self) -> str:
        return self.stoi.__str__()

    def __len__(self) -> int:
        return len(self.itos)

    def write_vocabulary_to_json(self, file_path: str) -> None:
        with open(file_path, "w") as fout:
            dump(self.stoi, fout)


if __name__ == "__main__":
    print(Vocabulary(3).__str__())
