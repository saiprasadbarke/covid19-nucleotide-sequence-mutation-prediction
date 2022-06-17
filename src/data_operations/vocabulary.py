# Standard
from typing import List, Tuple
from json import dump
from itertools import combinations_with_replacement, permutations
from typing import Dict

# Local
from settings.constants import NUCLEOTIDES  # , BOS_IDX, EOS_IDX, NUM_SPECIAL_CHARS


class Vocabulary:
    """Vocabulary represents mapping between tokens and indices."""

    def __init__(self, kmer_length: int, file_path: str = None) -> None:
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
        self.kmer_length = kmer_length
        self.stoi, self.itos = self.generate_vocabulary()
        assert len(self.stoi) == len(self.itos), "Lengths of stoi and itos not equal"
        if file_path != None:
            self.write_vocabulary_to_json(file_path=file_path)

    def generate_vocabulary(self) -> Tuple[Dict[str, int], List[str]]:
        # The number of possible k-combinations of these nucleotides taken with repetition is 4^kmer_length
        vocabulary_list = [list(x) for x in combinations_with_replacement(NUCLEOTIDES, self.kmer_length)]
        permuted_vocabulary = []
        for word in vocabulary_list:
            permuted_vocabulary += [list(x) for x in permutations(word, self.kmer_length)]

        for permuted_word in permuted_vocabulary:
            if permuted_word not in vocabulary_list:
                vocabulary_list.append(permuted_word)

        vocabulary_dict = {}
        for index, kmer in enumerate(vocabulary_list):
            # We start the indexing from NUM_SPECIAL_CHARS as the first NUM_SPECIAL_CHARS indices are for the special characters BOS, EOS
            vocabulary_dict["".join(kmer)] = index
        # vocabulary_dict["<BOS>"] = BOS_IDX
        # vocabulary_dict["<EOS>"] = EOS_IDX
        vocabulary_dict = dict(sorted(vocabulary_dict.items(), key=lambda item: item[1]))
        vocab_list = [s_k for s_k in vocabulary_dict.keys()]
        return vocabulary_dict, vocab_list

    def __str__(self) -> str:
        return self.stoi.__str__()

    def __len__(self) -> int:
        return len(self.stoi)

    def write_vocabulary_to_json(self, file_path: str) -> None:
        with open(file_path, "w") as fout:
            dump(self.stoi, fout)


if __name__ == "__main__":
    print(Vocabulary(3).__str__())
