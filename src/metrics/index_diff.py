from typing import List


def index_diff(seq1: List[int], seq2: List[int], max_len: int = 498):

    return len([i for i in range(min(len(seq1), len(seq2))) if seq1[i] != seq2[i]])
