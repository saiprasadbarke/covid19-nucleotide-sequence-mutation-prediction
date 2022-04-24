from nltk.translate.bleu_score import sentence_bleu
from typing import List


def calculate_ngram_score(reference: List[List[str]], candidate: List, n_gram: int):
    if n_gram == 1:
        return sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
    elif n_gram == 2:
        return sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
    elif n_gram == 3:
        return sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
    elif n_gram == 4:
        return sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
