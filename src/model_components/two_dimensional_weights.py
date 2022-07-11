from typing import List
from numpy import nditer, reshape, zeros, newaxis, delete

from torch import Tensor

from data_operations.vocabulary import Vocabulary
from settings.constants import USE_CUDA


def compute_2d_weight_vector(targets: List[Tensor], vocabulary: Vocabulary) -> Tensor:
    sequence_length_kmerized = targets[0].size(0)
    number_of_classes = len(vocabulary.itos)
    number_of_datapoints = len(targets)
    weights_array = zeros((number_of_classes, sequence_length_kmerized))
    print()
    for target in targets:
        for index, kmer_value in enumerate(target.tolist()):
            weights_array[int(kmer_value), index] += 1
    with nditer(weights_array, op_flags=["readwrite"]) as iterator:
        for element in iterator:
            item = element.item()
            if item == 0:
                element[...] = 0
            elif item == number_of_datapoints:
                element[...] = number_of_datapoints
            else:
                element[...] = number_of_datapoints / item
    weights_array = delete(weights_array, 0, 1)
    weights_array = weights_array[newaxis, ...]

    return Tensor(weights_array).cuda() if USE_CUDA else Tensor(weights_array)
