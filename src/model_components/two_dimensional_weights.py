from typing import List
from numpy import nditer, reshape, zeros, newaxis, delete
from torch import Tensor

from data_operations.vocabulary import Vocabulary
from settings.constants import USE_CUDA
from visualization.heatmap import generate_heatmap


def compute_2d_weight_vector(targets: List[Tensor], vocabulary: Vocabulary) -> Tensor:
    sequence_length_kmerized = targets[0].size(0)
    number_of_classes = len(vocabulary.itos)
    number_of_datapoints = len(targets)
    weights_array = zeros((number_of_classes, sequence_length_kmerized))
    print()
    for target in targets:
        for index, kmer_value in enumerate(target.tolist()):
            weights_array[int(kmer_value), index] += 1
    generate_heatmap(
        weights_array, vocabulary.itos, list(range(sequence_length_kmerized)), "class_frequencies_per_position.png"
    )
    with nditer(weights_array, op_flags=["readwrite"]) as iterator:
        for element in iterator:
            item = element.item()
            if item == 0:
                element[...] = 0
            else:
                effective_num = 1.0 - pow(0.999, item)
                element[...] = (1.0 - 0.99) / effective_num
    generate_heatmap(
        weights_array, vocabulary.itos, list(range(sequence_length_kmerized)), "class_weights_per_position.png"
    )
    weights_array = delete(weights_array, 0, 1)
    # weights_array = weights_array[newaxis, ...]

    return weights_array  # .cuda() if USE_CUDA else Tensor(weights_array)
