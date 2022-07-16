from typing import List
from numpy import nditer, reshape, zeros, newaxis, delete
from torch import Tensor

from data_operations.vocabulary import Vocabulary
from settings.constants import USE_CUDA
from visualization.heatmap import generate_heatmap


def compute_2d_weight_vector(targets: List[Tensor], vocabulary: Vocabulary, type: str = None) -> Tensor:
    sequence_length_kmerized = len(targets[0].tolist()) if isinstance(targets[0], Tensor) else len(targets[0])
    number_of_classes = len(vocabulary.itos)
    number_of_datapoints = len(targets)
    weights_array = zeros((number_of_classes, sequence_length_kmerized))
    print()
    for target in targets:
        for index, kmer_value in enumerate(target.tolist() if isinstance(target, Tensor) else target):
            weights_array[int(kmer_value), index] += 1
    generate_heatmap(
        weights_array,
        vocabulary.itos,
        list(range(sequence_length_kmerized)),
        "class_frequencies_per_position.png" if type == None else f"class_frequencies_per_position_{type}.png",
    )
    if type == None:
        with nditer(weights_array, op_flags=["readwrite"]) as iterator:
            for element in iterator:
                item = element.item()
                if item == 0:
                    element[...] = 0
                elif item == number_of_datapoints:
                    element[...] = 1
                else:
                    slope = 1 / number_of_datapoints
                    element[...] = -slope * item + 1
        generate_heatmap(
            weights_array, vocabulary.itos, list(range(sequence_length_kmerized)), "class_weights_per_position.png", float_flag=True
        )
        weights_array = delete(weights_array, 0, 1)
        # weights_array = weights_array[newaxis, ...]

        return weights_array  # .cuda() if USE_CUDA else Tensor(weights_array)
