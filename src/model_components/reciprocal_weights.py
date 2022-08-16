from typing import List
from numpy import nditer, power, reshape, zeros, newaxis, delete, unique, zeros_like, square
from torch import Tensor, cat, hsplit, stack
from sklearn.utils.class_weight import compute_class_weight
from data_operations.vocabulary import Vocabulary
from settings.constants import USE_CUDA
from visualization.heatmap import generate_heatmap


def compute_2d_weight_vector(targets: List[Tensor], vocabulary: Vocabulary, type: str = None) -> Tensor:
    sequence_length_kmerized = len(targets[0].tolist()) if isinstance(targets[0], Tensor) else len(targets[0])
    number_of_classes = len(vocabulary.itos)
    number_of_datapoints = len(targets)
    freq_array = zeros((number_of_classes, sequence_length_kmerized))
    for target in targets:
        for index, kmer_value in enumerate(target.tolist() if isinstance(target, Tensor) else target):
            freq_array[int(kmer_value), index] += 1
    generate_heatmap(
        freq_array,
        vocabulary.itos,
        list(range(sequence_length_kmerized)),
        "class_frequencies_per_position.png" if type == None else f"class_frequencies_per_position_{type}.png",
    )
    if type == None:
        targets = stack(targets, dim=0)
        position_class_weights = zeros_like(freq_array)
        # position_class_freq_dict = {}
        for position_index in range(sequence_length_kmerized):
            column = targets.numpy()[:, position_index]
            unique_classes = unique(column).tolist()
            class_weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=column).tolist()
            # position_frequency_array = freq_array[:, i]
            # freq_dict = dict.fromkeys(range(number_of_classes), 0)
            # freq_dict.update(zip(freq_dict, position_frequency_array))
            for kmer, weight in zip(unique_classes, class_weights):
                position_class_weights[int(kmer), position_index] = weight
        # position_class_weights[:, 5] = 0
        generate_heatmap(
            position_class_weights,
            vocabulary.itos,
            list(range(sequence_length_kmerized)),
            "class_weights_per_position.png",
            float_flag=True,
        )
        position_class_weights = delete(position_class_weights, 0, 1)
        # weights_array = weights_array[newaxis, ...]

        return position_class_weights
