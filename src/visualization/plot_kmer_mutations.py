from typing import List

from torch import Tensor
from data_operations.tokenize_data import Tokenize

from settings.constants import SAVED_PLOTS_PATH
from settings.reference_sequence import REFERENCE_GENOME
from visualization.plot_mutation_sites import get_string_difference_indices


def plot_kmer_position_mutations_graph(
    targets: List[Tensor], data_type: str, kmer_size: int, sequence_start_postion: int, sequence_end_postion: int,
):
    graph_path = f"{SAVED_PLOTS_PATH}/kmer_position_mutations_{data_type}.png"
    tokenizer = Tokenize(kmer_length=kmer_size)
    kmerized_ref_genome = tokenizer.tokenize_encode_sequence(
        REFERENCE_GENOME[sequence_start_postion:sequence_end_postion]
    )
    targets_as_list = [target.tolist() for target in targets]

    for _ in targets_as_list:
        continue
