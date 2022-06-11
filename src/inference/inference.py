from json import dump
from pathlib import Path
from statistics import mean
from typing import List
from numpy import size
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_operations.get_dataloaders import get_dataloader
from data_operations.vocabulary import Vocabulary
from inference.greedy_search import greedy_decode
from metrics.index_diff import index_diff
from model_components.create_model import create_model
import matplotlib.pyplot as plt
from model_components.model import EncoderDecoder
from model_training.batch import rebatch
from settings.constants import (
    EMBEDDING_SIZE,
    LEN_VOCABULARY,
    RNN_DROPOUT,
    RNN_HIDDEN_SIZE,
    RNN_NUM_LAYERS,
    RUN_NAME,
    SAVED_MODELS_PATH,
)
from settings.reference_sequence import WUHAN_REF
from data_operations.search_space_minimization import get_string_difference_indices
from visualization.plot_mutation_sites import plot_mutations


def test_model(test_dataloader: DataLoader, model: EncoderDecoder):
    vocab = Vocabulary(3)
    ground_truth_sequences = []
    predicted_sequences = []
    alphas = []
    difference_indices_wuhan = {}
    for batch in test_dataloader:
        batch = rebatch(batch)
        ground_truth = batch.trg_y[:, :-1].tolist()[0]
        ground_truth_sequences.append(ground_truth)
        pred, attention = greedy_decode(model, batch.src_input, max_len=499)
        lol = [vocab.itos[idx] for idx in pred]
        concat = []
        for index, kmer in enumerate(lol, 1):
            if index != 498:
                concat.append(kmer[0])
            else:
                concat.append(kmer)
        concat = "".join(concat)
        difference_wuhan = get_string_difference_indices(WUHAN_REF[0:499], concat)
        for index in difference_wuhan:
            if index in difference_indices_wuhan.keys():
                difference_indices_wuhan[index] += 1
            else:
                difference_indices_wuhan[index] = 1
        predicted_sequences.append(concat)
    difference_indices_file_wuhan = f"{SAVED_MODELS_PATH}/{RUN_NAME}/difference_indices_wuhan.json"
    with open(difference_indices_file_wuhan, "w") as fout:
        dump(difference_indices_wuhan, fout)
    mutations_graph_path_wuhan = f"{SAVED_MODELS_PATH}/{RUN_NAME}/mutation_sites_wuhan.png"
    data_dump_path_wuhan = f"{SAVED_MODELS_PATH}/{RUN_NAME}/sorted_difference_indices_wuhan.json"
    plot_mutations(difference_indices_file_wuhan, mutations_graph_path_wuhan, data_dump_path_wuhan)
    return predicted_sequences


def plot_mismatch(mismatch_list: int, graph_path: str):
    plt.figure(size=(20, 20))
    plt.hist(mismatch_list, bins=500)
    plt.xlabel("Number of mismatches in test set")
    plt.ylabel("Frequency")
    plt.savefig(graph_path)
    plt.show()


def inference():
    # TODO: Get model parameters from saved json
    # streamline and add other necessary metrics
    dataset_file_path = f"{Path.cwd()}/data/21A_21J.json"
    print(f"Path to dataset : {dataset_file_path}")
    _, _, test_dataloader = get_dataloader(dataset_file_path)
    model = create_model(
        vocab_size=LEN_VOCABULARY,
        embedding_size=EMBEDDING_SIZE,
        hidden_size=RNN_HIDDEN_SIZE,
        num_layers=RNN_NUM_LAYERS,
        dropout=RNN_DROPOUT,
    )
    checkpoint = torch.load(f"{SAVED_MODELS_PATH}/{RUN_NAME}/MODEL_{35}.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    losses = checkpoint["returned_metrics"]
    test_model(test_dataloader, model)

