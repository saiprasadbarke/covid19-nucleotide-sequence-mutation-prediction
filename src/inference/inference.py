from json import dump, load
from pathlib import Path
from statistics import mean
from typing import List
from numpy import size
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_operations.get_dataloaders import get_dataloader
from data_operations.tokenize_data import Tokenize
from data_operations.vocabulary import Vocabulary
from inference.greedy_search import greedy_decode
from metrics.index_diff import index_diff
from visualization.plot_mutations_comparision_graphs import plot_mutations_comparision_graphs
from model_components.create_model import create_model
import matplotlib.pyplot as plt
from model_components.model import EncoderDecoder
from model_training.batch import rebatch
from settings.constants import (
    CURRENT_RUN_DATA_DIR,
    CURRENT_RUN_DIR,
    RUN_NAME,
    SAVED_MODELS_PATH,
    SAVED_PLOTS_PATH,
    SAVED_STATS_PATH,
)
from settings.reference_sequence import REFERENCE_GENOME
from data_operations.generate_datasets import get_string_difference_indices
from visualization.plot_mutation_sites import plot_mutations


def test_model(test_dataloader: DataLoader, model: EncoderDecoder, kmer_size: int):
    data_parameters = load(open(f"{CURRENT_RUN_DIR}/data_parameters.json"))
    vocab = Vocabulary(kmer_size)
    ground_truth_sequences = []
    predicted_sequences = []
    alphas = []
    difference_indices_refgen_pred = {}
    for i, batch in enumerate(test_dataloader, 1):
        print(f"Predicting {i}")
        batch = rebatch(batch)
        ground_truth = batch.trg_input.tolist()[0]
        max_len = len(ground_truth)
        ground_truth_sequences.append(ground_truth)
        pred, attention = greedy_decode(model, batch.src_input, max_len=max_len)
        pred_kmer = [vocab.itos[idx] for idx in pred]
        concat = []
        for index, kmer in enumerate(pred_kmer, 1):
            if index != max_len - 1:
                concat.append(kmer[0])
            else:
                concat.append(kmer)
        concat = "".join(concat)
        difference_refgen_pred = get_string_difference_indices(
            REFERENCE_GENOME[data_parameters["sequence_start_postion"] : data_parameters["sequence_end_postion"]],
            concat,
            data_parameters["sequence_start_postion"],
        )
        for index in difference_refgen_pred:
            if index in difference_indices_refgen_pred.keys():
                difference_indices_refgen_pred[index] += 1
            else:
                difference_indices_refgen_pred[index] = 1
        # predicted_sequences.append(concat)
    difference_indices_file_refgen_pred = f"{SAVED_STATS_PATH}/difference_indices_refgen_pred.json"
    with open(difference_indices_file_refgen_pred, "w") as fout:
        dump(difference_indices_refgen_pred, fout)
    mutations_graph_path_wuhan = f"{SAVED_PLOTS_PATH}/mutation_sites_refgen_pred.png"
    data_dump_path_wuhan = f"{SAVED_STATS_PATH}/sorted_difference_indices_refgen_pred.json"
    plot_mutations(difference_indices_file_refgen_pred, mutations_graph_path_wuhan, data_dump_path_wuhan)


def inference():
    dataset_file_path = f"{CURRENT_RUN_DATA_DIR}/test.json"
    print(f"Path to dataset : {dataset_file_path}")

    training_parameters_performance_dict_path = f"{CURRENT_RUN_DIR}/training_parameters_performance.json"
    training_parameters_performance_dict = load(open(training_parameters_performance_dict_path))
    model = create_model(
        vocab_size=training_parameters_performance_dict["vocab_size"],
        embedding_size=training_parameters_performance_dict["embedding_size"],
        hidden_size=training_parameters_performance_dict["hidden_size"],
        encoder_num_layers=training_parameters_performance_dict["encoder_num_layers"],
        decoder_num_layers=training_parameters_performance_dict["decoder_num_layers"],
        dropout=training_parameters_performance_dict["dropout"],
        emb_dropout=training_parameters_performance_dict["emb_dropout"],
    )
    checkpoint = torch.load(
        f"{SAVED_MODELS_PATH}/save_epoch_{training_parameters_performance_dict['last_best_epoch']}.pt"
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    kmer_size = training_parameters_performance_dict["kmer_size"]

    tokenizer = Tokenize(kmer_length=kmer_size)
    test_inputs, test_targets = tokenizer.kmerize_numericalize_pad_tensorize_sequences(dataset_type="test")
    test_dataloader = get_dataloader(test_inputs, test_targets, batch_size=1)
    test_model(test_dataloader, model, kmer_size)
    # plot_mutations_comparision_graphs()
