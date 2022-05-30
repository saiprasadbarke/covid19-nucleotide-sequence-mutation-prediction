from pathlib import Path
from statistics import mean
from typing import List
from numpy import size
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_operations.get_dataloaders import get_dataloader
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


def test_model(test_dataloader: DataLoader, model: EncoderDecoder):
    ground_truth_sequences = []
    predicted_sequences = []
    alphas = []
    index_mismatch = []
    for batch in test_dataloader:
        batch = rebatch(batch)
        ground_truth = batch.trg_y[:, :-1].tolist()[0]
        ground_truth_sequences.append(ground_truth)
        pred, attention = greedy_decode(model, batch.src_input, max_len=499)
        predicted_sequences.append(pred)
        alphas.append(attention)
        index_mismatch.append(index_diff(ground_truth, pred))
    average_mismatch = mean(index_mismatch)
    min_mismatch = min(index_mismatch)
    max_mismatch = max(index_mismatch)
    return ground_truth_sequences, predicted_sequences, alphas, index_mismatch


def plot_mismatch(mismatch_list: int, graph_path: str):
    plt.figure(size=(20, 20))
    plt.hist(mismatch_list, bins=500)
    plt.xlabel("Number of mismatches in test set")
    plt.ylabel("Frequency")
    plt.savefig(graph_path)
    plt.show()


def inference():
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
    ground_truth_sequences, predicted_sequences, alphas, mismatch_list = test_model(test_dataloader, model)
    graph_path = f"{Path.cwd()}/reports/plots/index_mismatch_post_prediction.png"
    plot_mismatch(mismatch_list, graph_path)
