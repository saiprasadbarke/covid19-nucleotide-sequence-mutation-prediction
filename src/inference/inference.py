from pathlib import Path
from typing import List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_operations.train_val_test_split import get_split_data
from inference.greedy_search import greedy_decode
from model_components.create_model import create_model

from model_components.model import EncoderDecoder
from settings.constants import (
    EMBEDDING_SIZE,
    LEN_VOCABULARY,
    RNN_DROPOUT,
    RNN_HIDDEN_SIZE,
    RNN_NUM_LAYERS,
    RUN_NAME,
    SAVED_MODELS_PATH,
)


def generate_sequence(model: nn.Module, src, search_type: str = "greedy"):
    model.eval()
    prediction = greedy_decode(model, src).flatten().tolist()

    return prediction


def test_model(test_dataloader: DataLoader, model: EncoderDecoder, search_type: str = "greedy"):
    ground_truth_sequences = []
    predicted_sequences = []

    for batch in test_dataloader:
        ground_truth_sequences.append(batch[1][:, 1:].tolist())
        generated_sequence = generate_sequence(model, batch[0])
        predicted_sequences.append(generated_sequence)
    return ground_truth_sequences, predicted_sequences


def inference():
    dataset_file_path = f"{Path.cwd()}/data/21A_21J.json"
    print(f"Path to dataset : {dataset_file_path}")
    _, _, test_dataloader = get_split_data(dataset_file_path)
    model = create_model(
        vocab_size=LEN_VOCABULARY,
        embedding_size=EMBEDDING_SIZE,
        hidden_size=RNN_HIDDEN_SIZE,
        num_layers=RNN_NUM_LAYERS,
        dropout=RNN_DROPOUT,
    )
    checkpoint = torch.load(f"{SAVED_MODELS_PATH}/{RUN_NAME}/MODEL_{35}.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    ground_truth_sequences, predicted_sequences = test_model(test_dataloader, model)
