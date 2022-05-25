# Local

from model_components.attention import BahdanauAttention
from model_components.generator import Generator
from model_components.decoder import Decoder
from model_components.encoder import Encoder
from model_components.model import EncoderDecoder
from settings.constants import (
    LEN_VOCABULARY,
    RNN_DROPOUT,
    RNN_HIDDEN_SIZE,
    RNN_INPUT_FEATURE_SIZE,
    RNN_NUM_LAYERS,
    USE_CUDA,
)

# External
from torch.nn import Embedding
import torch


def create_model(
    vocab_size=LEN_VOCABULARY,
    embedding_size=RNN_INPUT_FEATURE_SIZE,
    hidden_size=RNN_HIDDEN_SIZE,
    num_layers=RNN_NUM_LAYERS,
    dropout=RNN_DROPOUT,
):

    model = EncoderDecoder(
        Encoder(embedding_size, hidden_size, num_layers=num_layers, dropout=dropout),
        Decoder(embedding_size, hidden_size, BahdanauAttention(hidden_size), num_layers=num_layers, dropout=dropout),
        Embedding(vocab_size, embedding_size),
        Embedding(vocab_size, embedding_size),
        Generator(hidden_size, vocab_size),
    )

    return model.to("cuda", dtype=torch.float32) if USE_CUDA else model.to("cpu", dtype=torch.float32)
