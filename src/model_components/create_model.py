# Local

from model_components.attention import BahdanauAttention
from model_components.generator import Generator
from model_components.decoder import Decoder
from model_components.encoder import Encoder
from model_components.model import EncoderDecoder
from settings.constants import RNN_DROPOUT, RNN_HIDDEN_SIZE, RNN_INPUT_FEATURE_SIZE, RNN_NUM_LAYERS, USE_CUDA

# External
from torch.nn import Embedding
import torch


def create_model(
    src_vocab,
    tgt_vocab,
    input_size=RNN_INPUT_FEATURE_SIZE,
    hidden_size=RNN_HIDDEN_SIZE,
    num_layers=RNN_NUM_LAYERS,
    dropout=RNN_DROPOUT,
):

    attention = BahdanauAttention(hidden_size)

    model = EncoderDecoder(
        Encoder(input_size, hidden_size, num_layers=num_layers, dropout=dropout),
        Decoder(input_size, hidden_size, attention, num_layers=num_layers, dropout=dropout),
        # Embedding(src_vocab, emb_size),
        # Embedding(tgt_vocab, emb_size),
        Generator(hidden_size, tgt_vocab),
    )

    return model.to("cuda", dtype=torch.float32) if USE_CUDA else model.to("cpu", dtype=torch.float32)
