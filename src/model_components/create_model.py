# Local

from model_components.attention import BahdanauAttention
from model_components.generator import Generator
from model_components.decoder import Decoder
from model_components.encoder import Encoder
from model_components.kmer_embedding import KmerEmbedding
from model_components.model import EncoderDecoder
from settings.constants import (
    EMBEDDING_SIZE,
    LEN_VOCABULARY,
    RNN_DROPOUT,
    RNN_HIDDEN_SIZE,
    RNN_NUM_LAYERS,
    USE_CUDA,
)

# External
from torch import float32


def create_model(
    vocab_size=LEN_VOCABULARY,
    embedding_size=EMBEDDING_SIZE,
    hidden_size=RNN_HIDDEN_SIZE,
    num_layers=RNN_NUM_LAYERS,
    dropout=RNN_DROPOUT,
):

    attention_layer = BahdanauAttention(hidden_size)
    model = EncoderDecoder(
        Encoder(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ),
        Decoder(
            emb_size=embedding_size,
            hidden_size=hidden_size,
            attention=attention_layer,
            num_layers=num_layers,
            dropout=dropout,
        ),
        KmerEmbedding(
            vocab_size=vocab_size,
            emb_size=embedding_size,
            scale_gradient=True,
        ),
        KmerEmbedding(
            vocab_size=vocab_size,
            emb_size=embedding_size,
            scale_gradient=True,
        ),
        Generator(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        ),
    )

    return model.to("cuda", dtype=float32) if USE_CUDA else model.to("cpu", dtype=float32)
