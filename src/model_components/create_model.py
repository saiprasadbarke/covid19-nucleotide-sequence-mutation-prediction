# Local

from model_components.attention import BahdanauAttention
from model_components.generator import Generator
from model_components.decoder import Decoder
from model_components.encoder import Encoder
from model_components.kmer_embedding import KmerEmbedding
from model_components.model import EncoderDecoder
from settings.constants import USE_CUDA


# External
from torch import float32


def create_model(
    vocab_size: int,
    embedding_size: int,
    hidden_size: int,
    encoder_num_layers: int,
    decoder_num_layers: int,
    dropout: float,
    emb_dropout: float,
):

    attention_layer = BahdanauAttention(hidden_size)
    model = EncoderDecoder(
        Encoder(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=encoder_num_layers,
            dropout=dropout,
            emb_dropout=emb_dropout,
        ),
        Decoder(
            emb_size=embedding_size,
            hidden_size=hidden_size,
            attention=attention_layer,
            num_layers=decoder_num_layers,
            dropout=dropout,
            emb_dropout=emb_dropout,
        ),
        KmerEmbedding(vocab_size=vocab_size, emb_size=embedding_size,),
        KmerEmbedding(vocab_size=vocab_size, emb_size=embedding_size,),
        Generator(hidden_size=hidden_size, vocab_size=vocab_size,),
    )

    return model.to("cuda", dtype=float32) if USE_CUDA else model.to("cpu", dtype=float32)
