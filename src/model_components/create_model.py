# Local
from model_components.attention import BahdanauAttention
from model_components.generator import Generator
from model_components.decoder import Decoder
from model_components.encoder import Encoder
from model_components.model import EncoderDecoder
from settings.constants import USE_CUDA

# External
from torch.nn import Embedding


def create_model(
    src_vocab,
    tgt_vocab,
    emb_size=256,
    hidden_size=512,
    num_layers=1,
    dropout=0,
):

    attention = BahdanauAttention(hidden_size)

    model = EncoderDecoder(
        Encoder(emb_size, hidden_size, num_layers=num_layers, dropout=dropout),
        Decoder(emb_size, hidden_size, attention, num_layers=num_layers, dropout=dropout),
        # Embedding(src_vocab, emb_size),
        # Embedding(tgt_vocab, emb_size),
        Generator(hidden_size, tgt_vocab),
    )

    return model.cuda() if USE_CUDA else model
