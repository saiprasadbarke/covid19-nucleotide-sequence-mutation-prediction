# Local
from model_components.decoder import Decoder
from model_components.encoder import Encoder
from model_components.generator import Generator

# External
import torch.nn as nn

from model_components.kmer_embedding import KmerEmbedding

# from torch.nn import Embedding


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: KmerEmbedding,
        trg_embed: KmerEmbedding,
        generator: Generator,
    ):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.generator = generator

    def forward(self, src, trg):
        """Take in and process masked src and target sequences."""
        encoder_hidden, encoder_final = self.encode(src)
        return self.decode(encoder_hidden, encoder_final, trg)

    def encode(self, src):
        src_embedded = self.src_embed(src)
        return self.encoder(src_embedded)

    def decode(self, encoder_hidden, encoder_final, trg, decoder_hidden=None):
        return self.decoder(self.trg_embed(trg), encoder_hidden, encoder_final, hidden=decoder_hidden)
