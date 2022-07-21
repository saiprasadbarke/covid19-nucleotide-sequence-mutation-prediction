# Local

from model_components.decoder import Decoder
from model_components.encoder import Encoder
from model_components.generator import Generator
from model_components.kmer_embedding import KmerEmbedding

# External
import torch.nn as nn
from torch import Tensor, randn_like


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

    def forward(self, src: Tensor, trg: Tensor):
        """Take in and process masked src and target sequences."""
        encoder_output, encoder_hidden = self.encode(src)
        noise = randn_like(encoder_output)
        noisy_encoder_output = encoder_output + noise
        return self.decode(noisy_encoder_output, encoder_hidden, trg)

    def encode(self, src: Tensor):
        src_embedded = self.src_embed(src)
        return self.encoder(src_embedded)

    def decode(
        self,
        encoder_output: Tensor,
        encoder_hidden: Tensor,
        trg: Tensor,
        unroll_steps: int = None,
        decoder_hidden: Tensor = None,
    ):
        return self.decoder(
            trg_embed=self.trg_embed(trg),
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            hidden=decoder_hidden,
            unroll_steps=unroll_steps,
        )
