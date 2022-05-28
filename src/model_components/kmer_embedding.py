import torch.nn as nn
from torch import Tensor
import math


class KmerEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size, scale_gradient: bool = False):
        super(KmerEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size, scale_grad_by_freq= scale_gradient)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
