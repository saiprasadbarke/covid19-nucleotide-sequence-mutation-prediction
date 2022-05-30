# Local
from settings.constants import EMBEDDING_SIZE

# External
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Encodes a sequence of word embeddings"""

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)

    def forward(self, x):
        """
        Applies a bidirectional GRU to sequence of embeddings x.
        The input mini-batch x needs to be sorted by length.
        x should have dimensions [batch, time, dim].
        """
        # x = torch.Size([8, 3699, 66]), output = torch.Size([8, 3699, 1024]), final = torch.Size([2, 8, 512])
        output, final = self.rnn(x)
        # we need to manually concatenate the final states for both directions
        fwd_final = final[0 : final.size(0) : 2]  # torch.Size([1, 8, 512])
        bwd_final = final[1 : final.size(0) : 2]  # torch.Size([1, 8, 512])
        final = torch.cat([fwd_final, bwd_final], dim=2)  # [num_layers, batch, 2*dim]  ------>torch.Size([1, 8, 1024])

        return output, final
