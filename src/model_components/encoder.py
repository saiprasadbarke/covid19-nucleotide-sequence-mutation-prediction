# Local
from settings.constants import KMER_LENGTH

# External
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Encodes a sequence of word embeddings"""

    def __init__(
        self,
        input_size: int = KMER_LENGTH,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout=0.1,
    ):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)

    def forward(
        self,
        x,
        # mask,
        # lengths,
    ):
        """
        Applies a bidirectional GRU to sequence of embeddings x.
        The input mini-batch x needs to be sorted by length.
        x should have dimensions [batch, time, dim].
        """
        # packed = pack_padded_sequence(x, lengths, batch_first=True)
        # x = torch.Size([8, 3699, 66]), output = torch.Size([8, 3699, 1024]), final = torch.Size([2, 8, 512])
        output, final = self.rnn(x)

        # output, _ = pad_packed_sequence(output, batch_first=True)

        # we need to manually concatenate the final states for both directions
        fwd_final = final[0 : final.size(0) : 2]  # torch.Size([1, 8, 512])
        bwd_final = final[1 : final.size(0) : 2]  # torch.Size([1, 8, 512])
        final = torch.cat([fwd_final, bwd_final], dim=2)  # [num_layers, batch, 2*dim]  ------>torch.Size([1, 8, 1024])

        return output, final
