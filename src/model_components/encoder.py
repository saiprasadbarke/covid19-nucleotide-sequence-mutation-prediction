# Local

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
        emb_dropout: float = 0.0,
    ):
        super(Encoder, self).__init__()
        self.emb_dropout = nn.Dropout(p=emb_dropout, inplace=False)
        self.num_layers = num_layers
        self.rnn = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

    def forward(self, embedded_src):
        """
        Applies a bidirectional GRU to sequence of embeddings.
        embedded_src should have dimensions [batch, seq_len, embedding_dim].
        """
        embedded_src = self.emb_dropout(embedded_src)
        output, final = self.rnn(embedded_src)
        # we need to manually concatenate the final states for both directions
        fwd_final = final[0 : final.size(0) : 2]  # torch.Size([1, 8, 512])
        bwd_final = final[1 : final.size(0) : 2]  # torch.Size([1, 8, 512])
        final = torch.cat([fwd_final, bwd_final], dim=2)  # [num_layers, batch, 2*dim]  ------>torch.Size([1, 8, 1024])

        return output, final
