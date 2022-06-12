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
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, embedded_src):
        """
        Applies a bidirectional GRU to sequence of embeddings.
        embedded_src should have dimensions [batch, seq_len, embedding_dim].
        """
        # embedded_src = self.emb_dropout(embedded_src)
        # hidden: dir*layers x batch x hidden
        # output: batch x max_length x directions*hidden
        output, hidden = self.rnn(embedded_src)
        batch_size = hidden.size(1)
        # separate final hidden states by layer and direction
        hidden_layerwise = hidden.view(
            self.rnn.num_layers, 2 if self.rnn.bidirectional else 1, batch_size, self.rnn.hidden_size,
        )
        # hidden_layerwise: layers x directions x batch x hidden
        # we need to manually concatenate the final states of the last layer for both directions
        fwd_hidden_last = hidden_layerwise[-1:, 0]
        bwd_hidden_last = hidden_layerwise[-1:, 1]
        # fwd_final = hidden[0 : hidden.size(0) : 2]
        # bwd_final = hidden[1 : hidden.size(0) : 2]
        # only feed the final state of the top-most layer to the decoder
        hidden_concat = torch.cat([fwd_hidden_last, bwd_hidden_last], dim=2).squeeze(0)
        # hidden_concat: batch x directions*hidden
        # output: batch x max_length x directions*hidden
        return output, hidden_concat
