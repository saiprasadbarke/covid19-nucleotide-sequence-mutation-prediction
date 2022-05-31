import torch
import torch.nn as nn
from torch import Tensor


class Decoder(nn.Module):
    """A conditional RNN decoder with attention."""

    def __init__(
        self,
        emb_size,
        hidden_size,
        attention,
        num_layers=1,
        dropout=0.5,
        emb_dropout=0.5,
        bridge=True,
    ):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout
        self.emb_dropout = nn.Dropout(p=emb_dropout, inplace=False)
        self.rnn = nn.GRU(
            emb_size + 2 * hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # to initialize from the final encoder state
        self.bridge = nn.Linear(2 * hidden_size, hidden_size, bias=True) if bridge else None

        self.attention_dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(hidden_size + 2 * hidden_size + emb_size, hidden_size, bias=False)

    def forward_step(self, prev_embed: Tensor, encoder_hidden: Tensor, proj_key: Tensor, hidden: Tensor):
        """Perform a single decoder step (1 word)"""

        # compute context vector using attention mechanism
        query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]
        context, attn_probs = self.attention(query=query, proj_key=proj_key, value=encoder_hidden)

        rnn_input = self.emb_dropout(rnn_input)
        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)

        pre_output = torch.cat([prev_embed, output, context], dim=2)
        pre_output = self.attention_dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        return output, hidden, pre_output

    def forward(
        self,
        trg_embed: Tensor,
        encoder_hidden: Tensor,
        encoder_final: Tensor,
        hidden: Tensor = None,
        unroll_steps: int = None,
    ):
        """Unroll the decoder one step at a time for `unroll_steps` steps.
        For every step, the `_forward_step` function is called internally.

        During training, the target inputs (`trg_embed') are already known for
        the full sequence, so the full unrol is done.
        In this case, `hidden` and `prev_att_vector` are None.

        For inference, this function is called with one step at a time since
        embedded targets are the predictions from the previous time step.
        In this case, `hidden` and `prev_att_vector` are fed from the output
        of the previous call of this function (from the 2nd step on).
        """

        # the maximum number of steps to unroll the RNN
        if unroll_steps is None:
            unroll_steps = trg_embed.size(1)

        # initialize decoder hidden state
        # hidden =  [1, batch_size, decoder_hidden_size]
        if hidden is None:
            hidden = self.init_hidden(encoder_final)

        # pre-compute projected encoder hidden states
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        proj_key = self.attention.key_layer(encoder_hidden)  # encoder_hidden = [N, SEQ_L, RNN_IN_HIDDEN]

        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []
        pre_output_vectors = []

        # unroll the decoder RNN for `unroll_steps` steps
        for i in range(unroll_steps):
            prev_embed = trg_embed[:, i].unsqueeze(1)
            output, hidden, pre_output = self.forward_step(
                prev_embed,
                encoder_hidden,
                proj_key,
                hidden,
            )
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return decoder_states, hidden, pre_output_vectors  # [B, N, D]

    def init_hidden(self, encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""

        if encoder_final is None:
            return None  # start with zeros

        return torch.tanh(self.bridge(encoder_final))
