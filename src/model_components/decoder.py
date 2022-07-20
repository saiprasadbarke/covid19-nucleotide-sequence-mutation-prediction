import torch
import torch.nn as nn
from torch import Tensor

from model_components.attention import BahdanauAttention


class Decoder(nn.Module):
    """A conditional RNN decoder with attention."""

    def __init__(
        self,
        emb_size: int,
        hidden_size: int,
        attention: BahdanauAttention,
        num_layers: int = 1,
        dropout: float = 0.5,
        emb_dropout: float = 0.5,
        bridge: bool = True,
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

    def _forward_step(
        self, prev_embed: Tensor, encoder_output: Tensor, proj_key: Tensor, hidden: Tensor,
    ):
        """Perform a single decoder step (1 word)"""

        # compute context vector using attention mechanism
        query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]
        context, attn_prob = self.attention(query=query, proj_key=proj_key, value=encoder_output)

        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, context], dim=2)
        rnn_input = self.emb_dropout(rnn_input)
        output, hidden = self.rnn(rnn_input, hidden)

        att_vector_input = torch.cat([prev_embed, output, context], dim=2)
        att_vector_input = self.attention_dropout_layer(att_vector_input)
        att_vector = torch.tanh(self.pre_output_layer(att_vector_input))

        return output, hidden, att_vector, attn_prob

    def forward(
        self,
        trg_embed: Tensor,
        encoder_output: Tensor,
        encoder_hidden: Tensor,
        hidden: Tensor = None,
        unroll_steps: int = None,
    ):
        """Unroll the decoder one step at a time for `unroll_steps` steps.
        For every step, the `_forward_step` function is called internally.

        During training, the target inputs (`trg_embed') are already known for
        the full sequence, so the full unroll is done.

        For inference, this function is called with one step at a time since
        embedded targets are the predictions from the previous time step.

        The `encoder_output` are the hidden states from the encoder and are
        used as context for the attention.

        The `encoder_hidden` is the last encoder hidden state that is used to
        initialize the first hidden decoder state

        :param trg_embed: embedded target inputs,
            shape (batch_size, trg_length, embed_size)
        :param encoder_output: hidden states from the encoder,
            shape (batch_size, src_length, encoder.output_size)
        :param encoder_hidden: last state from the encoder,
            shape (batch_size, encoder.output_size)
        :param unroll_steps: number of steps to unroll the decoder RNN
        :param hidden: previous decoder hidden state,
            if not given it's initialized as in `self.init_hidden`,
            shape (batch_size, num_layers, hidden_size)
        """

        # the maximum number of steps to unroll the RNN
        if unroll_steps is None:
            unroll_steps = trg_embed.size(1)

        # initialize decoder hidden state
        # hidden =  [1, batch_size, decoder_hidden_size]
        if hidden is None:
            hidden = self._init_hidden(encoder_hidden)

        # pre-compute projected encoder hidden states
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        # encoder_output = [N, SEQ_L, RNN_ENCODER_HIDDEN]
        proj_key = self.attention.key_layer(encoder_output)

        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []
        att_vectors = []
        att_probs = []

        # unroll the decoder RNN for `unroll_steps` steps
        for i in range(unroll_steps):
            prev_embed = trg_embed[:, i].unsqueeze(1)
            output, hidden, att_vector, att_prob = self._forward_step(prev_embed, encoder_output, proj_key, hidden,)
            decoder_states.append(output)
            att_vectors.append(att_vector)
            att_probs.append(att_prob)
        decoder_states = torch.cat(decoder_states, dim=1)
        # att_vectors: batch, unroll_steps, hidden_size
        att_vectors = torch.cat(att_vectors, dim=1)
        # att_probs: batch, unroll_steps, src_length
        att_probs = torch.cat(att_probs, dim=1)
        return decoder_states, hidden, att_vectors, att_probs  # [B, N, D]

    def _init_hidden(self, encoder_hidden: Tensor):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""
        batch_size = encoder_hidden.size(0)
        if encoder_hidden is None:
            # start with zeros
            return encoder_hidden.new_zeros(self.num_layers, batch_size, self.hidden_size)

        return torch.tanh(self.bridge(encoder_hidden)).unsqueeze(0).repeat(self.num_layers, 1, 1)
