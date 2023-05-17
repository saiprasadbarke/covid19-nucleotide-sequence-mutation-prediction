from typing import List
import torch
import numpy as np
from model_components.model import EncoderDecoder


def greedy_decode(model: EncoderDecoder, src: torch.Tensor, max_len: int = 500):
    """Greedily decode a sentence."""
    output = []
    attention_scores = []
    hidden = None
    model.eval()
    with torch.no_grad():
        encoder_hidden, encoder_final = model.encode(src)
        prev_y = torch.ones(1, 1).fill_(0).type_as(src)

        for _ in range(max_len):
            out, hidden, pre_output, _ = model.decode(
                encoder_output=encoder_hidden, encoder_hidden=encoder_final, trg=prev_y, decoder_hidden=hidden
            )

            # we predict from the pre-output layer, which is
            # a combination of Decoder state, prev emb, and context
            prob = model.generator(pre_output[:, -1])

            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data.item()
            output.append(next_word)
            prev_y = torch.ones(1, 1).type_as(src).fill_(next_word)
            attention_scores.append(model.decoder.attention.alphas.cpu().numpy())

    output = np.array(output)
    first_eos = np.where(output == 1)[0]
    if len(first_eos) > 0:
        output = output[: first_eos[0]]
    return output.tolist(), np.concatenate(attention_scores, axis=1)
