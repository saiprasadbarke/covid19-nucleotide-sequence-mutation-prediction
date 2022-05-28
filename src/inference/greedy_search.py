from torch import long, bool, ones, max, cat

from model_components.model import EncoderDecoder
from settings.constants import EOS_IDX, BOS_IDX


def greedy_decode(model: EncoderDecoder, src, max_len=501):
    output = []
    encoder_hidden, encoder_final = model.encode(src)
    ys = ones(1, 1).fill_(BOS_IDX).type(long)
    for i in range(max_len - 1):
        out, _, pre_output = model.decode(encoder_hidden, encoder_final, ys)
        logits = model.generator(pre_output)
        _, next_word = max(logits, dim=-1)
        next_word = next_word.item()

        ys = cat([ys, ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == EOS_IDX:
            break
    return ys
