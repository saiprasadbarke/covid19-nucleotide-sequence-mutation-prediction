import time, math

from model_components.model import EncoderDecoder
from settings.constants import KMER_LENGTH, NUM_SPECIAL_CHARS


def run_epoch(data_iter, model: EncoderDecoder, loss_compute, print_every=50):
    """Standard Training and Logging Function"""

    start = time.time()
    total_tokens = 0
    total_loss = 0
    print_tokens = 0
    sequence_len = 3700 - KMER_LENGTH + 1
    input_feature_len = 4**KMER_LENGTH + NUM_SPECIAL_CHARS
    for i, batch in enumerate(data_iter, 1):
        src, trg = batch[0].reshape(-1, sequence_len, input_feature_len), batch[1]
        out, _, pre_output = model.forward(
            src,
            trg,
            # batch.src_mask,
            # batch.trg_mask,
            # batch.src_lengths,
            # batch.trg_lengths,
        )
        loss = loss_compute(pre_output, trg[:, 1:], src.size(0))
        total_loss += loss
        total_tokens += trg[:, 1:].data.sum().item()
        print_tokens += trg[:, 1:].data.sum().item()

        if model.training and i % print_every == 0:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" % (i, loss / src.size(0), print_tokens / elapsed))
            start = time.time()
            print_tokens = 0

    return math.exp(total_loss / float(total_tokens))
