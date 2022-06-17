import time, math

from sklearn.model_selection import learning_curve

from model_components.model import EncoderDecoder
from model_training.loss_computation import SimpleLossCompute


def run_epoch(data_iter, model: EncoderDecoder, loss_compute: SimpleLossCompute, print_every=50, tb_writer=None):
    """Standard Training and Logging Function"""

    start = time.time()
    total_tokens = 0
    epoch_loss = 0
    print_tokens = 0
    epoch_learning_rate = []
    for i, batch in enumerate(data_iter, 1):
        _, _, pre_output, _ = model.forward(batch.src_input, batch.trg_input)
        batch_loss = loss_compute(pre_output, batch.trg_input, batch.nseqs)
        epoch_loss += batch_loss
        total_tokens += batch.ntokens
        print_tokens += batch.ntokens
        if loss_compute.optimizer is not None:
            epoch_learning_rate.append(loss_compute.optimizer.param_groups[0]["lr"])
        model.zero_grad()
        if model.training and i % print_every == 0:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" % (i, batch_loss / batch.nseqs, print_tokens / elapsed))
            start = time.time()
            print_tokens = 0

    perplexity = math.exp(epoch_loss / float(total_tokens))
    return epoch_loss, perplexity, epoch_learning_rate
