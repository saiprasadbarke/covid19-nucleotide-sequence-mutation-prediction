from settings.constants import RNN_INPUT_FEATURE_SIZE, RNN_INPUT_SEQUENCE_LENGTH, RNN_TARGET_SEQUENCE_LENGTH, USE_CUDA


class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a pytorch dataloader.
    """

    def __init__(
        self,
        src,
        trg,
    ):
        self.src_input = src
        self.nseqs = src.size(0)

        self.trg_input = None
        self.trg_y = None
        self.ntokens = None

        if trg is not None:
            # trg_input is used for teacher forcing, last one is cut off
            self.trg_input = trg[:, :-1]
            # trg_y is used for loss computation, shifted by one since BOS
            self.trg_y = trg[:, 1:]
            self.ntokens = self.trg_y.data.sum().item()

        if USE_CUDA:
            self.src_input = self.src_input.cuda()
            if trg is not None:
                self.trg_input = self.trg_input.cuda()
                self.trg_y = self.trg_y.cuda()


def rebatch(batch):
    """Wrap Dataloader batch into custom Batch class for pre-processing"""
    src = batch[0].reshape(-1, RNN_INPUT_SEQUENCE_LENGTH, 1)
    trg = batch[1].reshape(-1, RNN_TARGET_SEQUENCE_LENGTH, 1)
    return Batch(
        src,
        trg,
        #   pad_idx,
    )
