from settings.constants import RNN_INPUT_FEATURE_SIZE, RNN_INPUT_SEQUENCE_LENGTH, RNN_TARGET_SEQUENCE_LENGTH, USE_CUDA


class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a pytorch dataloader.
    """

    def __init__(
        self,
        src,
        trg,
        # pad_index=0,
    ):

        # src, src_lengths = src

        self.src = src
        # self.src_lengths = src_lengths
        # self.src_mask = (src != pad_index).unsqueeze(-2)
        self.nseqs = src.size(0)

        # self.trg = None
        # self.trg_y = None
        # self.trg_mask = None
        # self.trg_lengths = None
        # self.ntokens = None

        if trg is not None:
            # trg, trg_lengths = trg
            self.trg_input = trg[:, :-1]

            # trg_input is used for teacher forcing, last one is cut off
            # self.trg_lengths = trg_lengths
            self.trg_y = trg[:, 1:]  # trg_y is used for loss computation, shifted by one since BOS
            # self.trg_mask = self.trg_y != pad_index
            self.ntokens = self.trg_y.data.sum().item()

        # if USE_CUDA:
        #    self.src = self.src.cuda()
        #    self.src_mask = self.src_mask.cuda()

        #    if trg is not None:
        #        self.trg = self.trg.cuda()
        #        self.trg_y = self.trg_y.cuda()
        #        self.trg_mask = self.trg_mask.cuda()


def rebatch(
    # pad_idx,
    batch,
):
    """Wrap torchtext batch into our own Batch class for pre-processing"""
    src = batch[0].reshape(-1, RNN_INPUT_SEQUENCE_LENGTH, 1)
    trg = batch[1].reshape(-1, RNN_TARGET_SEQUENCE_LENGTH, 1)
    return Batch(
        src,
        trg,
        #   pad_idx,
    )
