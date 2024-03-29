from typing import Tuple
from torch import Tensor
from settings.constants import USE_CUDA


class Batch:
    """Object for holding a batch of data during training.
    Input is a batch from a pytorch dataloader.
    """

    def __init__(self, src: Tensor, trg: Tensor):
        self.src_input = src
        self.nseqs = src.size(0)
        # trg_input is used for teacher forcing, last one is cut off
        self.trg_input = trg[:, :-1]
        # trg_y is used for loss computation, shifted by one since BOS
        self.trg_y = trg[:, 1:]
        self.ntokens = self.trg_input.numel()

        if USE_CUDA:
            self.src_input = self.src_input.cuda()
            self.trg_input = self.trg_input.cuda()
            self.trg_y = self.trg_y.cuda()


def rebatch(batch: Tuple[Tensor, Tensor]) -> Batch:
    """Wrap Dataloader batch into custom Batch class for pre-processing"""
    return Batch(batch[0], batch[1])
