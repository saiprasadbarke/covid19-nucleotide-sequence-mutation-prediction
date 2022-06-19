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
        self.trg_input = trg
        self.ntokens = self.trg_input.numel()

        if USE_CUDA:
            self.src_input = self.src_input.cuda()
            self.trg_input = self.trg_input.cuda()


def rebatch(batch: Tuple[Tensor, Tensor]) -> Batch:
    """Wrap Dataloader batch into custom Batch class for pre-processing"""
    return Batch(batch[0], batch[1])
