from torch import Tensor, permute
from torch.optim import Optimizer
from torch.nn.utils.clip_grad import clip_grad_norm_

from model_components.model import EncoderDecoder


class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(
        self, model: EncoderDecoder, criterion, optimizer: Optimizer = None, scheduler=None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def __call__(self, x: Tensor, y: Tensor):
        x = self.model.generator(x)
        log_probs = permute(x, (0, 2, 1)).contiguous()
        targets = y.contiguous().long()
        loss = self.criterion(log_probs, targets)
        if self.optimizer is not None:
            loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.model.zero_grad()

        return loss.data.item()
