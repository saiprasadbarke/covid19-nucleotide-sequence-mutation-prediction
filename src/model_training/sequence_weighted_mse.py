from torch import nn, Tensor, sum
from torch.nn.functional import mse_loss


class SequenceWeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(SequenceWeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(
        self, inputs: Tensor, targets: Tensor,
    ):
        loss = mse_loss(inputs, targets, reduction="none")
        return sum(loss * self.weights)

