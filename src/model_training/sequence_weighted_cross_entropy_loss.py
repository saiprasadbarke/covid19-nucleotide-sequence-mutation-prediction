from torch import nn, Tensor, sum
from torch.nn.functional import cross_entropy


class SequenceWeightedCELoss(nn.Module):
    def __init__(self, weights):
        super(SequenceWeightedCELoss, self).__init__()
        self.weights = weights

    def forward(
        self, inputs: Tensor, targets: Tensor,
    ):
        loss = cross_entropy(inputs, targets, reduction="none")
        return sum(loss * self.weights)

