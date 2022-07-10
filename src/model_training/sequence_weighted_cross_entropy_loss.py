from torch import broadcast_to, nn, Tensor, permute, sum
from torch.nn.functional import cross_entropy


class SequenceWeightedCELoss(nn.Module):
    def __init__(self, weights: Tensor):
        super(SequenceWeightedCELoss, self).__init__()
        self.weights = weights

    def forward(
        self, inputs: Tensor, targets: Tensor,
    ):
        loss = cross_entropy(inputs, targets, reduction="none")
        loss = broadcast_to(loss, (self.weights.size(1), loss.size(0), loss.size(1)))
        loss = permute(loss, (1, 0, 2))
        return sum(loss * self.weights)

