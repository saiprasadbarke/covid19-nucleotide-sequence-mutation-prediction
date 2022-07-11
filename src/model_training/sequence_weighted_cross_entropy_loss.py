from torch import broadcast_to, nn, Tensor, permute, sum, gather
from torch.nn.functional import cross_entropy


class SequenceWeightedCELoss(nn.Module):
    def __init__(self, weights: Tensor):
        super(SequenceWeightedCELoss, self).__init__()
        self.weights = weights

    def forward(
        self, inputs: Tensor, targets: Tensor,
    ):
        n, c, l = inputs.shape

        # Gather log probabilities with respect to target
        logp = gather(inputs, 1, targets.view(n, 1, l))

        # Multiply with weights
        weighted_logp = (logp * self.weights).view(n, -1)

        # Rescale so that loss is in approx. same interval
        weighted_loss = weighted_logp.sum(1) / self.weights.view(1, -1).sum(1)

        # Average over mini-batch
        weighted_loss = -weighted_loss.mean()

        return weighted_loss

