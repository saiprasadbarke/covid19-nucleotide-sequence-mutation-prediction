from torch import broadcast_to, nn, Tensor, permute, sum, gather, set_printoptions
from torch.nn.functional import cross_entropy


class SequenceWeightedCELoss(nn.Module):
    def __init__(self, weights: Tensor):
        super(SequenceWeightedCELoss, self).__init__()
        self.weights = weights

    def forward(
        self, inputs: Tensor, targets: Tensor,
    ):
        n, c, l = inputs.shape
        # set_printoptions(profile="full")
        # print("Input")
        # print(inputs)
        # print()
        # print("Targets")
        # print(targets)
        # Gather log probabilities with respect to target
        logp = gather(inputs, 1, targets.view(n, 1, l))
        # print()
        # print("logp")
        # print(logp)
        # generate_heatmap(logp, vocabulary.itos, list(range(sequence_length_kmerized)), "logp.png")
        # Multiply with weights
        broadcasted_weights = broadcast_to(self.weights, (n, c, l))
        weight_map = gather(broadcasted_weights, 1, targets.view(n, 1, l))
        # print()
        # print("weights")
        # print(broadcasted_weights)
        # print()
        # print("weight_map")
        # print(weight_map)
        weighted_logp = (logp * weight_map).view(n, -1)
        # print()
        # print("weighted_logp")
        # print(weighted_logp)
        # Rescale so that loss is in approx. same interval
        weighted_loss = weighted_logp.sum(1) / weight_map.view(n, -1).sum(1)

        # Average over mini-batch
        weighted_loss = -weighted_logp.mean()

        return weighted_loss

