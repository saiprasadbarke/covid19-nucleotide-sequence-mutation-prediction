from numpy import nditer, zeros, zeros_like
from torch import broadcast_to, nn, Tensor, permute, sum, gather, set_printoptions
from torch.nn.functional import cross_entropy, nll_loss


class SequenceWeightedCELoss2(nn.Module):
    def __init__(self, weights: Tensor):
        super(SequenceWeightedCELoss2, self).__init__()
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
        # logp = gather(inputs, 1, targets.view(n, 1, l))  # logp: N, 1, D, targets : N, D
        logp = cross_entropy(inputs, targets, reduction="none")
        # print()
        # print("logp")
        # print(logp)
        # generate_heatmap(logp, vocabulary.itos, list(range(sequence_length_kmerized)), "logp.png")
        # Multiply with weights
        # broadcasted_weights = broadcast_to(self.weights, (n, c, l))
        # weight_map = gather(broadcasted_weights, 1, targets.view(n, 1, l))
        weight_map = zeros((n, l))
        for row, target in enumerate(targets.tolist()):
            for pos, kmer_value in enumerate(target):
                weight_map[row, pos] = self.weights[kmer_value, pos]
        # print()
        # print("weights")
        # print(broadcasted_weights)
        # print()
        # print("weight_map")
        # print(weight_map)
        weight_map = Tensor(weight_map).view(n, 1, l).cuda()
        weighted_loss = (logp * weight_map).view(n, -1)
        # print()
        # print("weighted_logp")
        # print(weighted_logp)
        # Rescale so that loss is in approx. same interval
        rescaled_loss = weighted_loss.sum(1) / weight_map.view(n, -1).sum(1)

        # Average over mini-batch
        rescaled_loss = rescaled_loss.mean()

        return rescaled_loss

