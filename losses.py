import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Takes embeddings of two samples and a target label == 1
    if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin=5.):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, ops, target, size_average=True):
        op1, op2 = ops[0], ops[1]
        dist = F.pairwise_distance(op1, op2)
        pdist = dist * target
        ndist = dist * (1 - target)
        loss = 0.5 * ((pdist ** 2) + (F.relu(self.margin - ndist) ** 2))
        return loss.mean()
