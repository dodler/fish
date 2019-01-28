import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output_pair, label):
        output1 = output_pair[0]
        output2 = output_pair[1]

        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) + (1 - label) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive
