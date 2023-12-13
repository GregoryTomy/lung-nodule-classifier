"""
https://amaarora.github.io/posts/2020-06-29-FocalLoss.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="none"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        targets = targets.type(torch.long)
        at = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt) ** self.gamma * at * bce_loss

        if self.reduction == "mean":
            return torch.mean(focal_loss)
        elif    self.reduction == "sum":
            return torch.sum(focal_loss)
        else:
            return focal_loss
