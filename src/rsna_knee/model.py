"""Engineering baseline: ResNet18 + separate attention pooling for each finding."""

import torch
from torch import nn
from torchvision.models import resnet18


class KneeMIL(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        # Offline by construction. This is NOT compatible with public CoAtNet/DINO checkpoints.
        self.encoder = resnet18(weights=None)
        self.encoder.fc = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.Linear(512, 12)
        self.finding_weight = nn.Parameter(torch.empty(12, 512))
        self.bias = nn.Parameter(torch.zeros(12))
        nn.init.normal_(self.finding_weight, std=0.01)

    def forward(self, images, mask):
        batch, windows = images.shape[:2]
        if not mask.any(dim=1).all():
            raise ValueError("Every study needs at least one unmasked MRI window")
        flat_mask = mask.reshape(-1)
        encoded = self.encoder(images.flatten(0, 1)[flat_mask])
        features = encoded.new_zeros((batch * windows, 512))
        features[flat_mask] = encoded
        features = self.dropout(features.reshape(batch, windows, 512))
        attention = self.attention(features).masked_fill(~mask.unsqueeze(-1), float("-inf"))
        attention = attention.softmax(dim=1)
        pooled = torch.einsum("bvk,bvd->bkd", attention, features)
        return torch.einsum("bkd,kd->bk", pooled, self.finding_weight) + self.bias
