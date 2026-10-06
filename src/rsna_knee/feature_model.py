"""Trainable two-level, finding-specific aggregation of generic frozen features."""

import torch
from torch import nn

from .contracts import TARGETS


def masked_attention(logits, mask, dim):
    weights = logits.masked_fill(~mask, -1e4).softmax(dim) * mask
    return weights / weights.sum(dim, keepdim=True).clamp_min(1e-12)


class FrozenFeatureHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config["head"]["hidden_dim"]
        self.pooling = config["head"]["pooling"]
        self.project = nn.Sequential(
            nn.LayerNorm(config["encoder"]["feature_dim"]), nn.Linear(config["encoder"]["feature_dim"], h), nn.GELU()
        )
        self.plane = nn.Embedding(4, h)
        self.geometry = nn.Linear(4, h)
        self.slice_score = nn.Linear(h, len(TARGETS))
        self.series_query = nn.Parameter(torch.empty(len(TARGETS), h))
        self.output_weight = nn.Parameter(torch.empty(len(TARGETS), h))
        self.output_bias = nn.Parameter(torch.zeros(len(TARGETS)))
        self.dropout = nn.Dropout(config["head"]["dropout"])
        nn.init.normal_(self.series_query, std=0.02)
        nn.init.normal_(self.output_weight, std=0.02)

    def forward(self, features, mask, positions, planes, flags):
        if not mask.flatten(1).any(1).all():
            raise ValueError("Every study requires at least one valid slice")
        # Mask before projection too, so corrupted padding cannot influence valid tokens.
        features = features.float().masked_fill(~mask[..., None], 0)
        position = positions.masked_fill(~mask, 0)
        metadata = torch.cat(
            [position[..., None], position.square()[..., None], flags[:, :, None, :].expand(-1, -1, mask.shape[2], -1)],
            -1,
        )
        x = self.project(features) + self.plane(planes)[:, :, None, :] + self.geometry(metadata)
        scores = self.slice_score(x) if self.pooling == "attention" else x.new_zeros((*mask.shape, len(TARGETS)))
        weights = masked_attention(scores, mask[..., None], dim=2)
        series = torch.einsum("bslt,bslh->bsth", weights, x)
        series_mask = mask.any(2)[..., None]
        scores = (
            torch.einsum("bsth,th->bst", series, self.series_query)
            if self.pooling == "attention"
            else series.new_zeros(series.shape[:3])
        )
        weights = masked_attention(scores, series_mask, dim=1)
        pooled = torch.einsum("bst,bsth->bth", weights, series)
        return (self.dropout(pooled) * self.output_weight).sum(-1) + self.output_bias
