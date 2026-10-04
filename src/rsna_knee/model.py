"""Engineering baseline: ResNet18 + separate attention pooling for each finding."""

import math
import re
from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet18

from .contracts import RESNET18_V1_SHA256_PREFIX, RESNET18_V1_URL, sha256


class KneeMIL(nn.Module):
    def __init__(self, dropout=0.2, *, bn_running_stats="update"):
        super().__init__()
        if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("Dropout must be a finite number in [0, 1)")
        if bn_running_stats not in ("update", "freeze"):
            raise ValueError("Unsupported BatchNorm running-statistics mode")
        self.bn_running_stats = bn_running_stats
        # Offline by construction. This is NOT compatible with public CoAtNet/DINO checkpoints.
        self.encoder = resnet18(weights=None)
        self.encoder.fc = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.Linear(512, 12)
        self.finding_weight = nn.Parameter(torch.empty(12, 512))
        self.bias = nn.Parameter(torch.zeros(12))
        nn.init.normal_(self.finding_weight, std=0.01)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.bn_running_stats == "freeze":
            for module in self.encoder.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def initialize_encoder(self, path, expected_sha256):
        """Explicit local training initialization; constructors and prediction never download."""
        if (
            not isinstance(expected_sha256, str)
            or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256)
            or not expected_sha256.lower().startswith(RESNET18_V1_SHA256_PREFIX)
        ):
            raise ValueError("Expected full official ResNet18 ImageNet1K V1 SHA-256")
        if not isinstance(path, (str, Path)) or not str(path).strip() or "://" in str(path):
            raise ValueError("Pretrained initialization requires an explicit local weight path")
        path = Path(path)
        if not path.is_file():
            raise ValueError("Local pretrained weight file does not exist")
        digest = sha256(path)
        if digest != expected_sha256.lower():
            raise ValueError("Local pretrained weight SHA-256 mismatch")
        state = torch.load(path, map_location="cpu", weights_only=True)
        # Keep the original MIL-head initialization and the next global RNG draw unchanged.
        with torch.random.fork_rng(devices=[]):
            encoder = resnet18(weights=None)
        encoder.load_state_dict(state, strict=True)  # Includes the official 1000-class classifier.
        encoder.fc = nn.Identity()
        original = next(self.encoder.parameters())
        encoder.to(device=original.device, dtype=original.dtype)
        self.encoder = encoder
        self.train(self.training)
        return {
            "name": "imagenet1k_v1",
            "source_url": RESNET18_V1_URL,
            "path": str(path.resolve()),
            "sha256": digest,
            "strict_load_including_fc": True,
            "mil_head_preserved": True,
        }

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
