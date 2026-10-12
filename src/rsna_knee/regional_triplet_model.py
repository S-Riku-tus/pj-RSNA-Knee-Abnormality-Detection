"""Isolated representation hypothesis: CLS plus sixteen fixed patch regions.

This component retains NativeTripletMIL's encoder/head parameterization. It is
not wired into any trainer, checkpoint schema, cache, or submission Notebook.
The identical state-dict shapes do not make CLS-only checkpoints semantically
interchangeable with this representation.
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from .finetune_model import NativeTripletMIL

PATCH_GRID = 28
REGIONAL_GRID = 4
REGIONS_PER_WINDOW = 1 + REGIONAL_GRID**2


def pool_region_tokens(normalized_tokens):
    """CLS, then row-major 4x4 means of the normalized 28x28 patch lattice.

    Each region averages a disjoint 7x7 patch block in FP32. All seventeen
    descriptors then use the unchanged FP16-storage/FP32-head contract.
    No learned region identity or extra spatial coordinates enter the head.
    """
    if (
        normalized_tokens.ndim != 3
        or normalized_tokens.shape[1:] != (1 + PATCH_GRID**2, 384)
        or not normalized_tokens.is_floating_point()
        or not torch.isfinite(normalized_tokens).all()
    ):
        raise ValueError("Expected finite normalized CLS plus28x28 patch tokens, dimension384")
    width = PATCH_GRID // REGIONAL_GRID
    patches = normalized_tokens[:, 1:].float().reshape(-1, REGIONAL_GRID, width, REGIONAL_GRID, width, 384)
    regional = patches.mean(dim=(2, 4)).reshape(-1, REGIONAL_GRID**2, 384)
    descriptors = torch.cat((normalized_tokens[:, :1].float(), regional), dim=1).to(torch.float16).float()
    if not torch.isfinite(descriptors).all():
        raise ValueError("Nonfinite regional FP16 feature storage")
    return descriptors


class RegionalTripletMIL(NativeTripletMIL):
    """The existing finding-specific head attends to17 instances per MRI window."""

    def __init__(self, encoder, config, *, trainable_blocks=0, window_chunk=4, checkpoint_last2=False):
        if config["preprocess"]["max_series"] != 3 or config["preprocess"]["slices_per_series"] != 48:
            raise ValueError("Keep the existing native392 three-series by48-window input contract")
        super().__init__(
            encoder,
            config,
            trainable_blocks=trainable_blocks,
            window_chunk=window_chunk,
            checkpoint_last2=checkpoint_last2,
        )

    def encode_normalized(self, images):
        """The original no_grad prefix/eval/suffix-checkpoint scopes, all tokens."""
        split = 12 - self.trainable_blocks
        with torch.no_grad():
            tokens = self.encoder.prepare_tokens_with_masks(images)
            for block in self.encoder.blocks[:split]:
                tokens = block(tokens)
            if self.trainable_blocks == 0:
                return pool_region_tokens(self.encoder.norm(tokens))
        for block in self.encoder.blocks[split:]:
            if self.checkpoint_last2 and torch.is_grad_enabled():
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        # The frozen norm parameters still transmit suffix gradients.
        return pool_region_tokens(self.encoder.norm(tokens))

    def forward(self, images, mask, positions, planes, flags):
        expected = (mask.shape[0], 3, 48)
        if mask.shape != expected or images.shape != (*expected, 3, 392, 392) or images.dtype != torch.uint8:
            raise ValueError("Expected uint8 padded native392 pixels with unchanged3x48 masks")
        if mask.dtype != torch.bool or not mask.flatten(1).any(1).all():
            raise ValueError("Every study requires at least one valid slice")
        if positions.shape != expected or planes.shape != expected[:2] or flags.shape != (*expected[:2], 2):
            raise ValueError("Keep the original window positions, series planes and sequence flags")
        flattened = images.reshape(-1, 3, 392, 392)
        indices = mask.reshape(-1).nonzero(as_tuple=False).flatten()
        encoded = []
        for offset in range(0, len(indices), self.window_chunk):
            batch = flattened.index_select(0, indices[offset : offset + self.window_chunk]).float().div(255)
            mean = batch.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
            std = batch.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
            with torch.autocast(
                device_type=batch.device.type,
                dtype=torch.float16,
                enabled=self.config["encoder"]["precision"] == "fp16",
            ):
                encoded.append(self.encode_normalized((batch - mean) / std))
        valid = torch.cat(encoded)
        if valid.shape != (len(indices), REGIONS_PER_WINDOW, 384) or not torch.isfinite(valid).all():
            raise ValueError("Nonfinite/malformed native encoder regional features")
        features = (
            valid.new_zeros((mask.numel(), REGIONS_PER_WINDOW, 384))
            .index_copy(0, indices, valid)
            .reshape(expected[0], expected[1], expected[2] * REGIONS_PER_WINDOW, 384)
        )
        regional_mask = mask.repeat_interleave(REGIONS_PER_WINDOW, dim=2)
        regional_positions = positions.repeat_interleave(REGIONS_PER_WINDOW, dim=2)
        return self.head(features, regional_mask, regional_positions, planes, flags)

    def parameter_contract(self):
        contract = super().parameter_contract()
        contract.update(
            representation="normalized_CLS_plus_fixed4x4_patch_means_v1",
            instances_per_window=REGIONS_PER_WINDOW,
            instances_per_series=48 * REGIONS_PER_WINDOW,
            patch_pooling="Disjoint7x7 FP32 means, row-major4x4; CLS first; no extra head coordinates/parameters",
            feature_quantization="All17 descriptors FP16 storage thenFP32; unchanged CLS quantization",
            trainer_checkpoint_or_serving_qualified=False,
        )
        return contract
