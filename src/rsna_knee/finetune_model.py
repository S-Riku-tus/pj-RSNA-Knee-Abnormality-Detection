"""One-factor native-triplet candidate: the same f002 head and DINO last two blocks.

The frozen prefix runs under no_grad, not inference_mode, so its ordinary output
tensors can be saved for the trainable suffix's backward pass. Encoder eval mode
and fp16 CLS storage quantization match the frozen feature path in both arms.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .feature_model import FrozenFeatureHead


class NativeTripletMIL(nn.Module):
    def __init__(self, encoder, config, *, trainable_blocks=0, window_chunk=4, checkpoint_last2=False):
        super().__init__()
        if trainable_blocks not in (0, 2) or type(window_chunk) is not int or window_chunk < 1:
            raise ValueError("Only frozen or last2 encoder arms and a positive window chunk are supported")
        if getattr(encoder, "chunked_blocks", True) or len(encoder.blocks) != 12:
            raise ValueError("The audited generic ViT-S/14 must have twelve unchunked transformer blocks")
        if getattr(encoder, "num_register_tokens", -1) != 0 or encoder.embed_dim != 384:
            raise ValueError("Unexpected generic DINO encoder architecture")
        if config["preprocess"]["image_size"] != 392 or config["encoder"]["feature_dim"] != 384:
            raise ValueError("Keep native 392px / CLS384 fixed")
        self.encoder, self.head = encoder, FrozenFeatureHead(config)
        self.config, self.trainable_blocks, self.window_chunk = config, trainable_blocks, window_chunk
        self.checkpoint_last2 = checkpoint_last2
        self.encoder.requires_grad_(False)
        if trainable_blocks:
            for block in self.encoder.blocks[-trainable_blocks:]:
                block.requires_grad_(True)
        self.encoder.eval()
        selected = [name for name, parameter in self.encoder.named_parameters() if parameter.requires_grad]
        if trainable_blocks and (
            not selected or any(not name.startswith(("blocks.10.", "blocks.11.")) for name in selected)
        ):
            raise ValueError("Only encoder blocks10/11 may require gradients; final norm must stay frozen")

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def encode_normalized(self, images):
        """Identical normalized CLS forward, with gradients only in the allowed suffix."""
        split = 12 - self.trainable_blocks
        with torch.no_grad():
            tokens = self.encoder.prepare_tokens_with_masks(images)
            for block in self.encoder.blocks[:split]:
                tokens = block(tokens)
            if self.trainable_blocks == 0:
                return self.encoder.norm(tokens)[:, 0].to(torch.float16).float()
        for block in self.encoder.blocks[split:]:
            if self.checkpoint_last2 and torch.is_grad_enabled():
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        # Frozen norm parameters still transmit the gradient to the suffix.
        return self.encoder.norm(tokens)[:, 0].to(torch.float16).float()

    def forward(self, images, mask, positions, planes, flags):
        p = self.config["preprocess"]
        expected = (mask.shape[0], p["max_series"], p["slices_per_series"])
        if mask.shape != expected or images.shape != (*expected, 3, 392, 392) or images.dtype != torch.uint8:
            raise ValueError("Expected uint8 padded native392 pixels with unchanged 3x48 masks")
        if mask.dtype != torch.bool or not mask.flatten(1).any(1).all():
            raise ValueError("Every study requires at least one valid slice")
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
        if valid.shape != (len(indices), 384) or not torch.isfinite(valid).all():
            raise ValueError("Nonfinite/malformed native encoder CLS features")
        features = valid.new_zeros((mask.numel(), 384)).index_copy(0, indices, valid).reshape(*expected, 384)
        return self.head(features, mask, positions, planes, flags)

    def parameter_contract(self):
        return {
            "trainable_encoder_names": [name for name, value in self.encoder.named_parameters() if value.requires_grad],
            "trainable_encoder_parameters": sum(
                value.numel() for value in self.encoder.parameters() if value.requires_grad
            ),
            "frozen_encoder_parameters": sum(
                value.numel() for value in self.encoder.parameters() if not value.requires_grad
            ),
            "trainable_head_parameters": sum(value.numel() for value in self.head.parameters() if value.requires_grad),
            "encoder_eval": not self.encoder.training,
            "final_norm_frozen": not any(value.requires_grad for value in self.encoder.norm.parameters()),
            "trainable_blocks": self.trainable_blocks,
            "window_chunk": self.window_chunk,
            "checkpoint_last2": self.checkpoint_last2,
            "chunk_factor": "same window microbatch in both arms; retained backward activations span all valid windows",
            "checkpoint_factor": "optional suffix activation recomputation; separate from the last2 biological factor",
            "feature_quantization": "fp16 CLS then fp32, matching frozen feature cache storage",
        }
