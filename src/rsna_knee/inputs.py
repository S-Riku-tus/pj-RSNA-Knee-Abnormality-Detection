"""Cache-to-tensor normalization shared by training and checkpoint-only inference."""

import numpy as np
import torch


def normalize_images(images, config):
    """Normalize uint8 NCHW windows; preserve the exact legacy arithmetic order."""
    if not isinstance(images, np.ndarray) or images.dtype != np.uint8 or images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Expected uint8 MRI windows with shape (windows, 3, height, width)")
    mode = config["model"].get("input_normalization", "legacy")
    if mode == "legacy":
        return torch.from_numpy(images.astype(np.float32) / 127.5 - 1)
    if mode == "imagenet":
        tensor = torch.from_numpy(images.astype(np.float32) / 255)
        mean = tensor.new_tensor((0.485, 0.456, 0.406)).reshape(1, 3, 1, 1)
        std = tensor.new_tensor((0.229, 0.224, 0.225)).reshape(1, 3, 1, 1)
        return (tensor - mean) / std
    raise ValueError("Unsupported input normalization")
