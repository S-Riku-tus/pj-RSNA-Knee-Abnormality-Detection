"""Separate F014 native392 CNN constructor; the DINO validators remain intact."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .contracts import TARGETS, sha256
from .feature_model import FrozenFeatureHead

MODEL = "convnext_tiny.fb_in22k_ft_in1k_384"
WEIGHTS_SHA = "a5cbe10cfa2e90ec787a082253033ac0a61de0264b68210815a988e53a1a5d7c"
CHECKPOINT_SCHEMA = "f014_convnext_native392_suffix_head_v1"
SOURCE_FILES = (
    "convnext_model.py", "convnext_training.py", "native_training_fast16.py", "finetune_model.py",
    "feature_model.py", "triplet_cache.py", "feature_imaging.py", "feature_contracts.py", "imaging.py",
    "contracts.py", "checkpoints.py", "metrics.py", "runtime.py", "diagnostics.py", "inputs.py", "model.py",
)
SUFFIX_PREFIXES = ("stages.3.blocks.1.", "stages.3.blocks.2.")


def require(value, message):
    if not value:
        raise ValueError(message)


def source_hashes():
    return {name: sha256(Path(__file__).with_name(name)) for name in SOURCE_FILES}


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def suffix_keys(model):
    return {k for k in model.encoder.state_dict() if k.startswith(SUFFIX_PREFIXES)}


def prefix_digest(model):
    return tensor_digest({k: v for k, v in model.encoder.state_dict().items() if k not in suffix_keys(model)})


def create_model(weights, config, provenance, *, device="cuda"):
    """Always offline, full strict generic initialization, classifier removed explicitly."""
    import safetensors
    import timm
    from safetensors.torch import load_file

    require(timm.__version__ == "1.0.22" and safetensors.__version__ == "0.6.2", "Exact F014 library versions")
    path = Path(weights)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size == 114374272
            and sha256(path) == WEIGHTS_SHA == provenance["weights_sha256"], "Exact generic ImageNet weights; no fallback")
    require(provenance["model_name"] == MODEL and provenance["competition_finetuned"] is False
            and provenance["gold_used_for_tuning"] is False, "Generic provenance only")
    encoder = timm.create_model(MODEL, pretrained=False, num_classes=1000)
    state = load_file(str(path), device="cpu")
    require(set(state) == set(encoder.state_dict()) and all(value.dtype == torch.float32
            and bool(torch.isfinite(value).all()) for value in state.values()), "Full finite FP32 generic state")
    encoder.load_state_dict(state, strict=True)
    require(encoder.num_features == 768 and [len(s.blocks) for s in encoder.stages] == [3, 3, 9, 3], "Exact Tiny architecture")
    encoder.reset_classifier(0)
    require(isinstance(encoder.head.fc, nn.Identity), "Explicit generic classification layer removal")
    result = ConvNeXtTripletMIL(encoder, config).to(device)
    result.bound_source_hashes = source_hashes()
    return result


class ConvNeXtTripletMIL(nn.Module):
    def __init__(self, encoder, config):
        super().__init__()
        require(config["schema_version"] == "f014_convnext_native392_v1"
                and config["encoder"]["model_name"] == MODEL and config["encoder"]["feature_dim"] == 768
                and config["preprocess"]["image_size"] == 392 and "augmentation" not in config, "Separate fixed F014 config")
        self.encoder, self.config = encoder, config
        self.window_chunk, self.trainable_blocks = config["online"]["window_chunk"], 2
        self.checkpoint_last2 = config["online"]["checkpoint_last2"]
        self.head = FrozenFeatureHead(config)
        self.study_gamma = None
        encoder.requires_grad_(False)
        for block in encoder.stages[3].blocks[1:]:
            block.requires_grad_(True)
        encoder.eval()
        require({n for n, p in encoder.named_parameters() if p.requires_grad} == suffix_keys(self), "Only exact last two CNN blocks trainable")

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def parameter_contract(self):
        return {"encoder_total_parameters": sum(p.numel() for p in self.encoder.parameters()),
                "trainable_encoder_parameters": sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
                "trainable_head_parameters": sum(p.numel() for p in self.head.parameters() if p.requires_grad),
                "trainable_encoder_names": sorted(n for n, p in self.encoder.named_parameters() if p.requires_grad),
                "trainable_blocks": 2, "window_chunk": self.window_chunk, "checkpoint_last2": self.checkpoint_last2,
                "feature_dim": 768, "generic_classifier_removed": True}

    def encode_normalized(self, images):
        with torch.no_grad():
            value = self.encoder.stem(images)
            for stage in self.encoder.stages[:3]:
                value = stage(value)
            value = self.encoder.stages[3].downsample(value)
            value = self.encoder.stages[3].blocks[0](value)
        for block in self.encoder.stages[3].blocks[1:]:
            value = checkpoint(block, value, use_reentrant=False) if self.checkpoint_last2 and torch.is_grad_enabled() else block(value)
        value = self.encoder.norm_pre(value)
        return self.encoder.forward_head(value, pre_logits=True).to(torch.float16).float()

    def forward(self, images, mask, positions, planes, flags):
        settings = self.config["preprocess"]
        expected = (mask.shape[0], settings["max_series"], settings["slices_per_series"])
        require(tuple(mask.shape) == expected and tuple(images.shape) == (*expected, 3, 392, 392)
                and images.dtype == torch.uint8 and mask.dtype == torch.bool, "Unchanged native392 uint8/mask contract")
        require(bool(mask.flatten(1).any(1).all()), "Every study needs at least one image")
        flattened = images.reshape(-1, 3, 392, 392)
        indices = mask.reshape(-1).nonzero(as_tuple=False).flatten()
        encoded = []
        for offset in range(0, len(indices), self.window_chunk):
            selected = indices[offset:offset + self.window_chunk]
            batch = flattened.index_select(0, selected).float().div(255)
            mean, std = [batch.new_tensor(v)[None, :, None, None] for v in ([.485, .456, .406], [.229, .224, .225])]
            with torch.autocast(device_type=batch.device.type, dtype=torch.float16, enabled=self.config["encoder"]["precision"] == "fp16"):
                encoded.append(self.encode_normalized((batch - mean) / std))
        valid = torch.cat(encoded)
        require(valid.shape == (len(indices), 768) and bool(torch.isfinite(valid).all()), "Finite native CNN768 features")
        features = valid.new_zeros((mask.numel(), 768)).index_copy(0, indices, valid).reshape(*expected, 768)
        return self.head(features, mask, positions, planes, flags)


def make_checkpoint(model, provenance, *, fold, epoch, weak_bce, prefix_sha256, audit, artificial=False):
    require(source_hashes() == model.bound_source_hashes and prefix_digest(model) == prefix_sha256, "Source/frozen prefix preserved")
    states = {"encoder_suffix": {k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items() if k in suffix_keys(model)},
              "head": {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}}
    require(all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for state in states.values() for v in state.values()), "Finite strict FP32 selected state")
    return {"schema_version": CHECKPOINT_SCHEMA, "targets": list(TARGETS), "config": model.config,
            "encoder_provenance": provenance, "source_sha256": model.bound_source_hashes, "fold": fold, "epoch": epoch,
            "weak_bce": weak_bce, "frozen_prefix_sha256": prefix_sha256, "selection_data": "weak_validation",
            "gold_used_for_training_or_selection": False, "gold_evaluated": False, "audit": audit,
            "artificial": artificial, **states}


def load_checkpoint(path, *, allow_artificial=False):
    value = torch.load(path, map_location="cpu", weights_only=True)
    require(value["schema_version"] == CHECKPOINT_SCHEMA and value["targets"] == list(TARGETS)
            and value["source_sha256"] == source_hashes() and value["gold_used_for_training_or_selection"] is False,
            "Exact distinct CNN source/label/checkpoint identity")
    require(value["artificial"] is False or allow_artificial, "Artificial checkpoint refused for genuine inference")
    return value


def apply_checkpoint(model, value, provenance, *, allow_artificial=False):
    require(type(model) is ConvNeXtTripletMIL and value["schema_version"] == CHECKPOINT_SCHEMA
            and model.config == value["config"] and provenance == value["encoder_provenance"]
            and value["targets"] == list(TARGETS) and (value["artificial"] is False or allow_artificial)
            and value["source_sha256"] == source_hashes() and prefix_digest(model) == value["frozen_prefix_sha256"], "Strict current CNN constructor identity")
    suffix = {k: v for k, v in model.encoder.state_dict().items() if k in suffix_keys(model)}
    for name, current in (("encoder_suffix", suffix), ("head", model.head.state_dict())):
        require(set(current) == set(value[name]), "Complete CNN selected key inventory")
        for key, tensor in value[name].items():
            require(tensor.dtype == torch.float32 and tensor.shape == current[key].shape
                    and bool(torch.isfinite(tensor).all()), "Strict CNN tensor shape/dtype/finiteness")
        if name == "encoder_suffix":
            state = model.encoder.state_dict()
            state.update(value[name])
            model.encoder.load_state_dict(state, strict=True)
        else:
            model.head.load_state_dict(value[name], strict=True)
    require(prefix_digest(model) == value["frozen_prefix_sha256"], "CNN frozen prefix unchanged after apply")
    return model


def use_vendor(path):
    path = Path(path).resolve(strict=True)
    require(path.is_dir() and not path.is_symlink(), "Explicit isolated vendor directory")
    sys.path.insert(0, str(path))
