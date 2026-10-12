"""Artificial last2 gradient, checkpoint and safe generic-asset extraction contracts."""

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from rsna_knee.feature_contracts import source_tree_hash
from rsna_knee.finetune_model import NativeTripletMIL

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("profile_dinov2_last2", ROOT / "scripts/profile_dinov2_last2.py")
profiler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiler)
CONFIG = json.loads((ROOT / "configs/experiments/f002-dinov2-attention.json").read_text())


class ArtificialEncoder(nn.Module):
    """Small deterministic differentiable stand-in; never called a real DINO test."""

    def __init__(self):
        super().__init__()
        self.chunked_blocks, self.num_register_tokens, self.embed_dim = False, 0, 384
        self.blocks = nn.ModuleList(nn.Sequential(nn.Linear(384, 384), nn.GELU()) for _ in range(12))
        self.norm = nn.LayerNorm(384)
        self.patch_embed = nn.Linear(3, 384)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 384))

    def prepare_tokens_with_masks(self, images):
        pooled = images.mean((-2, -1))
        return self.patch_embed(pooled)[:, None, :] + self.cls_token


class LastTwoContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="last2-artificial-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_CUDA_guard_runs_before_output_assets_or_config_access(self):
        with patch.object(profiler, "ensure_cuda", side_effect=ValueError("CUDA required")):
            with self.assertRaisesRegex(ValueError, "CUDA required"):
                profiler.profile(SimpleNamespace(output=self.root / "never-created"))
        self.assertFalse((self.root / "never-created").exists())

    def test_generic_archive_traversal_and_symlink_rejected_before_writing(self):
        for unsafe in ("../escape.py", "encoder/dinov2-source/../../escape.py", "encoder\\unsafe.py", "C:/escape.py"):
            path = self.root / (hashlib.sha256(unsafe.encode()).hexdigest() + ".zip")
            with zipfile.ZipFile(path, "w") as archive:
                item = zipfile.ZipInfo("placeholder")
                item.filename = item.orig_filename = unsafe
                archive.writestr(item, b"synthetic")
            with self.subTest(unsafe=unsafe), self.assertRaisesRegex(ValueError, "Unsafe"):
                profiler.extract_generic_assets(path, self.root / "never-created")
        self.assertFalse((self.root / "never-created").exists())
        path = self.root / "symlink.zip"
        with zipfile.ZipFile(path, "w") as archive:
            item = zipfile.ZipInfo(profiler.WEIGHTS_MEMBER)
            item.external_attr = 0o120777 << 16
            archive.writestr(item, b"/outside")
        with self.assertRaisesRegex(ValueError, "symlink"):
            profiler.extract_generic_assets(path, self.root / "never-created")

    def test_whitelist_excludes_labels_other_models_and_non_generic_weights(self):
        for name in (
            "manifest/weak.csv",
            "manifest/gold.csv",
            "head/best.pt",
            "encoder/other.pth",
            "train/a.dcm",
            profiler.SOURCE_PREFIX + "docs/Cell-DINO.png",
        ):
            self.assertFalse(profiler.allowed_member(name), name)
        for name in (
            profiler.WEIGHTS_MEMBER,
            profiler.PROVENANCE_MEMBER,
            profiler.SOURCE_PREFIX + "hubconf.py",
            profiler.SOURCE_PREFIX + "LICENSE",
            profiler.SOURCE_PREFIX + "dinov2/layers/attention.py",
        ):
            self.assertTrue(profiler.allowed_member(name), name)

    def test_small_synthetic_generic_asset_extract_hashes_and_no_overwrite(self):
        # An artificial file bundle exercises safe I/O; it is never imported or loaded as a checkpoint.
        source = self.root / "fake-source"
        (source / "dinov2").mkdir(parents=True)
        (source / "hubconf.py").write_bytes(b"# artificial inert fixture\n")
        (source / "dinov2/__init__.py").write_bytes(b"# artificial inert fixture\n")
        weight = b"not a torch checkpoint: artificial extraction fixture"
        provenance = {
            "source_url": "https://github.com/facebookresearch/dinov2",
            "source_revision": "artificial-extraction-fixture",
            "weights_url": "https://example.invalid/artificial-fixture",
            "license": "artificial fixture",
            "reviewed_at": "artificial test",
            "exposure_audit": "No model/data contents in this fixture",
            "weights_sha256": hashlib.sha256(weight).hexdigest(),
            "source_tree_sha256": source_tree_hash(source),
            "pretraining": "generic_dinov2_lvd142m",
            "competition_finetuned": False,
            "gold_used_for_tuning": False,
        }
        path = self.root / "fake-generic.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(profiler.WEIGHTS_MEMBER, weight)
            archive.writestr(profiler.PROVENANCE_MEMBER, json.dumps(provenance))
            archive.writestr(profiler.SOURCE_PREFIX + "LICENSE", b"artificial fixture notice")
            for name in ("hubconf.py", "dinov2/__init__.py"):
                archive.writestr(profiler.SOURCE_PREFIX + name, (source / name).read_bytes())
            archive.writestr("manifest/weak.csv", b"must not extract")
            archive.writestr("another-model/best.pt", b"must not extract")
        repo, weights, actual, receipt = profiler.extract_generic_assets(path, self.root / "extracted")
        self.assertEqual(actual, provenance)
        self.assertEqual(weights.read_bytes(), weight)
        self.assertEqual(source_tree_hash(repo), provenance["source_tree_sha256"])
        self.assertFalse((self.root / "extracted/manifest").exists())
        self.assertFalse(receipt["competition_checkpoints_extracted"])
        with self.assertRaisesRegex(ValueError, "immutable"):
            profiler.extract_generic_assets(path, self.root / "extracted")

    def test_prefix_and_norm_frozen_last2_gradient_and_checkpoint_equivalence(self):
        torch.manual_seed(91)
        config = copy.deepcopy(CONFIG)
        config["encoder"]["precision"] = "fp32"  # CPU stand-in test; real CUDA profiler keeps original fp16.
        model = NativeTripletMIL(ArtificialEncoder(), config, trainable_blocks=2, window_chunk=1)
        model.eval()
        image = torch.rand(2, 3, 392, 392)
        output = model.encode_normalized(image)
        output.square().mean().backward()
        baseline = {name: value.grad.clone() for name, value in model.encoder.named_parameters() if value.requires_grad}
        self.assertTrue(baseline)
        self.assertTrue(all(name.startswith(("blocks.10.", "blocks.11.")) for name in baseline))
        self.assertTrue(all(value.grad is None for value in model.encoder.norm.parameters()))
        self.assertTrue(all(value.grad is None for value in model.encoder.patch_embed.parameters()))
        model.zero_grad(set_to_none=True)
        model.checkpoint_last2 = True
        second = model.encode_normalized(image)
        second.square().mean().backward()
        self.assertTrue(torch.equal(output, second))
        for name, value in model.encoder.named_parameters():
            if value.requires_grad:
                self.assertTrue(torch.equal(value.grad, baseline[name]), name)
        model.train()
        self.assertTrue(model.head.training)
        self.assertFalse(model.encoder.training)
        self.assertTrue(model.parameter_contract()["final_norm_frozen"])

    def test_fixed_3x48_microbatch_head_padding_and_all_masked_rejection(self):
        torch.manual_seed(17)
        config = copy.deepcopy(CONFIG)
        config["encoder"]["precision"] = "fp32"
        encoder = ArtificialEncoder()
        model = NativeTripletMIL(encoder, config, trainable_blocks=0, window_chunk=1).eval()
        inputs, _ = profiler.synthetic_study(1, "cpu", config["seed"])
        output = model(**inputs)
        self.assertEqual(output.shape, (1, 12))
        self.assertTrue(torch.isfinite(output).all())
        changed = dict(inputs)
        changed["images"] = inputs["images"].clone()
        changed["images"][~inputs["mask"]] = 249
        changed["positions"] = inputs["positions"].clone().masked_fill(~inputs["mask"], float("nan"))
        self.assertTrue(torch.equal(output, model(**changed)))
        inputs["mask"][:] = False
        with self.assertRaisesRegex(ValueError, "valid slice"):
            model(**inputs)


if __name__ == "__main__":
    unittest.main()
