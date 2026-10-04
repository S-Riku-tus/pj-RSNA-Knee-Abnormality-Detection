"""Artificial model/config/normalization checks; no public weights or competition images."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsna_knee.contracts import RESNET18_V1_SHA256_PREFIX, load_config, validate_model_config

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "baseline.json"
HAS_TORCH = all(importlib.util.find_spec(name) is not None for name in ("torch", "torchvision", "numpy"))
SYNTHETIC_OFFICIAL_DIGEST = RESNET18_V1_SHA256_PREFIX + "0" * 56


class ModelConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_legacy_config_remains_unmodified(self):
        legacy = copy.deepcopy(self.config)
        for name in (
            "initialization",
            "input_normalization",
            "bn_running_stats",
            "pretrained_path",
            "pretrained_sha256",
        ):
            legacy["model"].pop(name, None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-config.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")
            actual = load_config(path)
        self.assertEqual(actual, legacy)
        self.assertNotIn("initialization", actual["model"])
        self.assertNotIn("input_normalization", actual["model"])

    def test_invalid_modes_and_dropout_fail_before_loading(self):
        for key, value in (
            ("initialization", "download"),
            ("input_normalization", "center_crop"),
            ("bn_running_stats", "anything"),
            ("dropout", -0.1),
            ("dropout", 1),
            ("dropout", True),
            ("dropout", float("nan")),
        ):
            with self.subTest(key=key, value=value):
                config = copy.deepcopy(self.config)
                config["model"][key] = value
                with self.assertRaises(ValueError):
                    validate_model_config(config)

    def test_pretrained_requires_local_file_full_official_hash_and_matching_mode(self):
        config = copy.deepcopy(self.config)
        config["model"]["initialization"] = "imagenet1k_v1"
        with self.assertRaisesRegex(ValueError, "local pretrained_path"):
            validate_model_config(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artificial.pth"
            path.write_bytes(b"artificial, not actual ImageNet weights")
            config["model"]["pretrained_path"] = str(path)
            for digest in (None, "f37072fd", "z" * 64, "0" * 64):
                config["model"]["pretrained_sha256"] = digest
                with self.subTest(digest=digest), self.assertRaises(ValueError):
                    validate_model_config(config)
            config["model"]["pretrained_sha256"] = SYNTHETIC_OFFICIAL_DIGEST
            validate_model_config(config)  # Content SHA is verified when the encoder is initialized.
            config["model"]["initialization"] = "random"
            with self.assertRaisesRegex(ValueError, "Random initialization"):
                validate_model_config(config)
            config["model"]["initialization"] = "imagenet1k_v1"
            config["model"]["pretrained_path"] = str(path.with_name("absent.pth"))
            with self.assertRaisesRegex(ValueError, "does not exist"):
                validate_model_config(config)


@unittest.skipUnless(HAS_TORCH, "Optional torch/torchvision/numpy unavailable")
class NormalizationTests(unittest.TestCase):
    def test_legacy_is_exact_and_does_not_change_cache(self):
        import numpy as np
        import torch

        from rsna_knee.inputs import normalize_images

        images = np.arange(256, dtype=np.uint8).reshape(1, 1, 16, 16).repeat(3, axis=1)
        original = images.copy()
        expected = torch.from_numpy(images.astype(np.float32) / 127.5 - 1)
        for model in ({}, {"input_normalization": "legacy"}):
            actual = normalize_images(images, {"model": model})
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(actual.dtype, torch.float32)
        np.testing.assert_array_equal(images, original)

    def test_imagenet_normalizes_each_adjacent_slice_channel(self):
        import numpy as np
        import torch

        from rsna_knee.inputs import normalize_images

        images = np.array([0, 127, 255], dtype=np.uint8).reshape(1, 3, 1, 1).repeat(2, axis=2)
        actual = normalize_images(images, {"model": {"input_normalization": "imagenet"}})
        values = torch.tensor([0, 127, 255], dtype=torch.float32) / 255
        expected = (values - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
        torch.testing.assert_close(actual[0, :, 0, 0], expected, rtol=0, atol=0)
        self.assertEqual(tuple(actual.shape), tuple(images.shape))

    def test_invalid_input_and_unknown_mode_are_rejected(self):
        import numpy as np

        from rsna_knee.inputs import normalize_images

        for images in (np.zeros((1, 3, 4, 4)), np.zeros((3, 4, 4), np.uint8), np.zeros((1, 1, 4, 4), np.uint8)):
            with self.assertRaises(ValueError):
                normalize_images(images, {"model": {}})
        with self.assertRaisesRegex(ValueError, "normalization"):
            normalize_images(np.zeros((1, 3, 4, 4), np.uint8), {"model": {"input_normalization": "unknown"}})


@unittest.skipUnless(HAS_TORCH, "Optional torch/torchvision/numpy unavailable")
class EncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        torch.set_num_threads(1)

    def test_random_constructor_matches_old_parameters_rng_and_checkpoint(self):
        import torch
        from torch import nn
        from torchvision.models import resnet18

        from rsna_knee.model import KneeMIL

        class LegacyKneeMIL(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = resnet18(weights=None)
                self.encoder.fc = nn.Identity()
                self.dropout = nn.Dropout(0.2)
                self.attention = nn.Linear(512, 12)
                self.finding_weight = nn.Parameter(torch.empty(12, 512))
                self.bias = nn.Parameter(torch.zeros(12))
                nn.init.normal_(self.finding_weight, std=0.01)

        torch.manual_seed(42)
        legacy = LegacyKneeMIL()
        previous_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(42)
        with patch("torch.hub.load_state_dict_from_url", side_effect=AssertionError("No downloads")):
            model = KneeMIL()
        self.assertTrue(torch.equal(torch.random.get_rng_state(), previous_rng))
        self.assertEqual(model.state_dict().keys(), legacy.state_dict().keys())
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, legacy.state_dict()[key]), key)
        model.load_state_dict(legacy.state_dict(), strict=True)
        model.eval()
        with torch.inference_mode():
            scores = model(torch.zeros(1, 2, 3, 32, 32), torch.ones(1, 2, dtype=torch.bool))
        self.assertEqual(tuple(scores.shape), (1, 12))
        self.assertTrue(torch.isfinite(scores).all())

    def test_local_initialization_is_strict_preserves_head_and_rng(self):
        import torch
        from torchvision.models import resnet18

        from rsna_knee.model import KneeMIL

        synthetic = resnet18(weights=None).state_dict()
        synthetic["conv1.weight"] = torch.full_like(synthetic["conv1.weight"], 0.123)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artificial-full-resnet.pt"
            torch.save(synthetic, path)
            model = KneeMIL()
            head = {key: value.clone() for key, value in model.state_dict().items() if not key.startswith("encoder.")}
            original_rng = torch.random.get_rng_state().clone()
            # Only hash identity is mocked; strict loading uses an actual artificial 1000-class state_dict.
            with patch("rsna_knee.model.sha256", return_value=SYNTHETIC_OFFICIAL_DIGEST):
                result = model.initialize_encoder(path, SYNTHETIC_OFFICIAL_DIGEST)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), original_rng))
            self.assertEqual(result["name"], "imagenet1k_v1")
            self.assertTrue(result["strict_load_including_fc"])
            self.assertTrue(torch.equal(model.encoder.conv1.weight, synthetic["conv1.weight"]))
            self.assertNotIn("encoder.fc.weight", model.state_dict())
            for key, value in head.items():
                self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
            broken = dict(synthetic)
            del broken["fc.weight"]
            torch.save(broken, path)
            before = {key: value.clone() for key, value in model.state_dict().items()}
            with patch("rsna_knee.model.sha256", return_value=SYNTHETIC_OFFICIAL_DIGEST):
                with self.assertRaises(RuntimeError):
                    model.initialize_encoder(path, SYNTHETIC_OFFICIAL_DIGEST)
            for key, value in before.items():
                self.assertTrue(torch.equal(value, model.state_dict()[key]), key)

    def test_bad_weight_hash_fails_without_loading_file(self):
        from rsna_knee.model import KneeMIL

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "not-a-checkpoint.pt"
            path.write_bytes(b"artificial")
            model = KneeMIL()
            with patch("torch.load", side_effect=AssertionError("Hash must be checked first")):
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    model.initialize_encoder(path, SYNTHETIC_OFFICIAL_DIGEST)
            with self.assertRaisesRegex(ValueError, "official"):
                model.initialize_encoder(path, "0" * 64)

    def test_bn_freeze_survives_train_and_preserves_affine_gradients(self):
        import torch
        from torch import nn

        from rsna_knee.model import KneeMIL

        model = KneeMIL(bn_running_stats="freeze").train()
        bns = [module for module in model.encoder.modules() if isinstance(module, nn.BatchNorm2d)]
        before = [
            (module.running_mean.clone(), module.running_var.clone(), module.num_batches_tracked.clone())
            for module in bns
        ]
        scores = model(torch.randn(1, 3, 3, 32, 32), torch.ones(1, 3, dtype=torch.bool))
        scores.square().mean().backward()
        for module, (mean, variance, count) in zip(bns, before):
            self.assertFalse(module.training)
            self.assertTrue(module.weight.requires_grad)
            self.assertIsNotNone(module.weight.grad)
            self.assertTrue(torch.equal(module.running_mean, mean))
            self.assertTrue(torch.equal(module.running_var, variance))
            self.assertTrue(torch.equal(module.num_batches_tracked, count))
        model.eval().train()
        self.assertTrue(model.dropout.training)
        self.assertTrue(all(not module.training for module in bns))

    def test_invalid_constructor_modes_fail_fast(self):
        from rsna_knee.model import KneeMIL

        for kwargs in ({"dropout": -1}, {"dropout": True}, {"bn_running_stats": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                KneeMIL(**kwargs)

    def test_cuda_amp_forward_backward_with_normalized_artificial_windows(self):
        import numpy as np
        import torch

        from rsna_knee.inputs import normalize_images
        from rsna_knee.model import KneeMIL

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        images = np.random.default_rng(123).integers(0, 256, (4, 3, 32, 32), dtype=np.uint8)
        tensor = normalize_images(images, {"model": {"input_normalization": "imagenet"}}).unsqueeze(0).cuda()
        mask = torch.tensor([[True, True, False, False]], device="cuda")
        model = KneeMIL(bn_running_stats="freeze").cuda().train()
        with torch.autocast(device_type="cuda"):
            logits = model(tensor, mask)
            loss = logits.float().square().mean()
        loss.backward()
        self.assertEqual(tuple(logits.shape), (1, 12))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(model.finding_weight.grad).all())
        changed = tensor.clone()
        changed[:, 2:] = 999
        model.eval()
        with torch.inference_mode():
            torch.testing.assert_close(model(tensor, mask), model(changed, mask))


if __name__ == "__main__":
    unittest.main()
