"""Independent F010 CPU stand-in and fail-closed checks; no real assets or GPU."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rsna_knee import f010_last4_training as candidate
from rsna_knee import native_training_fast16 as control
from rsna_knee.f010_last4_model import F010Last4MIL
from rsna_knee.finetune_model import NativeTripletMIL

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("f010_independent_artificial", ROOT / "tests/test_dinov2_last2.py")
FIXTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURE)
CONFIG = json.loads((ROOT / "configs/experiments/f010-dinov2-native-last4-late-cosine.json").read_text(encoding="utf-8"))
BASE = json.loads((ROOT / "configs/experiments/f006-dinov2-native-last2-late-cosine.json").read_text(encoding="utf-8"))


class F010IndependentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(61)
        self.temp = tempfile.TemporaryDirectory(prefix="f010-independent-artificial-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def model(self, encoder=None):
        return F010Last4MIL(encoder or FIXTURE.ArtificialEncoder(), copy.deepcopy(CONFIG),
                           trainable_blocks=4, window_chunk=16, checkpoint_last2=True)

    def test_control_config_one_factor_only_and_gamma_prohibited(self):
        left, right = copy.deepcopy(BASE), copy.deepcopy(CONFIG)
        for value in (left, right):
            value.pop("reason")
            value["online"].pop("schema_version")
            value["online"].pop("trainable_blocks")
        self.assertEqual(left, right)
        for field, value in (("augmentation", {}), ("augmentation", {"minimum": .85, "maximum": 1.15})):
            bad = copy.deepcopy(CONFIG)
            bad[field] = value
            with self.assertRaisesRegex(ValueError, "augmentation-free"):
                candidate.validate_native_config(bad)

    def test_suffix_gradient_and_frozen_prefix_norm_encode_equal_to_control(self):
        encoder = FIXTURE.ArtificialEncoder()
        model = self.model(copy.deepcopy(encoder)).eval()
        baseline = NativeTripletMIL(copy.deepcopy(encoder), copy.deepcopy(BASE), trainable_blocks=2,
                                    window_chunk=16, checkpoint_last2=True).eval()
        images = torch.rand(2, 3, 8, 8)
        actual, original = model.encode_normalized(images), baseline.encode_normalized(images)
        self.assertTrue(torch.equal(actual, original))
        before = candidate.prefix_digest(model)
        actual.square().mean().backward()
        selected = {name: tensor for name, tensor in model.encoder.named_parameters() if tensor.requires_grad}
        self.assertEqual({name.split(".")[1] for name in selected}, {"8", "9", "10", "11"})
        self.assertTrue(all(t.grad is not None and torch.isfinite(t.grad).all() for t in selected.values()))
        self.assertTrue(all(t.grad is None for t in model.encoder.parameters() if not t.requires_grad))
        self.assertTrue(all(not t.requires_grad for t in model.encoder.norm.parameters()))
        self.assertEqual(before, candidate.prefix_digest(model))

    def test_missing_extra_prefix_and_nonfinite_suffix_refused(self):
        model = self.model()
        receipt = candidate.make_native_checkpoint(model, {"artificial": True}, fold=0, epoch=0, weak_bce=0.,
                   prefix_sha256=candidate.prefix_digest(model), audit={"artificial": True}, artificial=True)
        for change in ("omit8", "extra7", "norm", "nonfinite", "dtype", "oldschema", "stalesource"):
            bad = copy.deepcopy(receipt)
            key = next(k for k in bad["encoder_suffix"] if k.startswith("blocks.8."))
            if change == "omit8":
                del bad["encoder_suffix"][key]
            elif change in ("extra7", "norm"):
                newkey = "blocks.7.0.weight" if change == "extra7" else "norm.weight"
                bad["encoder_suffix"][newkey] = model.encoder.state_dict()[newkey].clone()
            elif change == "nonfinite":
                bad["encoder_suffix"][key].view(-1)[0] = float("nan")
            elif change == "dtype":
                bad["encoder_suffix"][key] = bad["encoder_suffix"][key].double()
            elif change == "oldschema":
                bad["schema_version"] = control.CHECKPOINT_SCHEMA
            else:
                bad["source_sha256"]["f010_last4_model.py"] = "0" * 64
            with self.subTest(change=change), self.assertRaises(ValueError):
                candidate.apply_native_checkpoint(model, bad, {"artificial": True})

    def test_artificial_checkpoint_target_order_and_real_inference_refusal(self):
        model = self.model()
        value = candidate.make_native_checkpoint(model, {"artificial": True}, fold=0, epoch=0, weak_bce=0.,
                prefix_sha256=candidate.prefix_digest(model), audit={"artificial": True}, artificial=True)
        path = self.root / "artificial.pt"
        torch.save(value, path)
        with self.assertRaisesRegex(ValueError, "Artificial checkpoints"):
            candidate.load_native_checkpoint(path)
        value["targets"] = value["targets"][::-1]
        torch.save(value, path)
        with self.assertRaisesRegex(ValueError, "target order"):
            candidate.load_native_checkpoint(path, allow_artificial=True)

    def test_weak_mask_and_effective_gradient_exactly_match_control(self):
        y, mask = torch.rand(2, 12), torch.rand(2, 12) > .5
        y[~mask] = float("nan")
        a = torch.randn(2, 12, requires_grad=True)
        b = a.detach().clone().requires_grad_()
        denominator = int(mask.sum())
        loss_a = candidate.backward_observed_micro(a, y, mask, denominator)
        loss_b = control.backward_observed_micro(b, y, mask, denominator)
        self.assertEqual(loss_a, loss_b)
        self.assertTrue(torch.equal(a.grad, b.grad))
        self.assertTrue(torch.equal(a.grad[~mask], torch.zeros_like(a.grad[~mask])))

    def test_real_entry_points_cuda_first_before_paths_outputs(self):
        sentinel = ValueError("independent CUDA sentinel")
        for function, args, kwargs in (
            (candidate.preflight_native_training, (self.root / "no", None, None, 0, None), {}),
            (candidate.train_native, (self.root / "no", None, self.root / "output", None, 0, None, None, None, None), {}),
            (candidate.predict_native, (self.root / "no", None, self.root / "output", None, None, None), {"reader": None}),
        ):
            with self.subTest(function=function.__name__), patch.object(candidate, "ensure_cuda", side_effect=sentinel):
                with self.assertRaisesRegex(ValueError, "independent CUDA sentinel"):
                    function(*args, **kwargs)
        self.assertFalse((self.root / "output").exists())
        spec = importlib.util.spec_from_file_location("f010_independent_cli", ROOT / "scripts/train_f010_last4_fast16.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with patch.object(cli, "ensure_cuda", side_effect=sentinel):
            with self.assertRaisesRegex(ValueError, "independent CUDA sentinel"):
                cli.qa(SimpleNamespace(output=self.root / "output"))


if __name__ == "__main__":
    unittest.main()
