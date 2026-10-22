"""Synthetic study-level gamma, RNG isolation and inference compatibility contracts."""
from __future__ import annotations

import copy
import importlib.util
import json
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from rsna_knee.finetune_model import STUDY_GAMMA_CONFIG, NativeTripletMIL, validate_study_gamma
from rsna_knee.native_training_fast16 import (
    apply_native_checkpoint,
    make_native_checkpoint,
    prefix_digest,
    validate_native_config,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gamma_artificial_encoder_fixture", ROOT / "tests/test_dinov2_last2.py")
FIXTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURE)
BASE = json.loads((ROOT / "configs/experiments/f006-dinov2-native-last2-late-cosine.json").read_text(encoding="utf-8"))
GAMMA = json.loads((ROOT / "configs/experiments/f009-dinov2-native-last2-late-cosine-study-gamma.json").read_text(encoding="utf-8"))


def model(config=GAMMA, chunk=16):
    return NativeTripletMIL(FIXTURE.ArtificialEncoder(), copy.deepcopy(config), trainable_blocks=2, window_chunk=chunk)


def inputs(studies=1):
    images = torch.zeros((studies, 3, 48, 3, 392, 392), dtype=torch.uint8)
    mask = torch.zeros((studies, 3, 48), dtype=torch.bool)
    for study in range(studies):
        for series, window in ((0, 0), (1, 2), (2, 3)):
            mask[study, series, window] = True
            images[study, series, window] = torch.tensor([32, 128, 224], dtype=torch.uint8)[:, None, None]
            images[study, series, window, :, 0, 0] = 0
            images[study, series, window, :, 0, 1] = 255
    return {"images": images, "mask": mask, "positions": torch.zeros(studies, 3, 48),
            "planes": torch.zeros(studies, 3, dtype=torch.long), "flags": torch.zeros(studies, 3, 2)}


class StudyGammaContracts(unittest.TestCase):
    def test_config_is_only_augmentation_factor_and_reason(self):
        self.assertEqual(validate_native_config(GAMMA), GAMMA)
        candidate, baseline = copy.deepcopy(GAMMA), copy.deepcopy(BASE)
        self.assertEqual(candidate.pop("augmentation"), STUDY_GAMMA_CONFIG)
        candidate.pop("reason"), baseline.pop("reason")
        self.assertEqual(candidate, baseline)
        self.assertEqual(validate_native_config(BASE), BASE)

    def test_schema_rejects_ranges_types_and_extra_controls(self):
        for change in ({}, {**STUDY_GAMMA_CONFIG, "minimum": .8}, {**STUDY_GAMMA_CONFIG, "maximum": float("nan")},
                       {**STUDY_GAMMA_CONFIG, "minimum": "0.85"}, {**STUDY_GAMMA_CONFIG, "probability": .5},
                       {**STUDY_GAMMA_CONFIG, "schema_version": "other"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_study_gamma(change)
        constant = copy.deepcopy(GAMMA)
        constant["train"].pop("lr_schedule")
        with self.assertRaisesRegex(ValueError, "F006"):
            validate_native_config(constant)

    def test_sampler_reproducible_private_rng_and_eval_does_not_draw(self):
        first, second = model(), model()
        first.train(), second.train()
        torch.manual_seed(479)
        global_before = torch.get_rng_state().clone()
        actual = first.sample_training_gamma(8, "cpu")
        self.assertTrue(torch.equal(global_before, torch.get_rng_state()))
        self.assertTrue(torch.equal(actual, second.sample_training_gamma(8, "cpu")))
        self.assertTrue(bool(((actual >= .85) & (actual <= 1.15)).all()))
        self.assertGreater(len(set(actual.tolist())), 1)
        private_before = first._gamma_generator.get_state().clone()
        first.eval()
        self.assertIsNone(first.sample_training_gamma(2, "cpu"))
        self.assertTrue(torch.equal(private_before, first._gamma_generator.get_state()))
        self.assertEqual(first.training_augmentation_contract()["draws"], 8)
        plain = model(BASE)
        self.assertIsNone(plain.sample_training_gamma(2, "cpu"))
        self.assertIsNone(plain.training_augmentation_contract())

    def test_same_gamma_for_all_windows_series_channels_before_normalization_and_exact_endpoints(self):
        net = model(chunk=2).train()
        batch = inputs(studies=2)
        captured = []

        def encode(value):
            captured.append(value.detach().clone())
            return value.mean((1, 2, 3))[:, None].expand(-1, 384)

        generator = torch.Generator().manual_seed(GAMMA["seed"])
        gamma = torch.empty(2).uniform_(.85, 1.15, generator=generator)
        with patch.object(net, "encode_normalized", side_effect=encode):
            output = net(**batch)
        self.assertEqual(tuple(output.shape), (2, 12))
        self.assertTrue(bool(torch.isfinite(output).all()))
        actual = torch.cat(captured)
        selected = batch["mask"].reshape(-1).nonzero().flatten()
        raw = batch["images"].reshape(-1, 3, 392, 392).index_select(0, selected).float().div(255)
        exponents = gamma.index_select(0, torch.div(selected, 144, rounding_mode="floor"))[:, None, None, None]
        mean = raw.new_tensor([.485, .456, .406])[None, :, None, None]
        std = raw.new_tensor([.229, .224, .225])[None, :, None, None]
        expected = (raw.pow(exponents) - mean) / std
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual[:, :, 0, 0], (-mean / std)[0, :, 0, 0].expand(6, -1)))
        self.assertTrue(torch.equal(actual[:, :, 0, 1], ((1 - mean) / std)[0, :, 0, 0].expand(6, -1)))
        self.assertEqual(net.training_augmentation_contract()["draws"], 2)
        private_before = net._gamma_generator.get_state().clone()
        captured.clear()
        net.eval()
        with patch.object(net, "encode_normalized", side_effect=encode):
            net(**batch)
        self.assertTrue(torch.equal(torch.cat(captured), (raw - mean) / std))
        self.assertTrue(torch.equal(private_before, net._gamma_generator.get_state()))

    def test_chunking_does_not_resample_gamma(self):
        one, many = model(chunk=1).train(), model(chunk=16).train()
        batch = inputs()
        captured = [[], []]
        for index, net in enumerate((one, many)):
            def encode(value, index=index):
                captured[index].append(value.detach().clone())
                return value.mean((1, 2, 3))[:, None].expand(-1, 384)
            with patch.object(net, "encode_normalized", side_effect=encode):
                net(**batch)
            self.assertEqual(net.training_augmentation_contract()["draws"], 1)
        self.assertTrue(torch.equal(torch.cat(captured[0]), torch.cat(captured[1])))

    def test_gamma_checkpoint_constructor_and_eval_roundtrip(self):
        torch.manual_seed(41)
        original = model().eval()
        destination = model().eval()
        state = make_native_checkpoint(original, {"artificial_only": True}, fold=0, epoch=0, weak_bce=.69,
                                       prefix_sha256=prefix_digest(original), audit={"synthetic": True}, artificial=True)
        # Both models must share exactly the same untrained generic prefix.
        destination.encoder.load_state_dict(original.encoder.state_dict())
        apply_native_checkpoint(destination, state, {"artificial_only": True})
        batch = inputs()
        with torch.no_grad():
            expected, observed = original(**batch), destination(**batch)
        self.assertTrue(torch.equal(expected, observed))
        self.assertEqual(original.training_augmentation_contract()["draws"], 0)
        self.assertEqual(destination.training_augmentation_contract()["draws"], 0)

    def test_historical_F006_default_and_gamma_eval_logits_are_exact(self):
        snapshot = ROOT / "artifacts/research/20261017-f009-gamma-preparation-v1/snapshots/finetune_model.py"
        if not snapshot.is_file():
            self.skipTest("Historical source snapshot is local preparation evidence")
        old = types.ModuleType("rsna_knee._historical_f006_finetune_model")
        old.__package__ = "rsna_knee"
        exec(compile(snapshot.read_bytes(), str(snapshot), "exec"), old.__dict__)
        torch.manual_seed(73)
        encoder = FIXTURE.ArtificialEncoder()
        legacy = old.NativeTripletMIL(copy.deepcopy(encoder), copy.deepcopy(BASE), trainable_blocks=2, window_chunk=16)
        current = NativeTripletMIL(copy.deepcopy(encoder), copy.deepcopy(BASE), trainable_blocks=2, window_chunk=16)
        gamma = NativeTripletMIL(copy.deepcopy(encoder), copy.deepcopy(GAMMA), trainable_blocks=2, window_chunk=16)
        current.load_state_dict(legacy.state_dict()), gamma.load_state_dict(legacy.state_dict())
        batch = inputs()
        for training in (False, True):
            legacy.train(training), current.train(training)
            state = torch.get_rng_state().clone()
            left = legacy(**batch)
            torch.set_rng_state(state)
            right = current(**batch)
            self.assertTrue(torch.equal(left, right))
        legacy.eval(), gamma.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(legacy(**batch), gamma(**batch)))


if __name__ == "__main__":
    unittest.main()
