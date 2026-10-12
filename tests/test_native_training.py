"""Actual production loss/checkpoint/cache contracts tested with artificial inputs."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from rsna_knee.contracts import ID, TARGETS, write_csv
from rsna_knee.finetune_model import NativeTripletMIL
from rsna_knee.native_training import (
    apply_native_checkpoint,
    backward_observed_micro,
    effective_observation_count,
    load_native_checkpoint,
    make_native_checkpoint,
    observed_bce_sum,
    prefix_digest,
    preflight_native_training,
    read_prediction_ids,
    validate_native_config,
    validate_resource_gate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/experiments/f005-dinov2-native-last2.json").read_text(encoding="utf-8"))
spec = importlib.util.spec_from_file_location("artificial_last2_tests", ROOT / "tests/test_dinov2_last2.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


class NativeTrainContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="native-training-artificial-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_effective_masked_gradient_equals_fullbatch_with_unequal_labels_and_short_tail(self):
        torch.manual_seed(203)
        for count in (16, 3):
            baseline = torch.nn.Linear(4, 12).double()
            accumulated, naive = copy.deepcopy(baseline), copy.deepcopy(baseline)
            inputs = torch.rand(count, 4, dtype=torch.float64)
            targets = torch.rand(count, 12, dtype=torch.float64)
            mask = torch.zeros(count, 12, dtype=torch.bool)
            for index in range(count):
                mask[index, : (index % 12) + 1] = True
            total = int(mask.sum())
            full = observed_bce_sum(baseline(inputs), targets, mask) / total
            full.backward()
            total_numerator = 0.0
            for index in range(count):
                micro_targets = targets[index : index + 1].clone()
                micro_targets[~mask[index : index + 1]] = float(
                    "nan"
                )  # These are not negatives, even with bad placeholders.
                total_numerator += backward_observed_micro(
                    accumulated(inputs[index : index + 1]), micro_targets, mask[index : index + 1], total
                )
                wrong = observed_bce_sum(
                    naive(inputs[index : index + 1]), targets[index : index + 1], mask[index : index + 1]
                )
                (wrong / int(mask[index].sum()) / count).backward()
            self.assertAlmostEqual(total_numerator / total, float(full.detach()), places=14)
            for left, right in zip(baseline.parameters(), accumulated.parameters()):
                torch.testing.assert_close(left.grad, right.grad, rtol=1e-13, atol=1e-14)
            self.assertGreater(float((baseline.weight.grad - naive.weight.grad).abs().max()), 1e-4)

    def test_unobserved_labels_and_zero_effective_batch_rejected(self):
        row = {ID: "synthetic", **{target: "" for target in TARGETS}}
        with self.assertRaisesRegex(ValueError, "zero observed"):
            effective_observation_count([row])
        row[TARGETS[0]], row[TARGETS[1]] = "0", "1"
        self.assertEqual(effective_observation_count([row]), 2)
        logits = torch.zeros(1, 12, requires_grad=True)
        with self.assertRaisesRegex(ValueError, "No observed"):
            observed_bce_sum(logits, torch.zeros_like(logits), torch.zeros_like(logits, dtype=torch.bool))

    def test_fixed_config_only_trainability_diff_and_prior_LR_locked(self):
        self.assertEqual(validate_native_config(CONFIG), CONFIG)
        frozen = json.loads(
            (ROOT / "configs/experiments/f005-dinov2-native-frozen-control.json").read_text(encoding="utf-8")
        )
        left, right = copy.deepcopy(frozen), copy.deepcopy(CONFIG)
        left.pop("reason"), right.pop("reason")
        left["online"].pop("trainable_blocks"), right["online"].pop("trainable_blocks")
        self.assertEqual(left, right)
        for location, name, value in (
            ("online", "microbatch_studies", 16),
            ("online", "encoder_learning_rate", 3e-4),
            ("train", "learning_rate", 1e-4),
            ("train", "selection", "auc"),
        ):
            changed = copy.deepcopy(CONFIG)
            changed[location][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_native_config(changed)

    def test_artificial_suffix_head_checkpoint_roundtrip_and_strict_corruptions(self):
        torch.manual_seed(71)
        original = fixtures.ArtificialEncoder()
        source = NativeTripletMIL(copy.deepcopy(original), copy.deepcopy(CONFIG), trainable_blocks=2, window_chunk=2)
        destination = NativeTripletMIL(
            copy.deepcopy(original), copy.deepcopy(CONFIG), trainable_blocks=2, window_chunk=2
        )
        digest = prefix_digest(source)
        with torch.no_grad():
            next(source.encoder.blocks[10].parameters()).add_(0.0002)
            source.head.output_bias.add_(0.1)
        provenance = {"artificial_only": True}
        checkpoint = make_native_checkpoint(
            source,
            provenance,
            fold=0,
            epoch=0,
            weak_bce=0.0,
            prefix_sha256=digest,
            audit={"synthetic": True},
            artificial=True,
        )
        path = self.root / "artificial.pt"
        torch.save(checkpoint, path)
        with self.assertRaisesRegex(ValueError, "Artificial checkpoints"):
            load_native_checkpoint(path)
        loaded = load_native_checkpoint(path, allow_artificial=True)
        apply_native_checkpoint(destination, loaded, provenance)
        for name, tensor in source.state_dict().items():
            self.assertTrue(torch.equal(tensor, destination.state_dict()[name]), name)
        for change in ("extra-prefix", "missing-suffix", "wrong-head", "nonfinite", "dtype", "target-order"):
            invalid = copy.deepcopy(checkpoint)
            if change == "extra-prefix":
                invalid["encoder_suffix"]["norm.weight"] = source.encoder.norm.weight.detach().clone()
            elif change == "missing-suffix":
                invalid["encoder_suffix"].pop(next(iter(invalid["encoder_suffix"])))
            elif change == "wrong-head":
                invalid["head"]["output_bias"] = torch.zeros(13)
            elif change == "nonfinite":
                invalid["head"]["output_bias"][0] = float("nan")
            elif change == "dtype":
                invalid["head"]["output_bias"] = invalid["head"]["output_bias"].half()
            else:
                invalid["targets"] = list(reversed(TARGETS))
            if change == "target-order":
                torch.save(invalid, self.root / "wrong.pt")
                with self.assertRaisesRegex(ValueError, "target order"):
                    load_native_checkpoint(self.root / "wrong.pt", allow_artificial=True)
            else:
                with self.subTest(change=change), self.assertRaises(ValueError):
                    apply_native_checkpoint(destination, invalid, provenance)
        with torch.no_grad():
            source.encoder.norm.bias.add_(0.001)
        with self.assertRaisesRegex(ValueError, "frozen generic prefix"):
            make_native_checkpoint(
                source, provenance, fold=0, epoch=0, weak_bce=0.0, prefix_sha256=digest, audit={}, artificial=True
            )

    def test_CUDA_first_guard_and_real_resource_gate_cannot_use_template(self):
        with patch("rsna_knee.native_training.ensure_cuda", side_effect=ValueError("CUDA first")):
            with self.assertRaisesRegex(ValueError, "CUDA first"):
                preflight_native_training(self.root, self.root / "missing-index", {}, 0, self.root / "missing-gate")
        gate = ROOT / "artifacts/kaggle/f005-native-train-handoff-v1-20261008/resource-gate-template.json"
        with self.assertRaisesRegex(ValueError, "must pass"):
            validate_resource_gate(gate)

    def test_prediction_ID_only_csv_requires_no_Report_or_targets(self):
        write_csv(self.root / "test.csv", (ID,), [{ID: "synthetic-A"}, {ID: "synthetic-B"}])
        self.assertEqual(read_prediction_ids(self.root / "test.csv"), [{ID: "synthetic-A"}, {ID: "synthetic-B"}])
        write_csv(
            self.root / "with-report.csv", (ID, "Report"), [{ID: "synthetic-A", "Report": "must not reach model"}]
        )
        self.assertEqual(read_prediction_ids(self.root / "with-report.csv"), [{ID: "synthetic-A"}])

    def test_Gold_UID_or_group_conflict_rejected_before_teacher_conversion(self):
        manifest = {"seed": 20261002, "n_folds": 5, "label_provenance": {"gold_used_for_tuning": False}}
        (self.root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        for conflict in ("UID", "group"):
            weak = [
                {
                    ID: "A",
                    "source": "report_weak",
                    "group_id": "groupA",
                    "fold": "1",
                    **{t: "SENTINEL-MUST-NOT-PARSE" for t in TARGETS},
                },
                {
                    ID: "B",
                    "source": "report_weak",
                    "group_id": "groupB",
                    "fold": "0",
                    **{t: "SENTINEL-MUST-NOT-PARSE" for t in TARGETS},
                },
            ]
            gold = [
                {
                    ID: "A" if conflict == "UID" else "G",
                    "source": "official_image_review",
                    "group_id": "other" if conflict == "UID" else "groupA",
                }
            ]
            write_csv(self.root / "weak.csv", (ID, "source", "group_id", "fold", *TARGETS), weak)
            write_csv(self.root / "gold.csv", (ID, "source", "group_id"), gold)
            with (
                patch("rsna_knee.native_training.ensure_cuda"),
                patch("rsna_knee.native_training.validate_resource_gate", return_value={}),
                patch("rsna_knee.native_training.validate_fold_audit", return_value={}),
                patch(
                    "rsna_knee.native_training.label_tensors",
                    side_effect=AssertionError("teacher conversion forbidden"),
                ),
                patch("rsna_knee.native_training.read_csv", side_effect=AssertionError("teacher pass forbidden")),
            ):
                with self.subTest(conflict=conflict), self.assertRaisesRegex(ValueError, "leaked"):
                    preflight_native_training(self.root, self.root / "index", CONFIG, 0, self.root / "gate")

    def test_real_checkpoint_requires_positive_integer_epoch(self):
        model = NativeTripletMIL(
            fixtures.ArtificialEncoder(), copy.deepcopy(CONFIG), trainable_blocks=2, window_chunk=2
        )
        value = make_native_checkpoint(
            model,
            {"fixture": True},
            fold=1,
            epoch=1,
            weak_bce=0.69,
            prefix_sha256=prefix_digest(model),
            audit={"fixture": True},
        )
        for epoch in (0, -1, True, 1.5):
            invalid = copy.deepcopy(value)
            invalid["epoch"] = epoch
            torch.save(invalid, self.root / "invalid-epoch.pt")
            with self.subTest(epoch=epoch), self.assertRaisesRegex(ValueError, "checkpoint epoch"):
                load_native_checkpoint(self.root / "invalid-epoch.pt")


if __name__ == "__main__":
    unittest.main()
