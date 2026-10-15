"""Actual production loss/checkpoint/cache contracts tested with artificial inputs."""

import copy
import hashlib
import importlib.util
import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from rsna_knee.contracts import ID, SUBMISSION_COLUMNS, TARGETS, index_studies, sha256, write_csv
from rsna_knee.finetune_model import NativeTripletMIL
from rsna_knee.metrics import evaluate_rows
from rsna_knee.native_training_fast16 import (
    ENGINEERING_LINEAGE,
    RESOURCE_GATE_SCHEMA,
    apply_native_checkpoint,
    backward_observed_micro,
    effective_observation_count,
    load_native_checkpoint,
    make_native_checkpoint,
    native_epoch_learning_rates,
    native_source_hashes,
    observed_bce_sum,
    prefix_digest,
    preflight_native_training,
    read_prediction_ids,
    retain_epoch_checkpoint,
    validate_native_config,
    validate_resource_gate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/experiments/f005-dinov2-native-last2-fast16.json").read_text(encoding="utf-8"))
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
            (ROOT / "configs/experiments/f005-dinov2-native-frozen-control-fast16.json").read_text(encoding="utf-8")
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
        source = NativeTripletMIL(copy.deepcopy(original), copy.deepcopy(CONFIG), trainable_blocks=2, window_chunk=16)
        destination = NativeTripletMIL(
            copy.deepcopy(original), copy.deepcopy(CONFIG), trainable_blocks=2, window_chunk=16
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
        with patch("rsna_knee.native_training_fast16.ensure_cuda", side_effect=ValueError("CUDA first")):
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
                patch("rsna_knee.native_training_fast16.ensure_cuda"),
                patch("rsna_knee.native_training_fast16.validate_resource_gate", return_value={}),
                patch("rsna_knee.native_training_fast16.validate_fold_audit", return_value={}),
                patch(
                    "rsna_knee.native_training_fast16.label_tensors",
                    side_effect=AssertionError("teacher conversion forbidden"),
                ),
                patch("rsna_knee.native_training_fast16.read_csv", side_effect=AssertionError("teacher pass forbidden")),
            ):
                with self.subTest(conflict=conflict), self.assertRaisesRegex(ValueError, "leaked"):
                    preflight_native_training(self.root, self.root / "index", CONFIG, 0, self.root / "gate")

    def test_real_checkpoint_requires_positive_integer_epoch(self):
        model = NativeTripletMIL(
            fixtures.ArtificialEncoder(), copy.deepcopy(CONFIG), trainable_blocks=2, window_chunk=16
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

    def named_optimizer(self):
        parameters = [
            torch.tensor([0.4, -0.2], dtype=torch.float64, requires_grad=True),
            torch.tensor([0.3, 0.1], dtype=torch.float64, requires_grad=True),
        ]
        optimizer = torch.optim.AdamW([
            {"params": [parameters[0]], "name": "head", "lr": CONFIG["train"]["learning_rate"]},
            {"params": [parameters[1]], "name": "last2", "lr": CONFIG["online"]["encoder_learning_rate"]},
        ], weight_decay=CONFIG["train"]["weight_decay"])
        return parameters, optimizer

    def test_default_constant_matches_existing_AdamW_updates_and_state_exactly(self):
        for explicit in (False, True):
            with self.subTest(explicit_constant=explicit):
                config = copy.deepcopy(CONFIG)
                if explicit:
                    config["train"]["lr_schedule"] = {"name": "constant"}
                self.assertIs(validate_native_config(config), config)
                original, baseline = self.named_optimizer()
                candidate, optimizer = self.named_optimizer()
                for epoch in range(1, 13):
                    initial_lrs = [group["lr"] for group in optimizer.param_groups]
                    rates = native_epoch_learning_rates(optimizer, config, epoch)
                    self.assertEqual(rates["learning_rate_multiplier"], 1.0)
                    for group, original_lr in zip(optimizer.param_groups, initial_lrs):
                        self.assertIs(group["lr"], original_lr)  # Constant never rewrites the original value.
                    for index, (left, right) in enumerate(zip(original, candidate)):
                        left.grad = torch.full_like(left, (epoch + index) * 0.03)
                        right.grad = left.grad.clone()
                    baseline.step()
                    optimizer.step()
                    self.assertEqual(baseline.state_dict()["param_groups"], optimizer.state_dict()["param_groups"])
                    for left, right in zip(original, candidate):
                        self.assertTrue(torch.equal(left, right))
                        for key, value in baseline.state[left].items():
                            self.assertTrue(torch.equal(value, optimizer.state[right][key]))

    def test_late_cosine_is_only_training_factor_preserves_group_ratio_and_first_six_updates(self):
        config = json.loads(
            (ROOT / "configs/experiments/f006-dinov2-native-last2-late-cosine.json").read_text(encoding="utf-8")
        )
        without_schedule = copy.deepcopy(config)
        without_schedule["train"].pop("lr_schedule")
        self.assertNotEqual(without_schedule["reason"], CONFIG["reason"])
        without_schedule["reason"] = CONFIG["reason"]  # Adoption reason is metadata, not a training factor.
        self.assertEqual(without_schedule, CONFIG)
        validate_native_config(config)
        original, baseline = self.named_optimizer()
        candidate, optimizer = self.named_optimizer()
        expected = [1.0] * 6 + [0.9397114317029974, 0.775, 0.55, 0.325, 0.16028856829700258, 0.1]
        for epoch, expected_factor in enumerate(expected, 1):
            rates = native_epoch_learning_rates(optimizer, config, epoch)
            self.assertAlmostEqual(rates["learning_rate_multiplier"], expected_factor, places=14)
            self.assertAlmostEqual(rates["learning_rates"]["head"] / rates["learning_rates"]["last2"], 30.0)
            for index, (left, right) in enumerate(zip(original, candidate)):
                left.grad = torch.full_like(left, (epoch + index) * 0.03)
                right.grad = left.grad.clone()
            baseline.step()
            optimizer.step()
            if epoch <= 6:
                for left, right in zip(original, candidate):
                    self.assertTrue(torch.equal(left, right))
            elif epoch == 7:
                for left, right in zip(original, candidate):
                    self.assertFalse(torch.equal(left, right))  # The changed LR reaches actual AdamW updates.
        self.assertEqual(rates["learning_rates"], {"head": 0.0003 * 0.1, "last2": 1e-5 * 0.1})

    def test_invalid_learning_rate_schedules_are_rejected_without_CUDA_or_optimizer_mutation(self):
        bad_schedules = (
            "late_cosine", {}, {"name": "linear"}, {"name": "constant", "min_factor": 0.1},
            {"name": "late_cosine", "constant_epochs": 5, "min_factor": 0.1},
            {"name": "late_cosine", "constant_epochs": 6.0, "min_factor": 0.1},
            {"name": "late_cosine", "constant_epochs": True, "min_factor": 0.1},
            {"name": "late_cosine", "constant_epochs": 6, "min_factor": 0.2},
            {"name": "late_cosine", "constant_epochs": 6, "min_factor": float("nan")},
        )
        _, optimizer = self.named_optimizer()
        initial_groups = copy.deepcopy(optimizer.state_dict()["param_groups"])
        with patch("rsna_knee.native_training_fast16.ensure_cuda", side_effect=AssertionError("CUDA forbidden")), \
             patch("torch.cuda.init", side_effect=AssertionError("CUDA forbidden")):
            for schedule in bad_schedules:
                config = copy.deepcopy(CONFIG)
                config["train"]["lr_schedule"] = schedule
                with self.subTest(schedule=schedule), self.assertRaisesRegex(ValueError, "learning rate schedule"):
                    validate_native_config(config)
                with self.assertRaisesRegex(ValueError, "learning rate schedule"):
                    native_epoch_learning_rates(optimizer, config, 7)
            for epoch in (0, 13, True, 1.5):
                with self.subTest(epoch=epoch), self.assertRaisesRegex(ValueError, "integer epoch"):
                    native_epoch_learning_rates(optimizer, CONFIG, epoch)
        self.assertEqual(initial_groups, optimizer.state_dict()["param_groups"])

    def test_QA_old_arm_configs_identical_and_candidate_last2_is_not_replaced(self):
        spec = importlib.util.spec_from_file_location("native_fast16_QA_CLI", ROOT / "scripts/train_native_dinov2_fast16.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        directory = ROOT / "configs/experiments"
        configs = {
            blocks: json.loads((directory / name).read_text(encoding="utf-8"))
            for blocks, name in ((0, "f005-dinov2-native-frozen-control-fast16.json"),
                                 (2, "f005-dinov2-native-last2-fast16.json"))
        }
        for requested in configs.values():
            for blocks in (0, 2):
                self.assertEqual(cli.qa_arm_config(requested, directory, blocks), configs[blocks])
        candidate = json.loads((directory / "f006-dinov2-native-last2-late-cosine.json").read_text(encoding="utf-8"))
        original = copy.deepcopy(candidate)
        self.assertIs(cli.qa_arm_config(candidate, directory, 2), candidate)
        self.assertEqual(cli.qa_arm_config(candidate, directory, 0), configs[0])
        self.assertEqual(candidate, original)

    def test_epoch_retention_roundtrips_both_schemas_and_preserves_best_last_and_RNG(self):
        from rsna_knee.regional_training_fast16 import (
            apply_regional_checkpoint,
            load_regional_checkpoint,
            make_regional_checkpoint,
        )
        from rsna_knee.regional_triplet_model import RegionalTripletMIL

        regional_config = json.loads(
            (ROOT / "configs/experiments/f005-dinov2-native-regional4x4-last2-fast16.json").read_text(encoding="utf-8")
        )
        provenance = {
            **{key: "artificial contract fixture" for key in (
                "source_url", "source_revision", "weights_url", "license", "reviewed_at", "exposure_audit"
            )},
            "weights_sha256": "1" * 64,
            "source_tree_sha256": "2" * 64,
            "pretraining": "generic_dinov2_lvd142m",
            "competition_finetuned": False,
            "gold_used_for_tuning": False,
        }
        for name, model_class, config, make, load, apply in (
            ("CLS", NativeTripletMIL, CONFIG, make_native_checkpoint, load_native_checkpoint, apply_native_checkpoint),
            ("REG", RegionalTripletMIL, regional_config, make_regional_checkpoint,
             load_regional_checkpoint, apply_regional_checkpoint),
        ):
            with self.subTest(schema=name):
                output = self.root / name
                output.mkdir()
                encoder = fixtures.ArtificialEncoder()
                if name == "REG":
                    # Inert artificial tensors satisfy the production suffix-count guard;
                    # no genuine DINO assets are loaded and no encoder forward is run.
                    for block in encoder.blocks[-2:]:
                        extra = 3550464 // 2 - sum(parameter.numel() for parameter in block.parameters())
                        block.register_parameter("artificial_contract_padding", torch.nn.Parameter(torch.zeros(extra)))
                model = model_class(copy.deepcopy(encoder), copy.deepcopy(config), trainable_blocks=2,
                                    window_chunk=16, checkpoint_last2=True).eval()
                digest = prefix_digest(model)
                best_bce = None
                original_epoch_sha = None
                for epoch, probability_value in ((1, 0.15), (2, 0.20), (3, 0.15)):
                    with torch.no_grad():
                        next(model.encoder.blocks[10].parameters()).add_(0.001)
                        model.head.output_bias.add_(0.01)
                    predictions = [
                        {ID: "artificial-negative", **dict.fromkeys(TARGETS, probability_value)},
                        {ID: "artificial-positive", **dict.fromkeys(TARGETS, 1 - probability_value)},
                    ]
                    truth = [
                        {ID: "artificial-negative", **dict.fromkeys(TARGETS, "0")},
                        {ID: "artificial-positive", **dict.fromkeys(TARGETS, "1")},
                    ]
                    predictions_path = output / f"epoch_{epoch:03d}_weak.csv"
                    write_csv(predictions_path, SUBMISSION_COLUMNS, predictions)
                    bce = -math.log(1 - probability_value)
                    metrics = evaluate_rows(index_studies(truth), index_studies(predictions))
                    record = {"epoch": epoch, "weak_bce": bce, "weak_macro_auc_12": metrics["macro_auc_12"],
                              "metrics": metrics, "prediction_sha256": sha256(predictions_path)}
                    checkpoint = make(model, provenance, fold=0, epoch=epoch, weak_bce=bce,
                                      prefix_sha256=digest, audit={"synthetic": True}, artificial=True)
                    torch.save(checkpoint, output / "last.pt")
                    if best_bce is None or bce < best_bce:
                        best_bce = bce
                        torch.save(checkpoint, output / "best.pt")
                    before = {alias: sha256(output / alias) for alias in ("best.pt", "last.pt")}
                    torch_rng, python_rng, numpy_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
                    entry = retain_epoch_checkpoint(output, checkpoint, record, predictions_path)
                    self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
                    self.assertEqual(python_rng, random.getstate())
                    now_numpy_rng = np.random.get_state()
                    self.assertEqual(numpy_rng[0], now_numpy_rng[0])
                    np.testing.assert_array_equal(numpy_rng[1], now_numpy_rng[1])
                    self.assertEqual(numpy_rng[2:], now_numpy_rng[2:])
                    self.assertEqual(before, {alias: sha256(output / alias) for alias in before})
                    self.assertEqual(entry["checkpoint_sha256"], before["last.pt"])
                    self.assertEqual(entry["prediction_sha256"], sha256(predictions_path))
                    self.assertEqual(entry["metrics"], metrics)
                    retained = output / entry["checkpoint"]
                    with self.assertRaisesRegex(ValueError, "Artificial"):
                        load(retained)
                    loaded = load(retained, allow_artificial=True)
                    restored = model_class(copy.deepcopy(encoder), copy.deepcopy(config), trainable_blocks=2,
                                           window_chunk=16, checkpoint_last2=True).eval()
                    kwargs = {"allow_artificial": True} if name == "REG" else {}
                    apply(restored, loaded, provenance, **kwargs)
                    for key, tensor in model.state_dict().items():
                        self.assertTrue(torch.equal(tensor, restored.state_dict()[key]), key)
                    features, mask = torch.zeros(1, 3, 48, 384), torch.zeros(1, 3, 48, dtype=torch.bool)
                    mask[0, 0, 0] = True
                    inputs = (features, mask, torch.zeros(1, 3, 48), torch.zeros(1, 3, dtype=torch.long),
                              torch.zeros(1, 3, 2))
                    with torch.inference_mode():
                        expected, actual = model.head(*inputs), restored.head(*inputs)
                    self.assertEqual(tuple(actual.shape), (1, 12))
                    self.assertTrue(torch.isfinite(actual).all())
                    self.assertTrue(torch.equal(expected, actual))
                    if epoch == 1:
                        original_epoch_sha = entry["checkpoint_sha256"]
                    self.assertEqual(sha256(output / "epoch_001.pt"), original_epoch_sha)
                index = json.loads((output / "epoch_checkpoint_index.json").read_text(encoding="utf-8"))
                self.assertEqual([entry["epoch"] for entry in index["checkpoints"]], [1, 2, 3])
                self.assertEqual(index["aliases"]["best.pt"]["epoch"], 1)  # Earliest exact BCE tie remains selected.
                self.assertEqual(index["aliases"]["last.pt"]["epoch"], 3)
                self.assertFalse(index["resume_supported"])
                self.assertEqual(load(output / "best.pt", allow_artificial=True)["epoch"], 1)
                self.assertEqual(load(output / "last.pt", allow_artificial=True)["epoch"], 3)
                before = {path.name: sha256(path) for path in output.iterdir()}
                with self.assertRaisesRegex(ValueError, "existing epochs cannot be overwritten"):
                    retain_epoch_checkpoint(output, checkpoint, record, predictions_path)
                self.assertEqual(before, {path.name: sha256(path) for path in output.iterdir()})

    def test_fast16_rejects_old_config_schema_and_every_wrong_chunk(self):
        old = json.loads((ROOT / "configs/experiments/f005-dinov2-native-last2.json").read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            validate_native_config(old)
        for chunk in (0, 1, 2, 4, 8, 32, True):
            wrong = copy.deepcopy(CONFIG)
            wrong["online"]["window_chunk"] = chunk
            with self.subTest(chunk=chunk), self.assertRaises(ValueError):
                validate_native_config(wrong)

    def test_fast16_checkpoint_loader_rejects_old_schema(self):
        from rsna_knee import native_training as old

        config = json.loads((ROOT / "configs/experiments/f005-dinov2-native-last2.json").read_text(encoding="utf-8"))
        model = NativeTripletMIL(fixtures.ArtificialEncoder(), config, trainable_blocks=2, window_chunk=2)
        checkpoint = old.make_native_checkpoint(model, {"artificial": True}, fold=0, epoch=0,
            weak_bce=0.0, prefix_sha256=old.prefix_digest(model), audit={}, artificial=True)
        path = self.root / "old-schema.pt"
        torch.save(checkpoint, path)
        with self.assertRaisesRegex(ValueError, "Unsupported native checkpoint"):
            load_native_checkpoint(path, allow_artificial=True)

    def test_checkpoint_creation_and_application_reject_runtime_chunk_mismatch(self):
        model = NativeTripletMIL(fixtures.ArtificialEncoder(), copy.deepcopy(CONFIG), trainable_blocks=2, window_chunk=2)
        with self.assertRaisesRegex(ValueError, "runtime window chunk"):
            make_native_checkpoint(model, {"artificial": True}, fold=0, epoch=0,
                weak_bce=0.0, prefix_sha256=prefix_digest(model), audit={}, artificial=True)
        with self.assertRaisesRegex(ValueError, "runtime window chunk"):
            apply_native_checkpoint(model, {}, {})

    def resource_fixture(self, folder):
        from rsna_knee.feature_contracts import json_hash

        source = native_source_hashes()
        configs = {
            str(blocks): json_hash(json.loads((ROOT / "configs/experiments" / name).read_text(encoding="utf-8")))
            for blocks, name in ((0, "f005-dinov2-native-frozen-control-fast16.json"), (2, "f005-dinov2-native-last2-fast16.json"))
        }
        cli_sha = hashlib.sha256((ROOT / "scripts/train_native_dinov2_fast16.py").read_bytes()).hexdigest()
        qa = {
            "status": "native392_label_free_QA_fast16_complete",
            "source_sha256": source, "real_processed_MRI_pixels_used": True, "real_teacher_labels_read": False,
            "optimizer_steps": 0, "studies_simultaneously_on_GPU": 1, "pilot": {"archive_sha256": "a" * 64},
            "checks": [{"passed": True}] * 6, "checks_passed": 6, "config_semantic_sha256": configs,
            "CLI_sha256": cli_sha, "engineering_lineage": ENGINEERING_LINEAGE,
            "arms": [{"arm": arm, "parameters": {"window_chunk": 16, "checkpoint_last2": True},
                "valid_triplets": 115, "optimizer_steps": 0} for arm in ("freeze", "last2")],
        }
        full144 = copy.deepcopy(qa)
        full144["real_processed_MRI_pixels_used"] = False
        for row in full144["arms"]:
            row["valid_triplets"] = 144
        evidence = {
            "pixel16": {"all_passed": True, "studies": 16, "checks": {"synthetic": True}, "archive_sha256": "a" * 64},
            "processed_QA": qa,
            "artificial_CUDA": {"cuda_checks_passed": 19, "cuda_checks": [{"passed": True}] * 19,
                "source_sha256": {"finetune_model.py": source["finetune_model.py"]}},
            "fast16_full144_QA": full144,
        }
        gate = {"schema_version": RESOURCE_GATE_SCHEMA, "source_sha256": source, "CLI_sha256": cli_sha,
            "config_semantic_sha256": configs, "evidence": {},
            **{key: True for key in ("native392_pixel16_complete", "native392_byte_QC_passed",
                "compression_measured", "direct_archive_read_speed_measured", "artificial_CUDA_gradient_and_roundtrip_passed")}}
        for name, value in evidence.items():
            path = folder / (name + ".json")
            path.write_text(json.dumps(value), encoding="utf-8")
            gate["evidence"][name] = {"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        path = folder / "gate.json"
        path.write_text(json.dumps(gate), encoding="utf-8")
        return path, gate, evidence

    def test_fast16_resource_gate_rejects_old_gate_and_old_processed_QA(self):
        old = ROOT / "artifacts/research/20261008-f005-native-training-preparation-v1/resource-gate-bound.json"
        with self.assertRaisesRegex(ValueError, "must pass"):
            validate_resource_gate(old, CONFIG)
        with tempfile.TemporaryDirectory(prefix="fast16-artificial-gate-", dir=ROOT / "artifacts/qa") as directory:
            folder = Path(directory)
            path, gate, evidence = self.resource_fixture(folder)
            self.assertEqual(validate_resource_gate(path, CONFIG), gate)
            evidence["processed_QA"]["status"] = "native392_label_free_QA_complete"
            (folder / "processed_QA.json").write_text(json.dumps(evidence["processed_QA"]), encoding="utf-8")
            gate["evidence"]["processed_QA"]["sha256"] = hashlib.sha256((folder / "processed_QA.json").read_bytes()).hexdigest()
            path.write_text(json.dumps(gate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "processed-pixel CUDA"):
                validate_resource_gate(path, CONFIG)

    def test_fast16_resource_gate_requires144_botharms_and_current_CLI(self):
        with tempfile.TemporaryDirectory(prefix="fast16-artificial-gate-", dir=ROOT / "artifacts/qa") as directory:
            folder = Path(directory)
            path, gate, evidence = self.resource_fixture(folder)
            evidence["fast16_full144_QA"]["arms"][0]["valid_triplets"] = 6
            item = folder / "fast16_full144_QA.json"
            item.write_text(json.dumps(evidence["fast16_full144_QA"]), encoding="utf-8")
            gate["evidence"]["fast16_full144_QA"]["sha256"] = hashlib.sha256(item.read_bytes()).hexdigest()
            path.write_text(json.dumps(gate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "full144 artificial"):
                validate_resource_gate(path, CONFIG)
            gate["CLI_sha256"] = "0" * 64
            path.write_text(json.dumps(gate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "CLI/lineage"):
                validate_resource_gate(path, CONFIG)

    def test_fast16_critical_accumulation_selection_Gold_and_report_functions_AST_identical(self):
        import ast

        original = ast.parse((ROOT / "src/rsna_knee/native_training.py").read_text(encoding="utf-8"))
        fast = ast.parse((ROOT / "src/rsna_knee/native_training_fast16.py").read_text(encoding="utf-8"))
        names = ("label_tensors", "observed_bce_sum", "backward_observed_micro", "effective_observation_count",
            "preflight_native_training", "evaluate_native", "read_prediction_ids", "predict_native",
            "new_native_model", "NativePixelStore", "read_manifest_structure")
        old_nodes = {node.name: ast.dump(node, include_attributes=False) for node in original.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        new_nodes = {node.name: ast.dump(node, include_attributes=False) for node in fast.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        for name in names:
            self.assertEqual(old_nodes[name], new_nodes[name], name)


if __name__ == "__main__":
    unittest.main()
