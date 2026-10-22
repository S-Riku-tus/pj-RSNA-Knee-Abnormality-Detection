"""CPU artificial fold2 boundaries; no GPU, MRI, generic weights or fit."""
import ast
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from rsna_knee import f006_fold2_training as N

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/experiments/f006-dinov2-native-last2-late-cosine-fold2-20261017-v1.json").read_bytes())


class ToyModel:
    """Tiny artificial state fixture, not a DINO model or scientific forward."""
    def __init__(self):
        self.config, self.window_chunk, self.trainable_blocks = copy.deepcopy(CONFIG), 16, 2
        self.encoder = torch.nn.Module()
        self.encoder.blocks = torch.nn.ModuleList(torch.nn.Linear(2, 2) for _ in range(12))
        self.encoder.norm = torch.nn.LayerNorm(2)
        self.head = torch.nn.Linear(2, 12)


class Fold2Tests(unittest.TestCase):
    def test_fold2_config_only_and_CUDA_before_any_data(self):
        self.assertEqual(N.validate_native_config(copy.deepcopy(CONFIG)), CONFIG)
        frozen = json.loads((ROOT / "configs/experiments/f006-fold2-frozen-control-fast16-QA-20261017-v1.json").read_bytes())
        N.validate_native_config(frozen)
        for folds in ([0], [1], [0, 1], [0, 1, 2]):
            invalid = copy.deepcopy(CONFIG)
            invalid["online"]["folds"] = folds
            with self.assertRaises(ValueError):
                N.validate_native_config(invalid)
        with self.assertRaises(ValueError):
            N.validate_native_config({**CONFIG, "augmentation": None})
        with patch.object(N, "ensure_cuda", side_effect=ValueError("CUDA-first-sentinel")), patch.object(
                N, "validate_native_config", side_effect=AssertionError("config read before CUDA")):
            with self.assertRaisesRegex(ValueError, "CUDA-first-sentinel"):
                N.preflight_native_training(Path("must-not-read"), Path("must-not-read"), {}, 2, Path("must-not-read"))
        cli = ast.parse((ROOT / "scripts/train_f006_fold2_fast16.py").read_text())
        main = next(node for node in cli.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
        cuda_line = next(node.lineno for node in calls if isinstance(node.func, ast.Name) and node.func.id == "ensure_cuda")
        config_line = next(node.lineno for node in calls if isinstance(node.func, ast.Name) and node.func.id == "load_feature_config")
        self.assertLess(cuda_line, config_line)
        self.assertIn('choices=(2,)', (ROOT / "scripts/train_f006_fold2_fast16.py").read_text())

    def test_artificial_fold2_checkpoint_roundtrip_and_legacy_source_config_rejection(self):
        source, destination = ToyModel(), ToyModel()
        destination.encoder.load_state_dict(source.encoder.state_dict())
        digest = N.prefix_digest(source)
        checkpoint = N.make_native_checkpoint(source, {"artificial": True}, fold=2, epoch=0, weak_bce=0.,
            prefix_sha256=digest, audit={"artificial": True}, artificial=True)
        for fold in (0, 1, 2.0):
            with self.assertRaises(ValueError):
                N.make_native_checkpoint(source, {}, fold=fold, epoch=0, weak_bce=0., prefix_sha256=digest, audit={}, artificial=True)
        with tempfile.TemporaryDirectory(prefix="fold2-CPU-") as temporary:
            path = Path(temporary) / "artificial.pt"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "Artificial checkpoints"):
                N.load_native_checkpoint(path)
            loaded = N.load_native_checkpoint(path, allow_artificial=True)
            N.apply_native_checkpoint(destination, loaded, {"artificial": True})
            for name, tensor in source.head.state_dict().items():
                self.assertTrue(torch.equal(tensor, destination.head.state_dict()[name]))
            for name, tensor in source.encoder.state_dict().items():
                self.assertTrue(torch.equal(tensor, destination.encoder.state_dict()[name]))
            for field in ("fold0", "fold1", "oldschema", "source", "config", "targets"):
                bad = copy.deepcopy(checkpoint)
                if field.startswith("fold"):
                    bad["fold"] = int(field[-1])
                elif field == "oldschema":
                    bad["schema_version"] = "native392_generic_dino_suffix_head_fast16_v1"
                elif field == "source":
                    bad["source_sha256"]["f006_fold2_training.py"] = "0" * 64
                elif field == "config":
                    bad["config"]["online"]["folds"] = [0, 1]
                else:
                    bad["targets"] = list(reversed(bad["targets"]))
                torch.save(bad, path)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    N.load_native_checkpoint(path, allow_artificial=True)

    def test_old_numeric_LR_and_missing_mask_gradient_parity(self):
        original = (ROOT / "artifacts/research/20261017-f009-gamma-preparation-v1/snapshots/native_training_fast16.py").read_text()
        names = ("validate_native_learning_rate_schedule", "native_epoch_learning_rates", "observed_bce_sum", "backward_observed_micro")
        nodes = [node for node in ast.parse(original).body if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = {"torch": torch, "F": F, "math": math}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "<pinned-original-F006-functions>", "exec"), namespace)
        left, right = (SimpleNamespace(param_groups=[{"name": "head", "lr": .0003}, {"name": "last2", "lr": .00001}]) for _ in (0, 1))
        for epoch in range(1, 13):
            self.assertEqual(namespace["native_epoch_learning_rates"](left, CONFIG, epoch), N.native_epoch_learning_rates(right, CONFIG, epoch))
        generator = torch.Generator(device="cpu").manual_seed(104)
        a = torch.randn((3, 12), generator=generator, dtype=torch.float64, requires_grad=True)
        b = a.detach().clone().requires_grad_(True)
        targets = torch.randint(0, 2, (3, 12), generator=generator).double()
        mask = torch.arange(36).reshape(3, 12) % 3 != 0
        targets[~mask] = float("nan")
        old = namespace["observed_bce_sum"](a, targets, mask)
        new = N.observed_bce_sum(b, targets, mask)
        old.backward()
        new.backward()
        self.assertTrue(torch.equal(old, new) and torch.equal(a.grad, b.grad))
        self.assertTrue(torch.equal(b.grad[~mask], torch.zeros_like(b.grad[~mask])))

    def test_gate_exact_source_config_CLI_and_receipt_bindings(self):
        with tempfile.TemporaryDirectory(prefix="fold2-fakegate-", dir=ROOT / "artifacts") as temporary:
            directory = Path(temporary).resolve()
            self.assertTrue(directory.is_relative_to((ROOT / "artifacts").resolve()))
            sources, cli = N.native_source_hashes(), ROOT / "scripts/train_f006_fold2_fast16.py"
            configs = {"0": "artificial-frozen-semantic", "2": N.json_hash(CONFIG)}
            checks = [{"passed": True} for _ in range(6)]
            qa = {"status": "f006_fold2_label_free_QA_fast16_complete", "real_processed_MRI_pixels_used": True,
                "source_sha256": sources, "optimizer_steps": 0, "real_teacher_labels_read": False,
                "studies_simultaneously_on_GPU": 1, "pilot": {"archive_sha256": "synthetic"}, "checks": checks,
                "checks_passed": 6, "arms": [{"arm": arm, "parameters": {"window_chunk": 16, "checkpoint_last2": True}, "valid_triplets": 144, "optimizer_steps": 0} for arm in ("freeze", "last2")],
                "config_semantic_sha256": configs, "CLI_sha256": N.sha256(cli), "engineering_lineage": N.ENGINEERING_LINEAGE}
            evidence = {"pixel16": {"all_passed": True, "studies": 16, "checks": {"synthetic": True}, "archive_sha256": "synthetic"},
                "processed_QA": qa, "fast16_full144_QA": {**qa, "real_processed_MRI_pixels_used": False},
                "artificial_CUDA": {"cuda_checks_passed": 19, "cuda_checks": [{"passed": True}],
                    "source_sha256": {"finetune_model.py": sources["f006_fold2_model.py"]}}}
            refs = {}
            for name, value in evidence.items():
                path = directory / (name + ".json")
                path.write_text(json.dumps({**value, "synthetic_boundary_test_only_not_actual_GPU_evidence": True}))
                refs[name] = {"path": str(path.relative_to(ROOT)), "sha256": N.sha256(path)}
            gate = {"schema_version": N.RESOURCE_GATE_SCHEMA, "source_sha256": sources,
                "config_semantic_sha256": configs, "CLI_sha256": N.sha256(cli), "evidence": refs,
                **{name: True for name in ("native392_pixel16_complete", "native392_byte_QC_passed", "compression_measured", "direct_archive_read_speed_measured", "artificial_CUDA_gradient_and_roundtrip_passed")}}
            path = directory / "gate.json"
            path.write_text(json.dumps(gate))
            N.validate_resource_gate(path, CONFIG)
            for change in ("schema", "source", "config", "CLI", "receipt"):
                bad = copy.deepcopy(gate)
                if change == "schema":
                    bad["schema_version"] = "native392_training_resource_gate_fast16_v1"
                elif change == "source":
                    bad["source_sha256"]["f006_fold2_training.py"] = "0" * 64
                elif change == "config":
                    bad["config_semantic_sha256"]["2"] = "0" * 64
                elif change == "CLI":
                    bad["CLI_sha256"] = "0" * 64
                else:
                    bad["evidence"]["processed_QA"]["sha256"] = "0" * 64
                path.write_text(json.dumps(bad))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    N.validate_resource_gate(path, CONFIG)
        self.assertFalse(directory.exists())

    def test_original_model_bytes_and_shared_six_unchanged(self):
        diff = json.loads((ROOT / "artifacts/research/20261017-f006-fold2-implementation-v1/source-diff.json").read_bytes())
        self.assertEqual(N.sha256(ROOT / "src/rsna_knee/f006_fold2_model.py"), "6befcaa880e13581ef776fc52e3d47d0d4f6d4776e73ee67e2ed8e25ab70193f")
        for relative, expected in diff["protected_originals"].items():
            self.assertEqual(N.sha256(ROOT / relative), expected["sha256"])
        original = json.loads((ROOT / "artifacts/runs/f006-late-cosine-fold0-20261012-v1/run.json").read_bytes())["source_sha256"]
        current = N.native_source_hashes()
        for name in ("feature_model.py", "triplet_cache.py", "feature_imaging.py", "feature_contracts.py", "imaging.py", "contracts.py"):
            self.assertEqual(original[name], current[name])
        self.assertTrue(diff["train_loop_AST_exact_after_run_schema_normalization"])


if __name__ == "__main__":
    unittest.main()
