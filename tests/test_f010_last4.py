"""CPU-only artificial F010 boundary checks; no real assets, data or GPU calls."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from rsna_knee import f010_last4_training as train
from rsna_knee.f010_last4_model import F010Last4MIL
from rsna_knee.finetune_model import NativeTripletMIL

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/f010-dinov2-native-last4-late-cosine.json"
CONTROL = ROOT / "configs/experiments/f006-dinov2-native-last2-late-cosine.json"


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.chunked_blocks, self.num_register_tokens, self.embed_dim = False, 0, 384
        self.input = nn.Linear(3, 384)
        self.blocks = nn.ModuleList(nn.Linear(384, 384) for _ in range(12))
        self.norm = nn.LayerNorm(384)

    def prepare_tokens_with_masks(self, images):
        return self.input(images.mean(dim=(-2, -1)))[:, None, :].expand(-1, 2, -1)


def config(blocks=4):
    value = json.loads(CONFIG.read_text(encoding="utf8"))
    value["online"]["trainable_blocks"] = blocks
    return value


def model(blocks=4):
    torch.manual_seed(20261002)
    return F010Last4MIL(TinyEncoder(), config(blocks), trainable_blocks=blocks, window_chunk=16, checkpoint_last2=True)


def artificial_inputs():
    mask = torch.zeros((1, 3, 48), dtype=torch.bool)
    mask[0, 0, :2] = True
    return {
        "images": torch.full((1, 1, 1, 3, 392, 392), 117, dtype=torch.uint8).expand(1, 3, 48, 3, 392, 392),
        "mask": mask,
        "positions": torch.zeros((1, 3, 48)),
        "planes": torch.tensor([[0, 3, 3]], dtype=torch.int64),
        "flags": torch.zeros((1, 3, 2)),
    }


class Last4BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_only_control_difference_is_suffix_and_schema(self):
        candidate = config()
        control = json.loads(CONTROL.read_text(encoding="utf8"))
        candidate.pop("reason")
        control.pop("reason")
        candidate["online"]["schema_version"] = control["online"]["schema_version"]
        candidate["online"]["trainable_blocks"] = control["online"]["trainable_blocks"]
        self.assertEqual(candidate, control)
        train.validate_native_config(config())

    def test_rejects_last2_gamma_constant_schedule_and_wrong_seed(self):
        for modify in (
            lambda c: c["online"].update(trainable_blocks=2),
            lambda c: c.update(augmentation={"schema_version": "native392_train_study_gamma_v1", "minimum": .85, "maximum": 1.15}),
            lambda c: c["train"].update(lr_schedule={"name": "constant"}),
            lambda c: c.update(seed=1),
        ):
            value = config()
            modify(value)
            with self.assertRaises(ValueError):
                train.validate_native_config(value)

    def test_last4_scope_and_frozen_norm(self):
        value = model()
        names = value.parameter_contract()["trainable_encoder_names"]
        self.assertEqual({int(name.split(".")[1]) for name in names}, {8, 9, 10, 11})
        self.assertFalse(any(p.requires_grad for p in value.encoder.norm.parameters()))
        value.train()
        self.assertFalse(value.encoder.training)
        self.assertTrue(value.head.training)
        self.assertIsNone(value.training_augmentation_contract())

    def test_no_unintended_initialization_change_vs_last2(self):
        candidate = model()
        torch.manual_seed(20261002)
        control = NativeTripletMIL(TinyEncoder(), json.loads(CONTROL.read_text(encoding="utf8")), trainable_blocks=2, window_chunk=16, checkpoint_last2=True)
        self.assertEqual(set(candidate.state_dict()), set(control.state_dict()))
        self.assertTrue(all(torch.equal(value, control.state_dict()[key]) for key, value in candidate.state_dict().items()))
        with torch.no_grad():
            x = torch.rand(2, 3, 4, 4)
            self.assertTrue(torch.equal(candidate.encode_normalized(x), control.encode_normalized(x)))

    def test_forward_backward_prefix_scope_and_missing_mask(self):
        value = model().train()
        before = train.prefix_digest(value)
        logits = value(**artificial_inputs())
        self.assertEqual(tuple(logits.shape), (1, 12))
        self.assertTrue(torch.isfinite(logits).all())
        target = torch.tensor([[0., 1.] * 6])
        observed = torch.ones_like(target, dtype=torch.bool)
        observed[0, 3] = False
        target[0, 3] = float("nan")
        train.backward_observed_micro(logits, target, observed, 11 * 16)
        for name, parameter in value.encoder.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            else:
                self.assertIsNone(parameter.grad, name)
        self.assertEqual(before, train.prefix_digest(value))

    def test_no_observation_no_step(self):
        with self.assertRaises(ValueError):
            train.observed_bce_sum(torch.zeros(1, 12), torch.full((1, 12), float("nan")), torch.zeros(1, 12, dtype=torch.bool))

    def test_artificial_submission_contract_does_not_require_Report(self):
        from rsna_knee.contracts import validate_submission, write_csv

        inputs = artificial_inputs()
        with torch.no_grad():
            probabilities = model().eval()(**inputs).sigmoid()[0].tolist()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ids = [{train.ID: "artificial-one"}, {train.ID: "artificial-two"}]
            write_csv(directory / "test.csv", (train.ID,), ids)
            predictions = [{**row, **dict(zip(train.TARGETS, probabilities))} for row in ids]
            write_csv(directory / "sample.csv", train.SUBMISSION_COLUMNS, predictions)
            write_csv(directory / "submission.csv", train.SUBMISSION_COLUMNS, predictions)
            self.assertEqual(train.read_prediction_ids(directory / "test.csv"), ids)
            self.assertTrue(validate_submission(directory / "submission.csv", directory / "test.csv", directory / "sample.csv")["valid"])

    def checkpoint(self):
        value = model().eval()
        checkpoint = train.make_native_checkpoint(value, {}, fold=0, epoch=0, weak_bce=0., prefix_sha256=train.prefix_digest(value), audit={}, artificial=True)
        return value, checkpoint

    def test_exact_suffix_checkpoint_and_roundtrip(self):
        value, checkpoint = self.checkpoint()
        self.assertEqual({int(key.split(".")[1]) for key in checkpoint["encoder_suffix"]}, {8, 9, 10, 11})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artificial.pt"
            torch.save(checkpoint, path)
            restored = train.load_native_checkpoint(path, allow_artificial=True)
            target = model().eval()
            train.apply_native_checkpoint(target, restored, {})
            self.assertTrue(all(torch.equal(tensor, target.state_dict()[key]) for key, tensor in value.state_dict().items()))
            with self.assertRaisesRegex(ValueError, "Artificial"):
                train.load_native_checkpoint(path)

    def test_checkpoint_rejects_last2_schema_wrong_source_and_missing_suffix(self):
        for modify in (
            lambda c: c.update(schema_version="native392_generic_dino_suffix_head_fast16_v1"),
            lambda c: c.update(source_sha256={}),
            lambda c: c["encoder_suffix"].pop(next(key for key in c["encoder_suffix"] if key.startswith("blocks.8."))),
            lambda c: c["encoder_suffix"].update({"norm.weight": torch.ones(384)}),
        ):
            value, checkpoint = self.checkpoint()
            modify(checkpoint)
            with self.assertRaises(ValueError):
                train.apply_native_checkpoint(value, checkpoint, {})

    def test_checkpoint_rejects_nonfinite_shape_and_prefix_change(self):
        for modification in ("nan", "shape", "prefix"):
            value, checkpoint = self.checkpoint()
            key = next(iter(checkpoint["encoder_suffix"]))
            if modification == "nan":
                checkpoint["encoder_suffix"][key].flatten()[0] = float("nan")
            elif modification == "shape":
                checkpoint["encoder_suffix"][key] = torch.ones(1)
            else:
                with torch.no_grad():
                    value.encoder.input.weight.add_(1)
            with self.assertRaises(ValueError):
                train.apply_native_checkpoint(value, checkpoint, {})

    def test_lr_matches_F006_schedule(self):
        value = model()
        opt = torch.optim.AdamW([
            {"params": value.head.parameters(), "lr": .0003, "name": "head"},
            {"params": (p for p in value.encoder.parameters() if p.requires_grad), "lr": 1e-5, "name": "last4"},
        ])
        self.assertEqual(train.native_epoch_learning_rates(opt, config(), 6)["learning_rate_multiplier"], 1.)
        self.assertEqual(train.native_epoch_learning_rates(opt, config(), 12)["learning_rate_multiplier"], .1)
        self.assertAlmostEqual(opt.param_groups[1]["lr"], 1e-6)

    def test_source_loss_data_evaluation_and_accumulation_helpers_unchanged(self):
        def definitions(path):
            tree = ast.parse(path.read_text(encoding="utf8"))
            return {node.name: ast.dump(node, include_attributes=False) for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        old = definitions(ROOT / "src/rsna_knee/native_training_fast16.py")
        new = definitions(ROOT / "src/rsna_knee/f010_last4_training.py")
        for name in ("label_tensors", "observed_bce_sum", "backward_observed_micro", "effective_observation_count",
                     "NativePixelStore", "read_manifest_structure", "preflight_native_training", "tensor_study",
                     "read_prediction_ids", "evaluate_native", "predict_native"):
            self.assertEqual(old[name], new[name], name)

    def test_CUDA_first_before_preflight_reads(self):
        with patch.object(train, "ensure_cuda", side_effect=RuntimeError("CUDA sentinel")), patch.object(train, "validate_native_config") as validate:
            with self.assertRaisesRegex(RuntimeError, "CUDA sentinel"):
                train.preflight_native_training("missing", "missing", {}, 0, "missing")
            validate.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "CUDA sentinel"):
                train.train_native("missing", "missing", "missing", {}, 0, "missing", "missing", {}, "missing")

    def test_frozen_QA_config_cannot_fit(self):
        with patch.object(train, "ensure_cuda"):
            with self.assertRaisesRegex(ValueError, "frozen config is QA only"):
                train.train_native("missing", "missing", "missing", config(0), 0, "missing", "missing", {}, "missing")

    def test_resource_gate_new_receipts_and_negative_boundaries(self):
        suffixes = [f"blocks.{block}.{key}" for block in (8, 9, 10, 11) for key in (
            "norm1.weight", "norm1.bias", "attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight", "attn.proj.bias",
            "ls1.gamma", "norm2.weight", "norm2.bias", "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias", "ls2.gamma")]
        configs = {str(blocks): train.json_hash(json.loads((ROOT / "configs/experiments" / name).read_text(encoding="utf8"))) for blocks, name in (
            (0, "f010-dinov2-native-frozen-QA-only.json"), (4, "f010-dinov2-native-last4-late-cosine.json"))}
        def qa(real):
            arms = []
            for blocks in (0, 4):
                arms.append({"arm": "freeze" if blocks == 0 else "last4", "valid_triplets": 115 if real else 144,
                    "optimizer_steps": 0, "effective_batch_studies": 16,
                    "gpu_peak_reserved_bytes": 1000, "gpu_peak_allocated_bytes": 900,
                    "parameters": {"trainable_blocks": blocks, "window_chunk": 16, "checkpoint_last2": True,
                        "encoder_eval": True, "final_norm_frozen": True, "trainable_head_parameters": 83352,
                        "trainable_encoder_parameters": 0 if blocks == 0 else 7100928,
                        "frozen_encoder_parameters": 22056576 if blocks == 0 else 14955648,
                        "trainable_encoder_names": [] if blocks == 0 else suffixes}})
            return {"status": "f010_last4_label_free_QA_fast16_complete", "real_processed_MRI_pixels_used": real,
                "source_sha256": train.native_source_hashes(), "CLI_sha256": train.sha256(ROOT / "scripts/train_f010_last4_fast16.py"),
                "engineering_lineage": train.ENGINEERING_LINEAGE, "config_semantic_sha256": configs,
                "real_teacher_labels_read": False, "real_fit": False, "optimizer_steps": 0,
                "studies_simultaneously_on_GPU": 1, "raw_DICOM_decoded_or_downloaded": False,
                "arms": arms, "checks_passed": 6, "checks": [{"passed": True}] * 6,
                "pilot": {"archive_sha256": "synthetic-test-only", "read_decompress_hash_QC_seconds": .1} if real else None}
        original_receipts = {"pixel16": {"all_passed": True, "studies": 16, "checks": {"synthetic-test-only": True}, "archive_sha256": "synthetic-test-only"},
                             "processed_QA": qa(True), "artificial144_QA": qa(False)}
        mutations = [None,
            lambda r: r["artificial144_QA"]["arms"][1]["parameters"].update(trainable_encoder_names=[name for name in suffixes if not name.startswith("blocks.8.")]),
            lambda r: r["artificial144_QA"].update(status="native392_label_free_QA_fast16_complete"),
            lambda r: r["processed_QA"].update(source_sha256={}),
            lambda r: r["artificial144_QA"]["arms"][1].update(valid_triplets=143),
            lambda r: r["processed_QA"].update(optimizer_steps=1),
            lambda r: r["processed_QA"]["arms"][1].update(gpu_peak_reserved_bytes=21_000_000_001),
        ]
        with tempfile.TemporaryDirectory(prefix="f010-artificial-gate-", dir=ROOT / "artifacts/research") as temporary:
            directory = Path(temporary)
            for mutate in mutations:
                receipts = copy.deepcopy(original_receipts)
                if mutate:
                    mutate(receipts)
                gate = {"schema_version": train.RESOURCE_GATE_SCHEMA, "source_sha256": train.native_source_hashes(),
                    "CLI_sha256": train.sha256(ROOT / "scripts/train_f010_last4_fast16.py"), "engineering_lineage": train.ENGINEERING_LINEAGE,
                    "approved_for_real_fit": True, "config_semantic_sha256": configs, "evidence": {}}
                for name, value in receipts.items():
                    path = directory / f"{name}.json"
                    payload = json.dumps(value).encode()
                    path.write_bytes(payload)
                    gate["evidence"][name] = {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(payload).hexdigest()}
                path = directory / "gate.json"
                path.write_text(json.dumps(gate), encoding="utf8")
                if mutate:
                    with self.assertRaises(ValueError):
                        train.validate_resource_gate(path, config())
                else:
                    self.assertEqual(train.validate_resource_gate(path, config()), gate)


if __name__ == "__main__":
    unittest.main()
