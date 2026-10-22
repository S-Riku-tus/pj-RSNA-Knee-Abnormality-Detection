"""Local artificial-byte sealer tests. No real checkpoint/run/CSV/labels."""
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsna_knee import f009_pair_stage as S

ROOT = Path(__file__).resolve().parents[1]


def save(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return {"path": relative, **S.pin(path)}


def fixture(root):
    for relative in [S.BASE, S.POLICY0, S.POLICY1, "src/rsna_knee/f009_source_namespace.py",
                     "scripts/train_native_dinov2_fast16.py", "configs/experiments/f009-dinov2-native-last2-late-cosine-study-gamma.json",
                     *("src/rsna_knee/" + name for name in {**S.sources.CORE_SHA, **S.sources.SUPPORT_SHA})]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, path)
    p0, p1 = S.read(root / S.POLICY0), S.read(root / S.POLICY1)
    config = S.read(root / "configs/experiments/f009-dinov2-native-last2-late-cosine-study-gamma.json")
    base = S.read(root / S.BASE)
    spec = {"schema_version": "f009_pair_stage_root_pins_v1", "folds": {}}
    completed = {}
    for fold, relative in ((0, p0["planned_run_directory"]), (1, p1["new_run"])):
        folder = root / relative
        folder.mkdir(parents=True)
        (folder / "epoch_012.pt").write_bytes(("ARTIFICIAL_NON_MODEL_FOLD" + str(fold)).encode())
        checkpoint = S.pin(folder / "epoch_012.pt")
        csv_sha = hashlib.sha256(("fakeCSV" + str(fold)).encode()).hexdigest()
        provenance = {"weights_sha256": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"}
        run = {"status": "complete", "fold": fold, "seed": 20261002, "config": config, "source_sha256": S.sources.CORE_SHA,
               "targets": S.TARGETS, "gold_used_for_training_or_selection": False, "gold_evaluated": False,
               "encoder_provenance": provenance,
               "history": [{"epoch": e, "amp_skipped_steps": 0, "weak_bce": .3, "prediction_sha256": csv_sha} for e in range(1, 13)]}
        run_ref = save(root, relative + "/run.json", run)
        index = {"fold": fold, "source_sha256": S.sources.CORE_SHA, "checkpoints": [{"epoch": 12,
                 "checkpoint": "epoch_012.pt", "predictions_csv": "epoch_012_weak.csv", "weak_bce": .3,
                 "checkpoint_bytes": checkpoint["bytes"], "checkpoint_sha256": checkpoint["sha256"], "prediction_sha256": csv_sha}]}
        index_ref = save(root, relative + "/epoch_checkpoint_index.json", index)
        run_pin, index_pin = ({k: r[k] for k in ("bytes", "sha256")} for r in (run_ref, index_ref))
        tensors = {family: [{"name": name, "shape": shape, "dtype": "torch.float32", "finite": True,
                            "tensor_sha256": hashlib.sha256((name + str(fold)).encode()).hexdigest()} for name, shape in expected.items()]
                   for family, expected in S.expected_tensor_shapes().items()}
        cpu = {"schema_version": S.CPU_SCHEMA, "fold": fold, "epoch": 12, "source_sha256": S.sources.CORE_SHA,
               "support_sha256": S.sources.SUPPORT_SHA, "namespace_helper_sha256": S.HELPER_SHA,
               "CLI_sha256": S.sources.CLI_SHA, "config_sha256": S.sources.CONFIG_SHA, "targets": S.TARGETS,
               "run_pin": run_pin, "retained_index_pin": index_pin, "checkpoint_pin": checkpoint,
               "frozen_prefix_sha256": base["native"]["model_fingerprints"]["REG:0"]["frozen_prefix_sha256"],
               "producer_sha256": "a" * 64, "evidence_scope": "artificial_fixture_only",
               "namespace": "_rsna_f009_fixture_fold" + str(fold),
               "model_class": "_rsna_f009_fixture_fold" + str(fold) + ".finetune_model.NativeTripletMIL",
               "head_class": "_rsna_f009_fixture_fold" + str(fold) + ".feature_model.FrozenFeatureHead",
               "actual_CPU_strict_reconstruction": False, "generic_factory_calls": 1, "model_forward_calls": 0,
               "optimizer_steps": 0, "model_training": False, "encoder_training": False, "gamma_draws": 0,
               "teacher_MRI_Gold_Report_read": False, "GPU_used": False, "tensor_metadata": tensors,
               "encoder_suffix_tensor_sha256": "b" * 64, "head_tensor_sha256": "c" * 64,
               "checkpoint_schema": "native392_generic_dino_suffix_head_fast16_v1", "weak_bce": .3,
               "encoder_provenance": provenance}
        spec["folds"][str(fold)] = {"run": run_ref, "index": index_ref,
             "CPU": save(root, "CPU/fold" + str(fold) + ".json", cpu), "expected_CPU_producer_sha256": "a" * 64}
        completed[str(fold)] = {"candidate": {"run": run_pin, "index": index_pin,
            "E12_prediction": {"bytes": 10, "sha256": csv_sha}, "E12_checkpoint_body_stream_hash": checkpoint}}
    inputpins = {"src/rsna_knee/" + key: {"sha256": value} for key, value in S.sources.CORE_SHA.items()}
    inputpins.update({p0[key]["path"]: {"sha256": p0[key]["sha256"]} for key in ("config", "resource_gate", "input_cache")})
    gbase = {"complete": True, "Gold_body_opened_or_hashed": False, "model_deserialized": False, "input_source_pins": inputpins}
    original = completed["0"]["candidate"]
    r0 = {"status": "F009_E12_WEAK_COMPARISON_ONLY_POLICY_V2", "candidate_epoch": 12, "control_epoch": 12,
          "policy_sha256": S.POLICY0_SHA, "fixed10_delta": .001, "candidate_AMP_skipped_steps": 0, "fold1_gate_conditions_met": True,
          "inputs": {"candidate": {"run_sha256": original["run"]["sha256"], "epoch_index_sha256": original["index"]["sha256"],
            "prediction_sha256": original["E12_prediction"]["sha256"], "checkpoint_sha256_reference_only_not_loaded": original["E12_checkpoint_body_stream_hash"]["sha256"]}}}
    spec["result0"] = save(root, "proof/result0.json", r0)
    g0 = {"complete": True, "original_evaluate_calls": 1, "wrapper_pin": {"sha256": S.F0_WRAPPER_SHA},
          "original_result_pin": {k: spec["result0"][k] for k in ("bytes", "sha256")}, "fold1_gate_conditions_met": True,
          "guards": {**gbase, "completed_E12_control_candidate": completed["0"]}}
    spec["guard0"] = save(root, "proof/guard0.json", g0)
    prior = {"result": {k: spec["result0"][k] for k in ("bytes", "sha256")},
             "guard_receipt": {k: spec["guard0"][k] for k in ("bytes", "sha256")},
             "fixed10_delta": .001, "engineering_gate_passed": True}
    r1 = {"status": "F009_FOLD1_FIXED12_PAIRED_AND_INDEPENDENT_TWO_FOLD_SCREEN_V1", "fold": 1,
          "candidate_epoch": 12, "control_epoch": 12, "policy0_sha256": S.POLICY0_SHA, "policy1_sha256": S.POLICY1_SHA,
          "fixed10_delta": .002, "joint_candidate_screen_passed": True, "fold1_machine_engineering_passed": True,
          "AMP_skipped_steps": {"control": 0, "candidate": 0}, "fold0": prior, "inputs": completed["1"]}
    spec["result1"] = save(root, "proof/result1.json", r1)
    g1 = {"complete": True, "comparison_calls": 1, "evaluator_pin": {"sha256": S.F1_EVALUATOR_SHA},
          "result_pin": {k: spec["result1"][k] for k in ("bytes", "sha256")}, "joint_candidate_screen_passed": True,
          "guards": {**gbase, "fold0_positive_sha_bound": prior, "machine_engineering_passed": True,
                     "completed_E12_control_candidate": completed["1"]}}
    spec["guard1"] = save(root, "proof/guard1.json", g1)
    return spec


class PairStage(unittest.TestCase):
    def test_artificial_local_pair_exact_copies_REG_refs_atomic_complete_no_real_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = fixture(root)
            original = {p: S.pin(root / p) for p in (S.BASE, S.POLICY0, S.POLICY1)}
            pair = S._seal(root, spec, root / "stage", fixture_only=True)
            self.assertTrue(pair["artificial_fixture_only"])
            self.assertFalse(pair["actual_CLS_stage_constructed"])
            base = S.read(root / S.BASE)
            for key in ("REG:0", "REG:1"):
                self.assertEqual(pair["fixed_REG_references_only"][key]["fingerprint"], base["native"]["model_fingerprints"][key])
            for fold in (0, 1):
                folder = root / "stage" / ("fold" + str(fold))
                expected = S.read(root / spec["folds"][str(fold)]["CPU"]["path"])["checkpoint_pin"]
                self.assertEqual(S.pin(folder / "trained/epoch_012.pt"), expected)
                self.assertFalse(S.read(folder / "bundle.json")["actual_strict_CPU_accepted"])
            self.assertTrue(S.read(root / "stage/BUILD_COMPLETE.json")["complete"])
            self.assertFalse(list((root / "stage").rglob("*.tmp")))
            self.assertEqual(original, {p: S.pin(root / p) for p in original})
            with self.assertRaises(ValueError):
                S._seal(root, spec, root / "stage", fixture_only=True)

    def test_actual_producer_missing_or_negative_or_tensor_tamper_before_model_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = fixture(root)
            original_pin = S.pin
            def no_model(path):
                self.assertNotEqual(Path(path).suffix, ".pt", "Model touched before gate/CPU checks")
                return original_pin(path)
            with patch.object(S, "pin", side_effect=no_model):
                with self.assertRaisesRegex(ValueError, "producer is not implemented"):
                    S._seal(root, spec, root / "actual")
                self.assertFalse((root / "actual").exists())
                bad = copy.deepcopy(spec)
                value = S.read(root / bad["result0"]["path"])
                value["fixed10_delta"] = -.1
                bad["result0"] = save(root, "proof/negative.json", value)
                with self.assertRaisesRegex(ValueError, "positive fold0"):
                    S._seal(root, bad, root / "negative", fixture_only=True)
                value = S.read(root / spec["folds"]["0"]["CPU"]["path"])
                value["tensor_metadata"]["head"][0]["shape"] = [383]
                bad = copy.deepcopy(spec)
                bad["folds"]["0"]["CPU"] = save(root, "CPU/badshape.json", value)
                with self.assertRaisesRegex(ValueError, "tensor key/shape"):
                    S._seal(root, bad, root / "badshape", fixture_only=True)

    def test_mid_copy_failure_no_complete_or_pair_sidecar_and_default_disabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = fixture(root)
            original_copy = S.copy_exact
            def fail_second(source, destination, expected):
                if Path(source).suffix == ".pt" and "fold1" in str(source):
                    raise OSError("Artificial interrupted copy")
                return original_copy(source, destination, expected)
            with patch.object(S, "copy_exact", side_effect=fail_second):
                with self.assertRaises(OSError):
                    S._seal(root, spec, root / "partial", fixture_only=True)
            self.assertFalse((root / "partial/BUILD_COMPLETE.json").exists())
            self.assertFalse((root / "partial/pair-sidecar.json").exists())
            self.assertFalse(S.read(root / "partial/BUILD_FAILED.json")["complete"])
        result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/seal_f009_E12_pair.py")], capture_output=True,
                                text=True, check=True)
        self.assertFalse(json.loads(result.stdout)["actual_stage_constructed"])
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
