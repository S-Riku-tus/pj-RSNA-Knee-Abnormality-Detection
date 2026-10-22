"""Synthetic192 result objects only; no real diagnostic/model/image/label read."""
import copy
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsna_knee import f009_diagnostic_audit as A

ROOT = Path(__file__).resolve().parents[1]


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def average(left, right):
    return [f32(f32(a + b) / 2) for a, b in zip(left, right)]


def fixture():
    order = ["ARTIFICIAL_" + str(index).zfill(3) for index in range(192)]
    fingerprints = {role: {"model_class": "rsna_knee.finetune_model.NativeTripletMIL" if role.startswith("CLS") else
                           "rsna_knee.regional_triplet_model.RegionalTripletMIL",
                           "checkpoint_sha256": str(index + 1) * 64, "frozen_prefix_sha256": "a" * 64,
                           "selected_epoch": 12 if role.startswith("CLS") else (8 if role.endswith("0") else 10)}
                    for index, role in enumerate(("CLS:0", "CLS:1", "REG:0", "REG:1"))}
    component = {"complete": True, "phase_contract_passed": True, "actual_CPU_four_model_reconstruction": True,
                 "CPU_strict_four_model_reconstruction": True, "artificial_stub_components": False,
                 "generic_factory_calls": 1, "logical_CPU_fold_models": 4, "selected_checkpoint_loads": 4,
                 "strictapply_calls": 4, "CPU_storage_alias_count": 0, "model_fingerprints": fingerprints}
    old = {"schema_version": A.OLD_SCHEMA, "artificial_fixture_only": True, "complete": True, "studies": 192,
           "operational_recovery_contract_passed": True, "targets": list(A.TARGETS),
           "coverage": {"N": 192, "S": 192, "F_structural": 0, "F_operational": 0},
           "structural_failures": {}, "operational_failures": {}, "failure_UID_union": [],
           "full_fold_model_forward_calls": 768, "reconstruction_component": component,
           "probabilities_test_order": [], "perUID_receipts": {}}
    for index, uid in enumerate(order):
        hashes = {name: hashlib.sha256((uid + name).encode()).hexdigest() for name in A.ARRAYS}
        receipt = {"array_sha256": hashes, "valid_windows": 13, "native_SHA_verified_before_model": True,
                   "same_native_array_SHA_after_model": True, "logical_full_fold_model_forward_calls": 4,
                   "fallback": False, "inference_mode_in_consumer": True, "family_order": ["CLS", "REG"],
                   "fold_order": [0, 1], "families": {}, "family_mean_probabilities": {}}
        for family_index, family in enumerate(("CLS", "REG")):
            folds = [[f32(((index + label + fold + family_index * 3) % 101) / 113)
                      for label in range(12)] for fold in range(2)]
            receipt["families"][family] = {"fold_probabilities": folds, "array_sha256": dict(hashes),
                "logical_full_fold_model_forward_calls": 2, "fold_order": [0, 1], "fallback": False,
                "inference_mode_in_consumer": True}
            receipt["family_mean_probabilities"][family] = average(*folds)
        old["perUID_receipts"][uid] = receipt
        old["probabilities_test_order"].append({"StudyInstanceUID": uid, "probabilities": average(
            receipt["family_mean_probabilities"]["CLS"], receipt["family_mean_probabilities"]["REG"])})
    new = copy.deepcopy(old)
    new.update(schema_version=A.SCHEMA, actual_native_inference_executed=False, diagnostic_engineering_passed=True,
        fixed_selection_sha256=A.SELECTION_PIN["sha256"], fixed_UID_order_sha256=A.order_sha(order),
        fixed_train_series_sha256=A.SERIES_SHA256, baseline_kind="synthetic_constant_0.5_for_engineering_only",
        P033_parent_cells_executed=0, accuracy_or_weak_Gold_metric_evaluated=False,
        ranks_or_final_submission_written=False, fixture_cleanup_complete=True, work_cleanup_complete=True,
        resources_passed=True, spawn_workers_remaining=0, native_temp_files_removed=True,
        fixture_receipt={"schema_version": A.FIXTURE_SCHEMA, "selection_pin": dict(A.SELECTION_PIN),
            "UIDs": order, "studies": 192, "ordered_UID_sha256": A.order_sha(order),
            "ordered_UID_hash_encoding": "compact UTF8 JSON list", "train_series_csv_sha256": A.SERIES_SHA256,
            "fixture_complete": True, "actual_fixture_created": False, "artificial_fixture_only": True,
            "cleanup_complete": True})
    for fold in (0, 1):
        fp = new["reconstruction_component"]["model_fingerprints"]["CLS:" + str(fold)]
        fp.update(model_class="_rsna_f009_runtime_cls_f" + str(fold) + ".finetune_model.NativeTripletMIL",
                  checkpoint_sha256=str(6 + fold) * 64)
    # New CLS predictions differ by design. No accuracy claim is implied.
    for index, uid in enumerate(order):
        receipt = new["perUID_receipts"][uid]
        receipt["families"]["CLS"]["fold_probabilities"][0][0] = f32(0.03125)
        refresh(new, index)
    return new, old, order


def refresh(value, index):
    row = value["probabilities_test_order"][index]
    receipt = value["perUID_receipts"][row["StudyInstanceUID"]]
    for family in ("CLS", "REG"):
        receipt["family_mean_probabilities"][family] = average(*receipt["families"][family]["fold_probabilities"])
    row["probabilities"] = average(receipt["family_mean_probabilities"]["CLS"], receipt["family_mean_probabilities"]["REG"])


class DiagnosticAuditTests(unittest.TestCase):
    def test_default_reads_nothing_and_FP32_known_bits(self):
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/audit_f009_fixed192_diagnostic.py"),
                                 "--new-result", "NEVER_OPEN"], env=env, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["input_bodies_read"], 0)
        self.assertEqual(A.bits(A.mean_reference([f32(.1)] * 12, [f32(.2)] * 12)[0]).hex(), "9a99193e")

    def test_synthetic192_pass_is_not_actual_and_CLS_changes_allowed(self):
        new, old, order = fixture()
        result = A._audit_objects(new, old, order, fixture_only=True)
        self.assertTrue(result["passed"])
        self.assertFalse(result["actual_result_files_audited"])
        self.assertTrue(result["artificial_fixture_only"])
        self.assertEqual(result["independent_FP32_new_cells_checked"], 6912)
        self.assertEqual(result["REG_fold_FP32_bit_mismatches"], 0)
        self.assertGreater(result["CLS_fold_FP32_bit_changes_report_only"], 0)
        with self.assertRaises(ValueError):
            A._audit_objects(new, old, order, fixture_only=False)

    def test_contract_and_numeric_corruptions_rejected(self):
        for kind in ("UID", "NaN", "foldmean", "finalmean", "nestedhash", "newoldCLS", "boolcount"):
            with self.subTest(kind=kind):
                new, old, order = fixture()
                receipt = new["perUID_receipts"][order[0]]
                if kind == "UID":
                    new["probabilities_test_order"][0]["StudyInstanceUID"] = order[1]
                elif kind == "NaN":
                    receipt["families"]["CLS"]["fold_probabilities"][0][0] = float("nan")
                elif kind == "foldmean":
                    receipt["family_mean_probabilities"]["CLS"][0] = f32(.75)
                elif kind == "finalmean":
                    new["probabilities_test_order"][0]["probabilities"][0] = f32(.75)
                elif kind == "nestedhash":
                    receipt["families"]["REG"]["array_sha256"]["images"] = "0" * 64
                elif kind == "newoldCLS":
                    new["reconstruction_component"]["model_fingerprints"]["CLS:0"] = old["reconstruction_component"]["model_fingerprints"]["CLS:0"]
                else:
                    new["coverage"]["F_structural"] = False
                with self.assertRaises(ValueError):
                    A._audit_objects(new, old, order, fixture_only=True)

    def test_one_structural_failure_validates_partition_but_HOLD(self):
        new, old, order = fixture()
        uid = order[7]
        new["structural_failures"] = {uid: {"artificial_reason": "synthetic geometry failure"}}
        new["failure_UID_union"] = [uid]
        new["coverage"].update(S=191, F_structural=1)
        new["full_fold_model_forward_calls"] = 764
        new["diagnostic_engineering_passed"] = False
        new["probabilities_test_order"][7]["probabilities"] = [.5] * 12
        new["perUID_receipts"][uid] = {"logical_full_fold_model_forward_calls": 0, "GPU_forward_skipped": True,
            "native_baseline_placeholder": True, "excluded_from_native_success_subset_ranks": True}
        result = A._audit_objects(new, old, order, fixture_only=True)
        self.assertFalse(result["passed"])
        self.assertEqual(result["decision"], "HOLD")
        self.assertEqual(result["input_hash_common_successes_compared"], 191)
        new["perUID_receipts"][uid]["logical_full_fold_model_forward_calls"] = 1
        with self.assertRaises(ValueError):
            A._audit_objects(new, old, order, fixture_only=True)

    def test_coherent_input_or_REG_changes_hold_and_cleanup_is_required(self):
        for kind in ("array", "REG", "cleanup", "resources"):
            with self.subTest(kind=kind):
                new, old, order = fixture()
                receipt = new["perUID_receipts"][order[0]]
                if kind == "array":
                    receipt["array_sha256"]["images"] = "0" * 64
                    for family in ("CLS", "REG"):
                        receipt["families"][family]["array_sha256"]["images"] = "0" * 64
                elif kind == "REG":
                    receipt["families"]["REG"]["fold_probabilities"][0][0] = f32(.875)
                    refresh(new, 0)
                elif kind == "cleanup":
                    new["work_cleanup_complete"] = False
                else:
                    new["resources_passed"] = False
                self.assertFalse(A._audit_objects(new, old, order, fixture_only=True)["passed"])

    def test_fixed_selection_order_series_scope_and_mask_reject(self):
        for kind in ("selection", "series", "order", "artificial", "overlap"):
            with self.subTest(kind=kind):
                new, old, order = fixture()
                if kind == "selection":
                    new["fixture_receipt"]["selection_pin"]["sha256"] = "0" * 64
                elif kind == "series":
                    new["fixture_receipt"]["train_series_csv_sha256"] = "0" * 64
                elif kind == "order":
                    order.reverse()
                elif kind == "artificial":
                    new["actual_native_inference_executed"] = True
                else:
                    new["structural_failures"] = new["operational_failures"] = {order[0]: {}}
                with self.assertRaises(ValueError):
                    A._audit_objects(new, old, order, fixture_only=True)

    def test_public_hash_and_json_bounds_before_bodies(self):
        with patch.object(Path, "read_bytes", side_effect=AssertionError("No body allowed")):
            with self.assertRaisesRegex(ValueError, "historical"):
                A.audit_diagnostic("absent.json", "a" * 64, "absent.json", "b" * 64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pt = root / "not-json.pt"
            pt.write_bytes(b"ARTIFICIAL")
            with patch.object(Path, "read_bytes", side_effect=AssertionError("No PT body allowed")):
                with self.assertRaises(ValueError):
                    A.read_result(pt, "a" * 64)
            duplicate = root / "duplicate.json"
            raw = b'{"complete":true,"complete":false}'
            duplicate.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                A.read_result(duplicate, hashlib.sha256(raw).hexdigest())
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
