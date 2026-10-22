"""Independent stdlib FP32/input oracle for the fixed F009 engineering192.

This reads only explicitly pinned result JSON files. It does not import a
model, decoder, array library, training code or network client. A passing
result-file audit does not qualify saved Notebook source, Inputs or accuracy.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from pathlib import Path

SCHEMA = "f009_fixed192_native_diagnostic_result_v1"
AUDIT_SCHEMA = "f009_fixed192_independent_result_file_audit_v1"
OLD_SCHEMA = "native_isolated_uid_operational_recovery_v1"
FIXTURE_SCHEMA = "f009_fixed192_metadata_fixture_v1"
COUNT = 192
OLD_RESULT_SHA256 = "ca3bb34ff9290254bcb81d1ac8adf47e473f1a251ee2b984f8db32a61e6966e0"
OLD_RESULT_BYTES = 1580101
SELECTION_PIN = {"bytes": 14406, "sha256": "2f004a7cf359f62bc1fa56926f47bd63e5bc909388a304dcbc5080043bea54b0"}
ORDER_SHA256 = "2815c9fc459f5a02ade7d305e9acb17ca3b2e4028125b923d9e931b02abfd7c4"
SERIES_SHA256 = "573c1d80772bf41211c91b149c95677385a1c22d63f485c347f1b46c0177aef3"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
           "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
ARRAYS = {"images", "mask", "positions", "planes", "flags"}
ROLES = {"CLS:0", "CLS:1", "REG:0", "REG:1"}
MAX_RESULT_BYTES = 8 * 1024**2


def require(condition, message):
    if condition is not True:
        raise ValueError(message)


def order_sha(uids):
    return hashlib.sha256(json.dumps(uids, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def fp32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bits(value):
    return struct.pack("<f", value)


def mean_reference(left, right):
    """Independent FP32 add followed by FP32 multiply by one half."""
    return [fp32(fp32(a + b) * 0.5) for a, b in zip(left, right)]


def _probabilities(values):
    require(type(values) is list and len(values) == 12, "Exactly twelve ordered probabilities required")
    require(all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
                and fp32(value) == value for value in values), "Finite exact FP32 unit-interval probabilities required")
    return values


def _same(left, right):
    return len(left) == len(right) and all(bits(a) == bits(b) for a, b in zip(left, right))


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def read_result(path, expected_sha256):
    """JSON-only before any body read; reject duplicate keys and nonfinite JSON."""
    path = Path(path)
    require(path.suffix == ".json" and path.is_file() and not path.is_symlink()
            and 0 < path.stat().st_size <= MAX_RESULT_BYTES and _sha(expected_sha256), "Bounded explicitly pinned result JSON required")
    raw = path.read_bytes()
    pin = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    require(0 < len(raw) <= MAX_RESULT_BYTES and pin["sha256"] == expected_sha256, "Result SHA or size differs")

    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "Duplicate JSON key")
            value[key] = item
        return value

    value = json.loads(raw, object_pairs_hook=pairs,
                       parse_constant=lambda _: require(False, "Nonfinite JSON"))
    require(type(value) is dict, "Result JSON object required")
    return value, pin


def _partition(value, order, *, old=False):
    require(value["targets"] == TARGETS and value["studies"] == COUNT and value["complete"] is True,
            "Completed ordered192 result required")
    rows, receipts = value["probabilities_test_order"], value["perUID_receipts"]
    require(type(rows) is list and [row["StudyInstanceUID"] for row in rows] == order
            and type(receipts) is dict and set(receipts) == set(order), "Exact fixed192 UID order and receipt coverage required")
    structural, operational = value["structural_failures"], value["operational_failures"]
    require(type(structural) is dict and type(operational) is dict
            and not set(structural).intersection(operational), "Failure categories must be disjoint")
    failed = set(structural).union(operational)
    require(failed <= set(order) and value["failure_UID_union"] == [uid for uid in order if uid in failed],
            "Exact ordered failure union required")
    success = [uid for uid in order if uid not in failed]
    require(type(value["coverage"]) is dict and all(type(number) is int for number in value["coverage"].values())
            and value["coverage"] == {"N": COUNT, "S": len(success), "F_structural": len(structural), "F_operational": len(operational)}
            and type(value["full_fold_model_forward_calls"]) is int
            and value["full_fold_model_forward_calls"] == 4 * len(success), "N=S+F and four successful full forwards required")
    if old:
        require(not failed, "The exact historical P035192 must have 192 successes")
    probabilities = {}
    for row in rows:
        uid = row["StudyInstanceUID"]
        probabilities[uid] = _probabilities(row["probabilities"])
        if uid in failed:
            receipt = receipts[uid]
            require(type(receipt.get("logical_full_fold_model_forward_calls")) is int
                    and receipt["logical_full_fold_model_forward_calls"] == 0
                    and receipt.get("GPU_forward_skipped") is True
                    and receipt.get("native_baseline_placeholder") is True
                    and receipt.get("excluded_from_native_success_subset_ranks") is True
                    and "families" not in receipt and "family_mean_probabilities" not in receipt,
                    "Failed UID must have zero forwards and no successful family output")
            require(all(bits(value) == bits(0.5) for value in probabilities[uid]), "Diagnostic failed UID placeholder must stay synthetic0.5")
    return success, probabilities


def _success(receipt, values):
    require(receipt.get("logical_full_fold_model_forward_calls") == 4 and receipt.get("fallback") is False
            and receipt.get("family_order") == ["CLS", "REG"] and receipt.get("fold_order") == [0, 1]
            and receipt.get("inference_mode_in_consumer") is True
            and receipt.get("native_SHA_verified_before_model") is True
            and receipt.get("same_native_array_SHA_after_model") is True, "Full four-role successful receipt required")
    hashes = receipt["array_sha256"]
    require(type(hashes) is dict and set(hashes) == ARRAYS and all(_sha(value) for value in hashes.values()),
            "Exact five native array SHA fields required")
    require(type(receipt["valid_windows"]) is int and 0 < receipt["valid_windows"] <= 144, "Actual valid windows required")
    require(set(receipt["families"]) == set(receipt["family_mean_probabilities"]) == {"CLS", "REG"}, "Exact two families required")
    computed, folds = {}, {}
    for family in ("CLS", "REG"):
        record = receipt["families"][family]
        require(record.get("logical_full_fold_model_forward_calls") == 2 and record.get("fold_order") == [0, 1]
                and record.get("fallback") is False and record.get("inference_mode_in_consumer") is True
                and record.get("array_sha256") == hashes, "Each family must use the same arrays and two full folds")
        values_by_fold = record["fold_probabilities"]
        require(type(values_by_fold) is list and len(values_by_fold) == 2, "Exactly two fold vectors required")
        folds[family] = [_probabilities(values) for values in values_by_fold]
        computed[family] = mean_reference(*folds[family])
        require(_same(computed[family], _probabilities(receipt["family_mean_probabilities"][family])), "FP32 fold mean differs")
    require(_same(mean_reference(computed["CLS"], computed["REG"]), values), "FP32 family mean differs")
    return hashes, computed, folds


def _reconstruction(new, old):
    for value in (new, old):
        component = value["reconstruction_component"]
        require(component.get("complete") is True and component.get("phase_contract_passed") is True
                and component.get("actual_CPU_four_model_reconstruction") is True
                and component.get("CPU_strict_four_model_reconstruction") is True
                and component.get("artificial_stub_components") is False, "Four actual strict CPU model receipts required")
        for name, count in (("generic_factory_calls", 1), ("logical_CPU_fold_models", 4),
                            ("selected_checkpoint_loads", 4), ("strictapply_calls", 4), ("CPU_storage_alias_count", 0)):
            require(type(component.get(name)) is int and component[name] == count, "Reconstruction count differs: " + name)
        require(set(component["model_fingerprints"]) == ROLES, "Four model fingerprint roles required")
    a, b = (value["reconstruction_component"]["model_fingerprints"] for value in (new, old))
    for fold in (0, 1):
        require(a["REG:" + str(fold)] == b["REG:" + str(fold)], "Fixed REG model fingerprints differ")
        role = "CLS:" + str(fold)
        require(a[role]["model_class"] == "_rsna_f009_runtime_cls_f" + str(fold) + ".finetune_model.NativeTripletMIL"
                and a[role]["selected_epoch"] == 12
                and _sha(a[role]["checkpoint_sha256"]) and a[role]["checkpoint_sha256"] != b[role]["checkpoint_sha256"]
                and a[role]["frozen_prefix_sha256"] == b[role]["frozen_prefix_sha256"], "New fixedE12 CLS/source namespace required")
    require(a["CLS:0"]["checkpoint_sha256"] != a["CLS:1"]["checkpoint_sha256"], "Distinct new CLS folds required")


def _audit_objects(new, old, order, *, fixture_only=False):
    """Private artificial route cannot issue an actual result-file attestation."""
    require(type(order) is list and len(order) == len(set(order)) == COUNT and all(type(uid) is str and bool(uid) for uid in order),
            "Fixed ordered192 unique UIDs required")
    if fixture_only:
        require(all(uid.startswith("ARTIFICIAL_") for uid in order)
                and new.get("artificial_fixture_only") is old.get("artificial_fixture_only") is True,
                "Private artificial order/scope required")
    else:
        require(order_sha(order) == ORDER_SHA256 and new.get("artificial_fixture_only") is False,
                "Exact historical ordered192 and actual result scope required")
    require(new["schema_version"] == SCHEMA and old["schema_version"] == OLD_SCHEMA
            and old.get("operational_recovery_contract_passed") is True, "Distinct diagnostic and historical schemas required")
    require(new.get("actual_native_inference_executed") is (not fixture_only)
            and new.get("fixed_selection_sha256") == SELECTION_PIN["sha256"]
            and new.get("fixed_UID_order_sha256") == order_sha(order)
            and new.get("fixed_train_series_sha256") == SERIES_SHA256
            and new.get("baseline_kind") == "synthetic_constant_0.5_for_engineering_only"
            and new.get("P033_parent_cells_executed") == 0
            and new.get("accuracy_or_weak_Gold_metric_evaluated") is False
            and new.get("ranks_or_final_submission_written") is False,
            "Diagnostic execution/order/reference-parent scope differs")
    fixture = new["fixture_receipt"]
    require(fixture["schema_version"] == FIXTURE_SCHEMA and fixture["selection_pin"] == SELECTION_PIN
            and fixture["studies"] == COUNT and fixture["UIDs"] == order
            and fixture["ordered_UID_sha256"] == order_sha(order)
            and fixture["ordered_UID_hash_encoding"] == "compact UTF8 JSON list"
            and fixture["train_series_csv_sha256"] == SERIES_SHA256
            and fixture["fixture_complete"] is True
            and fixture["actual_fixture_created"] is (not fixture_only)
            and fixture["artificial_fixture_only"] is fixture_only,
            "Fixed selection/order/series and genuine fixture receipt required")
    require(type(new["diagnostic_engineering_passed"]) is bool and type(new["work_cleanup_complete"]) is bool
            and type(fixture["cleanup_complete"]) is bool
            and new.get("fixture_cleanup_complete") is fixture["cleanup_complete"], "Explicit engineering/cleanup booleans required")
    successes, probabilities = _partition(new, order)
    old_successes, old_probabilities = _partition(old, order, old=True)
    _reconstruction(new, old)
    previous = {uid: _success(old["perUID_receipts"][uid], old_probabilities[uid]) for uid in old_successes}
    input_differences = []
    reg_fold_differences = reg_mean_differences = cls_fold_differences = 0
    for uid in successes:
        hashes, means, folds = _success(new["perUID_receipts"][uid], probabilities[uid])
        old_hashes, old_means, old_folds = previous[uid]
        if hashes != old_hashes:
            input_differences.append(uid)
        reg_fold_differences += sum(bits(a) != bits(b) for row, prior in zip(folds["REG"], old_folds["REG"]) for a, b in zip(row, prior))
        reg_mean_differences += sum(bits(a) != bits(b) for a, b in zip(means["REG"], old_means["REG"]))
        cls_fold_differences += sum(bits(a) != bits(b) for row, prior in zip(folds["CLS"], old_folds["CLS"]) for a, b in zip(row, prior))
    all_success = len(successes) == COUNT
    parity = not input_differences and reg_fold_differences == reg_mean_differences == 0
    cleanup = fixture["cleanup_complete"] is True and new["work_cleanup_complete"] is True
    resources = (new.get("resources_passed") is True and type(new.get("spawn_workers_remaining")) is int
                 and new["spawn_workers_remaining"] == 0 and new.get("native_temp_files_removed") is True)
    passed = all_success and parity and cleanup and resources and new["diagnostic_engineering_passed"] is True
    return {"schema_version": AUDIT_SCHEMA, "passed": passed, "decision": "PASS" if passed else "HOLD",
        "artificial_fixture_only": fixture_only, "actual_result_files_audited": not fixture_only,
        "coverage": new["coverage"], "fixed_order_sha256": order_sha(order), "selection_pin": SELECTION_PIN,
        "train_series_csv_sha256": SERIES_SHA256, "all192_success_required": True, "all192_success": all_success,
        "independent_FP32_vectors_per_success": 3, "independent_FP32_new_cells_checked": 36 * len(successes),
        "independent_FP32_old_cells_checked": 36 * COUNT, "input_hash_common_successes_compared": len(successes),
        "input_hash_UID_mismatches": input_differences, "REG_fold_FP32_bit_mismatches": reg_fold_differences,
        "REG_mean_FP32_bit_mismatches": reg_mean_differences,
        "CLS_fold_FP32_bit_changes_report_only": cls_fold_differences,
        "cleanup_complete": cleanup, "sampled_resources_and_joined_workers_passed": resources,
        "new_result_engineering_passed": new["diagnostic_engineering_passed"],
        "source_saved_Input_decoder_full_runtime_or_Submit_qualification_issued": False,
        "accuracy_Gold_weak_Public_evaluated": False,
        "limits": ["Result-file/hash/numeric/input parity only; no official saved-source/Input/decoder attestation",
                   "New CLS probabilities may differ; this does not establish accuracy or Public improvement",
                   "A valid failure partition can be recorded, but any new failed UID holds this fixed192 gate"]}


def audit_diagnostic(new_path, expected_new_sha256, old_path, expected_old_sha256):
    """Actual public route binds the one historical reference before reading files."""
    require(expected_old_sha256 == OLD_RESULT_SHA256, "Exact historical P035192 result SHA required")
    new, new_pin = read_result(new_path, expected_new_sha256)
    old, old_pin = read_result(old_path, expected_old_sha256)
    require(old_pin == {"bytes": OLD_RESULT_BYTES, "sha256": OLD_RESULT_SHA256}, "Exact historical reference bytes required")
    order = [row["StudyInstanceUID"] for row in old["probabilities_test_order"]]
    result = _audit_objects(new, old, order, fixture_only=False)
    result["inputs"] = {"new_diagnostic": {"path": str(Path(new_path).resolve()), **new_pin},
                        "old_P035192": {"path": str(Path(old_path).resolve()), **old_pin}}
    return result
