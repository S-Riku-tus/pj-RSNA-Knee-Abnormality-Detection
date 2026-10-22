"""New explicit operational UID recovery around an unchanged serving component.

The sealed router can use this proxy at its ``recovery`` global binding. Its
``failures`` dictionary is then a routing union for membership/skip purposes;
genuine structural records and executor-bound operational records stay in
separate ledgers and schemas. Successful payloads use the original consumer.

The legacy router's JSON is quarantined outside Kaggle Output. In particular,
its structural_failures/phase_contract_passed/fallback_count=0 assertions must
never be published as qualification for operational recovery. The final result
uses a new schema and explicit counts. No MRI, model, network or rank code is
implemented here; success-subset integration remains a separate fixed policy.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import tempfile
from pathlib import Path

from .contracts import TARGETS
from .isolated_uid_executor import IsolatedUIDMultiprocessingProxy, OperationalUIDFailure, TaskSourcePin

OPERATIONAL_REASONS = frozenset(("deadline_exceeded", "worker_exit", "missing_result"))
RESULT_SCHEMA = "native_isolated_uid_operational_recovery_v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


class OperationalRecoveryProxy:
    """Preserve original structural validation; independently bind operational loss."""

    def __init__(self, original, *, executor_receipts, task_source_sha256):
        self.original = original
        self.executor_receipts = executor_receipts
        self.task_source_sha256 = task_source_sha256
        self.operational = {}
        self._routing_failures = None

    def __getattr__(self, name):
        return getattr(self.original, name)

    def _verify_operational(self, failure):
        _require(type(failure) is OperationalUIDFailure and failure.reason in OPERATIONAL_REASONS,
                 "Only explicit executor operational losses are recoverable")
        _require(failure.task_sha256 == self.task_source_sha256 and type(failure.uid) is str and bool(failure.uid),
                 "Operational task source/UID differs")
        matches = []
        for executor in self.executor_receipts():
            _require(executor["task_pin"]["sha256"] == self.task_source_sha256,
                     "Executor evidence uses a different task source")
            matches.extend(row for row in executor["records"] if row["uid"] == failure.uid)
        _require(len(matches) == 1, "Operational UID needs exactly one owned executor record")
        record = matches[0]
        _require(record.get("returned_operational_failure") is True and record.get("cancelled") is False
                 and record.get("direct_child_reaped") is True and record.get("owned_files_cleaned") is True
                 and record.get("callback_count") == 1
                 and record.get("failure") == failure.receipt(),
                 "Operational failure is not positively bound to a reaped/cleaned executor result")

    def _split(self, failures, *, policy):
        _require(type(failures) is dict, "Explicit UID failure routing map required")
        structural, operational = {}, {}
        for uid, failure in failures.items():
            _require(type(uid) is str and getattr(failure, "uid", None) == uid, "Failure payload UID/key differs")
            if type(failure) is self.original.FailedUID:
                policy.verify_failure(failure)
                structural[uid] = failure
            elif type(failure) is OperationalUIDFailure:
                _require(self.operational.get(uid) is failure, "Operational failure was not consumed by this execution")
                self._verify_operational(failure)
                operational[uid] = failure
            else:
                raise ValueError("Unknown failure type; old structural provenance cannot be invented")
        _require(set(operational) == set(self.operational), "Operational ledger/routing coverage differs")
        return structural, operational

    def consume_uid(self, device, payload, *, policy, expected_uids, successful_consume,
                    predictions, receipts, failures, lock):
        if type(payload) is not OperationalUIDFailure:
            return self.original.consume_uid(
                device, payload, policy=policy, expected_uids=expected_uids, successful_consume=successful_consume,
                predictions=predictions, receipts=receipts, failures=failures, lock=lock)
        self._verify_operational(payload)
        with lock:
            _require(payload.uid in expected_uids and payload.uid not in failures
                     and payload.uid not in predictions and payload.uid not in receipts
                     and payload.uid not in self.operational, "Unknown, duplicate or overlapping operational UID")
            if self._routing_failures is None:
                self._routing_failures = failures
            _require(self._routing_failures is failures, "A recovery proxy cannot span execution failure maps")
            self.operational[payload.uid] = payload
            # The unchanged router skips all entries in this routing union.
            # Its raw structural_failures JSON is quarantined and reclassified.
            failures[payload.uid] = payload

    def impute_all_uid_native(self, order, baseline_strings, predictions, receipts, failures, *, policy):
        structural, operational = self._split(failures, policy=policy)
        self.original._baseline(order, baseline_strings)
        expected, success = set(order), set(predictions)
        _require(not success.intersection(failures) and success.union(failures) == expected
                 and set(receipts) == success, "S/F_structural/F_operational must exactly partition N")
        reduced_order = [uid for uid in order if uid not in operational]
        if reduced_order:
            reduced_matrix, old_summary = self.original.impute_all_uid_native(
                reduced_order, {uid: baseline_strings[uid] for uid in reduced_order}, predictions, receipts,
                structural, policy=policy)
            values = dict(zip(reduced_order, reduced_matrix))
            forwards = old_summary["logical_full_fold_model_forward_calls"]
        else:
            _require(not predictions and not receipts and not structural, "All-operational partition differs")
            values, forwards = {}, 0
        matrix = [([float(v) for v in baseline_strings[uid]] if uid in operational else values[uid]) for uid in order]
        summary = {
            "schema_version": "native_structural_operational_UID_partition_v1",
            "N": len(order), "S": len(success), "F": len(failures),
            "F_structural": len(structural), "F_operational": len(operational),
            "S_plus_F_structural_plus_F_operational_equals_N": True,
            "logical_full_fold_model_forward_calls": forwards, "failed_UID_model_forward_calls": 0,
            "placeholder_policy": "finite completed-parent values only; union mask excludes all failures from native ranks",
            "structural_failures": {uid: vars(structural[uid]) for uid in order if uid in structural},
            "operational_failures": {uid: operational[uid].receipt() for uid in order if uid in operational},
        }
        _require(forwards == 4 * len(success), "Successful UID full-forward count differs")
        return matrix, summary

    def all_uid_receipts(self, success_receipts, failures):
        structural = {uid: value for uid, value in failures.items() if type(value) is self.original.FailedUID}
        operational = {uid: value for uid, value in failures.items() if type(value) is OperationalUIDFailure}
        _require(set(structural).union(operational) == set(failures)
                 and set(operational) == set(self.operational)
                 and not set(success_receipts).intersection(failures), "All-UID receipt partition differs")
        result = self.original.all_uid_receipts(success_receipts, structural)
        for uid, failure in operational.items():
            _require(self.operational.get(uid) is failure, "Unconsumed operational receipt")
            self._verify_operational(failure)
            result[uid] = {"operational_failure": failure.receipt(), "logical_full_fold_model_forward_calls": 0,
                           "GPU_forward_skipped": True, "native_baseline_placeholder": True,
                           "excluded_from_native_success_subset_ranks": True, "old_structural_failure_claimed": False}
        return result

    def normalize_result(self, raw, *, order, baseline_strings, policy, executor_receipts):
        """Validate/reclassify a completed legacy component; never inherit its certificate."""
        _require(_canonical(executor_receipts) == _canonical(self.executor_receipts()),
                 "Published executor receipts differ from the positive recovery evidence")
        _require(type(executor_receipts) is list and bool(executor_receipts),
                 "Nonempty completed executor evidence is required")
        records = []
        for executor in executor_receipts:
            _require(executor.get("schema_version") == "isolated_uid_executor_v1"
                     and executor.get("failure_mode") == "return"
                     and executor.get("task_pin", {}).get("sha256") == self.task_source_sha256
                     and executor.get("monitor_joined") is True
                     and executor.get("temporary_root_cleaned") is True and not executor.get("internal_errors"),
                     "All positively bound isolated executors must be joined and cleaned")
            records.extend(executor["records"])
        record_uids = [record["uid"] for record in records]
        _require(len(record_uids) == len(set(record_uids)) == len(order) and set(record_uids) == set(order),
                 "Executor records must cover every test UID exactly once")
        for record in records:
            operational = record["uid"] in self.operational
            _require(record.get("direct_child_reaped") is True and record.get("owned_files_cleaned") is True
                     and record.get("cancelled") is False and record.get("callback_count") == 1
                     and record.get("returned_operational_failure") is operational
                     and (operational or "failure" not in record),
                     "Executor success/structural/operational classification or ownership differs")
        _require(raw.get("complete") is True and raw.get("phase_contract_passed") is True,
                 "Original router engineering invariants did not complete")
        _require(self._routing_failures is not None or not self.operational, "Operational routing ledger missing")
        routing = self._routing_failures or {}
        # Structural-only runs have no operational consume call to bind routing.
        if not self.operational:
            routing = {uid: self.original.FailedUID(**fields) for uid, fields in raw["structural_failures"].items()}
        structural, operational = self._split(routing, policy=policy)
        _require(raw["structural_failures"] == {uid: vars(value) for uid, value in routing.items()},
                 "Quarantined raw failure routing records differ")
        failed = set(structural).union(operational)
        rows = raw["probabilities_test_order"]
        _require(raw["studies"] == len(order) and [row["StudyInstanceUID"] for row in rows] == order
                 and set(raw["perUID_receipts"]) == set(order) and raw["targets"] == list(TARGETS),
                 "Native result UID/target dimensions/order differ")
        successes = {row["StudyInstanceUID"]: row["probabilities"] for row in rows if row["StudyInstanceUID"] not in failed}
        success_receipts = {uid: value for uid, value in raw["perUID_receipts"].items() if uid not in failed}
        matrix, partition = self.impute_all_uid_native(order, baseline_strings, successes, success_receipts, routing, policy=policy)
        _require([row["probabilities"] for row in rows] == matrix and raw["uid_recovery"] == partition,
                 "Native values/placeholder/partition changed after projection")
        _require(raw["perUID_receipts"] == self.all_uid_receipts(success_receipts, routing), "Per-UID zero-forward receipts differ")
        _require(raw["full_fold_model_forward_calls"] == 4 * len(successes), "Native success full-forward total differs")
        inherited_keys = ("schema_version", "phase_contract_passed", "fallback_count", "fallback_count_scope",
                          "P019_reader_recovery_enabled")
        result = {key: value for key, value in raw.items() if key not in (*inherited_keys, "structural_failures")}
        result.update(
            schema_version=RESULT_SCHEMA, operational_recovery_contract_passed=True,
            legacy_router_assertions={key: raw.get(key) for key in inherited_keys},
            legacy_assertions_scope="internal unchanged component checks only; not this new operational contract's qualification",
            structural_failures=partition["structural_failures"], operational_failures=partition["operational_failures"],
            failure_UID_union=[uid for uid in order if uid in failed],
            coverage={"N": len(order), "S": len(successes), "F_structural": len(structural), "F_operational": len(operational)},
            operational_UID_fallback_count=len(operational), successful_UID_model_fallback_count=0,
            executor_receipts=executor_receipts,
            qualification={"new_candidate_runtime_contract": True, "old_structural_policy_extended": False,
                           "old_parent_or_S6_qualification_inherited": False, "Public_accuracy_evaluated": False},
            runtime_changes={"reader_and_model_source_bytes_changed": False, "success_four_fold_FP32_math_changed": False,
                             "executor": "isolated_owned_UID_subprocess", "operational_failure_projection": "explicit new schema"},
        )
        return result


def execute_router_with_operational_recovery(router, args, plan, started, *, task_source_sha256,
                                             uid_deadline_seconds=180.0, temporary_parent=None):
    """Quarantine legacy JSON, run original execute, publish only a new qualified schema.

    This is prepared integration code. Calling it performs the router's actual
    workload; the artificial tests use source callbacks/projection with fakes.
    Unexpected exceptions still raise, leaving the outer parent fallback intact.
    """
    output = Path(args.output)
    _require(not output.exists() and not output.is_symlink(), "Fresh operational result path required")
    original_recovery, original_mp = router.recovery, router.multiprocessing
    pin = TaskSourcePin.from_callable(router.prepare_hidden_task, expected_sha256=task_source_sha256)
    mp_proxy = IsolatedUIDMultiprocessingProxy(original_mp, pin, uid_deadline_seconds=uid_deadline_seconds,
                                              temporary_parent=temporary_parent, operational_failure_mode="return")
    def snapshots():
        return [executor.receipt() for executor in mp_proxy.executors]

    recovery = OperationalRecoveryProxy(original_recovery, executor_receipts=snapshots, task_source_sha256=pin.sha256)
    router.recovery, router.multiprocessing = recovery, mp_proxy
    try:
        with tempfile.TemporaryDirectory(prefix="operational-native-quarantine-", dir=temporary_parent) as temporary:
            raw_args = copy.copy(args)
            raw_args.output = Path(temporary) / "legacy-native-unqualified.json"
            raw = router.execute(raw_args, plan, started)
            _require(_canonical(json.loads(raw_args.output.read_bytes())) == _canonical(raw),
                     "Quarantined legacy file/returned component differ")
            result = recovery.normalize_result(raw, order=plan["test"], baseline_strings=plan["baseline_strings"],
                                                policy=plan["recovery_policy"], executor_receipts=snapshots())
            result["quarantined_legacy_result_sha256"] = hashlib.sha256(raw_args.output.read_bytes()).hexdigest()
            result["quarantined_legacy_result_not_published"] = True
            startup = raw_args.output.parent / "P018_MIXED_STARTUP.json"
            if startup.is_file():
                _require(startup.stat().st_size <= 2 * 1024**2, "Startup metadata exceeds text limit")
                result["unchanged_model_startup_component"] = json.loads(startup.read_bytes())
                result["unchanged_model_startup_component_scope"] = "model/source startup evidence only; no operational recovery qualification"
        result["legacy_quarantine_removed"] = True
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return result
    except BaseException as error:
        for executor in mp_proxy.executors:
            executor.terminate()
            executor.join()
        if not output.exists():
            with output.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump({"schema_version": RESULT_SCHEMA, "complete": False,
                           "operational_recovery_contract_passed": False, "error_type": type(error).__name__,
                           "error": str(error), "legacy_assertions_published": False}, stream, indent=2, allow_nan=False)
                stream.write("\n")
        raise
    finally:
        router.recovery, router.multiprocessing = original_recovery, original_mp


def write_native_projection(result, output_dir):
    """Fresh native CSV and structural/operational union mask; never a submission."""
    _require(result.get("schema_version") == RESULT_SCHEMA and result.get("complete") is True
             and result.get("operational_recovery_contract_passed") is True, "A new operational result is required")
    output_dir = Path(output_dir)
    _require(not output_dir.exists() and not output_dir.is_symlink(), "Fresh native projection directory required")
    targets, rows = result["targets"], result["probabilities_test_order"]
    _require(targets == list(TARGETS), "Exact twelve ordered targets required")
    order = [row["StudyInstanceUID"] for row in rows]
    _require(len(order) == len(set(order)) == result["coverage"]["N"] and bool(order), "Unique nonempty native UID order required")
    structural, operational = set(result["structural_failures"]), set(result["operational_failures"])
    _require(not structural.intersection(operational) and structural.union(operational) == set(result["failure_UID_union"])
             and structural.union(operational) <= set(order), "Structural/operational union mask differs")
    _require(result["failure_UID_union"] == [uid for uid in order if uid in structural or uid in operational]
             and result["coverage"]["S"] == len(order) - len(structural) - len(operational)
             and result["coverage"]["F_structural"] == len(structural)
             and result["coverage"]["F_operational"] == len(operational), "Failure coverage counts/order differ")
    for row in rows:
        values = row["probabilities"]
        _require(type(values) is list and len(values) == 12
                 and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in values),
                 "Native projection requires finite probabilities even for masked cells")
    output_dir.mkdir(parents=True, exist_ok=False)
    native, mask = output_dir / "native-mean.csv", output_dir / "native-failures.csv"
    with native.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["StudyInstanceUID", *targets])
        writer.writerows([row["StudyInstanceUID"], *row["probabilities"]] for row in rows)
    with mask.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["StudyInstanceUID"])
        writer.writerows([uid] for uid in order if uid in structural or uid in operational)
    receipt = {"schema_version": "native_operational_union_mask_projection_v1", "complete": True,
               "native_csv": str(native), "native_csv_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
               "failure_mask_csv": str(mask), "failure_mask_sha256": hashlib.sha256(mask.read_bytes()).hexdigest(),
               "studies": len(order), "successful_studies": result["coverage"]["S"],
               "structural_failure_count": len(structural), "operational_failure_count": len(operational),
               "total_failure_count": len(structural) + len(operational),
               "structural_failure_UIDs": [uid for uid in order if uid in structural],
               "operational_failure_UIDs": [uid for uid in order if uid in operational],
               "coverage": result["coverage"], "failure_UID_union": result["failure_UID_union"],
               "all_failed_UIDs_excluded_from_native_ranking": True, "old_structural_failure_claimed_for_operational": False,
               "logical_full_fold_model_forward_calls": result["full_fold_model_forward_calls"],
               "rank_or_submission_CSV_written": False}
    with (output_dir / "projection-receipt.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt
