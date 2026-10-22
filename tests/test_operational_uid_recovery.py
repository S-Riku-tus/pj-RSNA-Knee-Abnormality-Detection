"""Artificial CPU integration with exact sealed callback/recovery/projection source."""

import ast
import copy
import csv
import importlib.util
import json
import math
import multiprocessing
import struct
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "artifacts/research/20261016-f006-execution-v1/static-native-source"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from operational_uid_fixtures import prepare_operational_fixture  # noqa: E402

from rsna_knee import success20_blend  # noqa: E402
from rsna_knee.contracts import ID, TARGETS  # noqa: E402
from rsna_knee.isolated_uid_executor import IsolatedUIDExecutor, OperationalUIDFailure, TaskSourcePin  # noqa: E402
from rsna_knee.operational_uid_recovery import (  # noqa: E402
    RESULT_SCHEMA,
    OperationalRecoveryProxy,
    execute_router_with_operational_recovery,
    write_native_projection,
)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


class OperationalRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not STATIC.exists():
            raise unittest.SkipTest("Ignored sealed historical sources are unavailable")
        cls.recovery = load("uid_recovery_adapter", STATIC / "serving/uid_recovery_adapter.py")
        cls.io = load("operational_exact_serving_io", STATIC / "common/serving_contracts.py")
        cls.pipeline = load("operational_exact_p010", STATIC / "common/reused/code/scripts/p010_shared_runtime.py")
        tree = ast.parse((STATIC / "serving/serving_router.py").read_bytes())
        execute = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "execute")
        cls.consume_node = next(node for node in ast.walk(execute)
                                if isinstance(node, ast.FunctionDef) and node.name == "consume")
        cls.skip_loop = next(node for node in ast.walk(execute) if isinstance(node, ast.For)
                             and isinstance(node.iter, ast.Name) and node.iter.id == "rows")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.pools = []
        self.addCleanup(self.stop_pools)
        self.path_entry = str(STATIC / "serving")
        sys.path.insert(0, self.path_entry)
        self.addCleanup(lambda: sys.path.remove(self.path_entry))
        self.policy = self.recovery.StructuralReaderPolicy(STATIC / "common")
        self.pin = TaskSourcePin.from_callable(prepare_operational_fixture)
        self.baseline = {str(i): [f"{(.1 * i):.8f}"] * 12 for i in range(1, 7)}

    def stop_pools(self):
        for pool in self.pools:
            pool.terminate()
            pool.join()

    def failure_fields(self, uid):
        (_, _, line, message), (reason, relative, sha) = next(iter(self.policy.sites.items()))
        return {"uid": uid, "reason": reason, "message": message, "source_relative": relative,
                "source_sha256": sha, "source_line": line}

    def run_source_path(self, modes, *, proxy=None, pool_factory=None):
        order = [str(i + 1) for i in range(len(modes))]
        predictions, receipts, failures, lock = {}, {}, {}, threading.Lock()
        family_predictions = {"CLS": {}, "REG": {}}
        if pool_factory is None:
            pool = IsolatedUIDExecutor(self.pin, max_workers=2, uid_deadline_seconds=.8,
                                       cleanup_seconds=.15, temporary_parent=self.temp.name,
                                       operational_failure_mode="return")
            self.pools.append(pool)
            if proxy is None:
                proxy = OperationalRecoveryProxy(
                    self.recovery, executor_receipts=lambda: [pool.receipt()], task_source_sha256=self.pin.sha256)
            def pool_factory():
                return pool
        called = []

        def consume_success(device, payload):
            uid, arrays, detail = payload
            self.assertEqual(arrays, {"artificial": b"same-array-bytes"})
            values = [f32(((int(uid) + i) % 7) / 8) for i in range(12)]
            if detail["force_nonfinite"]:
                values[0] = math.nan
            with lock:
                self.assertNotIn(uid, predictions)
                predictions[uid] = values
                receipts[uid] = {"logical_full_fold_model_forward_calls": 4, "fallback": False, "device": device}
                family_predictions["CLS"][uid] = family_predictions["REG"][uid] = values
                called.append(uid)

        namespace = {"recovery": proxy, "plan": {"recovery_policy": self.policy}, "expected_uids": set(order),
                     "consume_success": consume_success, "predictions": predictions, "receipts": receipts,
                     "failures": failures, "lock": lock}
        exec(compile(ast.Module(body=[copy.deepcopy(self.consume_node)], type_ignores=[]),
                     "sealed-serving-router-consume", "exec"), namespace)
        items = [("artificial-root", uid, mode,
                  json.dumps(self.failure_fields(uid)) if mode == "structural" else "{}", "isolated-reader-workspace")
                 for uid, mode in zip(order, modes)]
        transport = self.pipeline.run_bounded_pipeline(
            pool_factory, prepare_operational_fixture, items, ("artificial-cpu-0", "artificial-cpu-1"),
            namespace["consume"], queue_max=2, outstanding_limit=2, shutdown_seconds=5)
        baseline = {uid: self.baseline[uid] for uid in order}
        matrix, summary = proxy.impute_all_uid_native(order, baseline, predictions, receipts, failures, policy=self.policy)
        all_receipts = proxy.all_uid_receipts(receipts, failures)
        rows = self.io.project_predictions(order, list(reversed(order)), dict(zip(order, matrix)), all_receipts)
        # Execute the exact sealed membership-skip/family-mean source loop.
        def mean(left, right):
            return [f32(f32(a + b) * .5) for a, b in zip(left, right)]

        exec(compile(ast.Module(body=[copy.deepcopy(self.skip_loop)], type_ignores=[]),
                     "sealed-serving-router-failure-skip", "exec"),
             {"rows": rows, "failures": failures, "family_predictions": family_predictions,
              "core": types.SimpleNamespace(probability_mean_reference=mean),
              "c": types.SimpleNamespace(require=lambda condition, message: self.assertTrue(condition, message))})
        raw = {"schema_version": "OLD_SCHEMA_NOT_A_NEW_CERTIFICATE", "complete": True, "phase_contract_passed": True,
               "studies": len(order), "targets": list(TARGETS), "full_fold_model_forward_calls": 4 * len(called),
               "fallback_count": 0, "fallback_count_scope": "OLD_SCOPE_NOT_OPERATIONAL", "P019_reader_recovery_enabled": True,
               "uid_recovery": summary, "structural_failures": {uid: vars(failure) for uid, failure in failures.items()},
               "probabilities_test_order": rows, "sample_UID_order": list(reversed(order)),
               "perUID_receipts": all_receipts, "transport": transport}
        return raw, proxy, predictions, receipts, failures, called

    def normalize(self, raw, proxy):
        order = [row[ID] for row in raw["probabilities_test_order"]]
        return proxy.normalize_result(raw, order=order, baseline_strings={uid: self.baseline[uid] for uid in order},
                                      policy=self.policy, executor_receipts=[pool.receipt() for pool in self.pools])

    def test_all_valid_preserves_original_recovery_and_success20_exactly(self):
        raw, proxy, predictions, receipts, failures, called = self.run_source_path(["success"] * 4)
        order = list(predictions)
        matrix, _ = self.recovery.impute_all_uid_native(order, {uid: self.baseline[uid] for uid in order},
                                                       predictions, receipts, {}, policy=self.policy)
        self.assertEqual({row[ID]: row["probabilities"] for row in raw["probabilities_test_order"]}, dict(zip(order, matrix)))
        self.assertEqual(raw["perUID_receipts"], self.recovery.all_uid_receipts(receipts, {}))
        result = self.normalize(raw, proxy)
        self.assertEqual(result["coverage"], {"N": 4, "S": 4, "F_structural": 0, "F_operational": 0})
        self.assertNotIn("phase_contract_passed", result)
        self.assertNotIn("fallback_count", result)
        self.assertFalse(failures)
        self.assertEqual(set(called), {"1", "2", "3", "4"})
        projection = write_native_projection(result, Path(self.temp.name) / "native")
        with open(projection["failure_mask_csv"]) as stream:
            self.assertEqual(list(csv.DictReader(stream)), [])
        with open(projection["native_csv"]) as stream:
            native = list(csv.DictReader(stream))
        parent = [{ID: str(i), **dict(zip(TARGETS, self.baseline[str(i)]))} for i in range(1, 5)]
        original_native = [{ID: row[ID], **dict(zip(TARGETS, row["probabilities"]))}
                           for row in raw["probabilities_test_order"]]
        self.assertEqual(success20_blend.blend_rows(parent, native)[0],
                         success20_blend.blend_rows(parent, original_native)[0])

    def test_timeout_and_structural_union_exclude_placeholders_but_keep_success_contribution(self):
        raw, proxy, _, _, failures, called = self.run_source_path(["success", "timeout", "structural", "success"])
        result = self.normalize(raw, proxy)
        self.assertEqual(result["coverage"], {"N": 4, "S": 2, "F_structural": 1, "F_operational": 1})
        self.assertEqual(set(result["structural_failures"]), {"3"})
        self.assertEqual(set(result["operational_failures"]), {"2"})
        self.assertEqual(set(failures), {"2", "3"})
        self.assertEqual(set(called), {"1", "4"})
        self.assertEqual(result["full_fold_model_forward_calls"], 8)
        self.assertEqual(result["perUID_receipts"]["2"]["logical_full_fold_model_forward_calls"], 0)
        self.assertNotIn("structural_failure", result["perUID_receipts"]["2"])
        projection = write_native_projection(result, Path(self.temp.name) / "native")
        with open(projection["failure_mask_csv"]) as stream:
            self.assertEqual([row[ID] for row in csv.DictReader(stream)], ["2", "3"])
        parent = [{ID: str(i), **dict(zip(TARGETS, self.baseline[str(i)]))} for i in range(1, 5)]
        with open(projection["native_csv"]) as stream:
            native = list(csv.DictReader(stream))
        blended, receipt = success20_blend.blend_rows(parent, native, failure_mask={"2": True, "3": True})
        self.assertEqual(blended[1:3], parent[1:3])
        self.assertFalse(receipt["all_native_disabled"])
        self.assertEqual(receipt["native_blended_cell_count"], 20)
        self.assertNotEqual(blended[0]["ACL"], parent[0]["ACL"])

    def test_all_operational_has_zero_forward_and_parent_only_rank_contract(self):
        raw, proxy, _, _, _, called = self.run_source_path(["timeout", "timeout"])
        result = self.normalize(raw, proxy)
        self.assertEqual(result["coverage"], {"N": 2, "S": 0, "F_structural": 0, "F_operational": 2})
        self.assertEqual(result["full_fold_model_forward_calls"], 0)
        self.assertFalse(called)
        self.assertEqual([row["probabilities"] for row in result["probabilities_test_order"]],
                         [[float(v) for v in self.baseline[uid]] for uid in ("1", "2")])
        parent = [{ID: uid, **dict(zip(TARGETS, self.baseline[uid]))} for uid in ("1", "2")]
        native = [{ID: row[ID], **dict(zip(TARGETS, row["probabilities"]))}
                  for row in result["probabilities_test_order"]]
        blended, receipt = success20_blend.blend_rows(parent, native, failure_mask={"1": True, "2": True})
        self.assertEqual(blended, parent)
        self.assertTrue(receipt["all_native_disabled"])

    def test_unbound_operational_failure_cannot_invent_source_or_executor_evidence(self):
        proxy = OperationalRecoveryProxy(self.recovery, executor_receipts=lambda: [], task_source_sha256=self.pin.sha256)
        payload = OperationalUIDFailure("1", "deadline_exceeded", task_sha256=self.pin.sha256)
        with self.assertRaisesRegex(ValueError, "owned executor record"):
            proxy.consume_uid("cpu", payload, policy=self.policy, expected_uids={"1"},
                              successful_consume=lambda *_: self.fail("Forged operational failure reached model"),
                              predictions={}, receipts={}, failures={}, lock=threading.Lock())

    def test_nonfinite_success_wrong_uid_and_unexpected_task_stay_fatal(self):
        with self.assertRaisesRegex(ValueError, "Invalid successful"):
            self.run_source_path(["success", "nonfinite"])
        for mode in ("wrong_uid", "unexpected"):
            with self.subTest(mode=mode), self.assertRaises(self.pipeline.PipelineFailure):
                self.run_source_path(["success", mode])

    def test_projection_rejects_nonfinite_bad_uid_counts_and_labels_before_directory(self):
        raw, proxy, *_ = self.run_source_path(["success", "success"])
        result = self.normalize(raw, proxy)
        for change in ("nonfinite", "duplicate", "count", "labels"):
            bad = copy.deepcopy(result)
            if change == "nonfinite":
                bad["probabilities_test_order"][0]["probabilities"][0] = math.nan
            elif change == "duplicate":
                bad["probabilities_test_order"][0][ID] = bad["probabilities_test_order"][1][ID]
            elif change == "count":
                bad["coverage"]["S"] = 0
            else:
                bad["targets"] = list(reversed(bad["targets"]))
            path = Path(self.temp.name) / change
            with self.subTest(change=change), self.assertRaises(ValueError):
                write_native_projection(bad, path)
            self.assertFalse(path.exists())

    def test_normalization_requires_nonempty_identical_complete_executor_evidence(self):
        raw, proxy, *_ = self.run_source_path(["success", "success"])
        provider = proxy.executor_receipts
        evidence = provider()
        parameters = {"order": ["1", "2"], "baseline_strings": {uid: self.baseline[uid] for uid in ("1", "2")},
                      "policy": self.policy}
        with self.assertRaisesRegex(ValueError, "Published executor receipts"):
            proxy.normalize_result(raw, executor_receipts=[], **parameters)
        variants = [[]]
        for change in ("missing", "duplicate", "unknown", "not-reaped", "unclean", "wrong-classification"):
            bad = copy.deepcopy(evidence)
            if change == "missing":
                bad[0]["records"].pop()
            elif change == "duplicate":
                bad[0]["records"].append(copy.deepcopy(bad[0]["records"][0]))
            elif change == "unknown":
                bad[0]["records"][0]["uid"] = "OTHER"
            elif change == "not-reaped":
                bad[0]["records"][0]["direct_child_reaped"] = False
            elif change == "unclean":
                bad[0]["temporary_root_cleaned"] = False
            else:
                bad[0]["records"][0]["returned_operational_failure"] = True
            variants.append(bad)
        for bad in variants:
            proxy.executor_receipts = lambda: bad
            with self.subTest(evidence=bad), self.assertRaises(ValueError):
                proxy.normalize_result(raw, executor_receipts=bad, **parameters)
        proxy.executor_receipts = provider

    def test_wrapper_quarantines_legacy_assertions_and_restores_bindings_on_success_or_error(self):
        router = types.SimpleNamespace(recovery=self.recovery, multiprocessing=multiprocessing,
                                       prepare_hidden_task=prepare_operational_fixture)
        raw_paths = []

        def execute(args, plan, started):
            raw_paths.append(args.output)
            raw, *_ = self.run_source_path(
                ["success", "timeout", "success"], proxy=router.recovery,
                pool_factory=lambda: router.multiprocessing.get_context("spawn").Pool(processes=2))
            args.output.write_text(json.dumps(raw), encoding="utf-8")
            return raw

        router.execute = execute
        path = Path(self.temp.name) / "new-result.json"
        args = types.SimpleNamespace(output=path)
        plan = {"test": ["1", "2", "3"], "baseline_strings": {uid: self.baseline[uid] for uid in ("1", "2", "3")},
                "recovery_policy": self.policy}
        result = execute_router_with_operational_recovery(
            router, args, plan, time.monotonic(), task_source_sha256=self.pin.sha256,
            uid_deadline_seconds=.8, temporary_parent=self.temp.name)
        self.assertEqual(result["schema_version"], RESULT_SCHEMA)
        self.assertTrue(result["legacy_quarantine_removed"])
        self.assertFalse(raw_paths[0].parent.exists())
        self.assertNotIn("fallback_count", json.loads(path.read_bytes()))
        self.assertIs(router.recovery, self.recovery)
        self.assertIs(router.multiprocessing, multiprocessing)

        def broken(*_):
            raise RuntimeError("artificial unexpected global error")

        router.execute = broken
        args.output = Path(self.temp.name) / "error-result.json"
        with self.assertRaisesRegex(RuntimeError, "unexpected global"):
            execute_router_with_operational_recovery(
                router, args, plan, time.monotonic(), task_source_sha256=self.pin.sha256, temporary_parent=self.temp.name)
        self.assertFalse(json.loads(args.output.read_bytes())["complete"])
        self.assertIs(router.recovery, self.recovery)
        self.assertIs(router.multiprocessing, multiprocessing)


if __name__ == "__main__":
    unittest.main()
