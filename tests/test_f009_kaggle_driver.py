"""Stdlib route/control fixtures; no Torch import, model, MRI, GPU or actual release."""
import ast
import copy
import csv
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rsna_knee import f009_kaggle_driver as D

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "artifacts/research/20261016-f006-execution-v1/static-native-source/serving/serving_router.py"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
           "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def function(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def module(name, **attrs):
    value = types.ModuleType(name)
    value.__dict__.update(attrs)
    return value


class Owner:
    def __init__(self, calls, kind):
        self.closed, self.calls, self.kind = False, calls, kind

    def close(self):
        if self.closed:
            raise ValueError("double close")
        self.closed = True
        self.calls.append("close_"+self.kind)


def router_fixture(root, *, failure=None):
    """Execute the derived control flow using explicit fake GPU/model components."""
    calls = []
    names = ("CLS:0", "CLS:1", "REG:0", "REG:1")
    fingerprints = {name: {"fixture_only": True} for name in names}
    pair = {"pair_fingerprint_sha256": "a"*64, "family_fold_manifest_sha256": {},
            "model_classes": {name: "FAKE_MODEL" for name in names},
            "family_pairs": {"REG": {"bindings": {0: {"config": {"preprocess": {}}}}}}}
    cpu, gpu = Owner(calls, "CPU"), Owner(calls, "GPU")

    def reconstruct(*args, **kwargs):
        calls.append("reconstruct")
        assert args[3] == {"fixture_documents": True}
        assert kwargs["regional_original"] == "FAKE_REG_MODULE"
        assert kwargs["permission_path"] == root / "FAKE_CPU_permission.json"
        assert kwargs["dependency_pins"] == {"fixture_only": True}
        assert kwargs["execute"] is True and "common_src" not in kwargs
        if failure == "CPU":
            raise RuntimeError("ARTIFICIAL CPU construction failure")
        return cpu, {"model_fingerprints": fingerprints}

    def transfer(model, core):
        assert model is cpu
        calls.append("transfer")
        if failure == "transfer":
            raise RuntimeError("ARTIFICIAL GPU transfer failure")
        return gpu, 12

    def consume_four(device, payload, **kwargs):
        uid, arrays, receipt = payload
        calls.extend(["full_forward"]*4)
        kwargs["family_predictions"]["CLS"][uid] = [.25]*12
        kwargs["family_predictions"]["REG"][uid] = [.75]*12
        kwargs["predictions"][uid] = [.5]*12
        kwargs["receipts"][uid] = {**receipt, "logical_full_fold_model_forward_calls": 4}

    mixed = SimpleNamespace(reconstruct_cpu_models=reconstruct, models_on_two_devices=transfer,
                            model_fingerprints_from_pair=lambda _: fingerprints, consume_four=consume_four)
    recovery = SimpleNamespace(
        prepare_uid=lambda item, policy: (item[1], {"synthetic_five_array_contract": True}, {"array_sha256": "f"*64}),
        consume_uid=lambda device, payload, **kwargs: kwargs["successful_consume"](device, payload),
        impute_all_uid_native=lambda order, baseline, predictions, receipts, failures, **kwargs:
            ([predictions[uid] for uid in order], {"logical_full_fold_model_forward_calls": 4*len(order)}),
        all_uid_receipts=lambda receipts, failures: receipts)

    def pipeline(pool_factory, task, items, devices, consume, *, queue_max, outstanding_limit):
        calls.append("pipeline")
        assert (queue_max, outstanding_limit, devices) == (2, 2, ("cuda:0", "cuda:1"))
        assert pool_factory() == "FAKE_POOL_2"
        for index, item in enumerate(items):
            consume(devices[index % 2], task(item))
        return {"consumer_threads_joined": True, "queue_unfinished_tasks": 0, "delivery_unfinished_tasks": 0}

    class Sampler:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def result(self):
            return {"artificial_resource_sampler": True}

    torch = module("torch", __version__="ARTIFICIAL_NO_TORCH", version=SimpleNamespace(cuda="FAKE"),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 0)),
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
            get_device_name=lambda _: "Tesla T4", reset_peak_memory_stats=lambda _: None,
            max_memory_reserved=lambda _: 1024))
    imports = {"torch": torch,
        "frozen_streaming_runtime": module("frozen_streaming_runtime", install_offline_decoders=lambda *args: {},
                                           verify_compressed_phantoms=lambda *args: {"fixture_only": True}),
        "native_runtime": module("native_runtime", DECODER_MANIFEST_SHA="d"*64, PHANTOM_MANIFEST_SHA="e"*64),
        "online_diagnostic_runtime": module("online_diagnostic_runtime", resources_passed=lambda *args: True),
        "p006_profile_runtime": module("p006_profile_runtime", Resources=Sampler),
        "p010_shared_runtime": module("p010_shared_runtime", run_bounded_pipeline=pipeline),
        "rsna_knee.triplet_cache": module("rsna_knee.triplet_cache", array_hashes=lambda _: "f"*64)}
    tree = ast.parse(D.derive_router_source(OLD.read_bytes()))
    namespace = {"c": D, "json": json, "math": math, "time": D.time, "Path": Path,
        "recovery": recovery, "multiprocessing": SimpleNamespace(active_children=lambda: [],
            get_context=lambda _: SimpleNamespace(Pool=lambda processes: "FAKE_POOL_"+str(processes))),
        "threading": __import__("threading"), "partial": __import__("functools").partial, "gc": __import__("gc"),
        "verify_hidden_payload": lambda payload, *args: payload}
    exec(compile(ast.Module(body=[function(tree, name) for name in ("prepare_hidden_task", "execute")], type_ignores=[]),
                 "F009_artificial_router_only", "exec"), namespace)
    plan = {"boundary": SimpleNamespace(bootstrap=lambda: ("FAKE_REG_MODULE", {"fixture_only": True})),
        "mixed": mixed, "core": SimpleNamespace(probability_mean_reference=lambda a, b: [(x+y)*.5 for x, y in zip(a, b)]),
        "pair": pair, "p": SimpleNamespace(FAMILIES=("CLS", "REG")), "context_sha256": "1"*64,
        "source_inventory_pin": {"bytes": 1, "sha256": "2"*64}, "router_pin": {"bytes": 2, "sha256": "3"*64},
        "input_receipt": {"fixture_only": True}, "baseline_csv_sha256": "4"*64,
        "models_roots": {}, "views": {}, "documents": {"fixture_documents": True}, "permission": {"reconstruction_permission": {}},
        "regional": None, "CLS_pair": {}, "conditional_plan": {}, "CLS_transport": None, "namespace_helper": None,
        "dependency_pins": {"fixture_only": True}, "CPU_permission_path": root / "FAKE_CPU_permission.json",
        "CPU_permission_sha256": "5"*64, "test": ["ARTIFICIAL_0", "ARTIFICIAL_1", "ARTIFICIAL_2"],
        "sample": ["ARTIFICIAL_0", "ARTIFICIAL_1", "ARTIFICIAL_2"], "series": {"ARTIFICIAL_"+str(i): [] for i in range(3)},
        "input_root": root, "recovery_policy": "FAKE_READER_POLICY", "baseline_strings": {},
        "io": SimpleNamespace(TARGETS=TARGETS, project_predictions=lambda test, sample, predictions, receipts:
                              [{"StudyInstanceUID": uid, "probabilities": predictions[uid]} for uid in test])}
    args = SimpleNamespace(output=root / "raw-result.json", decoder_root=root / "FAKE_DECODER", budget_seconds=3600,
                           permission_sha="6"*64)
    return SimpleNamespace(args=args, plan=plan, imports=imports, execute=namespace["execute"], calls=calls, cpu=cpu, gpu=gpu)


class DriverSource(unittest.TestCase):
    def test_default_disabled_before_context_and_unready_before_support(self):
        result = subprocess.run([sys.executable, "-B", str(Path(D.__file__))], capture_output=True, text=True, check=True)
        self.assertFalse(json.loads(result.stdout)["context_model_MRI_GPU_read"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = root / "context.json"
            context.write_text(json.dumps({"schema_version": D.CONTEXT_SCHEMA, "actual_entry_enabled": False}), encoding="utf-8")
            args = SimpleNamespace(output=root / "native.json", context=context, context_sha=D.sha(context))
            with patch.object(D, "__package__", D.SUPPORT_PACKAGE), patch.object(D, "_support", side_effect=AssertionError("Support must stay closed")):
                with self.assertRaisesRegex(ValueError, "UNREADY"):
                    D.prepare_plan(args)
            args.context = root / "never_open_checkpoint.pt"
            with patch.object(D, "__package__", D.SUPPORT_PACKAGE), patch.object(D, "sha", side_effect=AssertionError("No model body hash")):
                with self.assertRaisesRegex(ValueError, "metadata JSON"):
                    D.prepare_plan(args)

    def test_router_changes_are_limited_and_task_queue_algorithms_retained(self):
        original = ast.parse(OLD.read_bytes())
        derived = ast.parse(D.derive_router_source(OLD.read_bytes()))
        for name in ("prepare_hidden_task", "verify_hidden_payload"):
            self.assertEqual(ast.dump(function(original, name)), ast.dump(function(derived, name)))
        old_execute, new_execute = function(original, "execute"), function(derived, "execute")
        def calls(tree, name):
            return [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name]
        self.assertEqual(ast.dump(calls(old_execute, "run_bounded_pipeline")[0]), ast.dump(calls(new_execute, "run_bounded_pipeline")[0]))
        for name in ("consume_success", "consume", "pool_factory"):
            self.assertEqual(ast.dump(next(n for n in ast.walk(old_execute) if isinstance(n, ast.FunctionDef) and n.name == name)),
                             ast.dump(next(n for n in ast.walk(new_execute) if isinstance(n, ast.FunctionDef) and n.name == name)))
        rendered = ast.unparse(derived)
        self.assertNotIn("def preflight", rendered)
        self.assertNotIn("isolated_contracts", rendered)
        self.assertNotIn("CPU_smoke'][", rendered)
        self.assertIn("regional_original=original", rendered)
        with self.assertRaisesRegex(ValueError, "Exact original"):
            D.derive_router_source(OLD.read_bytes()+b"\n")

    def test_derived_execution_new_constructor_lifetimes_and_error_cleanup(self):
        for failure in (None, "CPU", "transfer"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                f = router_fixture(Path(temporary), failure=failure)
                with patch.dict(sys.modules, f.imports):
                    if failure:
                        with self.assertRaisesRegex(RuntimeError, "ARTIFICIAL"):
                            f.execute(f.args, f.plan, D.time.monotonic()-1)
                    else:
                        result = f.execute(f.args, f.plan, D.time.monotonic()-1)
                        self.assertTrue(result["complete"])
                        self.assertEqual(result["full_fold_model_forward_calls"], 12)
                        self.assertEqual(f.calls.count("full_forward"), 12)
                        self.assertEqual(result["probabilities_test_order"][0]["probabilities"], [.5]*12)
                        self.assertLess(f.calls.index("close_CPU"), f.calls.index("pipeline"))
                        self.assertEqual(f.calls.count("close_GPU"), 1)
                    if failure != "CPU":
                        self.assertTrue(f.cpu.closed)
                    written = json.loads(f.args.output.read_bytes())
                    self.assertEqual(written["complete"], failure is None)

    def test_close_both_even_on_cleanup_error(self):
        closed = []
        cpu = SimpleNamespace(closed=False, close=lambda: closed.append("CPU"))
        gpu = SimpleNamespace(closed=False, close=lambda: (_ for _ in ()).throw(RuntimeError("GPU close failed")))
        with self.assertRaisesRegex(RuntimeError, "GPU close failed"):
            D.close_models(cpu, gpu)
        self.assertEqual(closed, ["CPU"])

    def test_new_result_projection_union_and_fatal_whole_error(self):
        op_path = ROOT / "src/rsna_knee/operational_uid_recovery.py"
        namespace = {"_require": D.require, "Path": Path, "TARGETS": TARGETS, "math": math, "csv": csv,
                     "hashlib": hashlib, "json": json, "RESULT_SCHEMA": "native_isolated_uid_operational_recovery_v1"}
        exec(compile(ast.Module(body=[function(ast.parse(op_path.read_bytes()), "write_native_projection")], type_ignores=[]), str(op_path), "exec"), namespace)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            order = ["ARTIFICIAL_SUCCESS", "ARTIFICIAL_STRUCTURAL", "ARTIFICIAL_OPERATIONAL"]
            raw = {"schema_version": namespace["RESULT_SCHEMA"], "complete": True, "operational_recovery_contract_passed": True,
                "targets": TARGETS, "probabilities_test_order": [{"StudyInstanceUID": uid, "probabilities": [.5]*12} for uid in order],
                "coverage": {"N": 3, "S": 1, "F_structural": 1, "F_operational": 1},
                "structural_failures": {order[1]: {"fixture_only": True}}, "operational_failures": {order[2]: {"fixture_only": True}},
                "failure_UID_union": order[1:], "full_fold_model_forward_calls": 4, "studies": 3,
                "reconstruction_component": {"model_fingerprints": {"fixture_only": True}},
                "executor_receipts": [{"fixture_only": True}], "legacy_quarantine_removed": True,
                "runtime_changes": {"reader_and_model_source_bytes_changed": False}}

            def run(router, args, plan, started, **kwargs):
                self.assertEqual(kwargs, {"task_source_sha256": "a"*64, "uid_deadline_seconds": 180.0})
                D.dump(args.output, raw)
                return copy.deepcopy(raw)

            operational = SimpleNamespace(RESULT_SCHEMA=namespace["RESULT_SCHEMA"],
                execute_router_with_operational_recovery=run, write_native_projection=namespace["write_native_projection"])
            args = SimpleNamespace(output=root / "native-result.json")
            plan = {"router_pin": {"sha256": "a"*64}, "root_release_pin": {"sha256": "b"*64},
                    "pair": {"pair_fingerprint_sha256": "c"*64}, "spawn_router_source": {"fixture_only": True}}
            result = D._finish_native(args, plan, None, operational, started=D.time.monotonic(), temporary_parent=root)
            self.assertEqual(result["schema_version"], D.RESULT_SCHEMA)
            self.assertTrue(result["runtime_changes"]["CLS_model_source_bytes_changed"])
            self.assertNotIn("reader_and_model_source_bytes_changed", result["runtime_changes"])
            adapter = json.loads((root / "adapter-receipt.json").read_bytes())
            self.assertEqual(adapter["schema_version"], D.ADAPTER_SCHEMA)
            self.assertEqual(adapter["failure_UID_union"], order[1:])
            with (root / "isolated-uid-projection/native-failures.csv").open(newline="") as stream:
                self.assertEqual(list(csv.reader(stream)), [["StudyInstanceUID"], *[[uid] for uid in order[1:]]])
            self.assertFalse(list(root.glob("f009-native-component-*")))
            operational.execute_router_with_operational_recovery = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fatal whole component"))
            args.output = root / "failed-native.json"
            with self.assertRaisesRegex(RuntimeError, "fatal whole"):
                D._finish_native(args, plan, None, operational, started=D.time.monotonic(), temporary_parent=root)
            self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
