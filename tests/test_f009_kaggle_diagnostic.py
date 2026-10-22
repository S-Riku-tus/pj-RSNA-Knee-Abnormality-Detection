"""Artificial control flow only; no actual metadata, models, MRI, Torch or GPU."""
import ast
import copy
import csv
import hashlib
import io
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rsna_knee import f009_kaggle_diagnostic as D
from rsna_knee import f009_kaggle_driver as S

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
           "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def projection_function():
    path = ROOT / "src/rsna_knee/operational_uid_recovery.py"
    tree = ast.parse(path.read_bytes())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "write_native_projection")
    namespace = {"_require": D.require, "Path": Path, "TARGETS": TARGETS, "math": math, "csv": csv,
                 "hashlib": hashlib, "json": json, "RESULT_SCHEMA": "native_isolated_uid_operational_recovery_v1"}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["write_native_projection"]


def artificial_component(root, *, failures=False):
    order = ["ARTIFICIAL_" + str(n) for n in range(192)]
    structural = {order[-2]: {"fixture": True}} if failures else {}
    operational = {order[-1]: {"fixture": True}} if failures else {}
    count = len(structural) + len(operational)
    raw = {"schema_version": "native_isolated_uid_operational_recovery_v1", "complete": True,
           "operational_recovery_contract_passed": True, "studies": 192, "targets": TARGETS,
           "sample_UID_order": order, "probabilities_test_order": [{"StudyInstanceUID": u, "probabilities": [.5] * 12} for u in order],
           "perUID_receipts": {u: {"fixture": True} for u in order},
           "coverage": {"N": 192, "S": 192-count, "F_structural": len(structural), "F_operational": len(operational)},
           "structural_failures": structural, "operational_failures": operational,
           "failure_UID_union": order[-2:] if failures else [], "full_fold_model_forward_calls": 4 * (192-count),
           "executor_receipts": [{"fixture": True}], "reconstruction_component": {"model_fingerprints": {"fixture": True}},
           "resources_passed": True, "spawn_workers_remaining": 0, "native_temp_files_removed": True,
           "selected_parent_notebook_sha256": "Must not become a parent-execution claim",
           "serving_candidate_gate_passed": True, "legacy_quarantine_removed": True}

    def execute(router, args, plan, started, **kwargs):
        assert kwargs["uid_deadline_seconds"] == 180.0 and kwargs["temporary_parent"] == root
        args.output.write_bytes(D.json_bytes(raw))
        return copy.deepcopy(raw)
    op = SimpleNamespace(RESULT_SCHEMA=raw["schema_version"], execute_router_with_operational_recovery=execute,
                         write_native_projection=projection_function())
    receipt = {"UIDs": order, "cleanup_complete": False, "artificial_fixture_only": True}
    plan = {"test": order, "router_pin": {"bytes": 10, "sha256": "a"*64}, "pair": {"pair_fingerprint_sha256": "b"*64},
            "root_release_pin": {"bytes": 10, "sha256": "c"*64}, "context_sha256": "d"*64,
            "spawn_router_source": {"fixture": True}}
    args = SimpleNamespace(output=root / "unused-final.json")
    return args, plan, op, receipt


class DiagnosticTests(unittest.TestCase):
    def test_default_no_body_and_actual_args_fail_before_import(self):
        with patch.object(Path, "read_bytes", side_effect=AssertionError("No data read")), redirect_stdout(io.StringIO()) as stdout:
            self.assertIsNone(D.main([]))
        self.assertIn("SOURCE_PREPARATION_ONLY", stdout.getvalue())
        with self.assertRaisesRegex(ValueError, "Explicit"):
            D.main(["--execute"])
        self.assertNotIn("torch", sys.modules)

    def test_startup_only_input_call_and_signature_change_no_global_mutation(self):
        original = (ROOT / "src/rsna_knee/f009_kaggle_driver.py").read_bytes()
        before = dict(S.__dict__)
        derived = D.derive_startup_source(original)
        tree = ast.parse(derived)
        f = tree.body[0]
        self.assertEqual(f.name, "prepare_diagnostic_plan")
        self.assertEqual([n.arg for n in f.args.args], ["args", "fixture_input_root"])
        resolvers = [n for n in ast.walk(f) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "resolve_test_input"]
        self.assertEqual(len(resolvers), 1)
        self.assertEqual(ast.dump(resolvers[0].args[0]), ast.dump(ast.Name(id="fixture_input_root", ctx=ast.Load())))
        for key in before:
            self.assertIs(S.__dict__[key], before[key])
        with self.assertRaisesRegex(ValueError, "Exact frozen"):
            D.derive_startup_source(original + b"\n# changed\n")

    def test_owned_work_cleanup_and_tampered_owner_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            with D._owned_work(parent) as root:
                (root / "FAKE.model-evidence").write_bytes(b"No real model")
            self.assertFalse(root.exists())
            try:
                with self.assertRaisesRegex(ValueError, "cleanup target"):
                    with D._owned_work(parent) as changed:
                        (changed / ".f009-owner").write_text("tampered", encoding="ascii")
                self.assertTrue(changed.exists())
            finally:
                # This is a generated, still-owned temporary directory.
                self.assertEqual(changed.resolve().parent, parent.resolve())
                for path in changed.iterdir():
                    path.unlink()
                changed.rmdir()

    def test_fixed192_projection_schemas_union_and_exception_no_publication(self):
        for failed in (False, True):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args, plan, op, receipt = artificial_component(root, failures=failed)
                result, exports = D._finish_diagnostic(args, plan, None, op, started=0, fixture_receipt=receipt, fixture_only=True)
                self.assertEqual(result["schema_version"], D.RESULT_SCHEMA)
                self.assertEqual(result["P033_parent_cells_executed"], 0)
                self.assertFalse(result["diagnostic_engineering_passed"])
                self.assertTrue(result["artificial_fixture_only"])
                self.assertNotIn("selected_parent_notebook_sha256", result)
                self.assertNotIn("serving_candidate_gate_passed", result)
                self.assertEqual(set(exports), {"native-mean.csv", "native-failures.csv"})
                self.assertEqual(len(list(csv.reader(io.StringIO(exports["native-mean.csv"].decode())))), 193)
                self.assertEqual(len(list(csv.reader(io.StringIO(exports["native-failures.csv"].decode())))), 3 if failed else 1)
                self.assertFalse(args.output.exists())
                (root / "operational-component.json").unlink()
                op.execute_router_with_operational_recovery = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("whole failure"))
                with self.assertRaisesRegex(RuntimeError, "whole failure"):
                    D._finish_diagnostic(args, plan, None, op, started=0, fixture_receipt=receipt, fixture_only=True)

    def test_atomic_publish_requires_cleanup_and_failure_withdraws_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "fresh/diagnostic-result.json"
            result = {"schema_version": D.RESULT_SCHEMA, "complete": True, "fixture_receipt": {},
                      "fixture_cleanup_complete": False, "work_cleanup_complete": True}
            exports = {"native-mean.csv": b"UID,ACL\n", "native-failures.csv": b"UID\n"}
            with self.assertRaisesRegex(ValueError, "Cleanup"):
                D._publish(output, result, exports)
            result["fixture_cleanup_complete"] = True
            with patch.object(D.os, "rename", side_effect=OSError("injected publication failure")):
                with self.assertRaisesRegex(OSError, "injected"):
                    D._publish(output, result, exports)
            self.assertFalse(output.parent.exists())
            self.assertFalse((root / "fresh.partial/diagnostic-result.json").exists())
            self.assertFalse(json.loads((root / "fresh.partial/DIAGNOSTIC_FAILED.json").read_bytes())["complete"])
            output = root / "success/diagnostic-result.json"
            published = D._publish(output, result, exports)
            self.assertEqual(json.loads(output.read_bytes()), published)
            self.assertEqual({p.name for p in output.parent.iterdir()}, {"diagnostic-result.json", "fixture-receipt.json", *exports})
            with self.assertRaisesRegex(ValueError, "Fresh"):
                D._publish(output, result, exports)


if __name__ == "__main__":
    unittest.main()
