"""Offline P035 opt-in wiring and source-package contracts; no MRI/models/CUDA."""

import ast
import base64
import csv
import hashlib
import io
import json
import lzma
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_p032_adapter as baseline_adapter  # noqa: E402
from test_p032_adapter import builder, extension, native  # noqa: E402


def unpack(folder):
    book = json.loads((folder / "source.ipynb").read_bytes())
    for cell in book["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "restore_runtime":
                encoded, expected = map(ast.literal_eval, node.args)
                bundle = lzma.decompress(base64.b85decode(encoded))
                if hashlib.sha256(bundle).hexdigest() != expected:
                    raise AssertionError("Runtime bundle SHA differs")
                with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
                    return {name: archive.read(name) for name in archive.namelist()}
    raise AssertionError("No literal runtime bundle")


class P035BoundaryTests(unittest.TestCase):
    def test_parent_uses_union_mask_and_separate_operational_count(self):
        receipt, before, actual = baseline_adapter.ParentBoundaryTests.run_extension(
            self, "operational", blend_policy="success-subset", native_executor="isolated-uid")
        self.assertTrue(receipt["complete"])
        self.assertTrue(receipt["native_contribution_used"])
        self.assertEqual(receipt["native_executor"], "isolated-uid")
        self.assertEqual(receipt["native_structural_failures"], 0)
        self.assertEqual(receipt["native_operational_failures"], 1)
        self.assertEqual(receipt["native_failure_UID_union"], ["2"])
        old, new = [list(csv.DictReader(io.StringIO(body.decode()))) for body in (before, actual)]
        self.assertEqual(old[1], new[1])
        self.assertNotEqual(old[0]["ACL"], new[0]["ACL"])

    def test_parent_rejects_legacy_result_as_new_qualification_and_preserves_parent(self):
        receipt, before, actual = baseline_adapter.ParentBoundaryTests.run_extension(
            self, "bad-typed", blend_policy="success-subset", native_executor="isolated-uid")
        self.assertFalse(receipt["complete"])
        self.assertFalse(receipt["native_contribution_used"])
        self.assertEqual(before, actual)
        self.assertIn("typed result", receipt["error"])

    def test_policy_and_executor_selection_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "fixes the existing"):
            extension.execute({}, runtime="unused", started=0, native_executor="isolated-uid")
        for choice in ("legacy-pool", "isolated-uid"):
            argv = ["native", "--mode", "diagnostic", "--output", "artificial.json", "--budget-seconds", "3600",
                    "--native-executor", choice]
            with patch.object(sys, "argv", argv), patch.object(native, "run") as run, \
                    patch("faulthandler.enable"), patch("faulthandler.dump_traceback_later"), \
                    patch("faulthandler.cancel_dump_traceback_later"):
                native.main()
            self.assertEqual(run.call_args.args[0].native_executor, choice)

    def test_default_router_path_remains_original_direct_execute(self):
        router = types.SimpleNamespace(execute=lambda *args: ("unchanged", args))
        args, active, plan = types.SimpleNamespace(native_executor="legacy-pool"), object(), {}
        result = native.execute_native_router(args, active, plan, router, task_source_sha256="unused", started=7)
        self.assertEqual(result, ("unchanged", (active, plan, 7)))


class P035PackagingTests(unittest.TestCase):
    def test_offline_pack_dependencies_parent_BOOTSTRAP_and_child_projection(self):
        if not (builder.RECOVERY / "P029M-source.ipynb").is_file():
            self.skipTest("Ignored recovered source notebooks are unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            invalid = root / "invalid"
            with self.assertRaises(ValueError):
                builder.prepare(invalid, native_executor="isolated-uid")
            self.assertFalse(invalid.exists())
            baseline = builder.prepare(root / "legacy", blend_policy="success-subset")
            isolated = builder.prepare(root / "isolated", blend_policy="success-subset", native_executor="isolated-uid")
            old, new = unpack(root / "legacy"), unpack(root / "isolated")
            self.assertEqual(set(new) - set(old), {"p032blend/isolated_uid_executor.py", "p032blend/operational_uid_recovery.py"})
            self.assertEqual(old["P029M-source.ipynb"], new["P029M-source.ipynb"])
            self.assertEqual(old["p032blend/success20_blend.py"], new["p032blend/success20_blend.py"])
            self.assertEqual(baseline["native"], isolated["native"])
            self.assertEqual(baseline["inputs"], isolated["inputs"])
            self.assertEqual(isolated["inputs"]["total"], 9)
            self.assertEqual(isolated["native_executor"], "isolated-uid")
            self.assertFalse(isolated["scope"]["new_visible_complete"])
            for name, body in new.items():
                self.assertEqual(hashlib.sha256(body).hexdigest(), isolated["runtime_files"][name]["sha256"])
                target = root / "runtime" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(body)
            child_tree = ast.parse(new["p032_native_child.py"])
            bootstrap = next(ast.literal_eval(node.value) for node in child_tree.body
                             if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "BOOTSTRAP" for t in node.targets))
            original_bootstrap, _ = builder.inspect_native(builder.RECOVERY / "F006-native64-source.ipynb")
            self.assertEqual(bootstrap, original_bootstrap)
            runner = root / "artificial_adapter.py"
            runner.write_text(textwrap.dedent('''\
                import hashlib, importlib.util, json, sys, types
                from pathlib import Path
                runtime, work = map(Path, sys.argv[1:])
                sys.path.insert(0, str(runtime))
                from p032blend.contracts import ID, TARGETS
                from p032blend.operational_uid_recovery import RESULT_SCHEMA
                spec = importlib.util.spec_from_file_location('packaged_child_artificial', runtime / 'p032_native_child.py')
                child = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(child)
                assert '_p18_source' not in vars(child), 'BOOTSTRAP must not run during import'
                work.mkdir()
                result = {'schema_version': RESULT_SCHEMA, 'complete': True, 'operational_recovery_contract_passed': True,
                    'studies': 3, 'targets': list(TARGETS), 'structural_failures': {},
                    'operational_failures': {'2': {'uid': '2'}}, 'failure_UID_union': ['2'],
                    'coverage': {'N': 3, 'S': 2, 'F_structural': 0, 'F_operational': 1},
                    'probabilities_test_order': [{ID: str(i), 'probabilities': [v] * 12} for i,v in ((1,.9),(2,.5),(3,.1))],
                    'full_fold_model_forward_calls': 8, 'reconstruction_component': {'model_fingerprints': {'artificial': True}},
                    'executor_receipts': [], 'legacy_quarantine_removed': True}
                output = work / 'native-result.json'
                output.write_text(json.dumps(result))
                io = types.SimpleNamespace(ID=ID,TARGETS=TARGETS,sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest())
                receipt = child.emit_native_adapter(types.SimpleNamespace(native_executor='isolated-uid',mode='serving',output=output),
                    result,work,io,{'pair_fingerprint_sha256':'artificial'}, {'artificial':True},started=0)
                assert receipt['schema_version']=='p035_f006_isolated_uid_adapter_v1'
                assert receipt['operational_failure_count']==1 and receipt['structural_failure_count']==0
                assert receipt['successful_studies']==2 and receipt['failure_UID_union']==['2']
                assert receipt['failure_mask_is_structural_operational_union'] is True
                assert '_p18_source' not in vars(child)
                assert 'torch' not in sys.modules
                print(json.dumps({'packaged_import_and_projection':True,'BOOTSTRAP_executed':False,'GPU_MRI_models':0}))
                '''), encoding="utf-8")
            completed = subprocess.run([sys.executable, "-B", str(runner), str(root / "runtime"), str(root / "work")],
                                       text=True, capture_output=True, check=True, timeout=30)
            self.assertTrue(json.loads(completed.stdout)["packaged_import_and_projection"])


if __name__ == "__main__":
    unittest.main()
