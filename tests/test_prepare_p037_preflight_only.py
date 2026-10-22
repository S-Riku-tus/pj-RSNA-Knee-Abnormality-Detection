"""CPU-only preflight packaging/control-flow tests with synthetic callbacks."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import traceback
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p037c_builder", ROOT / "scripts/prepare_p037_preflight_only.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class CpuPreflightPackTests(unittest.TestCase):
    def fixture(self, directory):
        output = Path(directory) / "pack"
        manifest = builder.prepare(output)
        notebook = json.loads((output / "source.ipynb").read_bytes())
        payload = json.loads((output / "save-payload.json").read_bytes())
        return output, manifest, notebook, payload

    def test_seven_inputs_exact_runtime_GPU_off_no_author_recipe(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, manifest, notebook, payload = self.fixture(temporary)
            self.assertEqual(manifest["inputs"]["total"], 7)
            self.assertFalse(payload["enable_gpu"])
            self.assertFalse(payload["enable_internet"])
            self.assertTrue(payload["is_private"])
            self.assertNotIn("machine_shape", payload)
            self.assertEqual(payload["text"].encode(), (output / "source.ipynb").read_bytes())
            self.assertEqual(manifest["P037_parent_source_sha256"], builder.PARENT_SOURCE)
            self.assertLess(manifest["source_bytes"], 950000)
            text = notebook["cells"][1]["source"]
            tree = ast.parse(text)
            calls = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
            runtime_calls = [ast.literal_eval(node.func.slice) for node in ast.walk(tree) if isinstance(node, ast.Call)
                             and isinstance(node.func, ast.Subscript) and isinstance(node.func.value, ast.Name)
                             and node.func.value.id == "_p037c_runtime_ns"]
            self.assertEqual(runtime_calls, ["preflight"])
            self.assertNotIn("execute_cells", calls)
            self.assertNotIn("restore_cells", calls)
            self.assertNotIn("get_ipython", calls)
            runtime_literal = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                                   and any(isinstance(target, ast.Name) and target.id == "_p037c_runtime" for target in node.targets))
            self.assertEqual(builder.digest(ast.literal_eval(runtime_literal).encode()), manifest["operational_runtime_sha256"])
            with self.assertRaises(FileExistsError):
                builder.prepare(output)

    def test_mocked_positive_and_failure_export_one_JSON_without_science(self):
        for fails in (False, True):
            with self.subTest(fails=fails), tempfile.TemporaryDirectory() as temporary:
                _, _, notebook, _ = self.fixture(temporary)
                tree = ast.parse(notebook["cells"][1]["source"])
                start = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                             and any(isinstance(target, ast.Name) and target.id == "_p037c_result" for target in node.targets))
                tree.body = tree.body[start:]
                output = Path(temporary) / "P037_PREFLIGHT.json"
                for node in ast.walk(tree):
                    if isinstance(node, ast.Constant) and node.value == "/kaggle/working/P037_PREFLIGHT.json":
                        node.value = str(output)
                calls = []
                def fake_preflight(config, *, work):
                    self.assertTrue(Path(work).is_dir())
                    calls.append((config, work))
                    if fails:
                        partial = Path(work) / "P037/preflight.json"
                        partial.parent.mkdir()
                        partial.write_text(json.dumps({"complete": False, "fixed_members": {"R6": "synthetic_guard_progress"}}))
                        raise RuntimeError("synthetic incompatible decoder fixture")
                    return {"complete": True, "fixed_members": "artificial-only", "decoder": {"complete": True}}
                namespace = {"_p037c_json": json, "_p037c_sys": sys, "_p037c_temp": tempfile,
                             "_p037c_trace": traceback, "_p037c_Path": Path, "_p037c_config": {"synthetic": True},
                             "_p037c_hash": hashlib, "_p037c_runtime_ns": {"preflight": fake_preflight}}
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(ast.fix_missing_locations(tree), "<synthetic-p037c-control-flow>", "exec"), namespace)
                result = json.loads(output.read_bytes())
                self.assertEqual(result["complete"], not fails)
                self.assertEqual(result["scientific_code_cells_run"], 0)
                self.assertEqual(result["GPU_calls"], 0)
                self.assertEqual(len(calls), 1)
                self.assertFalse(Path(calls[0][1]).exists())
                self.assertEqual([p.name for p in Path(temporary).glob("*.json")], ["P037_PREFLIGHT.json"])
                if fails:
                    self.assertEqual(result["error_type"], "RuntimeError")
                    self.assertIn("synthetic incompatible", result["error"])
                    self.assertEqual(result["partial_preflight"]["fixed_members"], {"R6": "synthetic_guard_progress"})
                    self.assertEqual(len(result["partial_preflight_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
