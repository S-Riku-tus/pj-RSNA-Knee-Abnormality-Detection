"""Actual child-process regression for the failed profile's observation import."""

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old_tests = load("_p010_profile_v2_old_test_dependencies", ROOT / "tests/test_p010_profile.py")
with mock.patch.dict(
    sys.modules,
    {
        "p006_metrics": old_tests.p006,
        "p007_metrics": old_tests.p007,
        "p009_metrics": old_tests.p009,
        "p010_shared_metrics": old_tests.shared,
        "p010_author224_metrics": old_tests.author,
    },
):
    runtime = load("_p010_profile_v2_test_runtime", ROOT / "scripts/p010_profile_runtime_v2.py")
builder = load("_p010_profile_v2_test_builder", ROOT / "scripts/build_p010_profile_v2.py")


def fake_helpers(directory):
    directory.mkdir()
    (directory / "p006_metrics.py").write_text("SENTINEL = 6\n", encoding="utf-8")
    (directory / "p007_metrics.py").write_text(
        "from p006_metrics import SENTINEL\n"
        "if SENTINEL != 6: raise RuntimeError('wrong helper dependency')\n"
        "def inference_observer(*args): return None\n",
        encoding="utf-8",
    )
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}


class P010ProfileV2Test(unittest.TestCase):
    def test_original_child_import_fails_but_exact_guard_preflight_succeeds(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            helpers, own = folder / "profile_helpers", folder / "own_src"
            own.mkdir()
            hashes = fake_helpers(helpers)
            unbound = own / "unbound.py"
            unbound.write_text("from p007_metrics import inference_observer\n", encoding="utf-8")
            env = dict(os.environ)
            env["PYTHONPATH"] = ""
            result = subprocess.run([sys.executable, str(unbound)], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("No module named 'p007_metrics'", result.stderr)
            with mock.patch.dict(os.environ, {"PYTHONPATH": ""}):
                receipt = runtime.reader_helper_preflight(helpers, hashes, probe_directory=own)
                self.assertEqual(os.environ["PYTHONPATH"], "")
            self.assertTrue(receipt["actual_subprocess_import_passed"])
            self.assertFalse(receipt["model_or_MRI_or_GPU_forward_invoked"])

    def test_hash_mismatch_is_fatal_in_actual_child(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            helpers, own = folder / "helpers", folder / "own_src"
            own.mkdir()
            hashes = fake_helpers(helpers)
            (helpers / "p007_metrics.py").write_text("raise RuntimeError('must not be executed')\n", encoding="utf-8")
            with self.assertRaises(subprocess.CalledProcessError) as caught:
                runtime.reader_helper_preflight(helpers, hashes, probe_directory=own)
            self.assertIn("SHA-256 mismatch: p007_metrics.py", caught.exception.stderr)
            self.assertNotIn("must not be executed", caught.exception.stderr)

    def test_stale_cached_module_outside_fixed_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            helpers = folder / "helpers"
            hashes = fake_helpers(helpers)
            probe = folder / "stale.py"
            probe.write_text(
                "import types,sys\n"
                "wrong=types.ModuleType('p007_metrics')\n"
                "wrong.__file__='unrelated/p007_metrics.py'\n"
                "sys.modules['p007_metrics']=wrong\n" + runtime.reader_helper_guard_source(helpers, hashes),
                encoding="utf-8",
            )
            result = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("outside fixed helper directory", result.stderr)

    def test_real_observation_helpers_import_in_actual_child_without_GPU_MRI(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            own, helpers = folder / "own_src", folder / "helpers"
            own.mkdir()
            helpers.mkdir()
            for target, original in (
                ("p006_metrics.py", "p006_profile_runtime.py"),
                ("p007_metrics.py", "p007_profile_runtime.py"),
            ):
                (helpers / target).write_bytes((ROOT / "scripts" / original).read_bytes())
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in helpers.iterdir()}
            receipt = runtime.reader_helper_preflight(helpers, hashes, probe_directory=own)
            self.assertTrue(receipt["actual_subprocess_import_passed"])

    def test_reader_run_and_decorator_literal_and_preflight_precedes_models(self):
        baseline = json.loads(builder.serving.CONTROL.read_bytes())
        reader = builder.serving.text(baseline["cells"][8]).split("\n", 1)[1]
        generated = runtime.bound_reader_observation(
            reader,
            ROOT / "unused_helpers",
            {
                "p006_metrics.py": "0" * 64,
                "p007_metrics.py": "1" * 64,
            },
        )

        def definitions(source):
            return {
                node.name: ast.dump(node, include_attributes=False)
                for node in ast.parse(source).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            }

        self.assertEqual(definitions(reader), definitions(generated))
        paired = ast.parse((ROOT / "scripts/p010_profile_runtime_v2.py").read_text())
        function = next(
            node for node in paired.body if isinstance(node, ast.FunctionDef) and node.name == "paired_shared224"
        )
        calls = {
            name: [
                node.lineno
                for node in ast.walk(function)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
            ]
            for name in ("reader_helper_preflight", "run_main")
        }
        self.assertLess(max(calls["reader_helper_preflight"]), min(calls["run_main"]))

    def test_v2_preserves_v1_all_cells_except_profile_helper_and_bound_hash(self):
        old, old_inherited = old_tests.builder.make_profile()
        new, new_inherited = builder.make_profile()
        self.assertEqual(len(old["cells"]), len(new["cells"]))
        differences = [i for i, (a, b) in enumerate(zip(old["cells"], new["cells"], strict=True)) if a != b]
        self.assertEqual(differences, [10, 18])
        for name in ("original", "bounded", "shared", "reader", "fusion", "uids", "groups"):
            self.assertEqual(old_inherited[name], new_inherited[name])
        old_source, new_source = map(builder.serving.text, (old["cells"][-1], new["cells"][-1]))
        old_sha = builder.serving.digest(ROOT / "scripts/p010_profile_runtime.py")
        new_sha = builder.serving.digest(ROOT / "scripts/p010_profile_runtime_v2.py")
        self.assertEqual(old_source.replace(old_sha, new_sha), new_source)
        self.assertEqual(builder.serving.digest(builder.serving.CONTROL), builder.serving.CONTROL_SHA)


if __name__ == "__main__":
    unittest.main()
