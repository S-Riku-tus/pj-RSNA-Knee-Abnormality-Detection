"""Replay the real fusion command shape with artificial CSVs, before GPU work."""

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dependencies = load("_p010_v3_dependencies", ROOT / "tests/test_p010_profile.py")
with mock.patch.dict(
    sys.modules,
    {
        "p006_metrics": dependencies.p006,
        "p007_metrics": dependencies.p007,
        "p009_metrics": dependencies.p009,
        "p010_shared_metrics": dependencies.shared,
        "p010_author224_metrics": dependencies.author,
    },
):
    runtime = load("_p010_v3_runtime", ROOT / "scripts/p010_profile_runtime_v3.py")
builder = load("_p010_v3_builder", ROOT / "scripts/build_p010_profile_v3.py")
v2_builder = load("_p010_v3_original_builder", ROOT / "scripts/build_p010_profile_v2.py")


class P010ReplayV3Test(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "own_src").mkdir()
        self.labels = [f"artificial_label_{index}" for index in range(12)]
        self.uids = ["artificial-a", "artificial-b"]
        self.cache = self.root / "cache.csv"
        runtime.write_submission_exclusive(self.cache, self.uids, [[0.25] * 12, [0.75] * 12], self.labels)
        self.hash = runtime.sha256(self.cache)
        self.checkpoints = [str(self.root / "fake_input" / f"cnxt_v0_fold{index}.pt") for index in range(3)]
        # Extract the actual immutable original-fusion subprocess call.
        book = json.loads(builder.serving.CONTROL.read_bytes())
        tree = ast.parse(builder.serving.text(book["cells"][9]))
        call = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "_o_sp"
            and node.func.attr == "run"
            and any(
                isinstance(part, ast.Constant) and part.value == "/kaggle/working/own_src/infer.py"
                for part in ast.walk(node.args[0])
            )
        )
        values = {
            "_o_sys": types.SimpleNamespace(executable=sys.executable),
            "_o_data": str(self.root / "p010_profile_data"),
            "_o_own_csv": str(self.root / "_own.csv"),
            "_o_ck": self.checkpoints,
            "_o_env": {"ARTIFICIAL": "1"},
        }
        self.command = eval(compile(ast.Expression(call.args[0]), "<actual-fusion-command>", "eval"), values)
        self.assertEqual(self.command[1], "/kaggle/working/own_src/infer.py")
        # Project only the fixed Kaggle script path into an isolated temporary directory.
        self.command[1] = str(self.root / "own_src/infer.py")
        self.kwargs = {
            keyword.arg: eval(compile(ast.Expression(keyword.value), "<actual-fusion-kwargs>", "eval"), values)
            for keyword in call.keywords
        }

    def context(self):
        return runtime.replay_reader_once(
            self.cache, self.hash, checkpoints=self.checkpoints, uids=self.uids, labels=self.labels, root=self.root
        )

    def test_actual_fusion_exact_command_and_kwargs_replay_once_then_restore(self):
        original = subprocess.run
        with self.context() as state:
            completed = subprocess.run(self.command, **self.kwargs)
            self.assertEqual(completed.returncode, 0)
            self.assertEqual(state["replay_calls"], 1)
            self.assertEqual(runtime.sha256(self.root / "_own.csv"), self.hash)
            with self.assertRaises(ValueError):
                subprocess.run(self.command, **self.kwargs)
        self.assertIs(subprocess.run, original)
        self.assertEqual(state["expected_data_alias"], str(self.root / "p010_profile_data"))

    def test_wrong_alias_split_output_executable_checkpoints_flags_are_rejected(self):
        mutations = [
            (2, str(self.root / "p009_profile_data")),
            (2, str(self.root / "competition_root")),
            (3, "train"),
            (4, str(self.root / "foreign.csv")),
            (0, "foreign_python"),
            (5, str(self.root / "foreign_checkpoint.pt")),
            (7, self.checkpoints[0]),
        ]
        for index, value in mutations:
            wrong = list(self.command)
            wrong[index] = value
            with self.subTest(index=index, value=value), self.context():
                with self.assertRaises(ValueError):
                    subprocess.run(wrong, **self.kwargs)
        for kwargs in (
            {"check": False, "env": {}},
            {"check": True, "env": None},
            {"check": True, "env": {}, "capture_output": True},
        ):
            with self.subTest(kwargs=kwargs), self.context(), self.assertRaises(ValueError):
                subprocess.run(self.command, **kwargs)
        self.assertFalse((self.root / "_own.csv").exists())

    def test_stale_CSV_hash_UID_labels_and_checkpoint_order_are_fail_closed(self):
        output = self.root / "_own.csv"
        output.write_bytes(b"stale")
        with self.context(), self.assertRaises(ValueError):
            subprocess.run(self.command, **self.kwargs)
        output.unlink()
        self.cache.write_bytes(self.cache.read_bytes() + b"\n")
        with self.assertRaises(ValueError), self.context():
            pass
        self.cache.write_bytes(b"bad UID/header/schema")
        with (
            self.assertRaises(ValueError),
            runtime.replay_reader_once(
                self.cache,
                runtime.sha256(self.cache),
                checkpoints=self.checkpoints,
                uids=self.uids,
                labels=self.labels,
                root=self.root,
            ),
        ):
            pass
        with (
            self.assertRaises(ValueError),
            runtime.replay_reader_once(
                self.cache,
                runtime.sha256(self.cache),
                checkpoints=list(reversed(self.checkpoints)),
                uids=self.uids,
                labels=self.labels,
                root=self.root,
            ),
        ):
            pass

    def test_actual_preflight_removes_artifacts_restores_and_precedes_models(self):
        original = subprocess.run
        receipt = runtime.reader_replay_preflight(self.labels, working=self.root)
        self.assertTrue(receipt["exact_command_contract_passed"])
        self.assertFalse(receipt["real_subprocess_or_model_or_MRI_or_labels_consumed"])
        self.assertIs(subprocess.run, original)
        self.assertFalse(list(self.root.glob("p010-artificial-replay-*")))
        tree = ast.parse((ROOT / "scripts/p010_profile_runtime_v3.py").read_text(encoding="utf-8"))
        paired = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "paired_shared224"
        )
        called = {
            name: [
                node.lineno
                for node in ast.walk(paired)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
            ]
            for name in ("reader_replay_preflight", "run_main")
        }
        self.assertLess(max(called["reader_replay_preflight"]), min(called["run_main"]))

    def test_v3_only_observation_helper_and_hash_change_all_prediction_literals_same(self):
        old, old_values = v2_builder.make_profile()
        new, new_values = builder.make_profile()
        self.assertEqual(
            [i for i, (a, b) in enumerate(zip(old["cells"], new["cells"], strict=True)) if a != b], [10, 18]
        )
        for name in ("original", "bounded", "shared", "reader", "fusion", "uids", "groups"):
            self.assertEqual(old_values[name], new_values[name])
        old_tail, new_tail = map(builder.serving.text, (old["cells"][-1], new["cells"][-1]))
        self.assertEqual(
            old_tail.replace(
                builder.serving.digest(ROOT / "scripts/p010_profile_runtime_v2.py"),
                builder.serving.digest(ROOT / "scripts/p010_profile_runtime_v3.py"),
            ),
            new_tail,
        )
        require_hashes = {
            "14_submit_p005_public950_control.ipynb": "75c3abe77e8f0ccdaec2f57a8946d2baf297db8cc6fbdb275e5fcb1153b57546",
            "19_submit_p009_coat_fov160.ipynb": "35b64ae2396054077fb46d4991e05185e506bd9e731b016c414777e244f5148c",
            "21_submit_p010_swa224_rank10.ipynb": "4adf9088b97c795cb50e382b41d4a158950352ef4d5316fda40551111c39c41c",
            "21a_profile_p010_shared224_32.ipynb": "eaa9ec444a884ccb6f5f9c1fec427a6d85febd710cbc226b20b770c68bbce960",
            "21a_profile_p010_shared224_32_v2.ipynb": "86b16a97fec35c5c345b56b2eac479b4e70f66a04afaf97921837d6274290b05",
        }
        for name, expected in require_hashes.items():
            self.assertEqual(hashlib.sha256((ROOT / "notebooks/public" / name).read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
