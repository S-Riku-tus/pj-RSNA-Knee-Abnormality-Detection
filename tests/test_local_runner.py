"""Launch-record failure handling using standard mocks; no torch or GPU execution."""

import importlib.util
import io
import json
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_local_experiment.py"
START_SOURCE = {"artificial-module.py": "unchanged-artificial-source"}


class LocalRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(
            reset_peak_memory_stats=Mock(), max_memory_allocated=Mock(return_value=1024)
        )
        fake_runtime = types.ModuleType("rsna_knee.runtime")
        fake_runtime.ensure_cuda = Mock()
        fake_runtime.train = Mock()
        spec = importlib.util.spec_from_file_location("local_runner_under_test", SCRIPT)
        self.runner = importlib.util.module_from_spec(spec)
        # Importing the launch helper must not import real torch/runtime or probe CUDA.
        with patch.dict("sys.modules", {"torch": fake_torch, "rsna_knee.runtime": fake_runtime}):
            spec.loader.exec_module(self.runner)
        self.runner.load_config = Mock(return_value={"seed": 123, "scope": "artificial fixture"})
        self.runner.source_hashes = Mock(return_value=START_SOURCE.copy())
        self.git = patch.object(self.runner.subprocess, "check_output", return_value="artificial-git-head\n")
        self.git.start()
        self.addCleanup(self.git.stop)

    def args(self, name):
        config = self.root / f"{name}-config.json"
        config.write_text('{"seed":123,"scope":"artificial fixture"}', encoding="utf-8")
        return types.SimpleNamespace(
            config=config,
            run_dir=self.root / name,
            manifest_dir=self.root / "unused-manifest",
            cache_dir=self.root / "unused-cache",
            fold=0,
            reason="Mock launch-record verification",
            diagnostic_studies=None,
        )

    def fake_training(self, error=None):
        def train(manifest, cache, out, config, fold, *, diagnostic_studies):
            out.mkdir()
            if error is not None:
                raise error
            return {
                "run_dir": str(out),
                "best_weak_valid_bce": 0.25,
                "checkpoint": str(out / "not-a-real-model.pt"),
                "diagnostic_mode": False,
            }

        self.runner.train.side_effect = train

    def records(self, args):
        launch = json.loads((args.run_dir.parent / "launches" / f"{args.run_dir.name}.json").read_text())
        saved = json.loads((args.run_dir / "execution.json").read_text())
        self.assertEqual(saved, launch)
        self.assertIn("finished_at_utc", launch)
        self.assertGreaterEqual(launch["elapsed_seconds"], 0)
        self.assertEqual(launch["peak_allocated_bytes"], 1024)
        return launch

    def test_keyboard_interrupt_and_system_exit_are_recorded_and_reraised(self):
        for error in (KeyboardInterrupt(), SystemExit("artificial stop")):
            with self.subTest(exception=type(error).__name__):
                args = self.args(f"interrupt-{type(error).__name__}")
                self.fake_training(error)
                with self.assertRaises(type(error)) as caught:
                    self.runner.run(args)
                self.assertIs(caught.exception, error)
                record = self.records(args)
                self.assertEqual(record["status"], "interrupted")
                self.assertIn(type(error).__name__, record["error"])
                self.assertNotIn("result", record)

    def test_training_failure_remains_failed_with_completion_record(self):
        args, error = self.args("failed"), RuntimeError("artificial training failure")
        self.fake_training(error)
        with self.assertRaisesRegex(RuntimeError, "artificial training failure") as caught:
            self.runner.run(args)
        self.assertIs(caught.exception, error)
        record = self.records(args)
        self.assertEqual(record["status"], "failed")
        self.assertIn("RuntimeError", record["error"])
        self.assertNotIn("result", record)

    def test_source_change_after_success_invalidates_both_saved_records(self):
        args = self.args("source-changed")
        self.fake_training()
        changed = {"artificial-module.py": "changed-artificial-source"}
        self.runner.source_hashes.side_effect = [START_SOURCE.copy(), changed]
        with self.assertRaisesRegex(ValueError, "Source changed"):
            self.runner.run(args)
        record = self.records(args)
        self.assertEqual(record["status"], "invalid_source_changed")
        self.assertEqual(record["source_sha256"], START_SOURCE)
        self.assertEqual(record["source_sha256_at_finish"], changed)
        self.assertEqual(self.runner.train.call_count, 1)

    def test_success_records_source_confirmation_and_prevents_same_launch_reuse(self):
        args = self.args("completed")
        self.fake_training()
        with redirect_stdout(io.StringIO()):
            self.runner.run(args)
        record = self.records(args)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["source_sha256_at_finish"], START_SOURCE)
        self.assertEqual(record["result"]["best_weak_valid_bce"], 0.25)
        before = (args.run_dir / "execution.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "new empty run directory"):
            self.runner.run(args)
        self.assertEqual((args.run_dir / "execution.json").read_bytes(), before)
        self.assertEqual(self.runner.train.call_count, 1)
        # A failed-before-run launch also reserves its name, even without a run directory.
        reserved = self.args("reserved-name")
        launch = self.root / "launches" / "reserved-name.json"
        launch.write_text('{"status":"failed"}', encoding="utf-8")
        previous = launch.read_bytes()
        with self.assertRaisesRegex(ValueError, "launch already exists"):
            self.runner.run(reserved)
        self.assertEqual(launch.read_bytes(), previous)
        self.assertFalse(reserved.run_dir.exists())
        self.assertEqual(self.runner.train.call_count, 1)

    def test_cpu_guard_fails_before_files_config_or_training(self):
        args = self.args("cpu-refusal")
        self.runner.ensure_cuda.side_effect = ValueError("CUDA GPU unavailable")
        with self.assertRaisesRegex(ValueError, "CUDA GPU unavailable"):
            self.runner.run(args)
        self.runner.load_config.assert_not_called()
        self.runner.train.assert_not_called()
        self.runner.source_hashes.assert_not_called()
        self.runner.torch.cuda.reset_peak_memory_stats.assert_not_called()
        self.assertFalse(args.run_dir.exists())
        self.assertFalse((self.root / "launches").exists())


if __name__ == "__main__":
    unittest.main()
