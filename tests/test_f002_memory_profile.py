"""Artificial-only profile adapter contracts, restoration and prepared source."""

import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from rsna_knee.contracts import sha256, source_hashes

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load_script("bounded_streaming_adapter")
builder = load_script("build_f002_memory_profile")
reader = load_script("bounded_feature_reader")
HAS_IMAGING = all(importlib.util.find_spec(name) for name in ("torch", "numpy", "pydicom", "PIL"))


@unittest.skipUnless(HAS_IMAGING, "Optional imaging dependencies absent")
class MemoryProfileRuntimeTests(unittest.TestCase):
    def setUp(self):
        import test_frozen_streaming
        from test_bounded_feature_reader import BoundedReaderTests

        self.base = test_frozen_streaming.StreamingTests()
        self.base.setUp()
        self.addCleanup(self.base.doCleanups)
        self.fixture = BoundedReaderTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.series_dir = self.fixture.fixture(count=3)
        self.output = self.base.root / "memory-diagnostics"
        self.summary = self.base.root / "F002_STREAMING_SUMMARY.json"
        self.core_before = source_hashes()

    def profile(self, **overrides):
        import test_frozen_streaming

        arguments = {
            "mode": "profile_train",
            "profile_limit": 5,
            "weak_manifest": self.base.weak,
            "expected_weak_sha256": sha256(self.base.weak),
            "expected_checkpoint_sha256": sha256(self.base.checkpoint),
            "expected_sources": source_hashes(),
            "require_decoders": False,
            "device": self.base.device,
        }
        arguments.update(overrides)
        return adapter.run_memory_profile(
            test_frozen_streaming.streaming.run_streaming,
            reader.make_bounded_reader,
            self.base.root,
            self.base.checkpoint,
            self.base.provenance,
            "synthetic-encoder",
            "synthetic-weights",
            self.output,
            summary_path=self.summary,
            reader_sha256=sha256(ROOT / "scripts/bounded_feature_reader.py"),
            baseline_runtime_sha256=builder.BASELINE_RUNTIME_SHA256,
            resource_fn=test_frozen_streaming.streaming.resource_snapshot,
            **arguments,
        )

    def test_actual_artificial_streaming_profile_records_policy_and_restores_reader(self):
        import test_frozen_streaming

        from rsna_knee import feature_imaging

        original = feature_imaging.read_feature_windows

        def artificial_extract(_root, _split, key, _series, config, *_args):
            # Exercise the newly bound real reader on our artificial pixels;
            # the fake feature values remain the existing head fixture.
            windows = feature_imaging.read_feature_windows(self.series_dir, config["preprocess"])
            self.assertEqual(windows[0].shape[1:], (3, 28, 28))
            return self.base.arrays(key), []

        with (
            patch.object(feature_imaging, "load_frozen_encoder", return_value=None),
            patch.object(test_frozen_streaming.streaming, "decoder_inventory", return_value={}),
            patch.object(feature_imaging, "extract_study", side_effect=artificial_extract),
        ):
            result = self.profile()
        self.assertIs(feature_imaging.read_feature_windows, original)
        self.assertEqual(result["status"], "profile_complete_not_for_submission")
        self.assertEqual(result["processed"], 5)
        self.assertEqual(result["memory_reader"]["observed_series_allocations"], 5)
        self.assertTrue(result["memory_reader"]["reader_binding_restored"])
        self.assertEqual(result["memory_reader"]["temporary_files_remaining"], 0)
        self.assertFalse((self.base.root / "submission.csv").exists())
        self.assertFalse((self.base.root / "f002-native-workspace").exists())
        nested = json.loads((self.output / "F002_STREAMING_SUMMARY.json").read_text(encoding="utf-8"))
        root_summary = json.loads(self.summary.read_text(encoding="utf-8"))
        self.assertEqual(root_summary["memory_reader"], nested["memory_reader"])
        phases = [
            json.loads(line)["phase"]
            for line in (self.output / "memory-reader-events.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(phases.count("allocation_plan"), 5)
        self.assertEqual(phases.count("global_percentiles"), 5)
        self.assertEqual(phases.count("bounded_series_complete"), 5)
        self.assertEqual(source_hashes(), self.core_before)

    def test_runtime_exception_restores_reader_and_publishes_both_failed_summaries(self):
        import test_frozen_streaming

        from rsna_knee import feature_imaging

        original = feature_imaging.read_feature_windows
        with (
            patch.object(feature_imaging, "load_frozen_encoder", return_value=None),
            patch.object(test_frozen_streaming.streaming, "decoder_inventory", return_value={}),
            patch.object(feature_imaging, "extract_study", side_effect=ValueError("synthetic invalid geometry")),
        ):
            with self.assertRaisesRegex(ValueError, "synthetic invalid geometry"):
                self.profile()
        self.assertIs(feature_imaging.read_feature_windows, original)
        for path in (self.summary, self.output / "F002_STREAMING_SUMMARY.json"):
            summary = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "failed")
            self.assertTrue(summary["memory_reader"]["reader_binding_restored"])
            self.assertEqual(summary["memory_reader"]["temporary_files_remaining"], 0)
        self.assertFalse((self.base.root / "submission.csv").exists())

    def test_rejects_submission_mode_with_failed_root_summary(self):
        from rsna_knee import feature_imaging

        original = feature_imaging.read_feature_windows
        with self.assertRaisesRegex(ValueError, "not a submission notebook"):
            self.profile(mode="test")
        self.assertIs(feature_imaging.read_feature_windows, original)
        summary = json.loads(self.summary.read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "failed")
        self.assertFalse((self.base.root / "submission.csv").exists())

    def test_prior_nested_diagnostics_are_preserved(self):
        self.output.mkdir()
        prior = self.output / "F002_STREAMING_SUMMARY.json"
        prior.write_text('{"prior":"must remain untouched"}', encoding="utf-8")
        before = prior.read_bytes()
        with self.assertRaisesRegex(ValueError, "fresh memory-profile"):
            self.profile()
        self.assertEqual(prior.read_bytes(), before)
        self.assertEqual(json.loads(self.summary.read_text(encoding="utf-8"))["status"], "failed")


class MemoryProfileNotebookTests(unittest.TestCase):
    def test_profile1300_source_compiles_preserves_original_runtime_inputs_and_head(self):
        if not builder.BASE.is_file():
            self.skipTest("Local embedded-head artifact absent")
        core = source_hashes()
        original = json.loads(builder.BASE.read_text(encoding="utf-8"))
        book = builder.make_notebook()
        self.assertEqual(book["metadata"]["rsna_manual_inputs"], original["metadata"]["rsna_manual_inputs"])
        self.assertEqual(book["cells"][1:4], original["cells"][1:4])
        self.assertEqual(book["cells"][6], original["cells"][4])
        for index, cell in enumerate(book["cells"]):
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                compile("".join(cell["source"]), f"cell{index}", "exec")
        final = "".join(book["cells"][-1]["source"])
        self.assertIn("MODE = 'profile_train'", final)
        self.assertIn("PROFILE_LIMIT = 1300", final)
        self.assertIn("PROFILE_SEED = 20261007", final)
        self.assertIn("KAGGLE_IS_COMPETITION_RERUN", final)
        self.assertIn("700e3da8016d80fd87bde2d1c189ef755ce5e5f89717ebac44d011c967a8371a", final)
        self.assertIn("run_memory_profile(", final)
        self.assertEqual(source_hashes(), core)
        self.assertFalse(book["metadata"]["rsna_memory_profile"]["for_submission"])
        self.assertIn("head_dataset_required", book["metadata"]["rsna_delivery"])
        self.assertFalse(book["metadata"]["rsna_delivery"]["head_dataset_required"])


if __name__ == "__main__":
    unittest.main()
