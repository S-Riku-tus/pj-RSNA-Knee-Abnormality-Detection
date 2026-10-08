"""Synthetic directory layout checks; no MRI data, weights, GPU or network."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("frozen_test_input", ROOT / "scripts/frozen_test_input.py")
input_helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(input_helper)


class TestInputResolution(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.modern = self.root / "competitions/rsna-knee-abnormality-detection"
        self.legacy = self.root / "rsna-knee-abnormality-detection"

    def complete(self, path):
        (path / "test_series").mkdir(parents=True)
        (path / "test.csv").write_text("StudyInstanceUID\nsynthetic\n", encoding="utf-8")
        (path / "test_series.csv").write_text("StudyInstanceUID,SeriesInstanceUID\n", encoding="utf-8")

    def test_complete_modern_without_train_or_report(self):
        self.complete(self.modern)
        self.assertEqual(input_helper.resolve_test_input(self.root), self.modern.resolve())
        self.assertFalse((self.modern / "train.csv").exists())

    def test_empty_modern_placeholder_does_not_shadow_legacy_data(self):
        self.modern.mkdir(parents=True)
        self.complete(self.legacy)
        self.assertEqual(input_helper.resolve_test_input(self.root), self.legacy.resolve())

    def test_partial_legacy_does_not_shadow_modern_data(self):
        self.legacy.mkdir()
        (self.legacy / "test.csv").write_text("StudyInstanceUID\nplaceholder\n", encoding="utf-8")
        self.complete(self.modern)
        self.assertEqual(input_helper.resolve_test_input(self.root), self.modern.resolve())

    def test_missing_image_directory_is_incomplete(self):
        self.complete(self.modern)
        (self.modern / "test_series").rmdir()
        with self.assertRaisesRegex(RuntimeError, "found 0"):
            input_helper.resolve_test_input(self.root)

    def test_two_distinct_complete_roots_are_ambiguous(self):
        self.complete(self.modern)
        self.complete(self.legacy)
        with self.assertRaisesRegex(RuntimeError, "found 2"):
            input_helper.resolve_test_input(self.root)

    def test_alias_of_same_complete_root_is_deduplicated(self):
        self.complete(self.modern)
        try:
            self.legacy.symlink_to(self.modern, target_is_directory=True)
        except OSError:
            self.skipTest("Creating directory symlinks is unavailable on this Windows account")
        self.assertEqual(input_helper.resolve_test_input(self.root), self.modern.resolve())


if __name__ == "__main__":
    unittest.main()
