"""Synthetic source only: ensure a TTA control cannot change another factor."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "prepare_p030_tta_control", Path(__file__).resolve().parents[1] / "scripts/prepare_p030_tta_control.py"
)
P030 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P030)


class P030ControlTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.parent_dir = self.root / "parent"
        self.parent_dir.mkdir()
        self.output = self.root / "new-control"
        no_execute = "raise AssertionError('source must never execute')\n"
        self.book = {
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"papermill": {"duration": 444}, "kaggle": {"isGpuEnabled": False}},
            "cells": [
                {"cell_type": "markdown", "source": ["Parent title\n"], "metadata": {"tags": []}},
                {"cell_type": "markdown", "source": ["Original credits\n"], "metadata": {"tags": []}},
                *[{"cell_type": "code", "source": [no_execute], "execution_count": 99,
                   "outputs": [{"output_type": "stream", "text": ["old author result"]}],
                   "metadata": {"execution": {"end": "old"}, "papermill": {"status": "completed"}, "tags": []}}
                  for _ in range(7)]
            ],
        }
        self.book["cells"][2]["source"].insert(0, "%%writefile /tmp/raptor_run.py\n")
        self.book["cells"][4]["source"].insert(0, "%%writefile /tmp/raptor224_run.py\n")
        modules = {name: no_execute for name in P030.MODULE_NAMES}
        self.book["cells"][6]["source"] = ["ARM = 'E_lrprior'\n", "FLIP_TTA = True   # retain comment\n",
                                               "SOURCES = " + repr(modules) + "\n", no_execute]
        fixed = [{"source_id": number} for number in P030.EXPECTED_INPUT_IDS]
        fixed[-1].update(version=1, expected_manifest_sha256=P030.R6_MANIFEST_SHA)
        self.parent_manifest = {"fixed_inputs": fixed, "version_hazard": {"required_bundle_id": P030.R6_BUNDLE_ID},
                                "runtime_plan": {"GPU": "Tesla T4 x2", "Internet": False}}
        self.r6_manifest = {"bundle_id": P030.R6_BUNDLE_ID}

    def prepare(self, **changes):
        original = (json.dumps(self.book, ensure_ascii=False) + "\n").encode("utf-8")
        r6_metadata = json.dumps(self.r6_manifest).encode("utf-8")
        parent_document = copy.deepcopy(self.parent_manifest)
        parent_document["fixed_inputs"][-1]["expected_manifest_sha256"] = P030.digest(r6_metadata)
        parent_manifest = (json.dumps(parent_document) + "\n").encode("utf-8")
        (self.parent_dir / "source.ipynb").write_bytes(original)
        (self.parent_dir / "manifest.json").write_bytes(parent_manifest)
        r6_path = self.root / "r6-metadata.json"
        r6_path.write_bytes(r6_metadata)
        pins = {"SOURCE_SHA": P030.digest(original), "P029_MANIFEST_SHA": P030.digest(parent_manifest),
                "R6_MANIFEST_SHA": P030.digest(r6_metadata), **changes}
        with mock.patch.multiple(P030, **pins):
            return P030.prepare(self.output, parent_dir=self.parent_dir, r6_manifest_path=r6_path)

    def test_only_boolean_changes_and_old_results_are_cleared_without_execution(self):
        before = copy.deepcopy(self.book)
        manifest = self.prepare()
        candidate = json.loads((self.output / "source.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(self.book, before)
        for index in [2, 3, 4, 5, 7, 8]:
            self.assertEqual(P030.source_text(candidate["cells"][index]), P030.source_text(self.book["cells"][index]))
        self.assertIn("FLIP_TTA = False   # retain comment", P030.source_text(candidate["cells"][6]))
        self.assertEqual(P030.source_text(candidate["cells"][1]), "Original credits\n")
        self.assertTrue(P030.source_text(candidate["cells"][0]).startswith("# P030"))
        self.assertNotIn("papermill", candidate["metadata"])
        for cell in candidate["cells"][2:]:
            self.assertEqual(cell["outputs"], [])
            self.assertIsNone(cell["execution_count"])
            self.assertNotIn("execution", cell["metadata"])
            self.assertNotIn("papermill", cell["metadata"])
        self.assertIsNone(manifest["qualification"]["own_Public"])
        self.assertFalse(manifest["qualification"]["own_Public_verified"])
        self.assertEqual(P030.digest((self.output / "source.ipynb").read_bytes()), manifest["source"]["sha256"])
        self.assertEqual(len(manifest["static_checks"]["embedded_modules"]), 7)

    def test_duplicate_or_nested_assignment_is_not_a_unique_control(self):
        for source in ("FLIP_TTA=True\nFLIP_TTA=True\n", "if True:\n FLIP_TTA=True\n",
                       "FLIP_TTA=True\ndef f():\n global FLIP_TTA\n FLIP_TTA=False\n"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                P030.disable_flip(source)

    def test_nonliteral_or_already_false_controls_are_rejected(self):
        for value in ("False", "'True'", "bool(1)", "1", "None"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                P030.disable_flip("FLIP_TTA = " + value + "\n")

    def test_AST_fence_rejects_a_second_scientific_change(self):
        before = "ARM='E_lrprior'\nFLIP_TTA=True\nA224=0.35\n"
        after = "ARM='E_lrprior'\nFLIP_TTA=False\nA224=0.4\n"
        with self.assertRaisesRegex(ValueError, "beyond"):
            P030.assert_boolean_only_change(before, after)
        after, _ = P030.disable_flip(before)
        P030.assert_boolean_only_change(before, after)

    def test_wrong_parent_hashes_fail_before_output(self):
        for patch in ({"SOURCE_SHA": "0" * 64}, {"P029_MANIFEST_SHA": "0" * 64}):
            with self.subTest(patch=patch), self.assertRaisesRegex(ValueError, "SHA"):
                self.prepare(**patch)
            self.assertFalse(self.output.exists())

    def test_R7_input_or_manifest_enabling_TTA_cannot_issue_TTA_off_control(self):
        for change in ("R7_input", "manifest_TTA"):
            with self.subTest(change=change):
                original = copy.deepcopy(self.parent_manifest)
                if change == "R7_input":
                    self.parent_manifest["fixed_inputs"][-1]["source_id"] = 20731557
                    self.parent_manifest["fixed_inputs"][-1]["version"] = 2
                else:
                    self.r6_manifest["flip_tta"] = True
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assertFalse(self.output.exists())
                self.parent_manifest = original

    def test_existing_run_is_preserved_without_reading_the_parent(self):
        self.output.mkdir()
        marker = self.output / "old.txt"
        marker.write_text("keep")
        with mock.patch.object(P030, "pinned_json") as read, self.assertRaises(FileExistsError):
            P030.prepare(self.output, parent_dir=self.parent_dir)
        read.assert_not_called()
        self.assertEqual(marker.read_text(), "keep")
        self.assertEqual(list(self.output.iterdir()), [marker])

    def test_nonliteral_modules_or_unknown_magic_are_rejected_before_output(self):
        self.book["cells"][6]["source"] = ["FLIP_TTA=True\nSOURCES=dict()\n"]
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.output.exists())
        self.book["cells"][2]["source"] = ["%%bash\necho never-run\n"]
        with self.assertRaisesRegex(ValueError, "magic"):
            self.prepare()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
