"""Artificial, standard-library tests; no public model or notebook execution."""

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "prepare_p029_public957", Path(__file__).resolve().parents[1] / "scripts/prepare_p029_public957.py"
)
P029 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P029)


class P029PreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source_path = self.root / "original.ipynb"
        self.view_path = self.root / "view.json"
        self.dataset_path = self.root / "dataset.json"
        self.output = self.root / "new-run"
        self.no_execute = "raise AssertionError('notebook code must never execute')\n"
        modules = {name: self.no_execute for name in (
            "px_cache.py", "compact_train.py", "compact_infer.py", "probe_infer.py",
            "probe_dual.py", "knee_anchor_bank_v2.py", "side_common.py"
        )}
        self.book = {
            "nbformat": 4, "nbformat_minor": 5,
            "cells": [
                {"cell_type": "markdown", "source": ["Synthetic original title\n"]},
                {"cell_type": "markdown", "source": ["Synthetic credits with OAI acknowledgement\n"]},
                *[{"cell_type": "code", "source": [self.no_execute], "execution_count": 77,
                   "outputs": [{"output_type": "stream", "text": ["historical author output\n"]}]}
                  for _ in range(7)],
            ],
            "metadata": {"language_info": {"version": "3.13.15"}, "kaggle": {
                "isInternetEnabled": False, "isGpuEnabled": False,
                "dataSources": [{"sourceId": x["source_id"]} for x in P029.FIXED_INPUTS]
            }},
        }
        self.book["cells"][2]["source"].insert(0, "%%writefile /tmp/raptor_run.py\n")
        self.book["cells"][4]["source"].insert(0, "%%writefile /tmp/raptor224_run.py\n")
        self.book["cells"][6]["source"] = ["SOURCES = " + repr(modules) + "\n", self.no_execute]
        sources = []
        for expected in P029.FIXED_INPUTS:
            actual = {"sourceId": expected["source_id"], "mountSlug": expected["mount_slug"]}
            if expected["kind"] == "competition":
                actual["databundleVersionId"] = expected["databundle_version_id"]
            else:
                actual.update(sourceType="DATA_SOURCE_TYPE_DATASET_VERSION", datasetId=expected["dataset_id"])
            sources.append(actual)
        self.view = {
            "kernel": {"id": P029.KERNEL_ID, "url": "/code/" + P029.REF},
            "currentVersionNumber": 1,
            "kernelRun": {"id": P029.SCRIPT_VERSION, "kernelId": P029.KERNEL_ID,
                          "kernelVersionNumber": 1, "status": "COMPLETE", "dockerImageVersionId": 31481,
                          "runInfo": {"dockerImageDigest": P029.DOCKER_DIGEST, "runTimeSeconds": 458.3,
                                      "dockerImageName": "synthetic-observed-image"},
                          "isGpuEnabled": True, "acceleratorType": "NVIDIA_TESLA_T4", "dataSources": sources},
            "dataSources": copy.deepcopy(sources),
            "submission": {"id": P029.SUBMISSION_ID, "sourceScriptVersionId": P029.SCRIPT_VERSION,
                           "scoreFormatted": "0.957"},
            "bestSubmissionScore": {"scoreFormatted": "0.957", "kernelVersionNumber": 1},
        }
        self.dataset = {
            "id": 12524886, "ref": "heliosli/rsna-knee-r6-frozen-fe055406", "currentVersionNumber": 2,
            "description": P029.MANIFEST_SHA256 + " original V1; V2 " + P029.R7_MANIFEST_SHA256,
            "licenseName": "Other (specified in description)", "versions": [{"versionNumber": 1}, {"versionNumber": 2}]
        }
        metadata = {}
        for name, expected in (("Ortho-view.json", P029.FIXED_INPUTS[1]), ("SWA-view.json", P029.FIXED_INPUTS[2])):
            metadata[name] = {"id": expected["dataset_id"], "ref": expected["ref"].rsplit("/", 1)[0],
                              "currentVersionNumber": 1, "isPrivate": False}
        metadata["R6-v1-filelist.json"] = {
            "datasetFiles": [{"name": name, "totalBytes": size} for name, size in P029.R6_V1_FILES.items()],
            "nextPageToken": "", "errorMessage": ""
        }
        self.metadata = metadata

    def write_json(self, path, value):
        body = (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")
        path.write_bytes(body)
        return P029.digest(body)

    def prepare(self):
        patches = {
            "SOURCE_SHA256": self.write_json(self.source_path, self.book),
            "VIEW_SHA256": self.write_json(self.view_path, self.view),
            "DATASET_VIEW_SHA256": self.write_json(self.dataset_path, self.dataset),
            "INPUT_METADATA_SHA256": {name: self.write_json(self.root / name, value)
                                      for name, value in self.metadata.items()},
        }
        with mock.patch.multiple(P029, **patches):
            return P029.prepare(self.output, source=self.source_path, view_path=self.view_path,
                                dataset_view_path=self.dataset_path, input_metadata_dir=self.root)

    def test_preparation_copies_all_bytes_without_executing_any_code(self):
        manifest = self.prepare()
        self.assertEqual((self.output / "source.ipynb").read_bytes(), self.source_path.read_bytes())
        self.assertEqual(manifest["static_checks"]["code_cell_count"], 7)
        self.assertEqual(len(manifest["static_checks"]["embedded_modules"]), 7)
        self.assertEqual(manifest["credits"]["required_acknowledgement_verbatim"],
                         "".join(self.book["cells"][1]["source"]))
        self.assertFalse(manifest["qualification"]["own_visible_Serving_qualified"])
        self.assertFalse(manifest["license_and_Gold"]["Gold_clean_claimed"])
        self.assertIsNone(manifest["author_result"]["own_reproduction_score"])
        self.assertEqual(manifest["runtime_plan"]["GPU"], "Tesla T4 x2")
        self.assertFalse(manifest["runtime_plan"]["Internet"])
        self.assertEqual(manifest["version_hazard"]["required_internal_source_id"], 20611930)
        self.assertEqual(manifest["version_hazard"]["required_manifest_sha256"], P029.MANIFEST_SHA256)
        self.assertEqual((self.output / "manifest.sha256").read_text().split()[0],
                         P029.digest((self.output / "manifest.json").read_bytes()))

    def test_existing_output_is_rejected_and_preserved(self):
        self.output.mkdir()
        marker = self.output / "old.txt"
        marker.write_text("old")
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(list(self.output.iterdir()), [marker])
        self.assertEqual(marker.read_text(), "old")

    def test_wrong_source_hash_fails_before_output(self):
        self.source_path.write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, "SHA256"):
            P029.prepare(self.output, source=self.source_path)
        self.assertFalse(self.output.exists())

    def test_latest_V2_input_and_wrong_linked_score_are_rejected(self):
        for change in ("version_id", "submission", "Docker", "Internet", "private_input", "filelist"):
            with self.subTest(change=change):
                old_view, old_book, old_metadata = copy.deepcopy(self.view), copy.deepcopy(self.book), copy.deepcopy(self.metadata)
                if change == "version_id":
                    self.view["kernelRun"]["dataSources"][-1]["sourceId"] = 99999999
                elif change == "submission":
                    self.view["submission"]["sourceScriptVersionId"] += 1
                elif change == "Docker":
                    self.view["kernelRun"]["runInfo"]["dockerImageDigest"] = "0" * 64
                elif change == "Internet":
                    self.book["metadata"]["kaggle"]["isInternetEnabled"] = True
                elif change == "private_input":
                    self.metadata["Ortho-view.json"]["isPrivate"] = True
                else:
                    self.metadata["R6-v1-filelist.json"]["datasetFiles"][0]["totalBytes"] += 1
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assertFalse(self.output.exists())
                self.view, self.book, self.metadata = old_view, old_book, old_metadata

    def test_compile_rejects_unknown_magic_and_invalid_embedded_python(self):
        book = copy.deepcopy(self.book)
        book["cells"][2]["source"] = ["%%bash\necho should-not-run\n"]
        with self.assertRaisesRegex(ValueError, "magic"):
            P029.inspect_notebook(book)
        book = copy.deepcopy(self.book)
        book["cells"][6]["source"] = ["SOURCES = " + repr({
            name: "invalid python ???" for name in (
                "px_cache.py", "compact_train.py", "compact_infer.py", "probe_infer.py",
                "probe_dual.py", "knee_anchor_bank_v2.py", "side_common.py"
            )})]
        with self.assertRaises(SyntaxError):
            P029.inspect_notebook(book)

    def test_embedded_sources_must_be_literal_not_executable_expression(self):
        book = copy.deepcopy(self.book)
        book["cells"][6]["source"] = ["SOURCES = dict()\n"]
        with self.assertRaises(ValueError):
            P029.inspect_notebook(book)

    def test_separate_preflight_rejects_same_named_R7_manifest_without_touching_notebook(self):
        self.prepare()
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"bundle_id": "8c7b666e27dc0e8f"}), encoding="utf-8")
        before = (self.output / "source.ipynb").read_bytes()
        completed = subprocess.run(
            [sys.executable, str(self.output / "verify_frozen_input.py"), "--manifest", str(manifest)],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("R6 V1 manifest SHA mismatch", completed.stderr)
        self.assertEqual((self.output / "source.ipynb").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
