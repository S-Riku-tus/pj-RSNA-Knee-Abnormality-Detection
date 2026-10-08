"""Offline scored-version/source/env binding; no public code is executed."""

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("public944_control", ROOT / "scripts/build_p004_public_control.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


@unittest.skipUnless((builder.SOURCE_DIR / "source.ipynb").is_file(), "Immutable public source evidence absent")
class PublicControlTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_all_code_bytes_and_order_preserved_without_guard_injection(self):
        original, _, _, _ = builder.verified_evidence()
        prepared = builder.prepare_notebook(original)
        self.assertEqual(builder.code_sources(prepared), builder.code_sources(original))
        self.assertEqual(len(prepared["cells"]), len(original["cells"]))
        self.assertEqual(len(builder.code_sources(prepared)), 29)
        self.assertEqual(prepared["metadata"], original["metadata"])
        for cell in prepared["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
        self.assertTrue("".join(prepared["cells"][0]["source"]).endswith("".join(original["cells"][0]["source"])))
        compiled = builder.syntax_check(prepared)
        self.assertEqual(len(compiled), 29)
        self.assertEqual(sum(item["syntax"] == "ipython_writefile_python_body" for item in compiled), 3)

    def test_binding_to_author_scored_v2_original_environment_and19inputs(self):
        _, metadata, review, view = builder.verified_evidence()
        inputs = builder.manual_inputs(metadata, review)
        self.assertEqual(len(inputs), 19)
        self.assertEqual([item["sourceId"] for item in inputs], [item["sourceId"] for item in review["inputs"]])
        self.assertEqual([item["mountSlug"] for item in inputs], [item["mountSlug"] for item in review["inputs"]])
        self.assertEqual(view["kernelRun"]["id"], 355300588)
        self.assertEqual(view["submission"]["id"], 56837602)
        self.assertEqual(view["kernelRun"]["dockerImageVersionId"], 31430)
        self.assertEqual(view["kernelRun"]["runInfo"]["dockerImageDigest"], builder.DOCKER_DIGEST)
        notes = builder.manual_notes(inputs)
        self.assertIn("Copy & Edit", notes)
        self.assertIn("Original", notes)
        self.assertIn("最新Version5", notes)
        self.assertIn("Importするだけでは", notes)
        self.assertIn("品質確認", notes)
        self.assertIn("own reader: blended 3 checkpoints at weight 0.3", notes)
        log = builder.audit_author_reader_log(builder.SOURCE_DIR)
        self.assertTrue(log["reader_csv_written"])
        self.assertTrue(log["reader_blended3_at_weight030"])
        self.assertTrue(log["reader_failed_fallback_message_absent"])

    def copy_evidence(self):
        target = self.root / "evidence"
        target.mkdir()
        for name in ("source.ipynb", "metadata.json", "review.json", "view-model.json"):
            shutil.copyfile(builder.SOURCE_DIR / name, target / name)
        return target

    def test_source_tamper_rejected_before_any_handoff_write(self):
        evidence = self.copy_evidence()
        with (evidence / "source.ipynb").open("ab") as stream:
            stream.write(b"\n")
        output, notebook = self.root / "package", self.root / "book.ipynb"
        with self.assertRaisesRegex(ValueError, "SHA256"):
            builder.prepare_package(evidence, output, notebook)
        self.assertFalse(output.exists())
        self.assertFalse(notebook.exists())

    def test_latest_version_or_different_best_version_is_rejected(self):
        evidence = self.copy_evidence()
        review = builder.read_json(evidence / "review.json")
        review["best_submission_score"]["kernelVersionNumber"] = 5
        (evidence / "review.json").write_text(json.dumps(review), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "different version"):
            builder.verified_evidence(evidence)

    def test_package_hashes_and_raw_source_copy_match(self):
        output, notebook = self.root / "package", self.root / "book.ipynb"
        package = builder.prepare_package(output_dir=output, notebook_path=notebook)
        self.assertEqual(builder.digest(output / "original-source.ipynb"), builder.SOURCE_SHA256)
        self.assertEqual(builder.digest(notebook), builder.digest(output / "book.ipynb"))
        for name, entry in package["files"].items():
            self.assertEqual(builder.digest(output / name), entry["sha256"])
            self.assertEqual((output / name).stat().st_size, entry["bytes"])
        self.assertFalse((output / "kernel-metadata.json").exists())
        provenance = builder.read_json(output / "provenance.json")
        self.assertIsNone(provenance["own_public_lb"])
        self.assertFalse(provenance["own_visible_or_hidden_execution"])

    def test_existing_directory_and_notebook_are_never_overwritten(self):
        output, notebook = self.root / "package", self.root / "book.ipynb"
        output.mkdir()
        sentinel = output / "preserve.txt"
        sentinel.write_text("old artifact", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            builder.prepare_package(output_dir=output, notebook_path=notebook)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "old artifact")
        self.assertFalse(notebook.exists())
        other = self.root / "other-package"
        notebook.write_text("old notebook", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            builder.prepare_package(output_dir=other, notebook_path=notebook)
        self.assertEqual(notebook.read_text(encoding="utf-8"), "old notebook")
        self.assertFalse(other.exists())


if __name__ == "__main__":
    unittest.main()
