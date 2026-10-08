"""Offline Apex source/Input/container binding; published inference is never executed."""

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("public950_control", ROOT / "scripts/build_p005_public_control.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


@unittest.skipUnless((builder.SOURCE_DIR / "source.ipynb").is_file(), "Immutable Apex public evidence absent")
class ApexControlTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_all_original_code_preserved_and_magic_bodies_compile(self):
        original, _, _, _, _ = builder.verified_evidence()
        prepared = builder.prepare_notebook(original)
        self.assertEqual(builder.control_tools.code_sources(prepared), builder.control_tools.code_sources(original))
        self.assertEqual(len(prepared["cells"]), len(original["cells"]))
        self.assertEqual(prepared["metadata"], original["metadata"])
        code = builder.control_tools.code_sources(prepared)
        self.assertEqual(len(code), 7)
        for cell in prepared["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
        checks = builder.control_tools.syntax_check(prepared)
        self.assertEqual(len(checks), 7)
        self.assertEqual(sum(item["syntax"] == "ipython_writefile_python_body" for item in checks), 3)
        self.assertTrue("".join(prepared["cells"][0]["source"]).endswith("".join(original["cells"][0]["source"])))

    def test_actual_score3_public_inputs_original313container_and_decoder_failure_recorded(self):
        _, metadata, view, inputs, log = builder.verified_evidence()
        self.assertEqual(view["submission"]["id"], 56930357)
        self.assertEqual(view["submission"]["sourceScriptVersionId"], 356192954)
        self.assertEqual(view["kernelRun"]["dockerImageVersionId"], 31481)
        self.assertEqual(len(inputs), 3)
        self.assertEqual([item["sourceId"] for item in inputs], [154281, 20325538, 20440357])
        self.assertFalse(metadata["enableInternet"])
        self.assertTrue(log["python313_visible_log_path"])
        self.assertTrue(log["libjpeg_wheel_install_failure_seen"])
        self.assertTrue(log["found3_reader_checkpoints"])
        self.assertTrue(log["fusion_success"])
        self.assertTrue(log["fusion_exception_or_fallback_absent"])
        notes = builder.manual_notes(inputs)
        self.assertIn("条件付き採用", notes)
        self.assertIn("Original", notes)
        self.assertIn("品質確認", notes)

    def test_tampered_source_rejected_before_any_file_written(self):
        evidence = self.root / "evidence"
        evidence.mkdir()
        for name in ("source.ipynb", "metadata.json", "visible.log"):
            shutil.copyfile(builder.SOURCE_DIR / name, evidence / name)
        with (evidence / "source.ipynb").open("ab") as stream:
            stream.write(b"\n")
        output, notebook = self.root / "package", self.root / "book.ipynb"
        with self.assertRaisesRegex(ValueError, "SHA256"):
            builder.prepare_package(evidence, builder.VIEW_PATH, output, notebook)
        self.assertFalse(output.exists())
        self.assertFalse(notebook.exists())

    def test_immutable_package_hashes_audited_scope_and_no_fabricated_docker_metadata(self):
        output, notebook = self.root / "package", self.root / "book.ipynb"
        package = builder.prepare_package(output_dir=output, notebook_path=notebook)
        self.assertEqual(builder.digest(output / "original-source.ipynb"), builder.SOURCE_SHA256)
        self.assertEqual(builder.digest(notebook), builder.digest(output / "book.ipynb"))
        for name, entry in package["files"].items():
            self.assertEqual(builder.digest(output / name), entry["sha256"])
            self.assertEqual((output / name).stat().st_size, entry["bytes"])
        provenance = builder.read_json(output / "provenance.json")
        self.assertEqual(provenance["eligibility_status"], "conditionally_adopt_public_checkpoint_frozen_inference")
        self.assertEqual(provenance["eligibility_audit"]["sha256"], builder.digest(builder.ELIGIBILITY_PATH))
        self.assertIsNone(provenance["own_public_lb"])
        self.assertFalse(provenance["own_visible_or_hidden_execution"])
        self.assertFalse((output / "kernel-metadata.json").exists())
        before = builder.digest(notebook)
        with self.assertRaises(FileExistsError):
            builder.prepare_package(output_dir=output, notebook_path=notebook)
        self.assertEqual(builder.digest(notebook), before)

    def test_missing_or_wrong_eligibility_binding_rejected(self):
        with self.assertRaisesRegex(ValueError, "required"):
            builder.verified_eligibility(self.root / "missing.json")
        decision = builder.read_json(builder.ELIGIBILITY_PATH)
        decision["adopted_script_version_id"] = 1
        path = self.root / "wrong-source-decision.json"
        path.write_text(json.dumps(decision), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not adopt"):
            builder.verified_eligibility(path)


if __name__ == "__main__":
    unittest.main()
