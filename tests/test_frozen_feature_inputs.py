"""Offline packaging checks with artificial archives; no downloads or model loading."""

import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from rsna_knee.contracts import dump_json, sha256
from rsna_knee.feature_contracts import source_tree_hash

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("feature_packager", ROOT / "scripts/package_frozen_feature_inputs.py")
packager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packager)


class FeatureInputPackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "artifacts", prefix="feature-package-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "artifacts/kaggle/output"
        self.source = self.root / "source.zip"
        with zipfile.ZipFile(self.source, "w") as archive:
            for name, data in {
                "hubconf.py": "raise RuntimeError('Must never execute during packaging')\n",
                "dinov2/__init__.py": "# artificial source\n",
                "LICENSE": "artificial license fixture\n",
                "MODEL_CARD.md": "artificial model card fixture\n",
            }.items():
                archive.writestr(f"dinov2-{packager.REVISION}/{name}", data)
        self.weights = self.root / "dinov2_vits14_pretrain.pth"
        self.weights.write_bytes(b"artificial bytes, never torch.load\n")
        self.manifest = self.root / "manifest"
        self.manifest.mkdir()
        dump_json(self.manifest / "manifest.json", {"inputs_sha256": {"fixture": "not-real-data"}})
        (self.manifest / "weak.csv").write_text("StudyInstanceUID,fold\nsynthetic,2\n")
        (self.manifest / "gold.csv").write_text("StudyInstanceUID,fold\n")
        (self.manifest / "unrelated-private.txt").write_text("must not be packaged")
        dump_json(
            self.manifest / "fold-audit-v2.json",
            {
                "integrity_valid": True,
                "ready_for_training": True,
                "gold_used_for_tuning": False,
                "inputs_sha256": {"fixture": "not-real-data"},
                "manifest_sha256": {
                    name: sha256(self.manifest / name) for name in ("manifest.json", "weak.csv", "gold.csv")
                },
            },
        )
        self.bundle = self.root / "code.zip"
        with zipfile.ZipFile(self.bundle, "w") as archive:
            archive.writestr("src/rsna_knee/feature_contracts.py", "# synthetic\n")
        self.patches = patch.multiple(
            packager, ROOT=self.root, BUNDLE=self.bundle, BUNDLE_SHA=sha256(self.bundle), MANIFEST=self.manifest
        )
        self.patches.start()
        self.addCleanup(self.patches.stop)

    def test_archive_round_trip_hashes_notebook_and_manifest_allowlist(self):
        result = packager.package(self.source, self.weights, self.output)
        extracted = self.root / "extracted"
        with zipfile.ZipFile(result["upload_zip"]) as archive:
            self.assertNotIn("manifest/unrelated-private.txt", archive.namelist())
            self.assertEqual(
                {n for n in archive.namelist() if n.startswith("manifest/")},
                {"manifest/" + name for name in packager.MANIFEST_FILES},
            )
            archive.extractall(extracted)
        provenance = json.loads((extracted / "encoder/encoder-provenance.json").read_text())
        self.assertEqual(provenance["source_tree_sha256"], source_tree_hash(extracted / "encoder/dinov2-source"))
        self.assertEqual(provenance["weights_sha256"], sha256(extracted / "encoder/dinov2_vits14_pretrain.pth"))
        notebook = json.loads(Path(result["notebook"]).read_text(encoding="utf-8"))
        original = json.loads(packager.NOTEBOOK.read_text(encoding="utf-8"))
        # Feature extraction and QC cells must be byte-identical; only Input assignments change.
        for index in (4, 5):
            self.assertEqual(notebook["cells"][index]["source"], original["cells"][index]["source"])
        source = "\n".join("".join(c["source"]) for c in notebook["cells"])
        self.assertIn("PILOT = True", source)
        self.assertIn("CODE_INPUT = INPUT / 'code'", source)
        self.assertNotIn("your-reviewed-manifest", source)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "prepared06", "exec")
                self.assertEqual(cell["outputs"], [])
        record = json.loads((self.output / "package.json").read_text())
        self.assertEqual(record["input_zip_sha256"], sha256(result["upload_zip"]))
        self.assertFalse(record["uploaded_or_submitted"])

    def test_existing_output_refused_without_modification(self):
        self.output.mkdir(parents=True)
        sentinel = self.output / "keep.txt"
        sentinel.write_text("keep")
        with self.assertRaises(ValueError):
            packager.package(self.source, self.weights, self.output)
        self.assertEqual(sentinel.read_text(), "keep")

    def test_stale_fold_audit_refused_before_output(self):
        (self.manifest / "weak.csv").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Stale fold audit"):
            packager.package(self.source, self.weights, self.output)
        self.assertFalse(self.output.exists())

    def test_source_traversal_and_wrong_revision_refused(self):
        for member in (f"dinov2-{packager.REVISION}/../escape.py", "dinov2-other/hubconf.py"):
            with self.subTest(member=member):
                with zipfile.ZipFile(self.source, "w") as archive:
                    archive.writestr(member, "must not be extracted")
                with self.assertRaises(ValueError):
                    packager.package(self.source, self.weights, self.output)
                self.assertFalse(self.output.exists())

    def test_code_bundle_change_refused(self):
        self.bundle.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "code bundle has changed"):
            packager.package(self.source, self.weights, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
