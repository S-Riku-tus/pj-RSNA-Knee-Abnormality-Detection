"""Check that the scored p002 notebook rebuilds without ignored research artifacts."""

import ast
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import build_public_candidate as builder


class PublicCandidateRebuildTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for relative in (
            "docs/research/p002-effective-inputs-20261005.json",
            "docs/research/p002-output-contract-20261005.json",
            "notebooks/public/vendor/romantamrazov-dinosaur-v32.ipynb",
            "scripts/public_candidate_guard.py",
        ):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(builder.ROOT / relative, target)

    def test_rebuild_without_artifacts_preserves_scored_notebook(self):
        output = self.root / "handoff"
        with patch.object(builder, "ROOT", self.root):
            manifest = builder.build(output)
        self.assertFalse((self.root / "artifacts").exists())
        self.assertEqual(
            (output / "02_submit_p002.ipynb").read_bytes(),
            (builder.ROOT / "notebooks/public/01_submit_p002_scored.ipynb").read_bytes(),
        )
        self.assertFalse(manifest["inference_source_modified"])
        notebook = json.loads((output / "02_submit_p002.ipynb").read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])

    def test_changed_vendor_source_is_rejected_before_output(self):
        source = self.root / "notebooks/public/vendor/romantamrazov-dinosaur-v32.ipynb"
        with source.open("ab") as handle:
            handle.write(b"\n")
        output = self.root / "handoff"
        with patch.object(builder, "ROOT", self.root), self.assertRaisesRegex(ValueError, "source SHA mismatch"):
            builder.build(output)
        self.assertFalse(output.exists())

    def test_existing_handoff_is_preserved(self):
        output = self.root / "handoff"
        output.mkdir()
        sentinel = output / "existing.txt"
        sentinel.write_bytes(b"existing experiment")
        with patch.object(builder, "ROOT", self.root), self.assertRaises(FileExistsError):
            builder.build(output)
        self.assertEqual(sentinel.read_bytes(), b"existing experiment")


if __name__ == "__main__":
    unittest.main()
