"""Stdlib fake-header/transport tests. Never imports pydicom or reads DICOM."""
import ast
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "artifacts/kaggle/orthodiff-sag-header32-20261017-v1"
spec = importlib.util.spec_from_file_location("header32_test_runtime", ROOT / "scripts/orthodiffusion_header32_runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class HeaderOnly(unittest.TestCase):
    def row(self):
        return {"SeriesInstanceUID": "series", "Anatomical_Plane": "Sagittal", "Fluid_Sensitive": "1", "Fat_Suppression": "1"}

    def test_exact_first_one_header_allowlist_and_projection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "study" / "series"
            folder.mkdir(parents=True)
            for name in ("2.dcm", "10.dcm", "1.dcm"):
                (folder / name).write_bytes(b"fake-not-DICOM")
            calls = []
            def reader(path, **kwargs):
                calls.append((path.name, kwargs))
                return SimpleNamespace(StudyInstanceUID="study", SeriesInstanceUID="series", SOPInstanceUID="sop", SeriesDescription="DummySeriesDesc!",
                    SequenceName="*tse2d1_9", RepetitionTime=5000.0, EchoTime=28.0, ScanningSequence=["SE"],
                    file_meta=SimpleNamespace(TransferSyntaxUID="1.2.840.10008.1.2.1"))
            result, count = runtime.read_first_header("study", self.row(), root, reader)
            self.assertEqual(count, 1)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], "1.dcm")
            self.assertTrue(calls[0][1]["stop_before_pixels"])
            self.assertEqual(calls[0][1]["specific_tags"], list(runtime.TAGS))
            self.assertEqual(len(runtime.TAGS), 27)
            self.assertNotIn("PixelData", runtime.TAGS)
            self.assertNotIn("PatientName", runtime.TAGS)
            self.assertEqual(result["tags"]["RepetitionTime"], 5000.0)
            self.assertEqual(result["tags"]["ScanningSequence"], ["SE"])
            self.assertEqual(result["status"], "header_success")
            raw = (json.dumps(result["tags"], sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
            self.assertEqual(result["header_projection_sha256"], hashlib.sha256(raw).hexdigest())

    def test_UID_mismatch_and_reader_exception_are_recorded_without_second_try(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "study" / "series"
            folder.mkdir(parents=True)
            (folder / "1.dcm").write_bytes(b"fake")
            (folder / "2.dcm").write_bytes(b"fake")
            def wrong(*_, **__):
                return SimpleNamespace(StudyInstanceUID="other", SeriesInstanceUID="series")
            result, count = runtime.read_first_header("study", self.row(), root, wrong)
            self.assertEqual((result["status"], count), ("header_failure", 1))
            calls = []
            def failing(path, **_):
                calls.append(path)
                raise ValueError("fake-header-error")
            result, count = runtime.read_first_header("study", self.row(), root, failing)
            self.assertEqual((result["reason"], count, len(calls)), ("fake-header-error", 1, 1))

    def test_missing_and_nonfinite_do_not_create_false_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            result, count = runtime.read_first_header("study", self.row(), Path(temp), None)
            self.assertEqual((result["status"], count), ("header_failure", 0))
        self.assertIsNone(runtime.plain(float("nan")))
        self.assertEqual(runtime.plain([1, "SE", None]), [1, "SE", None])

    def test_transport_same_UID_and_cpu_competition_only(self):
        notebook = json.loads((PACK / "source.ipynb").read_bytes())
        payload = json.loads((PACK / "save-payload.json").read_bytes())
        config = json.loads((PACK / "config.json").read_bytes())
        original = json.loads((ROOT / "artifacts/kaggle/orthodiff-sag-mri32-pilot-20261017-v1/config.json").read_bytes())
        self.assertEqual(config["uids"], original["uids"])
        self.assertEqual(config["UID_order_sha256"], original["UID_order_sha256"])
        tree = ast.parse(notebook["cells"][1]["source"])
        node = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_raw" for t in node.targets))
        self.assertEqual(ast.literal_eval(node.value), (ROOT / "scripts/orthodiffusion_header32_runtime.py").read_bytes())
        self.assertEqual(payload["text"].encode(), (PACK / "source.ipynb").read_bytes())
        self.assertEqual(payload["competition_data_sources"], ["rsna-knee-abnormality-detection"])
        self.assertEqual(payload["dataset_data_sources"] + payload["kernel_data_sources"], [])
        self.assertTrue(payload["is_private"])
        self.assertFalse(payload["enable_gpu"] or payload["enable_internet"])
        self.assertNotIn("pydicom", sys.modules)
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
