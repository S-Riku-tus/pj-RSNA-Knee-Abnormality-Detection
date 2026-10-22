"""CPU synthetic arrays/metadata only; no DICOM decoder, Torch, model or GPU."""
import ast
import base64
import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "artifacts/kaggle/orthodiff-sag-fluid1080-20261017-v3"
PARENT = ROOT / "artifacts/kaggle/orthodiff-sag-fullweak4207-20261017-v2"


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


fluid = module("fluid_fallback_cpu_runtime", "scripts/orthodiffusion_fluid_fallback_runtime.py")
pilot = module("fluid_original_pilot_cpu", "scripts/orthodiffusion_mri_pilot_runtime.py")
full = module("fluid_original_full_cpu", "scripts/orthodiffusion_fullweak_runtime.py")
builder = module("fluid_fallback_cpu_builder", "scripts/prepare_orthodiffusion_fluid_fallback.py")


def pin(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def synthetic(folder, depth=20, description="DummySeriesDesc!", sid="fixed", photometric="MONOCHROME2"):
    headers, arrays, reads, decodes, resize_calls = {}, {}, [], [], []
    paths = []
    for index in range(depth):
        path = folder / "study" / sid / (str(index) + ".dcm")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic-array-reference")
        paths.append(path)
        headers[path] = SimpleNamespace(StudyInstanceUID="study", SeriesInstanceUID=sid, SOPInstanceUID="sop" + str(index),
            SeriesDescription=description, SequenceName="", NumberOfFrames=1, Rows=2, Columns=3, PixelSpacing=[.5, .5],
            ImageOrientationPatient=[0, 1, 0, 0, 0, -1], ImagePositionPatient=[-3 * index, 0, 0],
            PhotometricInterpretation=photometric, RescaleSlope=.100000019, RescaleIntercept=1.00000009,
            file_meta=SimpleNamespace(TransferSyntaxUID="uncompressed"))
        arrays[path] = np.array([[3, 18001, 60000], [32760, 65500, 8]], dtype=np.uint16) + index
    def dcmread(path, **kwargs):
        reads.append((path, kwargs))
        return headers[path]
    def decode(path, **_):
        decodes.append(path)
        return arrays[path]
    def resize(value, size, interpolation):
        resize_calls.append((value.dtype, size, interpolation))
        return np.full((256, 256), value.mean(), dtype=value.dtype)
    fake = SimpleNamespace(dcmread=dcmread, uid=SimpleNamespace(UID=lambda _: SimpleNamespace(is_compressed=False)),
                           pixels=SimpleNamespace(pixel_array=decode))
    return paths, headers, fake, SimpleNamespace(resize=resize, INTER_LINEAR=1), reads, decodes, resize_calls


def request(selected=True):
    return {"StudyInstanceUID": "study", "route": "fluid_sensitive_fallback", "confirmed_PD": False,
            "status": "metadata_candidate_selected" if selected else "metadata_unavailable",
            "selected_series": {"SeriesInstanceUID": "fixed", "metadata": {"Anatomical_Plane": "Sagittal", "Fluid_Sensitive": "1", "Fat_Suppression": "1"},
                                "Fluid_Sensitive": True, "Fat_Suppression": True} if selected else None}


class FluidFallbackProtocol(unittest.TestCase):
    def test_literal_reader_numeric_parity_carrier_and_environment(self):
        runtimes = {name: (PACK / (name + "-runtime.py")).read_bytes().decode() for name in ("base", "pilot", "full", "fluid")}
        parity = builder.science_parity(runtimes["pilot"], runtimes["full"], runtimes["fluid"])
        self.assertTrue(parity["restored_reader_AST_exact"])
        for name in ("base", "pilot", "full"):
            self.assertEqual((PACK / (name + "-runtime.py")).read_bytes(), (PARENT / (name + "-runtime.py")).read_bytes())
        self.assertEqual((PACK / "vendor.json").read_bytes(), (PARENT / "vendor.json").read_bytes())
        notebook = json.loads((PACK / "source.ipynb").read_bytes())
        payload, parent_payload = (json.loads((folder / "save-payload.json").read_bytes()) for folder in (PACK, PARENT))
        tree = ast.parse(notebook["cells"][1]["source"])
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "_ff_raw" for target in node.targets))
        raw = zlib.decompress(base64.b85decode(ast.literal_eval(assignment.value.args[0].args[0])))
        bundle = json.loads(raw)
        manifest = json.loads((PACK / "manifest.json").read_bytes())
        self.assertEqual({"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}, manifest["bundle_pin"])
        self.assertEqual(bundle["runtimes"], runtimes)
        self.assertEqual(bundle["config"], json.loads((PACK / "config.json").read_bytes()))
        reverted = copy.deepcopy(payload)
        for field in ("slug", "new_title", "text"):
            reverted[field] = parent_payload[field]
        self.assertEqual(reverted, parent_payload)
        self.assertEqual(payload["text"].encode(), (PACK / "source.ipynb").read_bytes())
        requests = bundle["config"]["fluid_requests"]
        self.assertEqual(len(requests), 1080)
        self.assertEqual(sum(row["selected_series"] is None for row in requests), 3)
        self.assertTrue(all(not {"group_id", "fold", "source", "Report", "labels"}.intersection(row) for row in requests))
        self.assertEqual(manifest["subset_proof"]["parent_records_fully_consumed"], 4207)
        self.assertEqual(manifest["subset_proof"]["parent_success_intersection"], 0)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("pydicom", sys.modules)

    def test_unknown_name_reader_preserves_central16_float64_and_original_PD_values(self):
        original_classifier = pilot.classify_pd
        new_reader = fluid.bind_reader(vars(pilot))
        self.assertIsNot(new_reader.__globals__, vars(pilot))
        self.assertNotIn("classify_pd", new_reader.__globals__)
        with tempfile.TemporaryDirectory() as temp:
            paths, headers, fake, cv2, _, decodes, calls = synthetic(Path(temp), photometric="MONOCHROME1")
            selected = {"SeriesInstanceUID": "fixed", "route": "fluid_sensitive_fallback", "confirmed_PD": False, "stratum": "fluid_sensitive_FS"}
            with self.assertRaisesRegex(pilot.PilotDataError, "mixed_or_unknown_PD_header"):
                pilot.read_clip("study", selected, paths, fake, np, cv2, {"file_pin": pin})
            self.assertEqual(decodes, [])
            actual, detail = new_reader("study", selected, paths, fake, np, cv2, {"file_pin": pin})
            self.assertEqual(calls, [(np.dtype("float64"), (256, 256), 1)] * 16)
            self.assertEqual([row["SOPInstanceUID"] for row in detail["selected_sources"]], ["sop" + str(i) for i in range(2, 18)])
            for header in headers.values():
                header.SeriesDescription = "PD_TSE"
            control, control_detail = pilot.read_clip("study", selected, paths, fake, np, cv2, {"file_pin": pin})
            np.testing.assert_array_equal(actual, control)
            self.assertEqual(detail, control_detail)
            headers[paths[0]].SOPInstanceUID = headers[paths[1]].SOPInstanceUID
            with self.assertRaisesRegex(pilot.PilotDataError, "duplicate_SOP"):
                new_reader("study", selected, paths, fake, np, cv2, {"file_pin": pin})
        self.assertIs(pilot.classify_pd, original_classifier)

    def test_fixed_series_no_retry_and_metadata_unavailable_has_no_header_or_pixels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths, _, fake, cv2, reads, decodes, _ = synthetic(root, depth=5)
            synthetic(root, depth=20, sid="other")
            row = {"StudyInstanceUID": "study", "SeriesInstanceUID": "fixed", **request()["selected_series"]["metadata"]}
            selected, chosen_paths = fluid.choose_fixed_series("study", request(), [row], root, pilot.check)
            self.assertEqual(chosen_paths, sorted(paths))
            self.assertEqual(reads, [])
            with self.assertRaisesRegex(pilot.PilotDataError, "fewer_than16"):
                fluid.bind_reader(vars(pilot))("study", selected, chosen_paths, fake, np, cv2, {"file_pin": pin})
            self.assertEqual(decodes, [])
            reads.clear()
            with self.assertRaisesRegex(pilot.PilotDataError, "metadata_unavailable"):
                fluid.choose_fixed_series("study", request(False), [], root, pilot.check)
            self.assertEqual(reads, [])
            self.assertEqual(decodes, [])
            wrong = copy.deepcopy(row)
            wrong["Fat_Suppression"] = "0"
            with self.assertRaisesRegex(pilot.PilotDataError, "metadata_mismatch"):
                fluid.choose_fixed_series("study", request(), [wrong], root, pilot.check)

    def test_reused_journal_failure_no_feature_and_new_status_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            journal = full.RecordJournal(Path(temp) / "records.jsonl", ["unavailable"])
            bad = {"StudyInstanceUID": "unavailable", "status": "data_failure", "reason": "no candidate",
                   "reason_type": "PilotDataError", "feature_forward_calls": 0, "fallback_features": False,
                   "pooled_float32_values": [0.] * 256}
            with self.assertRaisesRegex(ValueError, "fabricated"):
                journal.append(bad)
            del bad["pooled_float32_values"]
            journal.append(bad)
            status = fluid.status_snapshot({"complete": False, "phase": "data_failure_recorded"}, journal, fluid.time.monotonic())
            self.assertEqual(status["schema_version"], "orthodiff_fluid_fallback1080_small_status_v1")
            self.assertEqual(status["coverage"]["success"], 0)
            self.assertEqual(journal.finalize(), pin(journal.path))


if __name__ == "__main__":
    unittest.main()
