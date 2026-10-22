"""Artificial CPU arrays/metadata only: no Torch, DICOM decoder, MRI or GPU."""
import ast
import base64
import hashlib
import importlib.util
import json
import math
import random
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


runtime = module("ortho_pilot_test_runtime", "scripts/orthodiffusion_mri_pilot_runtime.py")
builder = module("ortho_pilot_test_builder", "scripts/prepare_orthodiffusion_mri_pilot.py")
PACK = ROOT / "artifacts/kaggle/orthodiff-sag-mri32-pilot-20261017-v1"


class PilotProtocol(unittest.TestCase):
    def test_PD_tokens_and_contradictions(self):
        for text in ("PD_TSE", "sag-pdw-spair", "PROTON DENSITY", "protondensity", "pdfse"):
            self.assertTrue(runtime.classify_pd(text, "")["eligible"], text)
        for text in ("updated sequence", "SPDW", "T1_PD", "PD T2w", "T2TSE PD", "unknown"):
            self.assertFalse(runtime.classify_pd(text, "")["eligible"], text)

    def test_orientation_eight_axis_permutations_and_oblique_rejection(self):
        desired = np.arange(5 * 7).reshape(5, 7)
        for transpose in (False, True):
            for row_sign in (-1, 1):
                for col_sign in (-1, 1):
                    raw = desired.copy()
                    if row_sign < 0:
                        raw = raw[::-1, :]
                    if col_sign < 0:
                        raw = raw[:, ::-1]
                    target_row, target_col = [0, 0, -row_sign], [0, col_sign, 0]
                    if transpose:
                        raw = raw.T
                        raw_col, raw_row = target_row, target_col
                    else:
                        raw_col, raw_row = target_col, target_row
                    plan = runtime.orientation_plan(raw_col + raw_row)
                    actual = raw.T if plan["transpose"] else raw
                    if plan["flip_rows"]:
                        actual = actual[::-1, :]
                    if plan["flip_columns"]:
                        actual = actual[:, ::-1]
                    np.testing.assert_array_equal(actual, desired)
                    self.assertLess(plan["depth_normal_LPS"][0], 0)
        half = math.sqrt(.5)
        for invalid in ([0, half, half, 0, -half, half], [0, 1, 0, 0, 1, 0], [1, 0, 0, 0, 0, -1]):
            with self.assertRaises(runtime.PilotDataError):
                runtime.orientation_plan(invalid)

    def test_actual_adjacent_central16_physical_order_and_rejections(self):
        for depth in (16, 17, 18, 19, 32):
            headers = [{"IOP": [0, 1, 0, 0, 0, -1], "IPP": [-3.0 * index, 0, 0], "index": index} for index in range(depth)]
            random.Random(depth).shuffle(headers)
            plan = runtime.physical_order(headers)
            expected = list(range(depth // 2 - 8, depth // 2 + 8))
            self.assertEqual([headers[index]["index"] for index in plan["selected_indices"]], expected)
            self.assertEqual(len(plan["selected_indices"]), 16)
        base = [{"IOP": [0, 1, 0, 0, 0, -1], "IPP": [-3.0 * index, 0, 0]} for index in range(20)]
        invalid_sets = [base[:15], [*base[:19], base[18]], [*base[:19], {"IOP": [0, 1, 0, 0, 0, -1], "IPP": [-80, 0, 0]}],
                        [*base[:19], {"IOP": [0, -1, 0, 0, 0, -1], "IPP": [-57, 0, 0]}]]
        for invalid in invalid_sets:
            with self.assertRaises(runtime.PilotDataError):
                runtime.physical_order(invalid)

    def test_series_choice_uses_header_and_FS_UID_before_geometry_pixels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            values, rows = {}, []
            for sid, fs, description, count in (("z", "0", "PD_TSE", 1), ("a", "1", "PD FS", 30), ("b", "0", "T1", 40)):
                folder = root / "study" / sid
                folder.mkdir(parents=True)
                for index in range(count):
                    (folder / (str(index) + ".dcm")).write_bytes(b"synthetic-not-DICOM")
                values[sid] = SimpleNamespace(StudyInstanceUID="study", SeriesInstanceUID=sid, SeriesDescription=description, SequenceName="")
                rows.append({"SeriesInstanceUID": sid, "Anatomical_Plane": "Sagittal", "Fat_Suppression": fs, "Fluid_Sensitive": "1"})
            fake = SimpleNamespace(dcmread=lambda path, **_: values[path.parent.name], errors=SimpleNamespace(InvalidDicomError=type("FakeInvalid", (Exception,), {})))
            chosen, paths = runtime.choose_series("study", rows, root, fake)
            self.assertEqual(chosen["SeriesInstanceUID"], "z")
            self.assertEqual(chosen["stratum"], "PD_nonFS")
            self.assertEqual(len(paths), 1)  # Insufficient depth is not a reason to select the other PD series.
            chosen_fs, _ = runtime.choose_series("study", rows[1:], root, fake)
            self.assertEqual(chosen_fs["SeriesInstanceUID"], "a")
            self.assertEqual(chosen_fs["stratum"], "PD_FS_fallback")

    def test_real_read_clip_fake_pixels_resize_spy_preserves_float64_then_cast(self):
        with tempfile.TemporaryDirectory() as temp:
            paths, headers, arrays = [], {}, {}
            raw = np.array([[3, 18001, 60000], [32760, 65500, 8]], dtype=np.uint16)
            slope = .100000019
            for index in range(20):
                path = Path(temp) / (str(index) + ".dcm")
                path.write_bytes(b"synthetic-array-reference")
                paths.append(path)
                headers[path] = SimpleNamespace(StudyInstanceUID="study", SeriesInstanceUID="series", SOPInstanceUID="sop" + str(index),
                    SeriesDescription="PD_TSE", SequenceName="", NumberOfFrames=1, Rows=2, Columns=3, PixelSpacing=[.5, .5],
                    ImageOrientationPatient=[0, 1, 0, 0, 0, -1], ImagePositionPatient=[-3 * index, 0, 0],
                    PhotometricInterpretation="MONOCHROME2", RescaleSlope=slope, RescaleIntercept=1.00000009,
                    file_meta=SimpleNamespace(TransferSyntaxUID="uncompressed"))
                arrays[path] = raw + index
            fake = SimpleNamespace(dcmread=lambda path, **_: headers[path], uid=SimpleNamespace(UID=lambda value: SimpleNamespace(is_compressed=False, __str__=lambda: value)),
                                   pixels=SimpleNamespace(pixel_array=lambda path, **_: arrays[path]))
            def pin(path):
                return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            calls = []
            def resize_spy(value, size, interpolation):
                calls.append((value.dtype, size, interpolation))
                # Artificial arithmetic only. This is not a local OpenCV numerical qualification.
                return np.full((256, 256), value.mean(), dtype=value.dtype)
            fake_cv2 = SimpleNamespace(resize=resize_spy, INTER_LINEAR=1)
            actual, detail = runtime.read_clip("study", {"SeriesInstanceUID": "series"}, paths, fake, np, fake_cv2, {"file_pin": pin})
            self.assertEqual(calls, [(np.dtype("float64"), (256, 256), 1)] * 16)
            expected = np.stack([np.full((256, 256), ((raw + index).astype(np.float64) * slope + 1.00000009).mean(), dtype=np.float64) for index in range(2, 18)])
            wrong = np.stack([np.full((256, 256), ((raw + index).astype(np.float64) * slope + 1.00000009).astype(np.float32).mean(), dtype=np.float32) for index in range(2, 18)])
            self.assertEqual(actual.dtype, np.float64)
            np.testing.assert_array_equal(actual, expected)
            self.assertGreater(np.count_nonzero(actual.astype(np.float32) != wrong), 0)
            self.assertEqual([item["SOPInstanceUID"] for item in detail["selected_sources"]], ["sop" + str(index) for index in range(2, 18)])
            self.assertEqual(len(detail["selected_sources"]), 16)

    def test_cohort_source_transport_and_zero_scientific_execution(self):
        cohort = builder.select_cohort()
        self.assertEqual(cohort["UID_order_sha256"], "4772418b195a729f6672ca35b42e3335bfa46f30a3c215232b9de13449c8ffb8")
        self.assertEqual(cohort["selected_gold_UID_overlap"], 0)
        self.assertEqual(cohort["selected_gold_group_overlap"], 0)
        self.assertEqual(len({row["group_id"] for row in cohort["selected"]}), 32)
        notebook = json.loads((PACK / "source.ipynb").read_bytes())
        payload = json.loads((PACK / "save-payload.json").read_bytes())
        tree = ast.parse(notebook["cells"][1]["source"])
        assign = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "_od_raw" for target in node.targets))
        raw = zlib.decompress(base64.b85decode(ast.literal_eval(assign.value.args[0].args[0])))
        bundle = json.loads(raw)
        self.assertEqual(bundle["runtimes"]["pilot"].encode(), (ROOT / "scripts/orthodiffusion_mri_pilot_runtime.py").read_bytes())
        self.assertEqual(bundle["config"], json.loads((PACK / "config.json").read_bytes()))
        self.assertEqual(payload["text"].encode(), (PACK / "source.ipynb").read_bytes())
        self.assertEqual(len(payload["dataset_data_sources"] + payload["kernel_data_sources"] + payload["competition_data_sources"]), 3)
        self.assertNotIn("gold_ids", bundle["config"])
        self.assertNotIn("selected", bundle["config"])
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("pydicom", sys.modules)
        self.assertFalse(any(key.startswith("od_qual_pinned") for key in sys.modules))


if __name__ == "__main__":
    unittest.main()
