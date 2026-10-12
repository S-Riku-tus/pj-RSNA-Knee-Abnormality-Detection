"""Meaningful pilot safety/byte tests; synthetic images, never patient data."""

import copy
import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rsna_knee.contracts import ID, SERIES_COLUMNS, SERIES_ID, write_csv
from rsna_knee.triplet_cache import (
    array_hashes,
    decode_pixel_member,
    encode_pixel_member,
    export_pixel_pilot,
    extract_pixel_study,
    load_archive_study,
    pixel_contract,
    raw_pixel_bytes,
    validate_cohort,
    validate_pixels,
)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "build_native_triplet_pilot", ROOT / "scripts/build_native_triplet_pilot.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
SETTINGS = json.loads((ROOT / "configs/experiments/f002-dinov2-attention.json").read_text())["preprocess"]


class NativePixelPilotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="native-pixel-artificial-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cohort = builder.select_cohort()

    @staticmethod
    def series(key, *, series_id="synthetic-series"):
        return {
            ID: key,
            SERIES_ID: series_id,
            "Fluid_Sensitive": "1",
            "Fat_Suppression": "true",
            "Anatomical_Plane": "Sagittal",
        }

    @staticmethod
    def fake_reader(directory, settings):
        windows = np.zeros((2, 3, 392, 392), np.uint8)
        windows[0, :, 3:27, 2:29] = 193
        windows[1, 1, 18:32, 7:28] = 78
        return (
            windows,
            np.array([0.0, 1.0], np.float32),
            {
                "native_shape": [2, 31, 27],
                "pixel_spacing_mm": [0.4, 0.7],
                "selected_centers": [0, 1],
                "physical_span_mm": 3.2,
                "order": "physical_position",
                "normalization": "synthetic",
            },
        )

    def arrays(self):
        key = self.cohort["studies"][0][ID]
        return extract_pixel_study(
            self.root, key, [self.series(key)], SETTINGS, reader=self.fake_reader, qc_reader=self.fake_reader
        )

    def test_fixed_whole_group_train_side_without_gold_and_no_targets(self):
        self.assertEqual(len(validate_cohort(self.cohort)), 16)
        self.assertEqual(builder.select_cohort(), self.cohort)
        self.assertTrue(all(row["fold"] != 0 for row in self.cohort["studies"]))
        self.assertFalse(self.cohort["targets_used_for_selection"])
        self.assertEqual(raw_pixel_bytes(16, SETTINGS), 1_062_125_568)
        for change in ("gold", "fold", "partial-group", "target"):
            invalid = copy.deepcopy(self.cohort)
            if change == "gold":
                invalid["gold_ids"].append(invalid["studies"][0][ID])
            elif change == "fold":
                invalid["studies"][0]["fold"] = 0
            elif change == "partial-group":
                group = invalid["studies"][0]["group_id"]
                invalid["complete_selected_groups"][group].append("missing-same-group-study")
            else:
                invalid["studies"][0]["ACL"] = "0"
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_cohort(invalid)

    def test_exact_mask_metadata_and_compressed_bytes_with_corruption_rejection(self):
        arrays, coverage = self.arrays()
        self.assertTrue(validate_pixels(arrays, SETTINGS))
        self.assertEqual(arrays["mask"].sum(), 2)
        self.assertEqual(arrays["planes"].tolist(), [0, 3, 3])
        self.assertEqual(arrays["flags"].tolist(), [[1, 1], [0, 0], [0, 0]])
        self.assertEqual(coverage[0]["shared_byte_qc"]["status"], "passed")
        payload = encode_pixel_member(arrays, "fixed-test-fingerprint", SETTINGS)
        restored = decode_pixel_member(payload, "fixed-test-fingerprint", SETTINGS)
        self.assertEqual(array_hashes(restored), array_hashes(arrays))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            decode_pixel_member(payload, "changed", SETTINGS)
        arrays["mask"][:] = False
        with self.assertRaisesRegex(ValueError, "all-masked"):
            validate_pixels(arrays, SETTINGS)

    def test_shared_reader_byte_QC_detects_one_pixel_delta(self):
        def changed(directory, settings):
            windows, positions, detail = self.fake_reader(directory, settings)
            windows[0, 0, 1, 1] ^= 1
            return windows, positions, detail

        key = self.cohort["studies"][0][ID]
        with self.assertRaisesRegex(ValueError, "byte QC"):
            extract_pixel_study(
                self.root, key, [self.series(key)], SETTINGS, reader=self.fake_reader, qc_reader=changed
            )

    def test_actual_artificial_DICOM_392_shared_vs_bounded_byte_QC(self):
        # Reuse the existing artificial DICOM fixture; no real MRI is accessed.
        fixture_spec = importlib.util.spec_from_file_location(
            "artificial_bounded_tests", ROOT / "tests/test_bounded_feature_reader.py"
        )
        fixtures = importlib.util.module_from_spec(fixture_spec)
        fixture_spec.loader.exec_module(fixtures)
        case = fixtures.BoundedReaderTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        directory = case.fixture(count=7, shape=(31, 27), mono1=True, signed=True, slope=-0.375)
        from rsna_knee.feature_imaging import read_feature_windows

        actual = fixtures.bounded.make_bounded_reader(case.workspace)
        expected = read_feature_windows(directory, SETTINGS)
        got = actual(directory, SETTINGS)
        self.assertEqual(got[0].shape, (7, 3, 392, 392))
        self.assertEqual(got[0].tobytes(), expected[0].tobytes())
        self.assertEqual(got[1].tobytes(), expected[1].tobytes())
        self.assertEqual(got[2], expected[2])
        self.assertEqual(list(case.workspace.iterdir()), [])

    def test_export16_never_opens_teacher_and_direct_member_hash_checks(self):
        rows = [self.series(row[ID]) for row in self.cohort["studies"]]
        write_csv(self.root / "train_series.csv", SERIES_COLUMNS, rows)
        for name in ("train.csv", "weak.csv", "gold.csv"):
            (self.root / name).write_text("DO NOT READ ANY TEACHER", encoding="utf-8")
        from rsna_knee import triplet_cache

        original = triplet_cache.read_csv
        opened = []

        def only_series(path, required):
            self.assertEqual(Path(path).name, "train_series.csv")
            opened.append(Path(path).name)
            return original(path, required)

        with patch.object(triplet_cache, "read_csv", side_effect=only_series):
            result = export_pixel_pilot(
                self.root,
                self.root / "output",
                SETTINGS,
                self.cohort,
                reader=self.fake_reader,
                qc_reader=self.fake_reader,
            )
        self.assertEqual(opened, ["train_series.csv"])
        self.assertTrue(result["complete"])
        self.assertEqual(result["processed"], 16)
        self.assertEqual(result["preprocessing_byte_QC_count"], 16)
        self.assertLess(result["archive"]["bytes"], 1024**3)
        self.assertFalse(result["teacher_CSV_read"])
        self.assertFalse(result["for_submission"])
        archive = self.root / "output/native392-pixel16.zip"
        key = self.cohort["studies"][0][ID]
        self.assertEqual(
            array_hashes(load_archive_study(archive, key, result)), result["files"][f"{key}.npz"]["array_sha256"]
        )
        changed = copy.deepcopy(result)
        changed["files"][f"{key}.npz"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_archive_study(archive, key, changed)
        with zipfile.ZipFile(archive) as source:
            self.assertEqual(len(source.namelist()), 16)
        with self.assertRaisesRegex(ValueError, "never overwritten"):
            export_pixel_pilot(self.root, self.root / "output", SETTINGS, self.cohort, reader=self.fake_reader)

    def test_generated_notebook_source_bundle_and_teacher_guard(self):
        import ast
        import base64
        import hashlib

        path = ROOT / "notebooks/public" / builder.NAME
        book = json.loads(path.read_text(encoding="utf-8"))
        codes = ["".join(cell["source"]) for cell in book["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(len(codes), 2)
        for code in codes:
            compile(code, str(path), "exec")
        tree = ast.parse(codes[0])
        payload_expr = next(
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "payload"
        )
        payload = base64.b64decode(ast.literal_eval(payload_expr.args[0]))
        self.assertEqual(hashlib.sha256(payload).hexdigest(), book["metadata"]["rsna_pixel_pilot"]["code_zip_sha256"])
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            self.assertNotIn("weak.csv", archive.namelist())
            for name in archive.namelist():
                self.assertFalse(name.endswith(".pth") or name.endswith(".pt"))
                if name.startswith("src/"):
                    self.assertEqual(archive.read(name), (ROOT / name).read_bytes())
        guard = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "prohibit_teacher_csv"
        )
        import os

        namespace = {"os": os, "Path": Path, "CSV_OPENED": []}
        exec(compile(ast.Module(body=[guard], type_ignores=[]), "guard-only", "exec"), namespace)
        for name in ("train.csv", "weak.csv", "gold.csv", "test.csv", "sample_submission.csv"):
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "forbidden"):
                namespace["prohibit_teacher_csv"]("open", (f"/kaggle/input/competitions/c/{name}",))
        namespace["prohibit_teacher_csv"]("open", ("/kaggle/input/c/train_series.csv",))
        self.assertEqual(namespace["CSV_OPENED"], ["/kaggle/input/c/train_series.csv"])
        self.assertFalse(book["metadata"]["rsna_pixel_pilot"]["for_submission"])

    def test_embedded_decoder_helpers_resource_snapshot_executes_on_cpu(self):
        namespace = {}
        exec(compile(builder.decoder_source(), "embedded-decoder-runtime", "exec"), namespace)
        result = namespace["resource_snapshot"]("cpu")
        self.assertGreater(result["free_disk_bytes"], 0)
        self.assertIn("cpu_peak_rss_bytes", result)
        self.assertNotIn("gpu_allocated_bytes", result)

    def test_preprocessing_changes_and_too_small_budget_rejected_before_decoding(self):
        changed = copy.deepcopy(SETTINGS)
        changed["image_size"] = 280
        with self.assertRaisesRegex(ValueError, "unchanged"):
            pixel_contract(changed)
        with self.assertRaisesRegex(ValueError, "budget before decoding"):
            export_pixel_pilot(self.root, self.root / "uncreated", SETTINGS, self.cohort, archive_budget_bytes=1024)
        self.assertFalse((self.root / "uncreated").exists())


if __name__ == "__main__":
    unittest.main()
