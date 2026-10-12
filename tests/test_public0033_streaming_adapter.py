"""Artificial pixel fixtures and global submission contracts; no real MRI/weights."""

import csv
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "public0033_streaming_adapter", ROOT / "scripts/public0033_streaming_adapter.py"
)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)
HAS_IMAGING = all(importlib.util.find_spec(name) for name in ("numpy", "torch"))


class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="meniscus10-contract-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.incumbent, self.branch, self.output = (self.root / name for name in ("base.csv", "raw.csv", "new.csv"))

    def write_csv(self, path, labels, rows):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["StudyInstanceUID", *labels])
            writer.writerows(rows)

    def base(self):
        rows = []
        for uid, medial, lateral in (("u3", "0.8", "0.1"), ("u1", "0.2", "0.4"), ("u2", "0.5", "0.9")):
            values = ["0.2500000000"] * 12
            values[2:4] = [medial, lateral]
            rows.append([uid, *values])
        self.write_csv(self.incumbent, adapter.LABELS, rows)
        self.write_csv(self.branch, adapter.MENISCUS, [["u1", 0.8, 0.6], ["u2", 0.5, 0.3], ["u3", 0.2, 0.9]])

    def test_overlay_keeps_incumbent_bytes_other_target_strings_and_uid_order(self):
        self.base()
        before = self.incumbent.read_bytes()
        result = adapter.fixed_meniscus_overlay(self.incumbent, self.branch, self.output)
        self.assertEqual(before, self.incumbent.read_bytes())
        original = adapter._read_csv(self.incumbent, adapter.LABELS)
        changed = adapter._read_csv(self.output, adapter.LABELS)
        self.assertEqual([row["StudyInstanceUID"] for row in original], [row["StudyInstanceUID"] for row in changed])
        for left, right in zip(original, changed):
            for label in set(adapter.LABELS) - set(adapter.MENISCUS):
                self.assertEqual(left[label], right[label])
        self.assertEqual(result["incumbent_sha256"], hashlib.sha256(before).hexdigest())
        self.assertFalse(result["public_improvement_verified"])

    def test_global_average_ties_and_order(self):
        self.assertEqual(adapter._rank_percentile([0.5, 0.5, 0.1, 0.9]), [0.625, 0.625, 0.25, 1.0])

    def test_missing_uid_and_nonfinite_raw_refuse_new_submission(self):
        self.base()
        self.write_csv(self.branch, adapter.MENISCUS, [["u1", 0.8, 0.6], ["u2", 0.5, 0.3]])
        with self.assertRaises(ValueError):
            adapter.fixed_meniscus_overlay(self.incumbent, self.branch, self.output)
        self.assertFalse(self.output.exists())
        self.write_csv(self.branch, adapter.MENISCUS, [["u1", "nan", 0.6], ["u2", 0.5, 0.3], ["u3", 0.2, 0.9]])
        with self.assertRaises(ValueError):
            adapter.fixed_meniscus_overlay(self.incumbent, self.branch, self.output)
        self.assertFalse(self.output.exists())

    def test_overwrite_or_input_alias_refused(self):
        self.base()
        with self.assertRaises(FileExistsError):
            adapter.fixed_meniscus_overlay(self.incumbent, self.branch, self.incumbent)
        self.output.write_text("previous", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            adapter.fixed_meniscus_overlay(self.incumbent, self.branch, self.output)
        self.assertEqual(self.output.read_text(), "previous")


@unittest.skipUnless(HAS_IMAGING, "Optional numpy/torch unavailable")
class PixelAndCacheTests(unittest.TestCase):
    def test_separate_branch_preflight_failure_keeps_output_directory_absent(self):
        with tempfile.TemporaryDirectory(prefix="meniscus10-preflight-") as temporary:
            output = Path(temporary) / "new-branch"
            module = SimpleNamespace(
                verify_bundle=lambda _root: {},
                _require_t4x2=lambda: (_ for _ in ()).throw(RuntimeError("artificial CUDA unavailable")),
            )
            with patch.object(adapter, "load_pinned_runtime", return_value=module), self.assertRaises(RuntimeError):
                adapter.run_separate_branch(
                    bundle_root=temporary,
                    series_rows=[],
                    test_series_root=temporary,
                    study_ids=["u1", "u2"],
                    output_directory=output,
                    scratch=temporary,
                )
            self.assertFalse(output.exists())

    def test_separate_branch_stock_api_mock_outputs_sorted_raw_only(self):
        import numpy as np

        with tempfile.TemporaryDirectory(prefix="meniscus10-separate-") as temporary:
            root, calls = Path(temporary), []
            base = root / "submission.csv"
            base.write_text("frozen-Apex-byte-fixture", encoding="utf-8")

            def build(uid):
                return np.full(adapter.CACHE_SHAPE, int(uid[-1]), np.uint8), np.ones(6, np.float32)

            def predict(cache, mask, studies, paths):
                calls.append((cache.shape, mask.dtype, studies, paths))
                with Path(paths["bag_raw_csv"]).open("x", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["StudyInstanceUID", *adapter.MENISCUS])
                    writer.writerows([[uid, 0.1, 0.2] for uid in studies])
                return {"status": "passed"}

            module = SimpleNamespace(
                verify_bundle=lambda _root: {}, _require_t4x2=lambda: ["T4", "T4"], run_cached_inference=predict
            )
            with (
                patch.object(adapter, "load_pinned_runtime", return_value=module),
                patch.object(adapter, "make_dicom_study_builder", return_value=build),
            ):
                result = adapter.run_separate_branch(
                    bundle_root=temporary,
                    series_rows=[],
                    test_series_root=temporary,
                    study_ids=["u3", "u1", "u2"],
                    output_directory=root / "branch",
                    scratch=root / "scratch",
                )
            self.assertEqual(result["status"], "passed")
            self.assertEqual(calls[0][2], ["u1", "u2", "u3"])
            self.assertEqual(base.read_text(encoding="utf-8"), "frozen-Apex-byte-fixture")
            self.assertFalse((root / "branch/submission.csv").exists())
            self.assertEqual(list((root / "scratch").iterdir()), [])

    def test_recovered_slot_classification_longest_series_and_laterality(self):
        headers = [
            {"SeriesDescription": "PD_FS", "plane": "Sagittal", "n_slices": 20},
            {"SeriesDescription": "PD_FS", "plane": "Sagittal", "n_slices": 30},
            {"SeriesDescription": "T1W", "plane": "Coronal", "n_slices": 25},
            {"SeriesDescription": "PD", "plane": "Sagittal", "n_slices": 29},
        ]
        selected = adapter.choose_recovered_slots([adapter.classify_header(row) for row in headers])
        self.assertEqual(set(selected), {"SAG_FLUID_FS", "SAG_FLUID_NOFS", "COR_T1"})
        self.assertEqual(selected["SAG_FLUID_FS"]["n_slices"], 30)
        self.assertEqual(adapter.laterality_from_headers([{"Laterality": "R"}, {"Laterality": "L"}]), "R")
        geometry = {
            "ImagePositionPatient": "-30|0|0",
            "ImageOrientationPatient": "1|0|0|0|1|0",
            "PixelSpacing": "1|1",
            "Rows": "10",
            "Columns": "10",
        }
        self.assertEqual(adapter.laterality_from_headers([geometry]), "R")
        geometry["ImagePositionPatient"] = "0|0|0"
        self.assertIsNone(adapter.laterality_from_headers([geometry]))

    @unittest.skipUnless(importlib.util.find_spec("pydicom"), "Optional pydicom unavailable")
    def test_lazy_dicom_caller_uses_raw_projection_order_and_all_six_slots(self):
        import numpy as np

        with tempfile.TemporaryDirectory(prefix="meniscus10-fake-dicom-") as temporary:
            root = Path(temporary)
            rows, descriptions, planes = [], {}, {}
            for index, (_, plane, fluid, fat_sat) in enumerate(adapter.SLOTS):
                series = f"series{index}"
                folder = root / "study" / series
                folder.mkdir(parents=True)
                for frame in range(29):
                    (folder / f"slice{frame}.dcm").write_text("artificial-placeholder", encoding="utf-8")
                descriptions[series] = ("PD" if fluid else "T1W") + (" FS" if fat_sat else "")
                planes[series] = plane
                rows.append({"StudyInstanceUID": "study", "SeriesInstanceUID": series, "Anatomical_Plane": plane})
            reads = []

            def read(path, stop_before_pixels=False, **_kwargs):
                path = Path(path)
                series, frame = path.parent.name, int(path.stem.removeprefix("slice"))
                reads.append((series, frame, stop_before_pixels))
                return SimpleNamespace(
                    SeriesDescription=descriptions[series],
                    SequenceName="",
                    ScanOptions=[],
                    ScanningSequence="SE",
                    RepetitionTime=2000,
                    EchoTime=30,
                    Laterality="R",
                    PixelSpacing=[0.5, 0.5],
                    Rows=31,
                    Columns=27,
                    RescaleSlope=1.125,
                    RescaleIntercept=-42.5,
                    ImagePositionPatient=[0, 0, float(frame)],
                    ImageOrientationPatient=[1, 0, 0, 0, 1, 0],
                    InstanceNumber=frame,
                    pixel_array=np.arange(31 * 27, dtype=np.float32).reshape(31, 27) + frame * 3.25,
                )

            with patch("pydicom.dcmread", side_effect=read):
                builder = adapter.make_dicom_study_builder(rows, root)
                self.assertEqual(reads, [])
                cache, mask = builder("study")
                self.assertTrue(np.array_equal(mask, np.ones(6, np.float32)))
                self.assertEqual(cache.shape, adapter.CACHE_SHAPE)
                selected = sorted(set(adapter.selected_indices(29)))
                for series in descriptions:
                    decoded = [frame for name, frame, header_only in reads if name == series and not header_only]
                    self.assertEqual(decoded, selected)
                expected = adapter.prepare_slot(
                    29,
                    lambda frame: (np.arange(31 * 27, dtype=np.float32).reshape(31, 27) + frame * 3.25) * 1.125 - 42.5,
                    0.5,
                    plane="Sagittal",
                    laterality="R",
                )
                self.assertTrue(np.array_equal(cache[0], expected))
                with self.assertRaises(ValueError):
                    builder("../outside")

    def test_twelve_sampling_preserves_reference_pad_last_policy(self):
        self.assertEqual(adapter.selected_indices(1).tolist(), [0] * 12)
        self.assertEqual(adapter.selected_indices(3).tolist(), [0] + [1] * 11)
        self.assertEqual(adapter.selected_indices(29).tolist(), [5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 22])

    def fixture(self, failed=()):
        import numpy as np

        reads = []

        def pixels(index):
            reads.append(index)
            if index in failed:
                raise ValueError("artificial decode failure")
            image = np.arange(31 * 27, dtype=np.float32).reshape(31, 27)
            return image + index * 3.25

        return reads, pixels

    def test_decode_only_unique_selected_frames_and_correct_right_laterality(self):
        import numpy as np

        reads, pixels = self.fixture()
        left = adapter.prepare_slot(29, pixels, 0.5, plane="Coronal", laterality="L")
        self.assertEqual(reads, sorted(set(adapter.selected_indices(29))))
        right = adapter.prepare_slot(29, pixels, 0.5, plane="Coronal", laterality="R")
        self.assertTrue(np.array_equal(right, left[..., ::-1]))
        sagittal = adapter.prepare_slot(29, pixels, 0.5, plane="Sagittal", laterality="R")
        self.assertTrue(np.array_equal(sagittal, left[::-1]))
        self.assertEqual(left.shape, (12, 336, 336))
        self.assertEqual(left.dtype, np.uint8)

    def test_partial_failed_slot_keeps_pixels_but_all_failed_is_unavailable(self):
        self.assertIsNone(
            adapter.prepare_slot(0, lambda _index: self.fail("empty slot must not decode"), 0.5, plane="Sagittal")
        )
        _, pixels = self.fixture(failed=(5, 8))
        valid = adapter.prepare_slot(29, pixels, 0.5, plane="Sagittal")
        self.assertIsNotNone(valid)
        _, unavailable = self.fixture(failed=tuple(adapter.selected_indices(29)))
        self.assertIsNone(adapter.prepare_slot(29, unavailable, 0.5, plane="Sagittal"))
        with self.assertRaises(ValueError):
            adapter.prepare_study({})

    def test_memmap_matches_per_study_callback_and_cleans_scratch(self):
        import numpy as np

        with tempfile.TemporaryDirectory(prefix="meniscus10-cache-") as temporary:
            visited = []

            def build(uid):
                visited.append(uid)
                value = int(uid[-1])
                return np.full(adapter.CACHE_SHAPE, value, np.uint8), np.ones(6, np.float32)

            with adapter.memmap_cache(["u3", "u1", "u2"], build, scratch=temporary) as (cache, mask, ordered):
                self.assertIsInstance(cache, np.memmap)
                self.assertEqual(ordered, ["u1", "u2", "u3"])
                self.assertEqual(visited, ordered)
                self.assertEqual(cache.shape, (3, *adapter.CACHE_SHAPE))
                self.assertTrue(np.array_equal(cache[:, 0, 0, 0, 0], [1, 2, 3]))
                self.assertEqual(mask.dtype, np.float32)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_cache_builder_failure_cleans_and_emits_no_partial_predictions(self):
        import numpy as np

        with tempfile.TemporaryDirectory(prefix="meniscus10-cache-fail-") as temporary:

            def build(_uid):
                return np.zeros(adapter.CACHE_SHAPE, np.uint8), np.zeros(6, np.float32)

            with self.assertRaises(ValueError), adapter.memmap_cache(["u1", "u2"], build, scratch=temporary):
                self.fail("unavailable study should not be yielded")
            self.assertEqual(list(Path(temporary).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
