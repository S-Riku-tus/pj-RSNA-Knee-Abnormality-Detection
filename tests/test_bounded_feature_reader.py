"""Artificial MRI fixtures only: exact preprocessing and temporary-file bounds."""

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsna_knee.contracts import SERIES_ID, source_hashes

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bounded_feature_reader", ROOT / "scripts/bounded_feature_reader.py")
bounded = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bounded)
HAS_IMAGING = all(importlib.util.find_spec(name) for name in ("numpy", "pydicom", "PIL", "torch"))


@unittest.skipUnless(HAS_IMAGING, "Optional imaging dependencies absent")
class BoundedReaderTests(unittest.TestCase):
    def setUp(self):
        from rsna_knee.feature_contracts import load_feature_config

        self.temporary = tempfile.TemporaryDirectory(prefix="bounded-reader-artificial-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "working"
        self.workspace.mkdir()
        self.config = load_feature_config(ROOT / "configs/experiments/f002-dinov2-attention.json")
        self.config["preprocess"].update(image_size=28, max_series=2, slices_per_series=48)

    def fixture(
        self,
        name="series",
        *,
        count=7,
        shape=(31, 27),
        slope=1.125,
        intercept=-42.5,
        mono1=False,
        signed=False,
        angle=20,
        spacing=(0.4, 0.7),
        constant=False,
        duplicate=False,
    ):
        import numpy as np
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

        rng = np.random.default_rng(982)
        theta = math.radians(angle)
        orient = [math.cos(theta), 0, math.sin(theta), 0, 1, 0]
        normal = np.cross(orient[:3], orient[3:])
        directory = self.root / "test_series/artificial-study" / name
        directory.mkdir(parents=True)
        for number in range(count):
            path = directory / f"{count - number:04d}.dcm"
            meta = FileMetaDataset()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID = MRImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
            ds.SOPClassUID = meta.MediaStorageSOPClassUID
            ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            ds.Rows, ds.Columns = shape
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME1" if mono1 else "MONOCHROME2"
            ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
            ds.PixelRepresentation = int(signed)
            ds.ImageOrientationPatient = orient
            ds.ImagePositionPatient = (normal * (0 if duplicate else number * 3.2)).tolist()
            ds.InstanceNumber = 100 - number
            if spacing is not None:
                ds.PixelSpacing = list(spacing)
            ds.RescaleSlope = slope
            ds.RescaleIntercept = intercept + number * 0.25
            dtype = np.int16 if signed else np.uint16
            pixels = (
                np.full(shape, 19, dtype=dtype)
                if constant
                else rng.integers(-1500 if signed else 0, 3000, size=shape, dtype=dtype)
            )
            if constant:
                ds.RescaleIntercept = intercept
            ds.PixelData = pixels.tobytes()
            ds.save_as(path, enforce_file_format=True)
        return directory

    def exact(self, directory, *, observer=None, settings=None):
        import numpy as np

        from rsna_knee.feature_imaging import read_feature_windows

        settings = self.config["preprocess"] if settings is None else settings
        old = read_feature_windows(directory, settings)
        new = bounded.read_bounded_feature_windows(
            directory, settings, workspace=self.workspace, copy_chunk_bytes=1024, observer=observer
        )
        np.testing.assert_array_equal(new[0], old[0])
        np.testing.assert_array_equal(new[1], old[1])
        self.assertEqual(new[2], old[2])
        self.assertEqual(list(self.workspace.iterdir()), [])
        return new

    def test_exact_oblique_geometry_anisotropic_spacing_and_fractional_rescale(self):
        self.exact(self.fixture())

    def test_exact_mono1_signed_pixels_and_negative_slope(self):
        self.exact(self.fixture(mono1=True, signed=True, slope=-0.375, intercept=319.625, angle=-32))

    def test_exact_sampling_and_quantile_interpolation_all59_native_slices(self):
        new = self.exact(self.fixture(count=59, shape=(17, 23), slope=0.03125))
        self.assertEqual(new[0].shape, (48, 3, 28, 28))
        self.assertEqual(new[2]["native_shape"], [59, 17, 23])

    def test_exact_single_slice_zero_position(self):
        new = self.exact(self.fixture(count=1, shape=(13, 19)))
        self.assertEqual(new[1].tolist(), [0])

    def test_allocation_uses_scratch_memmap_and_bounded_copy_not_native_stack(self):
        import numpy as np

        directory = self.fixture(count=19, shape=(37, 41))
        events = []
        original_percentile, original_stack = np.percentile, np.stack

        def checked_percentile(values, *args, **kwargs):
            self.assertIsInstance(values, np.memmap)
            self.assertTrue(kwargs.get("overwrite_input"))
            return original_percentile(values, *args, **kwargs)

        def checked_stack(values, *args, **kwargs):
            # Native pixels are float32. Only output uint8 letterbox/triplet
            # arrays may be stacked in this alternative reader.
            self.assertTrue(all(value.dtype == np.uint8 for value in values))
            return original_stack(values, *args, **kwargs)

        with (
            patch.object(np, "percentile", side_effect=checked_percentile),
            patch.object(np, "stack", side_effect=checked_stack),
        ):
            bounded.read_bounded_feature_windows(
                directory,
                self.config["preprocess"],
                workspace=self.workspace,
                copy_chunk_bytes=1024,
                observer=lambda phase, detail: events.append((phase, detail)),
            )
        copy = [event[1]["copied_bytes"] for event in events if event[0] == "quantile_copy"]
        self.assertGreater(len(copy), 1)
        self.assertLessEqual(max(copy), 1024)
        self.assertEqual(sum(copy), 19 * 37 * 41 * 4)
        self.assertEqual(list(self.workspace.iterdir()), [])

    def test_same_invalid_spacing_duplicate_geometry_and_constant_rejections(self):
        from rsna_knee.feature_imaging import read_feature_windows

        for name, kwargs, expected in (
            ("missing-spacing", {"spacing": None}, "PixelSpacing"),
            ("duplicate", {"duplicate": True}, "duplicate slice positions"),
            ("constant", {"constant": True}, "Constant-intensity"),
        ):
            with self.subTest(name=name):
                directory = self.fixture(name, **kwargs)
                with self.assertRaisesRegex(ValueError, expected):
                    read_feature_windows(directory, self.config["preprocess"])
                with self.assertRaisesRegex(ValueError, expected):
                    bounded.read_bounded_feature_windows(directory, self.config["preprocess"], workspace=self.workspace)
                self.assertEqual(list(self.workspace.iterdir()), [])

    def test_failure_during_later_pixel_decode_cleans_maps(self):
        import pydicom

        directory = self.fixture(count=3)
        original = pydicom.dcmread
        pixel_calls = []

        def fails_later(path, *args, **kwargs):
            if not kwargs.get("stop_before_pixels"):
                pixel_calls.append(path)
                if len(pixel_calls) == 2:
                    raise RuntimeError("Synthetic second-slice decoder failure")
            return original(path, *args, **kwargs)

        with (
            patch.object(pydicom, "dcmread", side_effect=fails_later),
            self.assertRaisesRegex(RuntimeError, "Synthetic second-slice"),
        ):
            bounded.read_bounded_feature_windows(directory, self.config["preprocess"], workspace=self.workspace)
        self.assertEqual(list(self.workspace.iterdir()), [])

    def test_insufficient_disk_rejected_before_memmap_creation(self):
        import numpy as np

        directory = self.fixture()
        with (
            patch.object(bounded.shutil, "disk_usage", return_value=type("Usage", (), {"free": 1})()),
            patch.object(np, "memmap") as allocate,
        ):
            with self.assertRaisesRegex(OSError, "Insufficient temporary disk"):
                bounded.read_bounded_feature_windows(directory, self.config["preprocess"], workspace=self.workspace)
            allocate.assert_not_called()
        self.assertEqual(list(self.workspace.iterdir()), [])

    def test_same_extract_study_arrays_selection_and_actual_cuda_head(self):
        import numpy as np
        import torch

        from rsna_knee import feature_imaging
        from rsna_knee.feature_model import FrozenFeatureHead

        self.fixture("sagittal", count=8, slope=1.125)
        self.fixture("coronal", count=5, signed=True, mono1=True, slope=0.375)
        self.fixture("unselected", count=3, spacing=None)
        rows = [
            {SERIES_ID: name, "Anatomical_Plane": plane, "Fluid_Sensitive": fluid, "Fat_Suppression": fat}
            for name, plane, fluid, fat in (
                ("unselected", "sagittal", "0", "0"),
                ("sagittal", "sagittal", "1", "1"),
                ("coronal", "coronal", "1", "0"),
            )
        ]

        def artificial_encode(_model, windows, _config, _device):
            means = windows.mean(axis=(2, 3), dtype=np.float32) / 255
            return np.tile(means, (1, 128)).astype(np.float16)

        before = source_hashes()
        with patch.object(feature_imaging, "encode_windows", side_effect=artificial_encode):
            old, old_coverage = feature_imaging.extract_study(
                self.root, "test", "artificial-study", rows, self.config, None, "cpu"
            )
            with patch.object(feature_imaging, "read_feature_windows", bounded.make_bounded_reader(self.workspace)):
                new, new_coverage = feature_imaging.extract_study(
                    self.root, "test", "artificial-study", rows, self.config, None, "cpu"
                )
        for name in old:
            np.testing.assert_array_equal(new[name], old[name])
        self.assertEqual(old_coverage, new_coverage)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(108)
        head = FrozenFeatureHead(self.config).eval().to(device)
        with torch.inference_mode():
            a = head(**{name: torch.from_numpy(value).unsqueeze(0).to(device) for name, value in old.items()})
            b = head(**{name: torch.from_numpy(value).unsqueeze(0).to(device) for name, value in new.items()})
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(source_hashes(), before)
        self.assertEqual(list(self.workspace.iterdir()), [])


class AllocationPlanTests(unittest.TestCase):
    def test_large_native_shape_budget_requires_no_volume_allocation(self):
        plan = bounded.allocation_plan((1000, 1444, 1280))
        self.assertEqual(plan["native_file_bytes"], 1000 * 1444 * 1280 * 4)
        self.assertEqual(plan["quantile_file_bytes"], plan["native_file_bytes"])
        self.assertLess(plan["copy_chunk_bytes"], 9 * 1024**2)
        self.assertFalse(plan["anonymous_native_volume_allocated"])
        self.assertGreater(plan["minimum_free_disk_bytes"], 14 * 1024**3)


if __name__ == "__main__":
    unittest.main()
