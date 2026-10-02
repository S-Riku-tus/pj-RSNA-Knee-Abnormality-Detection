"""Optional artificial image tests: no competition data, no optimization or training."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rsna_knee.contracts import ID, SERIES_COLUMNS, TARGETS, load_config, preprocess_fingerprint, write_csv

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_IMAGING = (
    HAS_NUMPY and importlib.util.find_spec("pydicom") is not None and importlib.util.find_spec("PIL") is not None
)
HAS_TORCH = (
    HAS_NUMPY and importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchvision") is not None
)
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "baseline.json"


@unittest.skipUnless(HAS_NUMPY, "Optional numpy unavailable")
class GeometryTests(unittest.TestCase):
    def test_physical_sort_fallback_and_bad_geometry(self):
        from rsna_knee.imaging import geometric_order

        headers = [
            SimpleNamespace(
                ImageOrientationPatient=[1, 0, 0, 0, 1, 0], ImagePositionPatient=[0, 0, z], InstanceNumber=i
            )
            for i, z in enumerate([10, 0, 5])
        ]
        self.assertEqual(geometric_order(headers), ([1, 2, 0], "physical_position"))
        self.assertEqual(geometric_order([SimpleNamespace(InstanceNumber=i) for i in [3, 1, 2]])[0], [1, 2, 0])
        with self.assertRaises(ValueError):
            geometric_order([headers[0], SimpleNamespace(InstanceNumber=1)])
        with self.assertRaises(ValueError):
            geometric_order([headers[0], headers[0]])


def synthetic_dicom(path, z, index):
    import numpy as np
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.Rows, dataset.Columns = 32, 32
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated, dataset.BitsStored, dataset.HighBit = 16, 16, 15
    dataset.PixelRepresentation = 0
    dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    dataset.ImagePositionPatient = [0, 0, z]
    dataset.InstanceNumber = index
    dataset.PixelData = (np.arange(1024).reshape(32, 32) + z * 10).astype(np.uint16).tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_as(path, enforce_file_format=True)


@unittest.skipUnless(HAS_IMAGING, "Optional DICOM dependencies unavailable")
class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = load_config(CONFIG)
        self.config["preprocess"].update(image_size=32, max_series=2, windows_per_series=2)
        write_csv(self.root / "test.csv", [ID, "Report"], [{ID: "001", "Report": "not an inference input"}])
        write_csv(
            self.root / "test_series.csv",
            SERIES_COLUMNS,
            [
                {
                    ID: "001",
                    "SeriesInstanceUID": "series-a",
                    "Anatomical_Plane": "Sagittal",
                    "Fluid_Sensitive": "1",
                    "Fat_Suppression": "1",
                }
            ],
        )
        # Filename order and InstanceNumber intentionally disagree with physical order.
        for index, z in enumerate([10, 0, 5]):
            synthetic_dicom(self.root / "test_series" / "001" / "series-a" / f"{index}.dcm", z, index)

    def tearDown(self):
        self.temp.cleanup()

    def test_real_dicom_cache_resume_config_and_cpu_inference(self):
        import numpy as np

        from rsna_knee.imaging import build_cache, load_cached

        cache_dir = self.root / "cache"
        result = build_cache(self.root, "test", cache_dir, self.config)
        self.assertEqual(result["requested"], 1)
        images, mask = load_cached(cache_dir, "001", self.config)
        self.assertEqual(images.shape, (4, 3, 32, 32))
        self.assertEqual(mask.tolist(), [True, True, False, False])
        np.testing.assert_array_equal(images[0, 0], images[0, 1])  # clipped lower endpoint
        before = (cache_dir / "001.npz").read_bytes()
        build_cache(self.root, "test", cache_dir, self.config)
        self.assertEqual(before, (cache_dir / "001.npz").read_bytes())
        altered = json.loads(json.dumps(self.config))
        altered["preprocess"]["image_size"] = 64
        with self.assertRaises(ValueError):
            load_cached(cache_dir, "001", altered)
        if HAS_TORCH:
            import torch

            from rsna_knee.model import KneeMIL
            from rsna_knee.runtime import predict

            checkpoint = self.root / "test-model.pt"
            torch.save(
                {
                    "schema_version": 1,
                    "model": KneeMIL().state_dict(),
                    "config": self.config,
                    "targets": list(TARGETS),
                    "preprocess_fingerprint": preprocess_fingerprint(self.config),
                },
                checkpoint,
            )
            result = predict(self.root / "test.csv", cache_dir, checkpoint, self.root / "submission.csv", "cpu")
            self.assertTrue(result["valid"])

    def test_study_with_no_series_fails_without_zero_prediction(self):
        from rsna_knee.imaging import build_cache

        write_csv(self.root / "test_series.csv", SERIES_COLUMNS, [])
        with self.assertRaisesRegex(ValueError, "no valid windows"):
            build_cache(self.root, "test", self.root / "empty-cache", self.config)
        self.assertFalse((self.root / "empty-cache" / "001.npz").exists())


@unittest.skipUnless(HAS_TORCH, "Optional torch/torchvision unavailable")
class ModelTests(unittest.TestCase):
    def test_masked_windows_cannot_influence_logits(self):
        import torch

        from rsna_knee.model import KneeMIL

        torch.set_num_threads(1)
        model = KneeMIL().eval()
        images = torch.rand(1, 4, 3, 32, 32)
        mask = torch.tensor([[True, True, False, False]])
        changed = images.clone()
        changed[:, 2:] = 999
        with torch.inference_mode():
            torch.testing.assert_close(model(images, mask), model(changed, mask))
            with self.assertRaises(ValueError):
                model(images, torch.zeros_like(mask))

    def test_missing_label_mask_and_cpu_training_guard(self):
        import torch

        from rsna_knee.runtime import masked_bce, train

        logits = torch.tensor([[1.0, 1000.0]])
        truth = torch.tensor([[1.0, 0.0]])
        observed = torch.tensor([[True, False]])
        self.assertAlmostEqual(float(masked_bce(logits, truth, observed)), 0.3132617, places=6)
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "GPU terminal"):
                train("does-not-exist", "does-not-exist", "does-not-exist", {}, 0)


if __name__ == "__main__":
    unittest.main()
