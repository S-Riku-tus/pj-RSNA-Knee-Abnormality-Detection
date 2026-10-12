"""Artificial contract tests; no MRI decoding, weights, training or remote jobs."""

import copy
import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import public0033_profile_runtime as profile  # noqa: E402


def cohort():
    uids = [f"artificial{index:02d}" for index in range(64)]
    return {
        "uids": uids,
        "study_uid_sha256": profile.uid_sha256(uids),
        "gold_overlap_count": 0,
        "gold_excluded": True,
        "Report_or_targets_consumed": False,
        "accuracy_evaluated": False,
    }


def observations():
    import numpy as np

    return {
        "loads": [{}, {}],
        "workers": [
            {"device_index": index, "indices": list(range(index * 32, (index + 1) * 32))} for index in range(2)
        ],
        "forwards": [
            {"device": f"cuda:{index}", "batch_studies": 32, "finite": True} for index in range(2) for _ in range(10)
        ],
        "full12": {index: np.full(12, (index + 1) / 100, dtype=np.float32) for index in range(64)},
    }


class MetadataContracts(unittest.TestCase):
    def test_fixed_cohort_rejects_drift_gold_and_path_escape(self):
        value = cohort()
        self.assertEqual(profile.validate_cohort(value), value["uids"])
        for change in (
            {"gold_overlap_count": 1},
            {"Report_or_targets_consumed": True},
            {"accuracy_evaluated": True},
            {"study_uid_sha256": "0" * 64},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                profile.validate_cohort({**value, **change})
        bad = copy.deepcopy(value)
        bad["uids"][0] = "../escape"
        with self.assertRaises(ValueError):
            profile.validate_cohort(bad)

    def test_metadata_reader_does_not_open_train_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            official = Path(directory) / "competitions/rsna-knee-abnormality-detection"
            official.mkdir(parents=True)
            (official / "train_series").mkdir()
            # Deliberately invalid UTF8/CSV: accessing train.csv would fail.
            (official / "train.csv").write_bytes(b"\xff\xfe\x00")
            with (official / "train_series.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane"])
                writer.writerows([["a", "s1", "Sagittal"], ["b", "s2", "Axial"]])
            rows, tree = profile.read_series(directory, ["b"])
            self.assertEqual(rows, [{"StudyInstanceUID": "b", "SeriesInstanceUID": "s2", "Anatomical_Plane": "Axial"}])
            self.assertEqual(tree, official / "train_series")
            with self.assertRaises(ValueError):
                profile.read_series(directory, ["missing"])

    def test_bundle_requires_exact_manifest_hash_and_single_mount(self):
        with tempfile.TemporaryDirectory() as directory:
            flat = Path(directory) / "rsna-knee-public0033-meniscus-bag-v1"
            flat.mkdir()
            manifest = flat / "bundle_manifest.json"
            manifest.write_text("artificial", encoding="utf-8")
            with self.assertRaises(ValueError):
                profile.find_bundle(directory)
            with mock.patch.object(profile, "BUNDLE_MANIFEST_SHA256", profile.sha256(manifest)):
                self.assertEqual(profile.find_bundle(directory), flat)
                other = Path(directory) / "datasets/renta0426/rsna-knee-public0033-meniscus-bag-v1"
                other.mkdir(parents=True)
                (other / "bundle_manifest.json").write_bytes(manifest.read_bytes())
                with self.assertRaises(ValueError):
                    profile.find_bundle(directory)

    def test_failure_has_event_but_no_passed_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fresh"
            with self.assertRaises(ValueError):
                profile.run_profile(
                    cohort(), input_root=directory, output_directory=output, scratch=Path(directory) / "scratch"
                )
            self.assertFalse((output / "PUBLIC0033_PROFILE_SUMMARY.json").exists())
            events = [
                json.loads(line)
                for line in (output / "PUBLIC0033_PROFILE_EVENTS.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(events[-1]["stage"], "failed")
            with self.assertRaises(FileExistsError):
                profile.run_profile(
                    cohort(), input_root=directory, output_directory=output, scratch=Path(directory) / "scratch"
                )


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy unavailable")
class PredictionContracts(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("pydicom") and importlib.util.find_spec("torch"), "pixel dependencies unavailable"
    )
    def test_decoder_observer_keeps_native_pixels_and_partial_failure_policy(self):
        import numpy as np

        def pixels(index):
            if index == 3:
                raise RuntimeError("artificial unreadable frame")
            return np.arange(64, dtype=np.float32).reshape(8, 8) + index

        native = profile.adapter.prepare_slot(7, pixels, 0.5, plane="Sagittal", laterality="R")
        original = profile.adapter.prepare_slot
        with profile.decoder_observer() as decoder:
            observed = profile.adapter.prepare_slot(7, pixels, 0.5, plane="Sagittal", laterality="R")
        np.testing.assert_array_equal(native, observed)
        self.assertIs(profile.adapter.prepare_slot, original)
        self.assertEqual(decoder["counts"]["slots_valid"], 1)
        self.assertEqual(decoder["counts"]["selected_frame_decode_errors"], 1)
        self.assertGreater(decoder["counts"]["selected_frame_success"], 0)

    def test_observed_contract_requires_real_maximum_batch_on_each_gpu(self):
        value = observations()
        profile.validate_observations(value, cohort()["uids"])
        for key in ("loads", "workers", "forwards", "full12"):
            bad = copy.deepcopy(value)
            if key == "full12":
                bad[key].pop(63)
            else:
                bad[key].pop()
            with self.subTest(key=key), self.assertRaises(ValueError):
                profile.validate_observations(bad, cohort()["uids"])
        bad = copy.deepcopy(value)
        bad["forwards"][0]["batch_studies"] = 16
        with self.assertRaises(ValueError):
            profile.validate_observations(bad, cohort()["uids"])

    def test_diagnostic_full12_csv_preserves_values_and_uid_order(self):
        import numpy as np

        uids, value = cohort()["uids"], observations()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostic_public0033_full12.csv"
            result = profile.write_full12(path, uids, value)
            self.assertEqual(result["rows"], 64)
            self.assertEqual(result["columns"], ["StudyInstanceUID", *profile.adapter.LABELS])
            rows = profile.adapter._read_csv(path, profile.adapter.LABELS)
            self.assertEqual([row["StudyInstanceUID"] for row in rows], uids)
            np.testing.assert_array_equal(
                np.asarray([[float(row[label]) for label in profile.adapter.LABELS] for row in rows]),
                np.stack(list(value["full12"].values())),
            )
            self.assertFalse((Path(directory) / "submission.csv").exists())
            with self.assertRaises(FileExistsError):
                profile.write_full12(path, uids, value)
            value["full12"][0][0] = np.nan
            with self.assertRaises(ValueError):
                profile.write_full12(Path(directory) / "other.csv", uids, value)


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch unavailable")
class StockObserverContracts(unittest.TestCase):
    def test_available_cuda_allocator_and_forward_observation_on_artificial_model(self):
        import numpy as np
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")

        class Artificial(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(1, 12)

            def forward(self, images, mask):
                return self.layer(torch.zeros((len(images), 1), device=images.device))

        runtime = types.SimpleNamespace(EXPECTED_STATE_TENSORS=2)
        runtime._load_fullfit_model = lambda root, device: Artificial().to(device).eval()

        def worker(**kwargs):
            device = torch.device("cuda:0")
            # The original stock worker initializes its device before loading.
            torch.cuda.set_device(device)
            model = runtime._load_fullfit_model(Path("/artificial"), device)
            images = torch.empty((32, 6, 3, 336, 336), dtype=torch.uint8, device=device)
            mask = torch.ones((32, 6), dtype=torch.float32, device=device)
            with torch.inference_mode():
                values = model(images, mask).sigmoid().cpu().numpy()
            return np.arange(32), values, {"strict_load": True, "artificial": True}

        runtime._worker_predict = worker
        with profile.stock_observer(runtime) as observed:
            runtime._worker_predict(device_index=0)
        self.assertEqual(observed["forwards"][0]["images_dtype"], "torch.uint8")
        self.assertEqual(observed["forwards"][0]["logits_dtype"], "torch.float32")
        self.assertTrue(observed["forwards"][0]["finite"])
        self.assertGreater(observed["workers"][0]["torch_peak_allocated_bytes"], 0)
        self.assertGreaterEqual(
            observed["workers"][0]["torch_peak_reserved_bytes"], observed["workers"][0]["torch_peak_allocated_bytes"]
        )

    def test_stock_return_identity_and_restoration_with_artificial_model(self):
        import numpy as np
        import torch

        class Artificial(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(1, 12)

            def forward(self, images, mask):
                return self.layer.bias.unsqueeze(0).expand(len(images), -1)

        model = Artificial().eval()
        returned = {}
        runtime = types.SimpleNamespace(EXPECTED_STATE_TENSORS=2)

        def load(root, device):
            return model

        def worker(**kwargs):
            loaded = runtime._load_fullfit_model(Path("/artificial"), torch.device("cuda:0"))
            images = torch.empty((32, 6, 3, 336, 336), dtype=torch.uint8)
            masks = torch.ones((32, 6), dtype=torch.float32)
            with torch.inference_mode():
                values = loaded(images, masks).sigmoid().numpy().copy()
            result = (np.arange(32), values, {"strict_load": True})
            returned["result"] = result
            return result

        runtime._load_fullfit_model, runtime._worker_predict = load, worker
        patches = {
            "reset_peak_memory_stats": mock.DEFAULT,
            "synchronize": mock.DEFAULT,
            "max_memory_allocated": mock.DEFAULT,
            "max_memory_reserved": mock.DEFAULT,
        }
        with mock.patch.multiple(torch.cuda, **patches) as cuda, profile.stock_observer(runtime) as observed:
            cuda["max_memory_allocated"].return_value = 123
            cuda["max_memory_reserved"].return_value = 456
            result = runtime._worker_predict(device_index=0)
            self.assertIs(result, returned["result"])
            self.assertEqual(observed["workers"][0]["torch_peak_allocated_bytes"], 123)
            self.assertEqual(observed["forwards"][0]["batch_studies"], 32)
            self.assertEqual(len(observed["full12"]), 32)
            self.assertEqual(len(observed["loads"]), 1)
        self.assertIs(runtime._worker_predict, worker)
        self.assertIs(runtime._load_fullfit_model, load)
        self.assertEqual(len(model._forward_hooks), 0)

    def test_full_profile_success_contract_with_synthetic_memmap_and_stock_stubs(self):
        import numpy as np
        import torch

        class Artificial(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(1, 12)

            def forward(self, images, mask):
                return self.layer.bias.unsqueeze(0).expand(len(images), -1)

        class ArtificialResources:
            def __init__(self, scratch):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def result(self):
                return {"artificial": True}

        @contextmanager
        def cache(uids, build, *, scratch):
            for uid in uids:
                build(uid)
            path = Path(scratch) / "artificial.npy"
            array = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(64, 6, 12, 336, 336))
            try:
                yield array, np.ones((64, 6), dtype=np.float32), uids
            finally:
                array._mmap.close()
                path.unlink()

        manifest = {"checkpoint": {"sha256": "artificial"}, "artifacts": {}, "runtime": {"sha256": "artificial"}}
        runtime = types.SimpleNamespace(
            TARGETS=profile.adapter.LABELS, BATCH_STUDIES=32, WINDOW_STARTS=tuple(range(10)), EXPECTED_STATE_TENSORS=2
        )
        runtime.verify_bundle = lambda root: manifest
        runtime._require_t4x2 = lambda: ["artificial T4", "artificial T4"]
        runtime._validate_cache_inputs = lambda cache, mask, uids: None
        runtime._load_fullfit_model = lambda root, device: Artificial().eval()

        def worker(**kwargs):
            model = runtime._load_fullfit_model(Path("/artificial"), torch.device(f"cuda:{kwargs['device_index']}"))
            images = torch.empty((32, 6, 3, 336, 336), dtype=torch.uint8)
            masks = torch.ones((32, 6), dtype=torch.float32)
            with torch.inference_mode():
                values = None
                for _ in range(10):
                    values = model(images, masks).sigmoid().numpy().copy()
            return kwargs["indices"], values, {"strict_load": True}

        runtime._worker_predict = worker

        def infer(cache, mask, uids, paths):
            all_values = np.zeros((64, 12), dtype=np.float32)
            for index in range(2):
                indices, values, _ = runtime._worker_predict(
                    device_index=index, indices=np.arange(index * 32, (index + 1) * 32)
                )
                all_values[indices] = values
            with Path(paths["bag_raw_csv"]).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(["StudyInstanceUID", *profile.adapter.MENISCUS])
                for uid, values in zip(uids, all_values, strict=True):
                    writer.writerow([uid, float(values[2]), float(values[3])])
            receipt = {
                "status": "passed",
                "fallback": 0,
                "neutralized_predictions": 0,
                "study_uid_sha256": profile.uid_sha256(uids),
                "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                "checkpoint": {"strict_load": True},
            }
            Path(paths["receipt_json"]).write_text(json.dumps(receipt), encoding="utf-8")
            return {"status": "passed"}

        runtime.run_cached_inference = mock.Mock(side_effect=infer)
        pixels = np.zeros((6, 12, 336, 336), dtype=np.uint8)
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.multiple(
                torch.cuda,
                reset_peak_memory_stats=mock.DEFAULT,
                synchronize=mock.DEFAULT,
                max_memory_allocated=mock.DEFAULT,
                max_memory_reserved=mock.DEFAULT,
            ) as cuda,
        ):
            cuda["max_memory_allocated"].return_value = 123
            cuda["max_memory_reserved"].return_value = 456
            root = Path(directory)
            bundle = root / "bundle"
            bundle.mkdir()
            scratch = root / "scratch"
            value = {**cohort(), "weak_manifest_sha256": "artificial", "gold_manifest_sha256": "artificial"}
            with (
                mock.patch.object(profile, "find_bundle", return_value=bundle),
                mock.patch.object(profile.adapter, "load_pinned_runtime", return_value=runtime),
                mock.patch.object(profile, "read_series", return_value=([], root)),
                mock.patch.object(
                    profile.adapter,
                    "make_dicom_study_builder",
                    return_value=lambda uid: (pixels, np.ones(6, dtype=np.float32)),
                ),
                mock.patch.object(profile.adapter, "memmap_cache", side_effect=cache),
                mock.patch.object(profile, "Resources", ArtificialResources),
                mock.patch.object(profile, "environment", return_value={"artificial": True}),
                mock.patch.object(profile, "sha256", return_value="artificial"),
                mock.patch("builtins.print"),
            ):
                summary = profile.run_profile(value, input_root=root, output_directory=root / "output", scratch=scratch)
            self.assertEqual(summary["status"], "passed")
            self.assertEqual(summary["stock_call_count"], 1)
            runtime.run_cached_inference.assert_called_once()
            self.assertEqual(len(summary["actual_forwards"]), 20)
            self.assertEqual(summary["full12_contract"]["rows"], 64)
            self.assertTrue((root / "output/PUBLIC0033_PROFILE_SUMMARY.json").is_file())
            self.assertFalse((root / "output/submission.csv").exists())


class BuilderContracts(unittest.TestCase):
    def test_code_payload_allowlist_no_large_or_label_assets(self):
        import io
        import zipfile

        import build_public0033_profile as builder

        body, members = builder.payload(cohort())
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            self.assertEqual(
                set(archive.namelist()),
                {"public0033_profile_runtime.py", "public0033_streaming_adapter.py", "public0033_profile_cohort.json"},
            )
            for name in archive.namelist():
                value = archive.read(name)
                self.assertEqual(hashlib.sha256(value).hexdigest(), members[name]["sha256"])
                self.assertEqual(len(value), members[name]["bytes"])
                self.assertLess(len(value), 100000)
                if name.endswith(".py"):
                    compile(value, name, "exec")


if __name__ == "__main__":
    unittest.main()
