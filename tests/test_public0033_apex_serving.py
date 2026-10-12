"""Serving contracts with artificial CSV/headers/model inputs; no real MRI/weights."""

import ast
import copy
import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import public0033_apex_serving as serving  # noqa: E402


def baseline(path, uids=("z", "a", "m")):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["StudyInstanceUID", *serving.adapter.LABELS])
        for index, uid in enumerate(uids):
            values = ["0.1250000"] * 12
            values[2] = "0.8" if index == 0 else "0.1"
            values[3] = "0.20" if index == 0 else "0.7"
            writer.writerow([uid, *values])


def fake_observations(uids):
    import numpy as np

    n = len(uids)
    midpoint = (n + 1) // 2
    values = {index: np.full(12, (index + 1) / (n + 1), dtype=np.float32) for index in range(n)}
    return {
        "study_masks": {uid: [1.0] * 6 for uid in uids},
        "stock": [
            {
                "loads": [{"parameters_fp32": True}, {"parameters_fp32": True}],
                "workers": [{"device_index": index} for index in range(2)],
                "forwards": [
                    {
                        "device": f"cuda:{index}",
                        "batch_studies": min(32, size - offset),
                        "finite": True,
                        "logits_dtype": "torch.float32",
                    }
                    for index, size in enumerate((midpoint, n - midpoint))
                    for offset in range(0, size, 32)
                    for _ in range(10)
                ],
                "full12": values,
            }
        ],
    }


def stock_artifacts(output, uids, observed):
    import numpy as np

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    ordered = sorted(uids)
    raw = output / "public0033_bag_raw.csv"
    receipt_path = output / "public0033_cached_inference_receipt.json"
    with raw.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["StudyInstanceUID", *serving.adapter.MENISCUS])
        for index, uid in enumerate(ordered):
            writer.writerow(
                [uid, float(observed["stock"][0]["full12"][index][2]), float(observed["stock"][0]["full12"][index][3])]
            )
    midpoint = (len(uids) + 1) // 2
    mask = np.ones((len(uids), 6), dtype=np.float32)
    receipt = {
        "status": "passed",
        "fallback": 0,
        "neutralized_predictions": 0,
        "study_count": len(uids),
        "study_uid_sha256": serving.uid_sha256(ordered),
        "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        "runtime_sha256": serving.adapter.PINNED_RUNTIME_SHA256,
        "bundle_manifest_sha256": serving.MANIFEST_SHA256,
        "checkpoint": {
            "sha256": serving.CHECKPOINT_SHA256,
            "bytes": 347965736,
            "state_tensor_count": 233,
            "strict_load": True,
        },
        "precision": "fp32",
        "autocast": False,
        "batch_studies": 32,
        "window_starts": list(range(10)),
        "cache": {"shape": [len(uids), 6, 12, 336, 336], "dtype": "uint8"},
        "devices": [{"device_name": "Tesla T4"}, {"device_name": "Tesla T4"}],
        "raw_output": {"sha256": serving.sha256(raw)},
        "partitions": [
            {
                "device_index": index,
                "index_start": start,
                "index_stop_exclusive": stop,
                "study_count": stop - start,
                "strict_load": True,
            }
            for index, start, stop in ((0, 0, midpoint), (1, midpoint, len(uids)))
        ],
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return {"status": "passed", "study_count": len(uids), "bag_raw_csv": str(raw), "receipt_json": str(receipt_path)}


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy unavailable")
class ServingContracts(unittest.TestCase):
    @contextmanager
    def setup(self, directory, uids=("z", "a", "m")):
        root = Path(directory)
        path = root / "submission.csv"
        baseline(path, uids)
        observed = fake_observations(uids)

        @contextmanager
        def observe():
            yield observed

        def branch(**kwargs):
            return stock_artifacts(kwargs["output_directory"], uids, observed)

        with (
            mock.patch.object(serving, "read_live_metadata", return_value=([], root)),
            mock.patch.object(serving, "find_bundle", return_value=root),
            mock.patch.object(serving, "release_inactive_cache", return_value=([], [])),
            mock.patch.object(serving, "gpu_memory", return_value=[]),
            mock.patch.object(serving, "branch_observer", side_effect=observe),
            mock.patch.object(serving.adapter, "run_separate_branch", side_effect=branch) as run,
            mock.patch("builtins.print"),
        ):
            yield root, path, observed, run

    def test_success_preserves10strings_UID_order_and_publishes_fixed_fusion(self):
        with tempfile.TemporaryDirectory() as directory, self.setup(directory) as (root, path, observed, run):
            before = path.read_bytes()
            result = serving.append_specialist(
                submission=path, input_root=root, output_directory=root / "run", scratch=root / "scratch"
            )
            self.assertEqual(result["status"], "passed")
            self.assertTrue(result["specialist_used"])
            self.assertTrue(result["actual_full12_contract"]["raw2_exact_projection"])
            self.assertEqual(result["actual_full12_contract"]["actual_forward_count"], 20)
            self.assertEqual((root / "run/apex_incumbent.csv").read_bytes(), before)
            rows = serving.adapter._read_csv(path, serving.adapter.LABELS)
            self.assertEqual([row["StudyInstanceUID"] for row in rows], ["z", "a", "m"])
            self.assertEqual([float(row["Medial Meniscus"]) for row in rows], [1.0, 1 / 3, 2 / 3])
            for label in serving.adapter.LABELS:
                if label not in serving.adapter.MENISCUS:
                    self.assertTrue(all(row[label] == "0.1250000" for row in rows))
            self.assertEqual(path.read_bytes(), (root / "run/apex_meniscus10_candidate.csv").read_bytes())
            run.assert_called_once()

    def test_variable65_UIDs_prove_repeated_batch32_and_final_batch1(self):
        uids = [f"artificial{index:03d}" for index in range(65)][::-1]
        with tempfile.TemporaryDirectory() as directory, self.setup(directory, uids) as (root, path, observed, run):
            result = serving.append_specialist(
                submission=path, input_root=root, output_directory=root / "run", scratch=root / "scratch"
            )
            self.assertEqual(result["actual_full12_contract"]["actual_forward_count"], 30)
            self.assertEqual(result["candidate_contract"]["rows"], 65)
            self.assertEqual(
                [row["StudyInstanceUID"] for row in serving.adapter._read_csv(path, serving.adapter.LABELS)], uids
            )
            bad = copy.deepcopy(observed)
            bad["stock"][0]["forwards"][10]["batch_studies"] = 32
            with self.assertRaises(ValueError):
                serving.verify_actual_full12(bad, uids, root / "run/branch/public0033_bag_raw.csv")

    def test_single_study_skips_stock_and_keeps_bytes_without_Inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "submission.csv"
            baseline(path, ("only",))
            before = path.read_bytes()
            with (
                mock.patch.object(serving.adapter, "run_separate_branch") as run,
                mock.patch.object(serving, "read_live_metadata") as metadata,
                mock.patch("builtins.print"),
            ):
                result = serving.append_specialist(
                    submission=path, input_root=root, output_directory=root / "run", scratch=root / "scratch"
                )
            self.assertEqual(result["status"], "skipped_incumbent_retained")
            self.assertEqual(result["stock_call_count"], 0)
            self.assertEqual(path.read_bytes(), before)
            run.assert_not_called()
            metadata.assert_not_called()

    def test_zero_rows_is_invalid_submission_and_no_fake_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "submission.csv"
            baseline(path, ())
            with self.assertRaises(ValueError):
                serving.append_specialist(submission=path, output_directory=root / "run")
            self.assertFalse((root / "run").exists())

    def test_allmissing_typed_input_retains_incumbent_and_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory, self.setup(directory) as (root, path, observed, run):
            before = path.read_bytes()
            run.side_effect = serving.ExpertSlotsUnavailable("z")
            result = serving.append_specialist(
                submission=path, input_root=root, output_directory=root / "run", scratch=root / "scratch"
            )
            self.assertEqual(result["status"], "fallback_incumbent_retained")
            self.assertFalse(result["specialist_used"])
            self.assertEqual(result["reason"], "ExpertSlotsUnavailable")
            self.assertIn("exception_traceback", result)
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse((root / "run/apex_meniscus10_candidate.csv").exists())

    def test_series_metadata_shortage_retains_incumbent_with_input_stage(self):
        with tempfile.TemporaryDirectory() as directory, self.setup(directory) as (root, path, observed, run):
            before = path.read_bytes()
            with mock.patch.object(
                serving,
                "read_live_metadata",
                side_effect=serving.ExpertInputContractError(
                    "live test series metadata does not cover all incumbent studies"
                ),
            ):
                result = serving.append_specialist(submission=path, input_root=root, output_directory=root / "run")
            self.assertEqual(result["status"], "fallback_incumbent_retained")
            self.assertEqual(result["stage"], "live_test_input_contract")
            self.assertFalse(result["specialist_used"])
            self.assertEqual(path.read_bytes(), before)
            run.assert_not_called()

    def test_checkpoint_drift_is_raised_and_incumbent_preserved(self):
        with tempfile.TemporaryDirectory() as directory, self.setup(directory) as (root, path, observed, run):
            before = path.read_bytes()
            original = run.side_effect

            def corrupt(**kwargs):
                result = original(**kwargs)
                receipt = Path(result["receipt_json"])
                data = json.loads(receipt.read_text())
                data["checkpoint"]["sha256"] = "0" * 64
                receipt.write_text(json.dumps(data), encoding="utf-8")
                return result

            run.side_effect = corrupt
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                serving.append_specialist(
                    submission=path, input_root=root, output_directory=root / "run", scratch=root / "scratch"
                )
            result = json.loads((root / "run/P008_MENISCUS_SUMMARY.json").read_text())
            self.assertEqual(result["status"], "failed_incumbent_preserved")
            self.assertEqual(path.read_bytes(), before)

    def test_CUDA_and_programming_errors_are_raised_not_fallback(self):
        for error in (RuntimeError("artificial CUDA out of memory"), TypeError("artificial programming error")):
            with (
                self.subTest(error=error),
                tempfile.TemporaryDirectory() as directory,
                self.setup(directory) as (root, path, observed, run),
            ):
                before = path.read_bytes()
                run.side_effect = error
                with self.assertRaises(type(error)):
                    serving.append_specialist(
                        submission=path, input_root=root, output_directory=root / "run", scratch=root / "scratch"
                    )
                result = json.loads((root / "run/P008_MENISCUS_SUMMARY.json").read_text())
                self.assertEqual(result["status"], "failed_incumbent_preserved")
                self.assertEqual(path.read_bytes(), before)

    def test_stale_output_is_refused_and_no_old_raw_prediction_used(self):
        with tempfile.TemporaryDirectory() as directory, self.setup(directory) as (root, path, observed, run):
            (root / "run").mkdir()
            (root / "run/public0033_bag_raw.csv").write_text("stale", encoding="utf-8")
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                serving.append_specialist(submission=path, input_root=root, output_directory=root / "run")
            self.assertEqual(path.read_bytes(), before)
            run.assert_not_called()


class LiveInputContracts(unittest.TestCase):
    def test_UID_only_metadata_without_Report_and_short_SeriesCSV_classified(self):
        with tempfile.TemporaryDirectory() as directory:
            official = Path(directory) / "competitions/rsna-knee-abnormality-detection"
            official.mkdir(parents=True)
            (official / "test_series").mkdir()
            (official / "train.csv").write_bytes(b"\xff\xfeinvalid")
            (official / "test.csv").write_text("StudyInstanceUID\nz\na\n", encoding="utf-8")
            series = official / "test_series.csv"
            series.write_text(
                "StudyInstanceUID,SeriesInstanceUID,Anatomical_Plane\nz,s1,Sagittal\na,s2,Axial\n", encoding="utf-8"
            )
            rows, _ = serving.read_live_metadata(directory, ["a", "z"])
            self.assertEqual(len(rows), 2)
            series.write_text("StudyInstanceUID,SeriesInstanceUID,Anatomical_Plane\nz,s1,Sagittal\n", encoding="utf-8")
            with self.assertRaisesRegex(serving.ExpertInputContractError, "cover"):
                serving.read_live_metadata(directory, ["a", "z"])
            series.write_text("StudyInstanceUID,SeriesInstanceUID\nz,s1\n", encoding="utf-8")
            with self.assertRaisesRegex(serving.ExpertInputContractError, "columns"):
                serving.read_live_metadata(directory, ["a", "z"])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy unavailable")
    def test_original_allslots_failure_translated_and_global_factory_restored(self):
        original = serving.adapter.make_dicom_study_builder

        def factory(*args, **kwargs):
            unknown = serving.adapter.classify_header(
                {"plane": "Sagittal", "SeriesDescription": "UnknownLocalizer", "n_slices": 0}
            )
            chosen = serving.adapter.choose_recovered_slots([unknown])
            # Original recovered rules can assign an unknown nonfluid Sagittal
            # series to SAG_T1. An empty series still yields no valid input slot.
            self.assertEqual(set(chosen), {"SAG_T1"})

            def unread(index):
                raise AssertionError("empty series must not decode any pixel")

            return lambda uid: serving.adapter.prepare_study(
                {"SAG_T1": {"file_count": 0, "read_pixels": unread, "pixel_spacing": float("nan")}}
            )

        with mock.patch.object(serving.adapter, "make_dicom_study_builder", side_effect=factory) as patched:
            with serving.branch_observer():
                build = serving.adapter.make_dicom_study_builder([], Path("/artificial"))
                with self.assertRaises(serving.ExpertSlotsUnavailable):
                    build("artificialUID")
            self.assertIs(serving.adapter.make_dicom_study_builder, patched)
        self.assertIs(serving.adapter.make_dicom_study_builder, original)


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"), "model dependencies unavailable"
)
class MissingMaskArchitectureContracts(unittest.TestCase):
    def test_inactive_cache_release_preserves_live_CUDA_tensor(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        held = torch.ones((1024, 1024), dtype=torch.float32, device="cuda:0")
        pointer = held.data_ptr()
        before, after = serving.release_inactive_cache()
        self.assertTrue(torch.all(held == 1).item())
        self.assertEqual(held.data_ptr(), pointer)
        self.assertGreaterEqual(before[0]["allocated_bytes"], held.nelement() * held.element_size())
        self.assertGreaterEqual(after[0]["allocated_bytes"], held.nelement() * held.element_size())

    def test_unmodified_head_all0mask_finite_but_stock_cache_rejects_it(self):
        import numpy as np
        import torch

        source = (
            ROOT
            / "artifacts/research/20261008-post950-public-research-v1/selected-provenance-files/public0033_runtime.py"
        ).read_text(encoding="utf-8")
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), serving.adapter.PINNED_RUNTIME_SHA256)
        tree = ast.parse(source)
        selected = [
            node
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in ("SlotHead", "_validate_cache_inputs")
        ]
        namespace = {
            "torch": torch,
            "nn": torch.nn,
            "np": np,
            "N_SLOTS": 6,
            "SLICES_PER_SLOT": 12,
            "IMAGE_SIZE": 336,
            "TARGETS": serving.adapter.LABELS,
            "Public0033InferenceError": RuntimeError,
        }
        selected.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), "audited-stock-functions", "exec"
            ),
            namespace,
        )
        head = namespace["SlotHead"](1536).eval()
        with torch.inference_mode():
            output = head(torch.zeros((2, 6, 1536)), torch.zeros((2, 6)))
        self.assertEqual(tuple(output.shape), (2, 12))
        self.assertTrue(torch.isfinite(output).all().item())
        cache = np.zeros((2, 6, 12, 336, 336), dtype=np.uint8)
        with self.assertRaisesRegex(RuntimeError, "whole-study mask failure"):
            namespace["_validate_cache_inputs"](cache, np.zeros((2, 6), dtype=np.float32), ["a", "b"])
        # Finite outputs from an allmasked head are not evidence of useful MRI
        # predictions and are never substituted into the runtime's cache route.


class BuilderContracts(unittest.TestCase):
    def test_only_append_changed_and_all7_original_code_cells_exact(self):
        import io
        import zipfile

        import build_p008_apex_meniscus10 as builder

        baseline = json.loads(builder.BASELINE.read_text(encoding="utf-8"))
        notebook, body, members = builder.prepare_notebook(baseline)
        self.assertEqual(len(notebook["cells"]), 11)
        self.assertEqual(notebook["metadata"], baseline["metadata"])
        self.assertEqual(
            [builder.source(cell) for cell in notebook["cells"][:-1] if cell["cell_type"] == "code"],
            [builder.source(cell) for cell in baseline["cells"] if cell["cell_type"] == "code"],
        )
        self.assertEqual(sum(cell["cell_type"] == "code" for cell in notebook["cells"]), 8)
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            self.assertEqual(
                set(archive.namelist()),
                {"public0033_streaming_adapter.py", "public0033_profile_runtime.py", "public0033_apex_serving.py"},
            )
            for name in archive.namelist():
                value = archive.read(name)
                self.assertEqual(hashlib.sha256(value).hexdigest(), members[name]["sha256"])
                compile(value, name, "exec")
        self.assertIn("scratch='/kaggle/temp'", builder.source(notebook["cells"][-1]))


if __name__ == "__main__":
    unittest.main()
