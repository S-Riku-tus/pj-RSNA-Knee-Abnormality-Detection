"""Synthetic data only: same head scores, decoder failures, geometry diagnostics and coverage."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsna_knee.contracts import ID, SERIES_COLUMNS, SERIES_ID, TARGETS, read_csv, sha256, source_hashes, write_csv
from rsna_knee.feature_contracts import feature_contract, head_implementation_hash, json_hash, load_feature_config

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("frozen_streaming_runtime", ROOT / "scripts/frozen_streaming_runtime.py")
streaming = importlib.util.module_from_spec(spec)
spec.loader.exec_module(streaming)
HAS_GPU_MODULES = all(importlib.util.find_spec(name) for name in ("torch", "numpy", "pydicom", "PIL"))


@unittest.skipUnless(HAS_GPU_MODULES, "Optional GPU/imaging packages absent")
class StreamingTests(unittest.TestCase):
    def setUp(self):
        import torch

        from rsna_knee.feature_model import FrozenFeatureHead

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = load_feature_config(ROOT / "configs/experiments/f002-dinov2-attention.json")
        self.config["preprocess"].update(image_size=28, max_series=2, slices_per_series=3)
        self.config["head"].update(hidden_dim=8, dropout=0)
        self.provenance = {
            "source_url": "synthetic",
            "source_revision": "synthetic",
            "weights_url": "synthetic",
            "weights_sha256": "a" * 64,
            "source_tree_sha256": "b" * 64,
            "license": "synthetic",
            "reviewed_at": "synthetic",
            "exposure_audit": "synthetic",
            "pretraining": "generic_dinov2_lvd142m",
            "competition_finetuned": False,
            "gold_used_for_tuning": False,
        }
        contract = feature_contract(self.config, self.provenance)
        torch.manual_seed(7)
        self.head = FrozenFeatureHead(self.config).eval()
        self.checkpoint = self.root / "best.pt"
        torch.save(
            {
                "schema_version": "frozen_feature_head_v1",
                "targets": list(TARGETS),
                "config": self.config,
                "model": self.head.state_dict(),
                "fold": 0,
                "epoch": 11,
                "selection_data": "weak_validation",
                "feature_contract": contract,
                "feature_fingerprint": json_hash(contract),
                "head_implementation_sha256": head_implementation_hash(),
            },
            self.checkpoint,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ids = [f"1.2.3.{number}" for number in range(1, 18)]
        for split in ("train", "test"):
            write_csv(self.root / f"{split}.csv", (ID,), [{ID: key} for key in self.ids])
            write_csv(
                self.root / f"{split}_series.csv",
                SERIES_COLUMNS,
                [
                    {
                        ID: key,
                        SERIES_ID: "1.2.4",
                        "Anatomical_Plane": "sagittal",
                        "Fluid_Sensitive": "1",
                        "Fat_Suppression": "1",
                    }
                    for key in self.ids
                ],
            )
        self.weak = self.root / "weak.csv"
        write_csv(self.weak, (ID,), [{ID: key} for key in self.ids])

    def arrays(self, key):
        import numpy as np

        rng = np.random.default_rng(self.ids.index(key))
        return {
            "features": rng.normal(size=(2, 3, 384)).astype(np.float16),
            "mask": np.array([[True, True, True], [False, False, False]]),
            "positions": np.array([[0, 0.5, 1], [0, 0, 0]], dtype=np.float32),
            "planes": np.array([0, 3], dtype=np.int64),
            "flags": np.ones((2, 2), np.float32),
        }

    def run_candidate(self, **kwargs):
        return streaming.run_streaming(
            self.root,
            self.checkpoint,
            self.provenance,
            "synthetic",
            "synthetic",
            self.root / "diagnostics",
            expected_checkpoint_sha256=sha256(self.checkpoint),
            expected_sources=source_hashes(),
            device=self.device,
            **kwargs,
        )

    def test_streaming_gpu_scores_match_batched_head_and_report_free_csv(self):
        import numpy as np
        import torch

        from rsna_knee import feature_imaging

        arrays = [self.arrays(key) for key in self.ids]
        with torch.inference_mode():
            baseline = (
                self.head.to(self.device)(
                    **{
                        name: torch.from_numpy(np.stack([a[name] for a in arrays])).to(self.device)
                        for name in arrays[0]
                    }
                )
                .float()
                .cpu()
                .sigmoid()
                .numpy()
            )
        with (
            patch.object(streaming, "decoder_inventory", return_value={}),
            patch.object(feature_imaging, "load_frozen_encoder", return_value=None),
            patch.object(
                feature_imaging, "extract_study", side_effect=lambda root, split, key, *args: (self.arrays(key), [])
            ),
        ):
            result = self.run_candidate(require_decoders=False)
        rows = read_csv(self.root / "submission.csv")[1]
        actual = np.array([[float(row[t]) for t in TARGETS] for row in rows])
        np.testing.assert_allclose(actual, baseline, rtol=0, atol=2e-6)
        self.assertEqual([row[ID] for row in rows], self.ids)
        self.assertEqual(
            (result["status"], result["processed"], result["intermediate_feature_files"]), ("passed", 17, 0)
        )
        self.assertFalse(list(self.root.rglob("*.npz")))
        self.assertFalse(result["report_required"])
        self.assertEqual((result["fallback_predictions"], result["skipped_series"]), (0, 0))

    def test_missing_decoder_fails_before_encoder_and_csv_with_summary(self):
        from rsna_knee import feature_imaging

        with (
            patch.object(streaming, "decoder_inventory", return_value={"lossless": {"available": False}}),
            patch.object(feature_imaging, "load_frozen_encoder") as load,
        ):
            with self.assertRaisesRegex(RuntimeError, "decoder unavailable"):
                self.run_candidate()
            load.assert_not_called()
        summary = json.loads((self.root / "diagnostics/F002_STREAMING_SUMMARY.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["phase"], "decoder_preflight")
        self.assertEqual(summary["processed"], 0)
        self.assertFalse((self.root / "submission.csv").exists())

    def test_missing_geometry_retains_uid_header_and_no_partial_submission(self):
        from test_imaging_runtime import synthetic_dicom

        from rsna_knee import feature_imaging

        original = feature_imaging.read_feature_windows
        synthetic_dicom(self.root / "test_series" / self.ids[0] / "1.2.4/0.dcm", 0, 0)
        with (
            patch.object(streaming, "decoder_inventory", return_value={}),
            patch.object(feature_imaging, "load_frozen_encoder", return_value=None),
        ):
            with self.assertRaisesRegex(ValueError, "PixelSpacing"):
                self.run_candidate(require_decoders=False)
        self.assertIs(feature_imaging.read_feature_windows, original)
        summary = json.loads((self.root / "diagnostics/F002_STREAMING_SUMMARY.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["current_context"]["study_uid"], self.ids[0])
        self.assertEqual(summary["current_context"]["series_uid"], "1.2.4")
        self.assertEqual(summary["current_context"]["header"]["transfer_syntax"], "1.2.840.10008.1.2.1")
        self.assertEqual(summary["phase"], "decode_and_preprocess_series")
        self.assertFalse((self.root / "submission.csv").exists())

    def test_profile_uses_manifest_subset_and_never_creates_submission(self):
        from rsna_knee import feature_imaging

        write_csv(self.weak, (ID,), [{ID: key} for key in self.ids[3:]])
        with (
            patch.object(streaming, "decoder_inventory", return_value={}),
            patch.object(feature_imaging, "load_frozen_encoder", return_value=None),
            patch.object(
                feature_imaging, "extract_study", side_effect=lambda root, split, key, *args: (self.arrays(key), [])
            ),
        ):
            result = self.run_candidate(
                mode="profile_train",
                weak_manifest=self.weak,
                expected_weak_sha256=sha256(self.weak),
                profile_limit=5,
                require_decoders=False,
            )
        rows = read_csv(self.root / "diagnostics/profile_predictions.csv")[1]
        self.assertEqual(len(rows), 5)
        self.assertTrue({row[ID] for row in rows} <= set(self.ids[3:]))
        self.assertEqual(result["status"], "profile_complete_not_for_submission")
        self.assertFalse((self.root / "submission.csv").exists())

    def test_checkpoint_change_rejected_before_predictions(self):
        with self.assertRaisesRegex(ValueError, "Code/checkpoint"):
            streaming.run_streaming(
                self.root,
                self.checkpoint,
                self.provenance,
                "synthetic",
                "synthetic",
                self.root / "diagnostics",
                expected_checkpoint_sha256="b" * 64,
                expected_sources=source_hashes(),
                device=self.device,
            )
        self.assertFalse((self.root / "submission.csv").exists())


class DiagnosticNotebookTests(unittest.TestCase):
    def test_generated_notebooks_compile_and_profile_has_submission_guard(self):
        spec = importlib.util.spec_from_file_location(
            "diagnostic_builder", ROOT / "scripts/build_frozen_serving_diagnostics.py"
        )
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        if not (ROOT / "artifacts/kaggle/dicom-decoders-py313-v1-20261007/decoder-manifest.json").exists():
            self.skipTest("Locally prepared immutable package absent")
        for name, book in builder.notebooks().items():
            saved = json.loads((ROOT / "notebooks/public" / name).read_text(encoding="utf-8"))
            self.assertEqual(saved["cells"], book["cells"])
            for item in saved["cells"]:
                if item["cell_type"] == "code":
                    self.assertEqual(item["outputs"], [])
                    self.assertIsNone(item["execution_count"])
                    compile("".join(item["source"]), name, "exec")
            if name.startswith("08"):
                self.assertIn("KAGGLE_IS_COMPETITION_RERUN", "".join(saved["cells"][-1]["source"]))


if __name__ == "__main__":
    unittest.main()
