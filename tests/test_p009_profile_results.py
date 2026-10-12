"""Compact-result failure checks without Kaggle, real model or images."""

import copy
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "p009_result_verifier", ROOT / "artifacts/research/20261008-p009-profile32-results-v2/verify_results.py")
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class CompactResultContracts(unittest.TestCase):
    def test_independent_fixed_rank_fusion_with_ties_and_spearman(self):
        self.assertEqual(verifier.ranks([.9, .1, .1, .5]), [1., .375, .375, .75])
        left, right = [[.9], [.1], [.1], [.5]], [[.1], [.5], [.9], [.5]]
        self.assertEqual(verifier.fixed_fusion(left, right, ["ACL"], {"ACL": .25}),
                         [[1.], [.25], [.5], [.75]])
        self.assertAlmostEqual(verifier.independent_spearman([1., 2., 2., 4.], [4., 2., 2., 1.]), -1.)

    def test_CSV_wrong_UID_column_nonfinite_constant_rejected(self):
        labels = ["target" + str(index) for index in range(12)]
        uids = ["artificial-a", "artificial-b", "artificial-c"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predict.csv"
            for invalid in (None, "UID", "columns", "nan", "constant"):
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["StudyInstanceUID", *(labels[::-1] if invalid == "columns" else labels)])
                    for index, uid in enumerate(uids):
                        values = [index / 2] * 12
                        if invalid == "nan" and index == 1:
                            values[4] = float("nan")
                        if invalid == "constant":
                            values[4] = .5
                        writer.writerow([uids[0] if invalid == "UID" else uid, *values])
                if invalid is None:
                    self.assertEqual(len(verifier.read_predictions(path, uids, labels)), 3)
                else:
                    with self.assertRaises(AssertionError, msg=invalid):
                        verifier.read_predictions(path, uids, labels)

    def test_actual_worker_main_forward_gradient_and_pickle_gates(self):
        uids = [str(index) for index in range(32)]
        slots = [["Sagittal", 1, 26], ["Sagittal", 0, 22], ["Coronal", 1, 18],
                 ["Coronal", 0, 12], ["Axial", -1, 18]]
        row = {
            "fov_mm": 160, "main_success_marker": True, "main_fallback_or_preprocess_failure": False,
            "successful_pair_calls": 32, "valid_slices_by_uid": dict.fromkeys(uids, 96),
            "forward_observations": [{"shape": [1, 94, 3, 320, 320], "grad_disabled": True,
                                      "finite_logits": True, "device": "cuda:" + str(index % 2)} for index in range(64)],
            "loaded_model_devices": ["cuda:0", "cuda:1"], "task_pickle_identity": {"_pre_one": True, "_pre_init": True},
            "registered_module_name": "_p009_profile_main_fov160", "registered_module_removed": True,
            "worker_initializers": [{"pid": 123, "actual_builder_CROP_MM": 160, "actual_builder_IMG": 384,
                                      "actual_builder_SLOTS": slots, "actual_builder_source_sha256": "expected"}],
            "fastread_binding": {"source_sha256": "expected", "no_other_cache_or_source_removed": True,
                                "specific_cache_file": "/kaggle/working/__pycache__/_fastread.cpython-313.pyc"},
        }
        self.assertEqual(verifier.validate_main(row, 160, uids, "expected")["forwards"], 64)
        for invalid in ("grad", "count", "FOV", "source", "mask", "pickle", "cleanup"):
            bad = copy.deepcopy(row)
            if invalid == "grad":
                bad["forward_observations"][0]["grad_disabled"] = False
            elif invalid == "count":
                bad["forward_observations"].pop()
            elif invalid == "FOV":
                bad["worker_initializers"][0]["actual_builder_CROP_MM"] = 140
            elif invalid == "source":
                bad["worker_initializers"][0]["actual_builder_source_sha256"] = "stale"
            elif invalid == "mask":
                bad["valid_slices_by_uid"][uids[0]] = 0
            elif invalid == "pickle":
                bad["task_pickle_identity"]["_pre_one"] = False
            elif invalid == "cleanup":
                bad["registered_module_removed"] = False
            with self.assertRaises(AssertionError, msg=invalid):
                verifier.validate_main(bad, 160, uids, "expected")

    def test_single_reader24forward_and_T4_memory_gate(self):
        state = {"mode": "plain", "complete": True, "model_count": 3, "load_grad_disabled": True,
                 "mirror_calls": [], "forwards": [{"model_index": index % 3, "phase": "plain",
                  "shape": [4, 6, 16, 3, 384, 384], "finite_logits": True, "grad_disabled": True}
                  for index in range(24)]}
        self.assertEqual(verifier.validate_inference(state, "plain", 32)["model_calls"], 24)
        bad = copy.deepcopy(state)
        bad["forwards"].extend(copy.deepcopy(state["forwards"]))
        with self.assertRaises(AssertionError):
            verifier.validate_inference(bad, "plain", 32)
        resources = {"elapsed_seconds": 1., "samples": 2, "sampled_peak_process_tree_rss_bytes": 10,
                     "disk_free_bytes_after": 2_000_000_000, "sampled_gpu_peak_used_mib": [8071, 543]}
        verifier.validate_resources(resources)
        resources["sampled_gpu_peak_used_mib"][0] = 13000
        with self.assertRaises(AssertionError):
            verifier.validate_resources(resources)


if __name__ == "__main__":
    unittest.main()
