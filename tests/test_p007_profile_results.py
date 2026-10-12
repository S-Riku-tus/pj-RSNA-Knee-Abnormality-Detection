"""Independent profile-result rejection tests; no real CSV, MRI or weights required."""

import copy
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "p007_results", ROOT / "artifacts/research/20261008-p007-profile32-results-v1/verify_results.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def observations(mode):
    views = 1 if mode == "plain" else 2
    return {
        "mode": mode, "complete": True, "model_count": 3, "load_grad_disabled": True,
        "forwards": [{"model_index": i % 3, "phase": "plain" if views == 1 or i % 6 < 3 else "mirror",
                      "shape": [4, 6, 16, 3, 384, 384], "finite_logits": True, "grad_disabled": True}
                     for i in range(8 * 3 * views)],
        "mirror_calls": [] if mode == "plain" else [
            {"shape": [4, 6, 16, 3, 384, 384], "valid_slots": [[True] * 6 for _ in range(4)],
             "mask_object_preserved": True, "position_contract_passed": True,
             "uint8_before_normalization": True} for _ in range(8)],
    }


class CompactProfileResultTests(unittest.TestCase):
    def test_average_tied_ranks_and_fixed_fusion(self):
        self.assertEqual(audit.ranks([0, 0, 1]), [0.5, 0.5, 1.0])
        self.assertEqual(audit.ranks([3, 2, 1, 0]), [1, 0.75, 0.5, 0.25])
        base, own = [[0.1], [0.2], [0.3], [0.4]], [[0.4], [0.3], [0.2], [0.1]]
        self.assertEqual(audit.fixed_fusion(base, own, ["ACL"], {"ACL": 0.5}), [[0.625]] * 4)
        self.assertEqual(audit.fixed_fusion(base, own, ["ACL"], {"ACL": 0.1}), [[0.25], [0.5], [0.75], [1]])

    def test_wrong_UID_column_nonfinite_range_and_constant_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic.csv"
            def write(fields, values):
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(fields)
                    writer.writerows(values)
            fields = ["StudyInstanceUID", "ACL"]
            write(fields, [["a", 0.1], ["b", 0.9]])
            self.assertEqual(audit.read_predictions(path, ["a", "b"], ["ACL"]), [[0.1], [0.9]])
            for changed_fields, values in (
                (fields, [["b", 0.1], ["a", 0.9]]),
                (["ACL", "StudyInstanceUID"], [[0.1, "a"], [0.9, "b"]]),
                (fields, [["a", "nan"], ["b", 0.9]]),
                (fields, [["a", -0.1], ["b", 0.9]]),
                (fields, [["a", 0.5], ["b", 0.5]]),
            ):
                write(changed_fields, values)
                with self.assertRaises(AssertionError):
                    audit.read_predictions(path, ["a", "b"], ["ACL"])

    def test_actual_views_forwards_mirror_gradient_rejection(self):
        for mode in ("plain", "mirror"):
            state = observations(mode)
            result = audit.validate_inference(state, mode, 32)
            self.assertEqual(result["model_calls"], 24 if mode == "plain" else 48)
            for mutation in ("load_grad", "forward_grad", "count", "phase", "model_order"):
                altered = copy.deepcopy(state)
                if mutation == "load_grad":
                    altered["load_grad_disabled"] = False
                elif mutation == "forward_grad":
                    altered["forwards"][0]["grad_disabled"] = False
                elif mutation == "count":
                    altered["forwards"].pop()
                elif mutation == "phase":
                    altered["forwards"][0]["phase"] = "mirror"
                else:
                    altered["forwards"][0]["model_index"] = 1
                with self.subTest(mode=mode, mutation=mutation), self.assertRaises(AssertionError):
                    audit.validate_inference(altered, mode, 32)
        bad = observations("mirror")
        bad["mirror_calls"][0]["position_contract_passed"] = False
        with self.assertRaises(AssertionError):
            audit.validate_inference(bad, "mirror", 32)

    def test_T4_margin_strictly_below13000_and_resource_validation(self):
        good = {"elapsed_seconds": 10, "samples": 20, "sampled_peak_process_tree_rss_bytes": 100,
                "disk_free_bytes_after": 20_000_000_000, "sampled_gpu_peak_used_mib": [8069, 2000]}
        self.assertEqual(audit.validate_resources(good)["elapsed_seconds"], 10)
        for bad in (
            {**good, "sampled_gpu_peak_used_mib": [13000, 2000]},
            {**good, "samples": 0}, {**good, "elapsed_seconds": float("nan")},
        ):
            with self.assertRaises(AssertionError):
                audit.validate_resources(bad)


if __name__ == "__main__":
    unittest.main()
