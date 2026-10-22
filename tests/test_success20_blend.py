"""Failure-isolation and submission contracts for the new success-subset rule."""

import csv
import io
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rsna_knee import fixed20_blend, success20_blend  # noqa: E402
from rsna_knee.contracts import ID, SUBMISSION_COLUMNS, TARGETS  # noqa: E402


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def rows(values):
    return [{ID: str(i + 1), **{target: str(value) for target in TARGETS}} for i, value in enumerate(values)]


class SuccessSubsetTests(unittest.TestCase):
    def setUp(self):
        self.parent = rows(["0.08000000", "0.72000000", "0.35000000", "0.91000000"])
        self.native = rows(["0.9", "0.5", "0.2", "0.2"])

    def test_complete_coverage_matches_previous_contract_exactly(self):
        old, _ = fixed20_blend.blend_rows(self.parent, self.native)
        new, receipt = success20_blend.blend_rows(self.parent, list(reversed(self.native)))
        self.assertEqual(new, old)
        self.assertEqual(receipt["native_blended_cell_count"], 40)
        self.assertEqual(receipt["blended_target_count"], 10)
        self.assertFalse(receipt["arithmetic"]["parent_reranked"])
        self.assertFalse(receipt["arithmetic"]["output_reranked"])

    def test_one_failed_study_preserves_parent_but_other_studies_contribute(self):
        actual, receipt = success20_blend.blend_rows(self.parent, self.native, failure_mask={"2": True})
        self.assertEqual(actual[1], self.parent[1])
        # Valid raw values .9, .2, .2 have ranks 3/3, 1.5/3, 1.5/3.
        for i, rank in [(0, 1.0), (2, 0.5), (3, 0.5)]:
            expected = f32(f32(f32(0.8) * f32(float(self.parent[i]["ACL"]))) + f32(f32(0.2) * f32(rank)))
            self.assertEqual(actual[i]["ACL"], repr(expected))
            for target in ("MCL", "Synovitis"):
                self.assertEqual(actual[i][target], self.parent[i][target])
        self.assertEqual(receipt["native_blended_cell_count"], 30)
        self.assertEqual(receipt["per_target"]["ACL"]["native_failure_count"], 1)
        self.assertEqual(receipt["per_target"]["ACL"]["native_rank_population_count"], 3)
        self.assertEqual(receipt["per_target"]["ACL"]["status"], "blended_success_subset")
        self.assertFalse(receipt["all_native_disabled"])
        previous, previous_receipt = fixed20_blend.blend_rows(self.parent, self.native, failure_mask={"2": True})
        self.assertEqual(previous, self.parent)
        self.assertTrue(previous_receipt["all_native_disabled"])

    def test_target_failure_does_not_change_other_target_population(self):
        baseline, _ = success20_blend.blend_rows(self.parent, self.native)
        self.native[1]["ACL"] = ""
        actual, receipt = success20_blend.blend_rows(self.parent, self.native)
        for i in range(4):
            for target in TARGETS:
                if target != "ACL":
                    self.assertEqual(actual[i][target], baseline[i][target])
        self.assertEqual(actual[1]["ACL"], self.parent[1]["ACL"])
        self.assertEqual(receipt["per_target"]["ACL"]["native_rank_population_count"], 3)
        self.assertEqual(receipt["per_target"]["Fracture"]["native_rank_population_count"], 4)

    def test_partial_singleton_all_failure_and_alpha_zero_preserve_parent(self):
        for mask, alpha in [({"1": True, "2": True, "3": True}, 0.2),
                            ({str(i): True for i in range(1, 5)}, 0.2), ({"2": True}, 0)]:
            with self.subTest(mask=mask, alpha=alpha):
                actual, receipt = success20_blend.blend_rows(self.parent, self.native, failure_mask=mask, alpha=alpha)
                self.assertEqual(actual, self.parent)
                self.assertTrue(receipt["all_native_disabled"])
                self.assertEqual(receipt["native_blended_cell_count"], 0)

    def test_partial_constant_native_cannot_shift_successes_past_failed_parent(self):
        parent = rows(["0.45000000", "0.50000000", "0.80000000"])
        native = rows(["0.49999999", "0.1", "0.50000001"])
        actual, receipt = success20_blend.blend_rows(parent, native, failure_mask={"2": True})
        self.assertEqual(actual, parent)
        self.assertTrue(receipt["all_native_disabled"])
        self.assertEqual(receipt["per_target"]["ACL"]["status"], "parent_only_constant_partial_native")
        # Two distinct Python floats collapse to the same FP32 native score.
        self.assertEqual(f32(0.49999999), f32(0.50000001))

    def test_invalid_masked_value_is_rejected_before_output_directory_exists(self):
        self.native[1]["ACL"] = str(math.nan)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            for name, data in [("parent", self.parent), ("native", self.native)]:
                with (base / (name + ".csv")).open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, SUBMISSION_COLUMNS)
                    writer.writeheader()
                    writer.writerows(data)
            (base / "failures.csv").write_text(ID + "\n2\n")
            with self.assertRaises(ValueError):
                success20_blend.blend_csv(base / "parent.csv", base / "native.csv", base / "new",
                                         failure_mask_csv=base / "failures.csv")
            self.assertFalse((base / "new").exists())

    def test_csv_contract_identity_bytes_and_fresh_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            for name, data in [("parent", self.parent), ("native", self.native), ("sample", self.parent)]:
                with (base / (name + ".csv")).open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, SUBMISSION_COLUMNS, lineterminator="\r\n")
                    writer.writeheader()
                    writer.writerows(data)
            (base / "test.csv").write_text(ID + "\n4\n2\n1\n3\n")
            (base / "failures.csv").write_text(ID + "\n2\n")
            with self.assertRaisesRegex(ValueError, "row order"):
                success20_blend.blend_csv(base / "parent.csv", base / "native.csv", base / "rejected",
                                         test_csv=base / "test.csv", sample_csv=base / "sample.csv")
            self.assertFalse((base / "rejected").exists())
            (base / "test.csv").write_text(ID + "\n1\n2\n3\n4\n")
            receipt = success20_blend.blend_csv(base / "parent.csv", base / "native.csv", base / "new",
                                              failure_mask_csv=base / "failures.csv",
                                              test_csv=base / "test.csv", sample_csv=base / "sample.csv")
            actual = list(csv.DictReader(io.StringIO((base / "new/submission.csv").read_text())))
            self.assertEqual(actual[1], self.parent[1])
            self.assertEqual([row[ID] for row in actual], ["1", "2", "3", "4"])
            self.assertEqual(receipt["native_blended_cell_count"], 30)
            with self.assertRaises(FileExistsError):
                success20_blend.blend_csv(base / "parent.csv", base / "native.csv", base / "new")
            success20_blend.blend_csv(base / "parent.csv", base / "native.csv", base / "identity", alpha=0)
            self.assertEqual((base / "identity/submission.csv").read_bytes(), (base / "parent.csv").read_bytes())


if __name__ == "__main__":
    unittest.main()
