"""Synthetic standard-library tests of the offline fixed20 numerical contract."""

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from rsna_knee.contracts import ID, SUBMISSION_COLUMNS, TARGETS, read_csv, sha256, write_csv

SPEC = importlib.util.spec_from_file_location(
    "preview_fixed20_candidate", Path(__file__).resolve().parents[1] / "scripts/preview_fixed20_candidate.py"
)
PREVIEW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREVIEW)


class Fixed20PreviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.parent_csv = self.root / "parent.csv"
        self.native_csv = self.root / "native.csv"
        self.failed_csv = self.root / "failed.csv"
        self.output = self.root / "new-preview"

    def rows(self, values):
        return [{ID: f"study-{index}", **{target: str(value) for target in TARGETS}} for index, value in enumerate(values)]

    def inputs(self, parent, native=None):
        write_csv(self.parent_csv, SUBMISSION_COLUMNS, parent)
        write_csv(self.native_csv, SUBMISSION_COLUMNS, parent if native is None else native)

    def run_preview(self, failed=None):
        if failed is not None:
            write_csv(self.failed_csv, (ID,), [{ID: key} for key in failed])
        manifest = PREVIEW.preview(
            self.parent_csv, self.native_csv, self.output, failed_uids_csv=None if failed is None else self.failed_csv
        )
        fields, rows = read_csv(self.output / "preview.csv")
        self.assertEqual(tuple(fields), SUBMISSION_COLUMNS)
        return rows, manifest

    def test_single_study_has_percentile_one_and_preserves_unsupported_strings(self):
        parent = self.rows(["0.00100"])
        parent[0]["MCL"], parent[0]["Synovitis"] = " 2.50e-1 ", "-0.000"
        self.inputs(parent, self.rows(["0.99"]))
        rows, manifest = self.run_preview()
        self.assertEqual(rows[0]["ACL"], "1.0")
        self.assertEqual(rows[0]["MCL"], " 2.50e-1 ")
        self.assertEqual(rows[0]["Synovitis"], "-0.000")
        self.assertFalse(manifest["qualification"]["Kaggle_serving_qualified"])

    def test_average_ties_use_one_based_rank_divided_by_N(self):
        self.assertEqual(PREVIEW.percentile_average_ranks([0.8, 0.2, 0.2, 0.9]), [0.75, 0.375, 0.375, 1.0])
        self.assertEqual(PREVIEW.percentile_average_ranks([0.2, 0.2, 0.2]), [2 / 3] * 3)
        parent = self.rows([0.2, 0.2, 0.8, 0.9])
        self.inputs(parent, self.rows([0.9, 0.9, 0.1, 0.2]))
        rows, _ = self.run_preview()
        self.assertEqual([float(row["ACL"]) for row in rows], [0.375, 0.375, 0.75, 1.0])

    def test_fixed20_overlay_changes_an_order_and_aligns_native_by_UID(self):
        parent = self.rows([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        native = self.rows([0.6, 0.1, 0.2, 0.3, 0.4, 0.5])
        self.inputs(parent, list(reversed(native)))
        rows, _ = self.run_preview()
        self.assertEqual([row[ID] for row in rows], [row[ID] for row in parent])
        self.assertEqual([float(row["ACL"]) for row in rows], [2 / 6, 1 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0])

    def test_partial_failure_restores_all_parent_strings_after_full_N_ranking(self):
        parent = self.rows(["1.000e-1", "0.2000", "0.30"])
        self.inputs(parent, self.rows([0.9, 0.5, 0.1]))
        rows, manifest = self.run_preview([parent[0][ID]])
        self.assertEqual(rows[0], parent[0])
        self.assertEqual(float(rows[1]["ACL"]), 2 / 3)
        self.assertEqual(float(rows[2]["ACL"]), 1.0)
        self.assertEqual(rows[1]["MCL"], "0.2000")
        self.assertEqual(rows[1]["Synovitis"], "0.2000")
        self.assertEqual(manifest["explicit_failed_UID_count"], 1)

    def test_all_failed_preserves_all_original_cell_strings(self):
        parent = self.rows([" 0.100 ", "+2e-1", "0.3000"])
        self.inputs(parent, self.rows([0.9, 0.8, 0.7]))
        rows, _ = self.run_preview([row[ID] for row in parent])
        self.assertEqual(rows, parent)

    def test_manifest_hashes_and_cli_describe_only_offline_preview(self):
        self.inputs(self.rows([0.2, 0.8]))
        captured = io.StringIO()
        with redirect_stdout(captured):
            PREVIEW.main(
                [
                    "--parent-csv", str(self.parent_csv), "--native-mean-csv", str(self.native_csv),
                    "--output-dir", str(self.output),
                ]
            )
        report = json.loads(captured.getvalue())
        self.assertEqual(report["stage"], "offline_preview_only")
        self.assertFalse(report["existing_inference_equivalence_verified"])
        manifest_path = self.output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["inputs"]["parent"]["sha256"], sha256(self.parent_csv))
        self.assertEqual(manifest["inputs"]["native_mean"]["sha256"], sha256(self.native_csv))
        self.assertEqual(manifest["output"]["sha256"], sha256(self.output / "preview.csv"))
        self.assertEqual(
            (self.output / "manifest.sha256").read_text(encoding="utf-8"), f"{sha256(manifest_path)}  manifest.json\n"
        )
        self.assertTrue(all(value is False for value in manifest["qualification"].values()))

    def test_existing_output_is_rejected_without_overwrite(self):
        self.inputs(self.rows([0.5]))
        self.output.mkdir()
        marker = self.output / "previous-result.txt"
        marker.write_text("old-run", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            PREVIEW.preview(self.parent_csv, self.native_csv, self.output)
        self.assertEqual(marker.read_text(encoding="utf-8"), "old-run")
        self.assertEqual(list(self.output.iterdir()), [marker])

    def test_missing_design_evidence_is_rejected_before_creating_output(self):
        self.inputs(self.rows([0.5]))
        missing = PREVIEW.ROOT / "experiments/absent-fixed20-test-design.json"
        with mock.patch.object(PREVIEW, "DESIGN", missing), self.assertRaises(FileNotFoundError):
            PREVIEW.preview(self.parent_csv, self.native_csv, self.output)
        self.assertFalse(self.output.exists())

    def test_invalid_probabilities_in_either_input_are_rejected_before_output(self):
        for role in ("parent", "native"):
            for value in ("nan", "inf", "-inf", "", "-0.01", "1.01", "not-a-number"):
                with self.subTest(role=role, value=value):
                    parent, native = self.rows([0.2]), self.rows([0.8])
                    (parent if role == "parent" else native)[0]["ACL"] = value
                    self.inputs(parent, native)
                    with self.assertRaises(ValueError):
                        PREVIEW.preview(self.parent_csv, self.native_csv, self.output)
                    self.assertFalse(self.output.exists())

    def test_duplicate_UID_in_either_input_is_rejected(self):
        for role in ("parent", "native"):
            with self.subTest(role=role):
                parent, native = self.rows([0.2]), self.rows([0.8])
                (parent if role == "parent" else native).append({ID: "study-0", **{t: "0.5" for t in TARGETS}})
                self.inputs(parent, native)
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    PREVIEW.preview(self.parent_csv, self.native_csv, self.output)
                self.assertFalse(self.output.exists())

    def test_UID_set_mismatch_is_rejected(self):
        parent, native = self.rows([0.2]), self.rows([0.8])
        native[0][ID] = "different-study"
        self.inputs(parent, native)
        with self.assertRaisesRegex(ValueError, "UID sets"):
            PREVIEW.preview(self.parent_csv, self.native_csv, self.output)
        self.assertFalse(self.output.exists())

    def test_empty_inputs_and_wrong_headers_are_rejected(self):
        for role in ("parent", "native"):
            for columns, rows in (
                (SUBMISSION_COLUMNS, []),
                (tuple(reversed(SUBMISSION_COLUMNS)), self.rows([0.5])),
                ((*SUBMISSION_COLUMNS, "extra"), self.rows([0.5])),
                ((*SUBMISSION_COLUMNS, "ACL"), self.rows([0.5])),
            ):
                with self.subTest(role=role, columns=columns, empty=not rows):
                    self.inputs(self.rows([0.5]))
                    write_csv(self.parent_csv if role == "parent" else self.native_csv, columns, rows)
                    with self.assertRaises(ValueError):
                        PREVIEW.preview(self.parent_csv, self.native_csv, self.output)
                    self.assertFalse(self.output.exists())

    def test_unknown_and_duplicate_failed_UIDs_are_rejected(self):
        self.inputs(self.rows([0.5]))
        for failed in (["unknown"], ["study-0", "study-0"]):
            with self.subTest(failed=failed):
                write_csv(self.failed_csv, (ID,), [{ID: key} for key in failed])
                with self.assertRaises(ValueError):
                    PREVIEW.preview(
                        self.parent_csv, self.native_csv, self.output, failed_uids_csv=self.failed_csv
                    )
                self.assertFalse(self.output.exists())

    def test_native_failed_row_still_requires_complete_finite_probabilities(self):
        parent, native = self.rows([0.5]), self.rows([0.8])
        native[0]["Fracture"] = "nan"
        self.inputs(parent, native)
        write_csv(self.failed_csv, (ID,), [{ID: "study-0"}])
        with self.assertRaises(ValueError):
            PREVIEW.preview(self.parent_csv, self.native_csv, self.output, failed_uids_csv=self.failed_csv)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
