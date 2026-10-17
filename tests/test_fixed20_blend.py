"""Synthetic contracts for full-N fixed20 integration, standard library only."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from rsna_knee.contracts import ID, SUBMISSION_COLUMNS, TARGETS, read_csv, sha256, write_csv
from rsna_knee.fixed20_blend import (
    BLENDED_TARGETS,
    blend_csv,
    blend_rows,
    main,
    percentile_average_ranks,
)


class Fixed20BlendTests(unittest.TestCase):
    def rows(self, values):
        return [{ID: f"study-{index}", **{target: value for target in TARGETS}} for index, value in enumerate(values)]

    def test_alpha_zero_preserves_every_parent_string_and_order(self):
        parent = self.rows([" 0.00100 ", "+2.50e-1", "-0.000", "0.99999000"])
        native = self.rows([0.99, None, 0.3, 0.1])
        output, receipt = blend_rows(parent, list(reversed(native)), alpha=0)
        self.assertEqual(output, parent)
        self.assertTrue(receipt["text_noop"])
        self.assertTrue(receipt["all_native_disabled"])
        self.assertEqual(receipt["accuracy_comparison_status"], "not_established_all_native_disabled")
        self.assertEqual(receipt["per_target"]["ACL"]["native_success_count"], 3)

    def test_parent_and_output_are_not_reranked_golden_fp32_values(self):
        parent = self.rows(["0", "0.25", "0.5", "1"])
        native = self.rows([0.4, 0.3, 0.2, 0.1])
        output, receipt = blend_rows(parent, native)
        self.assertEqual(
            [float(row["ACL"]) for row in output],
            [0.20000000298023224, 0.3500000238418579, 0.5, 0.8500000238418579],
        )
        self.assertFalse(receipt["arithmetic"]["parent_reranked"])
        self.assertFalse(receipt["arithmetic"]["output_reranked"])
        self.assertEqual(receipt["per_target"]["ACL"]["effective_weights"], {"parent_s": 0.8, "native_rank": 0.2})
        self.assertEqual(
            receipt["per_target"]["ACL"]["conditional_P029M_component_weights_nominal"],
            {"Raptor384": 0.26, "Raptor224": 0.14, "CompactR6": 0.4, "F006_native_rank": 0.2},
        )

    def test_ties_use_full_N_average_one_based_ranks(self):
        self.assertEqual(percentile_average_ranks([0.8, 0.2, 0.2, 0.9]), [0.75, 0.375, 0.375, 1.0])
        self.assertEqual(percentile_average_ranks([1, 1, 1]), [2 / 3] * 3)
        output, _ = blend_rows(self.rows([0, 0, 0, 0]), self.rows([0.8, 0.2, 0.2, 0.9]))
        self.assertEqual(
            [float(row["ACL"]) for row in output],
            [0.15000000596046448, 0.07500000298023224, 0.07500000298023224, 0.20000000298023224],
        )

    def test_native_quantization_precedes_ties_and_separate_fp32_arithmetic(self):
        output, _ = blend_rows(self.rows([0, 0]), self.rows(["0.500000001", "0.500000002"]))
        self.assertEqual([float(row["ACL"]) for row in output], [0.15000000596046448] * 2)

    def test_single_study_preserves_parent_gap_instead_of_ranking_to_one(self):
        output, _ = blend_rows(self.rows(["0"]), self.rows([0.99]))
        self.assertEqual(float(output[0]["ACL"]), 0.20000000298023224)

    def test_native_row_order_is_irrelevant_and_parent_permutation_only_changes_order(self):
        parent, native = self.rows([0.48, 0.5, 0.52]), self.rows([0.9, 0.2, 0.1])
        reference, _ = blend_rows(parent, native)
        output, _ = blend_rows(parent, list(reversed(native)))
        self.assertEqual(output, reference)
        permuted, _ = blend_rows(list(reversed(parent)), native)
        self.assertEqual(permuted, list(reversed(reference)))

    def test_excluded_MCL_and_synovitis_retain_exact_parent_text(self):
        parent, native = self.rows(["0.05", "0.95"]), self.rows([0.9, 0.1])
        parent[0]["MCL"], parent[0]["Synovitis"] = " 2.50e-1 ", "-0.000"
        native[1]["MCL"], native[1]["Synovitis"] = None, None
        output, receipt = blend_rows(parent, native)
        for old, new in zip(parent, output):
            for target in ("MCL", "Synovitis"):
                self.assertEqual(old[target], new[target])
                self.assertEqual(receipt["per_target"][target]["effective_weights"]["parent_s"], 1)
        self.assertEqual(receipt["blended_target_count"], 10)

    def test_one_missing_native_cell_falls_back_whole_column_not_success_subset(self):
        parent, native = self.rows(["0.30000", "0.31000", "0.32000"]), self.rows([0.9, 0.1, 0.2])
        native[1]["ACL"] = None
        output, receipt = blend_rows(parent, native)
        self.assertEqual([row["ACL"] for row in output], [row["ACL"] for row in parent])
        self.assertNotEqual([row["Effusion"] for row in output], [row["Effusion"] for row in parent])
        self.assertEqual(receipt["fallback_targets"], ["ACL"])
        self.assertEqual(receipt["per_target"]["ACL"]["native_success_count"], 2)
        self.assertEqual(receipt["per_target"]["ACL"]["native_failure_count"], 1)
        self.assertEqual(receipt["per_target"]["ACL"]["effective_weights"], {"parent_s": 1.0, "native_rank": 0.0})

    def test_explicit_per_target_failure_behaves_like_missing_cell(self):
        parent, native = self.rows(["0.3000", "0.7000"]), self.rows([0.9, 0.1])
        output, receipt = blend_rows(parent, native, failure_mask={"study-1": {"ACL": True, "Effusion": False}})
        self.assertEqual([row["ACL"] for row in output], ["0.3000", "0.7000"])
        self.assertEqual(receipt["fallback_targets"], ["ACL"])
        self.assertEqual(receipt["blended_target_count"], 9)

    def test_one_entire_failed_study_disables_all_ten_columns(self):
        parent, native = self.rows(["0.30", "0.70"]), self.rows([0.9, 0.1])
        output, receipt = blend_rows(parent, native, failure_mask={"study-0": True})
        self.assertEqual(output, parent)
        self.assertEqual(receipt["fallback_targets"], list(BLENDED_TARGETS))
        self.assertTrue(receipt["all_native_disabled"])
        self.assertEqual(receipt["accuracy_comparison_status"], "not_established_all_native_disabled")
        self.assertTrue(all(receipt["per_target"][target]["native_success_count"] == 1 for target in TARGETS))

    def test_all_missing_is_explicit_noop_and_never_becomes_zero_predictions(self):
        parent = self.rows(["0.300", "0.700"])
        output, receipt = blend_rows(parent, self.rows([None, None]))
        self.assertEqual(output, parent)
        self.assertTrue(receipt["all_native_disabled"])
        self.assertTrue(receipt["numeric_noop"])
        self.assertTrue(receipt["rank_noop"])
        self.assertTrue(all(target["native_success_count"] == 0 for target in receipt["per_target"].values()))

    def test_ranking_noop_is_identified_even_when_numeric_scores_change(self):
        output, receipt = blend_rows(self.rows([0.1, 0.4, 0.8]), self.rows([0.1, 0.4, 0.8]))
        self.assertFalse(receipt["all_native_disabled"])
        self.assertFalse(receipt["numeric_noop"])
        self.assertTrue(receipt["rank_noop"])
        self.assertEqual(receipt["accuracy_comparison_status"], "ranking_noop_under_auc")
        self.assertLess(float(output[0]["ACL"]), float(output[1]["ACL"]))
        self.assertLess(float(output[1]["ACL"]), float(output[2]["ACL"]))

    def test_exact_numeric_noop_and_changed_rankings_have_distinct_receipts(self):
        _, noop = blend_rows(self.rows([0.25, 0.5, 0.75, 1]), self.rows([1, 2 / 3, 1 / 3, 0]))
        self.assertFalse(noop["numeric_noop"])
        _, changed = blend_rows(self.rows([0.48, 0.5, 0.52]), self.rows([0.9, 0.2, 0.1]))
        self.assertFalse(changed["rank_noop"])
        self.assertEqual(changed["accuracy_comparison_status"], "changed_rankings_unscored")
        self.assertGreater(changed["parent_to_final"]["rank_changed_cell_count"], 0)
        _, exact = blend_rows(self.rows([0.25, 0.5, 0.75, 1]), self.rows([0.1, 0.2, 0.3, 0.4]))
        self.assertTrue(exact["numeric_noop"])

    def test_invalid_values_raise_even_when_masked_or_alpha_zero(self):
        for role in ("parent", "native"):
            for value in ("nan", "NaN", "inf", "-inf", "NA", "null", "oops", " ", -0.001, 1.001, True):
                for alpha in (0, 0.2):
                    with self.subTest(role=role, value=value, alpha=alpha):
                        parent, native = self.rows([0.5]), self.rows([0.5])
                        (parent if role == "parent" else native)[0]["ACL"] = value
                        with self.assertRaises(ValueError):
                            blend_rows(parent, native, failure_mask={"study-0": True}, alpha=alpha)
        for value in (None, ""):
            parent = self.rows([value])
            with self.assertRaises(ValueError):
                blend_rows(parent, self.rows([0.5]))

    def test_empty_mismatched_duplicate_and_unsafe_UIDs_raise(self):
        with self.assertRaises(ValueError):
            blend_rows([], [])
        for role in ("parent", "native"):
            for key in ("other-study", "", "../oops", "study\n1", " study", 123):
                with self.subTest(role=role, uid=key):
                    parent, native = self.rows([0.5]), self.rows([0.5])
                    (parent if role == "parent" else native)[0][ID] = key
                    with self.assertRaises(ValueError):
                        blend_rows(parent, native)
            parent, native = self.rows([0.5]), self.rows([0.5])
            rows = parent if role == "parent" else native
            rows.append(dict(rows[0]))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                blend_rows(parent, native)

    def test_column_order_and_failure_mask_are_strict(self):
        for role in ("parent", "native"):
            for change in ("reverse", "missing", "extra"):
                parent, native = self.rows([0.5]), self.rows([0.5])
                rows = parent if role == "parent" else native
                if change == "reverse":
                    rows[0] = dict(reversed(list(rows[0].items())))
                elif change == "missing":
                    del rows[0]["MCL"]
                else:
                    rows[0]["extra"] = 0.1
                with self.assertRaises(ValueError):
                    blend_rows(parent, native)
        for mask in ({"unknown": True}, {"study-0": 1}, {"study-0": {"unknown": True}}, {"study-0": {"ACL": 1}}):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                blend_rows(self.rows([0.5]), self.rows([0.5]), failure_mask=mask)
        for alpha in (True, 0.1, 1, "0.2", float("nan")):
            with self.assertRaises(ValueError):
                blend_rows(self.rows([0.5]), self.rows([0.5]), alpha=alpha)


class Fixed20BlendCsvTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.parent, self.native = self.root / "parent.csv", self.root / "native.csv"
        self.output = self.root / "new-run"
        self.rows = Fixed20BlendTests().rows
        write_csv(self.parent, SUBMISSION_COLUMNS, self.rows(["0.3000", "0.7000"]))
        write_csv(self.native, SUBMISSION_COLUMNS, self.rows(["0.9", "0.1"]))

    def test_CLI_outputs_fresh_contract_CSV_and_hashed_receipt_only(self):
        test, sample = self.root / "test.csv", self.root / "sample.csv"
        write_csv(test, (ID,), [{ID: "study-0"}, {ID: "study-1"}])
        write_csv(sample, SUBMISSION_COLUMNS, self.rows([0, 0]))
        captured = io.StringIO()
        with redirect_stdout(captured):
            main(
                [
                    "--parent-csv",
                    str(self.parent),
                    "--native-mean-csv",
                    str(self.native),
                    "--output-dir",
                    str(self.output),
                    "--test-csv",
                    str(test),
                    "--sample-csv",
                    str(sample),
                ]
            )
        self.assertEqual(json.loads(captured.getvalue())["stage"], "local_csv_integration_only")
        receipt = json.loads((self.output / "receipt.json").read_text(encoding="utf-8"))
        fields, rows = read_csv(self.output / "submission.csv")
        self.assertEqual(tuple(fields), SUBMISSION_COLUMNS)
        self.assertEqual([row[ID] for row in rows], ["study-0", "study-1"])
        self.assertEqual(receipt["inputs"]["parent"]["sha256"], sha256(self.parent))
        self.assertEqual(receipt["output"]["sha256"], sha256(self.output / "submission.csv"))
        self.assertEqual(
            (self.output / "receipt.sha256").read_text(encoding="utf-8"),
            f"{sha256(self.output / 'receipt.json')}  receipt.json\n",
        )
        self.assertTrue(receipt["submission_contract"]["valid"])
        self.assertTrue(all(value is False for value in receipt["qualification"].values()))

    def test_alpha_zero_keeps_exact_parent_CSV_bytes(self):
        self.parent.write_bytes(b"\xef\xbb\xbf" + self.parent.read_bytes().replace(b"\r\n", b"\n"))
        blend_csv(self.parent, self.native, self.output, alpha=0)
        self.assertEqual((self.output / "submission.csv").read_bytes(), self.parent.read_bytes())

    def test_empty_native_CSV_cell_means_explicit_column_missing(self):
        native = self.rows([0.9, 0.1])
        native[0]["ACL"] = ""
        write_csv(self.native, SUBMISSION_COLUMNS, native)
        receipt = blend_csv(self.parent, self.native, self.output)
        self.assertEqual(receipt["fallback_targets"], ["ACL"])
        _, rows = read_csv(self.output / "submission.csv")
        self.assertEqual([row["ACL"] for row in rows], ["0.3000", "0.7000"])

    def test_failure_UID_csv_or_per_target_flags_disables_native(self):
        failed = self.root / "failed.csv"
        write_csv(failed, (ID,), [{ID: "study-0"}])
        receipt = blend_csv(self.parent, self.native, self.output, failure_mask_csv=failed)
        self.assertTrue(receipt["all_native_disabled"])
        self.assertEqual((self.output / "submission.csv").read_bytes(), self.parent.read_bytes())
        flags = self.rows(["0", "0"])
        flags[1]["ACL"] = "1"
        write_csv(failed, SUBMISSION_COLUMNS, flags)
        receipt = blend_csv(self.parent, self.native, self.root / "second-run", failure_mask_csv=failed)
        self.assertEqual(receipt["fallback_targets"], ["ACL"])

    def test_existing_output_or_invalid_input_is_rejected_before_new_output(self):
        self.output.mkdir()
        marker = self.output / "prior.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            blend_csv(self.parent, self.native, self.output)
        self.assertEqual(list(self.output.iterdir()), [marker])
        invalid_output = self.root / "invalid-run"
        native = self.rows([0.5, 0.5])
        native[0]["ACL"] = "nan"
        write_csv(self.native, SUBMISSION_COLUMNS, native)
        with self.assertRaises(ValueError):
            blend_csv(self.parent, self.native, invalid_output)
        self.assertFalse(invalid_output.exists())

    def test_invalid_csv_headers_masks_and_test_UIDs_raise_before_writing(self):
        for fields in (tuple(reversed(SUBMISSION_COLUMNS)), (*SUBMISSION_COLUMNS, "ACL")):
            write_csv(self.native, fields, self.rows([0.5, 0.5]))
            with self.assertRaises(ValueError):
                blend_csv(self.parent, self.native, self.output)
            self.assertFalse(self.output.exists())
        write_csv(self.native, SUBMISSION_COLUMNS, self.rows([0.5, 0.5]))
        failed = self.root / "failed.csv"
        for keys in (["study-0", "study-0"], ["unknown"]):
            write_csv(failed, (ID,), [{ID: key} for key in keys])
            with self.assertRaises(ValueError):
                blend_csv(self.parent, self.native, self.output, failure_mask_csv=failed)
            self.assertFalse(self.output.exists())
        write_csv(failed, SUBMISSION_COLUMNS, self.rows(["True", "False"]))
        with self.assertRaises(ValueError):
            blend_csv(self.parent, self.native, self.output, failure_mask_csv=failed)
        test = self.root / "wrong-test.csv"
        write_csv(test, (ID,), [{ID: "other-study"}])
        with self.assertRaises(ValueError):
            blend_csv(self.parent, self.native, self.output, test_csv=test)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
