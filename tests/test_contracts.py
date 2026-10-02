import json
import tempfile
import unittest
from pathlib import Path

from rsna_knee.contracts import ID, SUBMISSION_COLUMNS, TARGETS, probability, validate_submission, write_csv
from rsna_knee.metrics import binary_auc, evaluate_rows
from rsna_knee.prepare import label_template, prepare


class ContractsTests(unittest.TestCase):
    def test_auc_ties_and_missing_classes(self):
        self.assertEqual(binary_auc([0, 1, 0, 1], [0.1, 0.5, 0.5, 0.9]), 0.875)
        self.assertIsNone(binary_auc([0, 0], [0.1, 0.2]))
        truth = {"a": {t: "0" for t in TARGETS}, "b": {t: "1" for t in TARGETS}}
        predictions = {"a": {t: ".1" for t in TARGETS}, "b": {t: ".9" for t in TARGETS}}
        self.assertEqual(evaluate_rows(truth, predictions)["macro_auc_12"], 1)
        truth["b"]["ACL"] = ".8"
        result = evaluate_rows(truth, predictions)
        self.assertIsNone(result["macro_auc_12"])
        self.assertEqual(result["defined_classes"], 11)
        self.assertEqual(result["per_target"]["ACL"]["soft_excluded"], 1)

    def test_finite_probability_and_missing_policy(self):
        for invalid in ("nan", "inf", "-0.1", "1.1"):
            with self.assertRaises(ValueError):
                probability(invalid)
        self.assertIsNone(probability("", missing=True))
        self.assertEqual(probability("0", missing=True), 0)

    def test_submission_identity_range_and_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            test = root / "test.csv"
            prediction = root / "submission.csv"
            write_csv(test, [ID], [{ID: "001"}, {ID: "002"}])
            rows = [{ID: key, **{t: 0.5 for t in TARGETS}} for key in ("001", "002")]
            write_csv(prediction, SUBMISSION_COLUMNS, rows)
            self.assertTrue(validate_submission(prediction, test)["valid"])
            for bad in ([rows[0], rows[0]], rows[::-1], rows[:1]):
                write_csv(prediction, SUBMISSION_COLUMNS, bad)
                with self.assertRaises(ValueError):
                    validate_submission(prediction, test)
            rows[0]["ACL"] = "nan"
            write_csv(prediction, SUBMISSION_COLUMNS, rows)
            with self.assertRaises(ValueError):
                validate_submission(prediction, test)
            rows[0]["ACL"] = 0.5
            write_csv(prediction, tuple(reversed(SUBMISSION_COLUMNS)), rows)
            with self.assertRaises(ValueError):
                validate_submission(prediction, test)


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.train_path = self.root / "train.csv"
        self.weak_path = self.root / "weak.csv"
        self.provenance = self.root / "provenance.json"
        self.rows = [
            {ID: f"s{i:02d}", "Report": f"synthetic report {i}", **{t: "" for t in TARGETS}} for i in range(30)
        ]
        self.rows[0]["ACL"] = "1"  # Partial official labels still reserve the whole study.
        self.rows[1]["Report"] = self.rows[0]["Report"]
        self.rows[3]["Report"] = "  SYNTHETIC   REPORT 2  "
        write_csv(self.train_path, [ID, "Report", *TARGETS], self.rows)
        write_csv(self.weak_path, [ID, *TARGETS], [{ID: r[ID], **{t: 0.7 for t in TARGETS}} for r in self.rows])
        self.provenance.write_text(
            json.dumps(
                {
                    "source_url": "synthetic://tests",
                    "source_version": "v1",
                    "license": "test-only",
                    "method": "synthetic",
                    "created_at": "2026-10-02",
                    "gold_used_for_tuning": False,
                }
            )
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_reserves_gold_and_duplicate_groups(self):
        from rsna_knee.contracts import read_csv

        summary = prepare(self.train_path, self.weak_path, self.provenance, self.root / "out")
        _, weak = read_csv(self.root / "out" / "weak.csv")
        _, gold = read_csv(self.root / "out" / "gold.csv")
        self.assertEqual([r[ID] for r in gold], ["s00"])
        self.assertNotIn("s01", {r[ID] for r in weak})
        duplicate = [r for r in weak if r[ID] in ("s02", "s03")]
        self.assertEqual(duplicate[0]["fold"], duplicate[1]["fold"])
        self.assertEqual(duplicate[0]["group_id"], duplicate[1]["group_id"])
        self.assertFalse(summary["patient_independence_verified"])
        prepare(self.train_path, self.weak_path, self.provenance, self.root / "out2")
        self.assertEqual((self.root / "out" / "weak.csv").read_bytes(), (self.root / "out2" / "weak.csv").read_bytes())

    def test_transitive_explicit_groups(self):
        from rsna_knee.contracts import read_csv

        groups = self.root / "groups.csv"
        # report links s02-s03; supplied mapping links s03-s04.
        write_csv(
            groups,
            (ID, "group_id"),
            [{ID: r[ID], "group_id": "same-patient" if i in (3, 4) else f"p{i}"} for i, r in enumerate(self.rows)],
        )
        prepare(self.train_path, self.weak_path, self.provenance, self.root / "out", groups_path=groups)
        _, weak = read_csv(self.root / "out" / "weak.csv")
        linked = [r for r in weak if r[ID] in ("s02", "s03", "s04")]
        self.assertEqual(len({r["group_id"] for r in linked}), 1)
        self.assertEqual(len({r["fold"] for r in linked}), 1)

    def test_empty_label_template_is_not_training_labels(self):
        label_template(self.train_path, self.weak_path)
        with self.assertRaisesRegex(ValueError, "No usable weak"):
            prepare(self.train_path, self.weak_path, self.provenance, self.root / "out")

    def test_unknown_id_and_provenance_are_rejected(self):
        write_csv(self.weak_path, (ID, *TARGETS), [{ID: "alien", **{t: 0.5 for t in TARGETS}}])
        with self.assertRaisesRegex(ValueError, "unknown"):
            prepare(self.train_path, self.weak_path, self.provenance, self.root / "out")
        write_csv(self.weak_path, (ID, *TARGETS), [{ID: r[ID], **{t: 0.5 for t in TARGETS}} for r in self.rows])
        self.provenance.write_text(json.dumps({"source_url": "TODO"}))
        with self.assertRaisesRegex(ValueError, "provenance"):
            prepare(self.train_path, self.weak_path, self.provenance, self.root / "out")


if __name__ == "__main__":
    unittest.main()
