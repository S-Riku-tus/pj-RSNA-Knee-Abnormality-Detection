"""CPU-only tests of paired group sampling, AUC ties and CSV privacy; standard library runner."""

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rsna_knee.contracts import ID, SUBMISSION_COLUMNS, TARGETS, dump_json, write_csv
from rsna_knee.metrics import binary_auc

SPEC = importlib.util.spec_from_file_location(
    "weak_bootstrap", Path(__file__).resolve().parents[1] / "scripts/bootstrap_weak_comparison.py"
)
BOOTSTRAP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BOOTSTRAP)


class WeakBootstrapTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.rows = [
            {
                ID: f"private-study-{index}",
                **{target: str(label) for target in TARGETS},
                "fold": "0",
                "group_id": f"private-group-{group}",
                "source": "report_weak",
            }
            for index, (label, group) in enumerate(((0, 0), (1, 0), (1, 1), (0, 2)))
        ]
        self.truth = {row[ID]: row for row in self.rows}
        self.left = {
            row[ID]: {ID: row[ID], **{target: score for target in TARGETS}}
            for row, score in zip(self.rows, (0.1, 0.7, 0.9, 0.2))
        }
        self.right = {
            row[ID]: {ID: row[ID], **{target: score for target in TARGETS}}
            for row, score in zip(self.rows, (0.8, 0.6, 0.9, 0.1))
        }
        dump_json(self.root / "manifest.json", {"seed": 123, "n_folds": 2, "inputs_sha256": {"train": "artificial"}})
        write_csv(self.root / "weak.csv", (*SUBMISSION_COLUMNS, "fold", "group_id", "source"), self.rows)
        self.left_path, self.right_path = self.root / "left.csv", self.root / "right.csv"
        write_csv(self.left_path, SUBMISSION_COLUMNS, self.left.values())
        write_csv(self.right_path, SUBMISSION_COLUMNS, self.right.values())
        self.output = self.root / "comparison.json"

    def compare(self, **overrides):
        options = {"seed": 123, "repeats": 30, "rng": "python_random"}
        options.update(overrides)
        return BOOTSTRAP.compare(self.root, 0, self.left_path, self.right_path, self.output, **options)

    def test_weighted_auc_ties_match_expanded_studies(self):
        self.assertEqual(BOOTSTRAP.weighted_auc([[[0, 1], [2, 3]]], [1] * 4), 0.5)
        values = [(0.2, 0, 0), (0.2, 1, 0), (0.5, 0, 1), (0.8, 1, 2)]
        weights = [2, 0, 3]
        buckets = [
            [[group for score, label, group in values if score == tied_score and label == y] for y in (0, 1)]
            for tied_score in (0.2, 0.5, 0.8)
        ]
        repeated = [(score, label) for score, label, group in values for _ in range(weights[group])]
        expected = binary_auc([label for score, label in repeated], [score for score, label in repeated])
        self.assertTrue(math.isclose(BOOTSTRAP.weighted_auc(buckets, weights), expected, abs_tol=1e-12))
        self.assertIsNone(BOOTSTRAP.weighted_auc(buckets, [0, 0, 3]))

    def test_shared_group_weights_keep_all_studies_and_pair_models(self):
        generator = mock.Mock()
        generator.randrange.side_effect = [0, 0, 1]
        with mock.patch.object(BOOTSTRAP.random, "Random", return_value=generator):
            result = BOOTSTRAP.bootstrap(self.truth, self.left, self.right, seed=9, repeats=1, rng="python_random")
        self.assertEqual(generator.randrange.call_args_list, [mock.call(3)] * 3)
        self.assertEqual(result["supplied_groups"], 3)
        self.assertEqual(result["studies"], 4)
        self.assertAlmostEqual(result["macro_auc_12_difference"]["point"], -0.25)
        self.assertAlmostEqual(result["macro_auc_12_difference"]["percentile_95_interval"][0], -2 / 3)

    def test_missing_masks_and_undefined_classes_keep_fixed_macro_denominators(self):
        for row in self.truth.values():
            row["Synovitis"] = "1"
        self.rows[0]["ACL"] = ""
        result = BOOTSTRAP.bootstrap(self.truth, self.left, self.right, seed=123, repeats=80, rng="python_random")
        self.assertIsNone(result["left_macro_auc_12"])
        self.assertIsNone(result["macro_auc_12_difference"]["point"])
        self.assertEqual(result["macro_auc_12_difference"]["undefined_resamples"], 80)
        self.assertIsNotNone(result["macro_auc_11_without_synovitis_difference"]["point"])
        self.assertEqual(result["per_target"]["ACL"]["observed"], 3)
        self.assertEqual(result["per_target"]["ACL"]["missing"], 1)
        self.assertGreater(result["macro_auc_11_without_synovitis_difference"]["undefined_resamples"], 0)

    def test_csv_validation_privacy_reproducibility_and_no_gold_access(self):
        original_open = Path.open

        def guarded_open(path, *args, **kwargs):
            if path.name == "gold.csv":
                self.fail("gold must not be opened")
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", guarded_open):
            result = self.compare(left_epoch=1, right_epoch=2)
        self.assertFalse(result["validation"]["gold_read"])
        self.assertFalse(result["environment"]["numpy_used"])
        self.assertEqual(result["algorithm"]["macro_auc_12_denominator"], 12)
        self.assertEqual(result["algorithm"]["auxiliary_denominator"], 11)
        text = self.output.read_text()
        self.assertNotIn("private-study", text)
        self.assertNotIn("private-group", text)
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.compare()
        again = BOOTSTRAP.compare(
            self.root,
            0,
            self.left_path,
            self.right_path,
            self.root / "again.json",
            seed=123,
            repeats=30,
            rng="python_random",
        )
        self.assertEqual(result["per_target"], again["per_target"])
        self.assertEqual(result["inputs_sha256"], again["inputs_sha256"])

    def test_invalid_csv_labels_scores_and_cross_fold_groups_rejected_before_output(self):
        original_weak = (self.root / "weak.csv").read_bytes()
        original_left = self.left_path.read_bytes()
        for value, message in (("0.2", "binary"), ("inf", "finite probability")):
            with self.subTest(label=value):
                self.rows[0]["ACL"] = value
                write_csv(self.root / "weak.csv", (*SUBMISSION_COLUMNS, "fold", "group_id", "source"), self.rows)
                with self.assertRaisesRegex(ValueError, message):
                    self.compare()
                self.assertFalse(self.output.exists())
        (self.root / "weak.csv").write_bytes(original_weak)
        self.left["private-study-0"]["ACL"] = "nan"
        write_csv(self.left_path, SUBMISSION_COLUMNS, self.left.values())
        with self.assertRaisesRegex(ValueError, "finite probability"):
            self.compare()
        self.left_path.write_bytes(original_left)
        rows = list(self.right.values())
        rows[0][ID] = "outside-validation"
        write_csv(self.right_path, SUBMISSION_COLUMNS, rows)
        with self.assertRaisesRegex(ValueError, "study set differs"):
            self.compare()
        self.rows[0]["ACL"] = "0"
        self.rows[1]["fold"] = "1"
        write_csv(self.root / "weak.csv", (*SUBMISSION_COLUMNS, "fold", "group_id", "source"), self.rows)
        with self.assertRaisesRegex(ValueError, "group crosses folds"):
            self.compare()
        self.assertFalse(self.output.exists())

    def test_input_change_during_sampling_prevents_publication(self):
        original = BOOTSTRAP.bootstrap

        for input_path in (self.root / "manifest.json", self.root / "weak.csv", self.left_path, self.right_path):
            with self.subTest(input=input_path.name):

                def mutate_input(*args, **kwargs):
                    result = original(*args, **kwargs)
                    with input_path.open("a", encoding="utf-8") as stream:
                        stream.write("\n")
                    return result

                with mock.patch.object(BOOTSTRAP, "bootstrap", side_effect=mutate_input):
                    with self.assertRaisesRegex(ValueError, "inputs changed"):
                        self.compare()
                self.assertFalse(self.output.exists())
                self.assertFalse(list(self.root.glob(".*.tmp")))

    def test_fsync_failure_never_exposes_partial_json_or_leaves_temporary(self):
        with mock.patch.object(BOOTSTRAP.os, "fsync", side_effect=OSError("synthetic write failure")):
            with self.assertRaisesRegex(OSError, "synthetic write failure"):
                self.compare()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".*.tmp")))

    def test_atomic_publish_race_preserves_competing_output(self):
        original_link = BOOTSTRAP.os.link
        competing = '{"preserve": "other writer"}\n'

        def publish_competing_output(source, destination):
            Path(destination).write_text(competing, encoding="utf-8")
            original_link(source, destination)

        with mock.patch.object(BOOTSTRAP.os, "link", side_effect=publish_competing_output):
            with self.assertRaisesRegex(ValueError, "already exists"):
                self.compare()
        self.assertEqual(self.output.read_text(encoding="utf-8"), competing)
        self.assertFalse(list(self.root.glob(".*.tmp")))


if __name__ == "__main__":
    unittest.main()
