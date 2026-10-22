import importlib.util
import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsna_knee.contracts import ID, TARGETS, write_csv
from rsna_knee.ensemble_diversity import compare_target, correlation, pair_transitions, ranks
from rsna_knee.metrics import binary_auc

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("diversity_cli", ROOT / "scripts/analyze_native_ensemble_diversity.py")
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


class DiversityMathTests(unittest.TestCase):
    def test_handworked_tie_correction_half_units(self):
        result = pair_transitions([1, 1, 0, 0], [.7, .2, .3, .2], [.3, .6, .3, .1])
        self.assertEqual(result["pairs"], 4)
        self.assertEqual((result["improved"], result["worsened"]), (2, 1))
        self.assertEqual(result["auc_delta"], .25)

    def test_exact_pair_auc_identity_random_with_ties(self):
        rng = random.Random(82)
        y = [0] * 13 + [1] * 11
        for _ in range(20):
            a, b = ([rng.randrange(5) / 4 for _ in y] for _ in range(2))
            result = pair_transitions(y, a, b)
            self.assertAlmostEqual(result["auc_delta"], binary_auc(y, b) - binary_auc(y, a), places=14)
            self.assertEqual(sum(map(sum, result["transition_counts"])), 143)

    def test_mean_improves_different_error_pairs(self):
        result = compare_target([1, 1, 0, 0], [.4, 1., .5, 0.], [1., .4, 0., .5])
        self.assertEqual(result["auc"]["left"], .75)
        self.assertEqual(result["auc"]["right"], .75)
        self.assertEqual(result["auc"]["fixed_FP32_probability_mean50"], 1.)
        self.assertEqual(result["pair_transitions"]["both_incorrect"], 0)

    def test_identical_predictions_do_not_improve(self):
        result = compare_target([1, 0, 1, 0], [.3, .7, .4, .2], [.3, .7, .4, .2])
        self.assertAlmostEqual(result["rank_correlation"], 1.)
        self.assertAlmostEqual(result["residual_correlation"], 1.)
        self.assertEqual(result["mean50_transitions_from_left"]["auc_delta"], 0.)

    def test_constant_and_single_class_explicit_undefined(self):
        result = compare_target([1, 1], [.5, .5], [.1, .1])
        self.assertIsNone(result["rank_correlation"])
        self.assertIsNone(result["auc"]["left"])
        self.assertIsNone(result["pair_transitions"]["auc_delta"])
        self.assertEqual(result["pair_transitions"]["pairs"], 0)

    def test_empty_binary_support_preserves_missingness(self):
        result = compare_target([], [], [])
        self.assertEqual(result["binary_observed"], 0)
        self.assertIsNone(result["auc"])

    def test_tie_ranks_and_monotonic_invariance(self):
        self.assertEqual(ranks([.2, .1, .2]), [2.5, 1., 2.5])
        self.assertAlmostEqual(correlation(ranks([1, 1, 3, 4]), ranks([5, 5, 8, 19])), 1.)

    def test_nonfinite_or_soft_pairs_rejected(self):
        for labels, a, b in (([.5, 0], [.2, .3], [.2, .3]), ([1, 0], [math.nan, .2], [.3, .4])):
            with self.assertRaises(ValueError):
                pair_transitions(labels, a, b)


class DiversityBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def run_fixture(self, name, fold=0, status="complete", augmentation=False):
        directory = self.root / name
        directory.mkdir()
        history = [{"epoch": epoch, "weak_bce": .3} for epoch in range(1, 13)]
        value = {"status": status, "fold": fold, "targets": list(TARGETS), "seed": 1,
                 "selection_data": "weak_validation", "selection_rule": "bce", "history": history,
                 "gold_used_for_training_or_selection": False, "gold_evaluated": False,
                 "selected": dict(history[0]), "audit": {"manifest_sha256": {}, "pixel_index_sha256": "same"}}
        if augmentation:
            value["training_augmentation"] = {"gamma": .85}
        (directory / "run.json").write_text(json.dumps(value), encoding="utf-8")
        return directory

    def test_incomplete_second_arm_refused_before_any_prediction_or_teacher_read(self):
        left, right = self.run_fixture("a"), self.run_fixture("b", status="running")
        with patch.object(CLI, "prediction_csv", side_effect=AssertionError("payload accessed")), \
                patch.object(CLI, "weak_partition", side_effect=AssertionError("teacher accessed")):
            with self.assertRaisesRegex(ValueError, "Both runs must be complete"):
                CLI.analyze(left, right, self.root / "weak.csv", "selected")

    def test_crossfold_and_outer2_refused_before_payload(self):
        left, right = self.run_fixture("a"), self.run_fixture("b", fold=1)
        with patch.object(CLI, "prediction_csv", side_effect=AssertionError("payload accessed")):
            with self.assertRaisesRegex(ValueError, "Same validation fold"):
                CLI.analyze(left, right, self.root / "weak.csv", "selected")
        outer = self.run_fixture("outer2", fold=2)
        with self.assertRaisesRegex(ValueError, "outer2 remains sealed"):
            CLI.read_run(outer, "fixed12")

    def test_augmented_candidate_requires_fixed12(self):
        run = self.run_fixture("gamma", augmentation=True)
        with self.assertRaisesRegex(ValueError, "fixed E12 only"):
            CLI.read_run(run, "selected")

    def test_non_requested_labels_never_interpreted_missing_and_soft_preserved(self):
        path = self.root / "weak.csv"
        rows = [{ID: "a", **dict.fromkeys(TARGETS, ""), "source": "report_weak", "group_id": "a", "fold": "0"},
                {ID: "sealed", **dict.fromkeys(TARGETS, "DO_NOT_PARSE"), "source": "report_weak",
                 "group_id": "b", "fold": "2"}]
        rows[0]["ACL"] = ".4"
        write_csv(path, CLI.WEAK_COLUMNS, rows)
        truth, counts, _ = CLI.weak_partition(path, 0)
        self.assertEqual(truth["a"]["ACL"], .4)
        self.assertIsNone(truth["a"]["MCL"])
        self.assertNotIn("sealed", truth)
        self.assertEqual(counts, {0: 1, 2: 1})

    def test_supplied_group_leakage_rejected(self):
        path = self.root / "weak.csv"
        rows = [{ID: key, **dict.fromkeys(TARGETS, ""), "source": "report_weak", "group_id": "same", "fold": fold}
                for key, fold in (("a", "0"), ("b", "1"))]
        write_csv(path, CLI.WEAK_COLUMNS, rows)
        with self.assertRaisesRegex(ValueError, "group spans folds"):
            CLI.weak_partition(path, 0)


if __name__ == "__main__":
    unittest.main()
