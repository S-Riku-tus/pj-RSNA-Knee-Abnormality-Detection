"""Artificial CSV/review/mask checks; never fit or evaluate real weak examples."""
import ast
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/fit_orthodiffusion_fluid_composite_head.py"
spec = importlib.util.spec_from_file_location("fluid_composite_head_test", SCRIPT)
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


class CompositeHeadTests(unittest.TestCase):
    def test_fixed_sources_policy_and_unmodified_math_delegation(self):
        loader_sha = C.H.pin(C.LOADER)["sha256"]
        policy, _, _, _ = C.prepare_sources(C.POLICY, C.POLICY_SHA, loader_sha)
        self.assertEqual(policy["seed"], 20261002)
        self.assertEqual([policy["folds"][x]["original_valid"] for x in ("0", "1")], [821, 891])
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "H"]
        for name in ("fit_numeric", "rank_blend", "evaluate", "predict_numeric"):
            self.assertEqual(calls.count(name), 1)
        with self.assertRaises(ValueError):
            C.prepare_sources(C.POLICY, C.POLICY_SHA, "0" * 64)
        with self.assertRaises(ValueError):
            C.prepare_sources(C.POLICY, "0" * 64, loader_sha)

    def test_UID_alignment_missing_head_and_unknown_UID_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow([C.ID, *C.TARGETS])
                writer.writerow(["b", *([.25] * len(C.TARGETS))])
                writer.writerow(["a", *([""] + [.75] * (len(C.TARGETS) - 1))])
            reference = {"path": str(path), **C.H.pin(path)}
            values, present = C.read_old_scores(reference, ["a", "b", "c"], allow_subset=True)
            self.assertEqual(present.tolist(), [True, True, False])
            self.assertTrue(np.isnan(values[0, 0]) and np.isnan(values[2]).all())
            self.assertEqual(values[0, 1], .75)
            with self.assertRaises(ValueError):
                C.read_old_scores(reference, ["a", "b", "c"])
            with self.assertRaises(ValueError):
                C.read_old_scores(reference, ["a", "c"], allow_subset=True)

    def test_secondary_uses_same_common_UID_target_mask(self):
        count = len(C.TARGETS)
        labels = np.tile(np.asarray([0, 1, 0, 1, np.nan, 1])[:, None], (1, count))
        baseline = np.tile(np.linspace(.1, .9, 6)[:, None], (1, count))
        old_head, new_head = baseline.copy(), baseline.copy()
        old_head[2:] = np.nan
        new_head[0] = np.nan
        old_head[1, 1] = np.nan
        report = C.secondary_comparison(labels, baseline, baseline, new_head, [False, True, True, True, True, True],
                                        baseline, old_head, [True, True, False, False, False, False])
        self.assertTrue(report["report_only_not_a_gate"])
        self.assertEqual(report["fixed10_full_holdout_delta_auc_vs_strict_PD"], 0)
        self.assertEqual(report["per_target"][C.TARGETS[0]]["common_success_UID_count"], 1)
        self.assertEqual(report["per_target"][C.TARGETS[1]]["common_success_UID_count"], 0)
        for target in C.TARGETS:
            observed = {value["observed"] for value in report["per_target"][target]["common_success_same_mask_metrics"].values()}
            self.assertEqual(len(observed), 1)

    def test_only_new_review_positive_same_sources_can_authorize_fold1(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "new-fd-fold0"
            run_dir.mkdir()
            pins = {"synthetic-source.py": {"bytes": 4, "sha256": "a" * 64}}
            bundle_sha = "b" * 64
            run = {"schema_version": C.RUN_SCHEMA, "status": "complete", "fold": 0, "seed": 20261002,
                   "engineering_qualified": True, "continuation_screen_passed": True,
                   "inputs": {"policy": {"sha256": C.POLICY_SHA},
                              "qualified_composite_bundle": {"sha256": bundle_sha}, "source_pins": pins}}
            evaluation = {"continuation_screen_passed": True, "fixed10_delta": .001}
            names = ("run.json", "evaluation.json", "numeric-model.json", "cv-rankmix20.csv", "head-valid-success.csv")

            def seal_review():
                (run_dir / "run.json").write_text(json.dumps(run))
                (run_dir / "evaluation.json").write_text(json.dumps(evaluation))
                for name in names[2:]:
                    (run_dir / name).write_text("synthetic")
                review = Path(temporary) / "root-review.json"
                review.write_text(json.dumps({"root_review_passed": True, "inputs": {name: C.H.pin(run_dir / name) for name in names}}))
                return SimpleNamespace(fold0_run=run_dir, fold0_review=review,
                    expected_fold0_review_sha256=C.H.pin(review)["sha256"], expected_bundle_sha256=bundle_sha)

            self.assertTrue(C.verify_new_fold0(seal_review(), pins)["approved"])
            evaluation["fixed10_delta"] = 0
            with self.assertRaises(ValueError):
                C.verify_new_fold0(seal_review(), pins)
            evaluation["fixed10_delta"] = .001
            run["inputs"]["policy"]["sha256"] = C.H.POLICY_SHA  # Old strict-PD root approval is insufficient.
            with self.assertRaises(ValueError):
                C.verify_new_fold0(seal_review(), pins)
            run["inputs"]["policy"]["sha256"] = C.POLICY_SHA
            with self.assertRaises(ValueError):
                C.verify_new_fold0(seal_review(), {})
            self.assertTrue(C.joint_gate(.001, .001, True))
            self.assertFalse(C.joint_gate(.001, -.001, True))
            self.assertFalse(C.joint_gate(.001, .001, False))


if __name__ == "__main__":
    unittest.main()
