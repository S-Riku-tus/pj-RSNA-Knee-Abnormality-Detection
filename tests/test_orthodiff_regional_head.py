"""Artificial1024 model and source/review contracts; no real feature/label fit."""
import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/fit_orthodiffusion_regional_head.py"
spec = importlib.util.spec_from_file_location("regional_head_test", SCRIPT)
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


class RegionalHeadTests(unittest.TestCase):
    def test_policy_exact_parent_population_and_same_math_delegation(self):
        self.assertEqual(C.H.pin(C.POLICY)["sha256"], C.POLICY_SHA)
        policy = C.H.read_json(C.POLICY)
        self.assertEqual(policy["feature_contract"]["feature_dim"], 1024)
        self.assertFalse(policy["feature_contract"]["FD_availability_extension"])
        self.assertEqual([policy["folds"][x]["successful_valid"] for x in ("0", "1")], [594, 637])
        self.assertEqual([policy["folds"][x]["successful_train"] for x in ("0", "1")], [2469, 2426])
        for relative, expected in policy["immutable_old_source_pins"].items():
            C.H.check_pin(ROOT / relative, expected)
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "H"]
        for name in ("fit_numeric", "predict_numeric", "rank_blend", "evaluate"):
            self.assertEqual(calls.count(name), 1)
        with self.assertRaises(ValueError):
            C.prepare_sources(C.POLICY, "0" * 64, "0" * 64)

    def test_artificial1024_numeric_model_roundtrip_and_last_region_used(self):
        heads = []
        for i, target in enumerate(C.TARGETS):
            coefficient = [0.] * 1024
            coefficient[0], coefficient[1023] = .25, -.5
            heads.append({"target": target, "status": "qualified", "coef": coefficient, "intercept": i / 16})
        model = {"schema_version": "orthodiff_weak_head_numeric_model_v1", "feature_dim": 1024,
                 "targets": list(C.TARGETS), "scaler": {"mean": [0.] * 1024, "scale": [1.] * 1024}, "heads": heads}
        features = np.zeros((3, 1024), dtype=np.float64)
        features[:, 0], features[:, 1023] = [0., 1., 2.], [0., 2., 4.]
        result = C.H.predict_numeric(model, features)
        self.assertEqual(result.shape, (3, len(C.TARGETS)))
        expected = 1 / (1 + np.exp(-np.asarray([0., -.75, -1.5])))
        np.testing.assert_array_equal(result[:, 0].astype(np.float32), expected.astype(np.float32))
        np.testing.assert_array_equal(C.H.predict_numeric(json.loads(json.dumps(model)), features), result)
        with self.assertRaises(ValueError):
            C.H.predict_numeric(model, features[:, :256])

    def test_same_population_secondary_comparison_does_not_change_primary_gate(self):
        labels = np.tile(np.asarray([0., 1., 0., 1.])[:, None], (1, len(C.TARGETS)))
        scores = np.tile(np.asarray([.1, .9, .2, .8])[:, None], (1, len(C.TARGETS)))
        success = np.asarray([True, False, True, True])
        head = scores.copy()
        head[~success] = np.nan
        result = C.secondary_comparison(labels, scores, scores, head, success, scores, head, success)
        self.assertTrue(result["report_only_not_a_gate"])
        self.assertTrue(result["pooling_only_same_success_UIDs_and_preprocessing"])
        self.assertEqual(result["fixed10_full_holdout_delta_auc_vs_strict_PD"], 0.)
        self.assertTrue(all(row["common_success_UID_count"] == 3 for row in result["per_target"].values()))
        self.assertTrue(C.joint_gate(.001, .001, True))
        self.assertFalse(C.joint_gate(.001, 0., True))
        self.assertFalse(C.joint_gate(.001, .001, False))

    def test_old_global_or_FD_review_cannot_authorize_regional_fold1(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "regional-fold0"
            run_dir.mkdir()
            pins = {"artificial.py": {"bytes": 1, "sha256": "a" * 64}}
            bundle_sha = "b" * 64
            run = {"schema_version": C.RUN_SCHEMA, "status": "complete", "fold": 0, "seed": 20261002,
                "engineering_qualified": True, "continuation_screen_passed": True,
                "inputs": {"policy": {"sha256": C.POLICY_SHA}, "qualified_regional_bundle": {"sha256": bundle_sha}, "source_pins": pins}}
            evaluation = {"continuation_screen_passed": True, "fixed10_delta": .001}
            names = ("run.json", "evaluation.json", "numeric-model.json", "cv-rankmix20.csv", "head-valid-success.csv")

            def seal_review():
                (run_dir / "run.json").write_text(json.dumps(run))
                (run_dir / "evaluation.json").write_text(json.dumps(evaluation))
                for name in names[2:]:
                    (run_dir / name).write_text("artificial")
                review = Path(temporary) / "root-review.json"
                review.write_text(json.dumps({"root_review_passed": True, "inputs": {name: C.H.pin(run_dir / name) for name in names}}))
                return SimpleNamespace(fold0_run=run_dir, fold0_review=review,
                    expected_fold0_review_sha256=C.H.pin(review)["sha256"], expected_bundle_sha256=bundle_sha)

            self.assertTrue(C.verify_new_fold0(seal_review(), pins)["approved"])
            for wrong_policy in (C.H.POLICY_SHA, "105bda1d86b677925356249d913a69abd9c03d4c950c86fc96589e8a79787abe"):
                run["inputs"]["policy"]["sha256"] = wrong_policy
                with self.assertRaises(ValueError):
                    C.verify_new_fold0(seal_review(), pins)
            run["inputs"]["policy"]["sha256"] = C.POLICY_SHA
            evaluation["fixed10_delta"] = -.001
            with self.assertRaises(ValueError):
                C.verify_new_fold0(seal_review(), pins)


if __name__ == "__main__":
    unittest.main()
