"""Minimal artificial numeric and qualification boundaries; never fits."""
import copy
import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("independent_new_head_audit", ROOT / "scripts/audit_orthodiffusion_new_head.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


class IndependentNewHeadAudit(unittest.TestCase):
    def test_ties_success_only_rank_fp32_products_and_missing_unsupported(self):
        raw = np.tile(np.asarray([.1, .2, .2, .9])[:, None], (1, 12))
        predictions = np.full((3, 12), np.nan)
        predictions[:, 0] = [.8, .2, .2]
        heads = [{"status": "qualified" if i == 0 else "insufficient_training_classes"} for i in range(12)]
        baseline, mixture = A.rank_mix(raw, predictions, [True, False, True, True], heads)
        self.assertEqual(baseline[:, 0].tolist(), [.25, .625, .625, 1.])
        self.assertEqual(mixture[:, 0].view(np.uint32).tolist(), [0x3ECCCCCD, 0x3F200000, 0x3F19999A, 0x3F666667])
        self.assertTrue(np.array_equal(mixture[:, 1:].view(np.uint32), baseline[:, 1:].view(np.uint32)))
        predictions[:, 0] = .3
        _, constant = A.rank_mix(raw, predictions, [True, False, True, True], heads)
        self.assertTrue(np.array_equal(constant.view(np.uint32), baseline.view(np.uint32)))

    def test_scalar_sigmoid_uses_last_dimension_256_and_1024_and_keeps_None_nan(self):
        mean, variance, scale = A.training_moments(np.asarray([[1., 0.], [1., 2.]]))
        self.assertEqual(mean.tolist(), [1., 1.])
        self.assertEqual(variance.tolist(), [0., 1.])
        self.assertEqual(scale.tolist(), [1., 1.])
        for dimension in (256, 1024):
            coef = [0.] * dimension
            coef[-1] = 1.
            model = {"feature_dim": dimension, "scaler": {"mean": [0.] * dimension, "scale": [1.] * dimension},
                     "heads": [{"status": "qualified", "coef": coef, "intercept": 0.},
                               *[{"status": "insufficient_training_classes", "coef": None, "intercept": None} for _ in range(11)]]}
            values = np.zeros((2, dimension))
            values[:, -1] = [-1000., 1000.]
            result = A.scalar_predict(model, values)
            self.assertEqual(result[:, 0].tolist(), [0., 1.])
            self.assertTrue(np.isnan(result[:, 1:]).all())

    def test_wrong_recipe_fold_incomplete_or_unqualified_gate_refuses_before_inputs(self):
        for recipe in ("fluid", "regional"):
            for fold in (0, 1):
                run = {"status": "complete", "schema_version": A.PRODUCERS[recipe]["run_schema"], "fold": fold,
                       "seed": 20261002, "engineering_qualified": True}
                self.assertEqual(A.initial_gate(run, recipe, fold)["dimension"], 256 if recipe == "fluid" else 1024)
                for key, value in (("status", "running"), ("engineering_qualified", False), ("fold", 1 - fold), ("schema_version", "old256_or_other_recipe")):
                    bad = copy.deepcopy(run)
                    bad[key] = value
                    with self.assertRaises(ValueError):
                        A.source_inputs(bad, recipe, fold)


if __name__ == "__main__":
    unittest.main()
