"""Small analytical CPU checks; no image/model/Gold/network access."""
import os
import sys
import unittest
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np  # noqa: E402 -- Set BLAS thread caps before importing NumPy.

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from cpu_teacher_screen import normalization, pool_planes, rank_scores, ridge_fit  # noqa: E402


class Tests(unittest.TestCase):
    def test_plane_pool_uses_observed_windows_and_presence(self):
        x = np.full((3, 2, 384), np.nan)
        x[0, 0], x[0, 1], x[1, 0] = 1, 3, 5
        mask = np.array([[True, True], [True, False], [False, False]])
        planes = np.array([0, 0, 2])
        pooled = pool_planes(x, mask, planes)
        self.assertEqual(pooled.shape, (1540,))
        np.testing.assert_array_equal(pooled[:384], np.full(384, 3.))
        np.testing.assert_array_equal(pooled[384:1536], np.zeros(1152))
        np.testing.assert_array_equal(pooled[1536:], [1, 0, 0, 0])
        with self.assertRaises(ValueError):
            pool_planes(x, np.zeros_like(mask), planes)
        x[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            pool_planes(x, mask, planes)

    def test_ridge_matches_analytical_penalized_slope_and_free_intercept(self):
        x = np.array([[-1.], [0.], [1.]])
        y = np.array([[0., 0.], [.5, 1.], [1., 2.]])
        beta, bias = ridge_fit(x, y, .1)
        np.testing.assert_allclose(beta, [[10 / 23, 20 / 23]], atol=1e-14, rtol=0)
        np.testing.assert_allclose(bias, [.5, 1.], atol=1e-14, rtol=0)
        for badx, bady in [(x, np.array([0., np.nan, 1.])), (np.array([[-1.], [np.inf], [1.]]), y)]:
            with self.assertRaises(ValueError):
                ridge_fit(badx, bady)

    def test_normalization_is_train_only_and_handles_constant_feature(self):
        train = np.array([[0., 2.], [2., 2.]])
        before = train.copy()
        mean, scale = normalization(train)
        np.testing.assert_array_equal(mean, [1, 2])
        np.testing.assert_array_equal(scale, [1, 1])
        np.testing.assert_array_equal(train, before)
        np.testing.assert_array_equal((np.array([[1000., 1000.]]) - mean) / scale, [[999., 998.]])

    def test_unbounded_linear_scores_have_exact_average_tie_submission_ranks(self):
        self.assertEqual(rank_scores([-9, 1, 1, 23]), [.25, .625, .625, 1])
        with self.assertRaises(ValueError):
            rank_scores([float('nan')])
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main()
