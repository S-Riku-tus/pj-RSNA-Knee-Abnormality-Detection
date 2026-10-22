import itertools
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from isotonic_teacher import apply, fit


class Isotonic(unittest.TestCase):
    def test_matches_independent_exhaustive_monotone_partition_optimum(self):
        # Enumerate every possible constant-block partition, without using PAVA.
        for y in itertools.product((0., 1.), repeat=5):
            best = float('inf')
            for cuts in itertools.product((False, True), repeat=4):
                bounds = [0] + [i+1 for i, cut in enumerate(cuts) if cut] + [5]
                means = [sum(y[a:b])/(b-a) for a, b in zip(bounds[:-1], bounds[1:])]
                if means != sorted(means):
                    continue
                error = sum((y[i]-mean)**2 for a, b, mean in zip(bounds[:-1], bounds[1:], means) for i in range(a, b))
                best = min(best, error)
            curve = fit([0., .25, .5, .75, 1.], y)
            actual = sum((a-b)**2 for a, b in zip(y, curve['probabilities']))
            self.assertAlmostEqual(actual, best, places=13)

    def test_score_ties_weighted_pooling(self):
        curve = fit([.2, .2, .4, .8], [0., 1., 0., 1.])
        self.assertEqual(curve['counts'], [2, 1, 1])
        self.assertEqual(curve['probabilities'], [1/3, 1/3, 1.])
        self.assertEqual(curve['n_observed_training'], 4)

    def test_interpolation_and_fixed_endpoint_clip(self):
        curve = fit([.2, .8], [0., 1.])
        result = apply(curve, [0., .2, .35, .5, .8, 1.]).tolist()
        for actual, wanted in zip(result, [.001, .001, .25, .5, .999, .999]):
            self.assertAlmostEqual(actual, wanted, places=12)

    def test_missing_or_nonbinary_cannot_be_negative_training(self):
        for x, y in [([.1, float('nan')], [0., 1.]), ([.1, .9], [0., float('nan')]), ([.1, .9], [0., .5])]:
            with self.assertRaises(ValueError):
                fit(x, y)
        with self.assertRaises(ValueError):
            apply(fit([.1, .9], [0., 1.]), [float('nan')])


if __name__ == '__main__':
    unittest.main()
