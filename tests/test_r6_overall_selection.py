import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_r6_overall_probe import CHOICES, choose, common_cohort


class Overall(unittest.TestCase):
    def test_only_both_positive_and_largest_worse_bank(self):
        d = {name: {'0': .001, '1': .003} for name in CHOICES}
        self.assertEqual(choose(d), 'tta_off')
        d['crop320'] = {'0': .002, '1': .002}
        self.assertEqual(choose(d), 'crop320')
        d['crop320']['1'] = 0.
        self.assertEqual(choose(d), 'tta_off')

    def test_nonpositive_or_invalid_deltas_do_not_select(self):
        d = {name: {'0': 0., '1': .1} for name in CHOICES}
        self.assertIsNone(choose(d))
        d['crop320']['0'] = math.nan
        with self.assertRaises(ValueError):
            choose(d)

    def test_separate98percent_does_not_imply_paired98percent(self):
        selection = {str(i): {'weak_fold': str(i//100)} for i in range(200)}
        left, right = set(selection), set(selection)
        left.remove('0')
        left.remove('1')
        right.remove('2')
        right.remove('3')
        with self.assertRaises(ValueError):
            common_cohort(selection, left, right)
        common, counts, paired = common_cohort(selection, left, left)
        self.assertEqual(counts, {'0': 100, '1': 100})
        self.assertEqual(paired, {'0': 98, '1': 100})
        self.assertEqual(common, left)


if __name__ == '__main__':
    unittest.main()
