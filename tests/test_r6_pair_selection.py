import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_r6_pair_probe import choose


class Selection(unittest.TestCase):
    def test_worst_bank_beats_mean_and_tie_order(self):
        d = {'tta_off': {'0': .1, '1': .002}, 'tta_quarter': {'0': .003, '1': .004},
             'no_calibration': {'0': .003, '1': .005}}
        self.assertEqual(choose(d), 'tta_quarter')
        d['tta_off'] = {'0': .003, '1': .006}
        self.assertEqual(choose(d), 'tta_off')

    def test_both_banks_must_be_strictly_positive(self):
        self.assertIsNone(choose({'tta_off': {'0': 1., '1': 0.}, 'tta_quarter': {'0': -.001, '1': 1.},
                                  'no_calibration': {'0': 0., '1': 0.}}))

    def test_missing_bank_or_nonfinite_refused(self):
        for bad in ({'0': 1.}, {'0': 1., '1': math.nan}):
            with self.assertRaises(ValueError):
                choose({'tta_off': bad, 'tta_quarter': {'0': 1., '1': 1.}, 'no_calibration': {'0': 1., '1': 1.}})


if __name__ == '__main__':
    unittest.main()
