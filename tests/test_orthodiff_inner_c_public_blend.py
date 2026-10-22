"""Artificial CSV fixtures; independent struct/Fraction arithmetic oracle."""
import copy
import math
import struct
import unittest
from fractions import Fraction

from rsna_knee import orthodiff_inner_c_public_blend as B


def f32(value):
    return struct.unpack('<f', struct.pack('<f', float(value)))[0]


def bits(value):
    return struct.pack('<f', float(value))


def parents(n=17):
    return [{B.ID: 'uid' + str(i), **{name: format((i * 19 + j * 23 + 17) % 997 / 997, '.17g')
             for j, name in enumerate(B.TARGETS)}} for i in range(n)]


def heads(rows):
    return {row[B.ID]: [((i * 7 + j * 11) % 19) / 19 for j in range(12)] for i, row in enumerate(rows)}


class InnerCPublicBlendTests(unittest.TestCase):
    def test_all_active_cells_match_independent_tie_rank_and_fp32_product_oracle(self):
        parent = parents()
        head = heads(parent[:13])
        before = copy.deepcopy(parent)
        actual, receipt = B.overlay_rows(parent, head, supported_targets=B.TARGETS)
        self.assertEqual(parent, before)
        self.assertEqual(receipt['active_cells'], 130)
        for j, target in enumerate(B.TARGETS):
            values = [f32(head[row[B.ID]][j]) for row in parent[:13]]
            for i, row in enumerate(parent):
                if i >= 13 or target not in B.FIXED10:
                    self.assertEqual(actual[i][target], row[target])
                    continue
                less = sum(v < values[i] for v in values)
                ties = sum(v == values[i] for v in values)
                rank = f32(float(Fraction(2 * less + ties + 1, 2 * len(values))))
                left = f32(f32(.8) * f32(row[target]))
                right = f32(f32(.2) * rank)
                expected = f32(left + right)
                self.assertEqual(bits(actual[i][target]), bits(expected))
        self.assertFalse(receipt['parent_or_output_reranked'])
        self.assertFalse(receipt['public_submission_contract_qualified'])

    def test_failed_excluded_and_unsupported_original_text_is_preserved(self):
        parent = parents(6)
        parent[0]['ACL'] = '0.10000000000000000555'
        head = heads(parent[1:5])
        supported = set(B.TARGETS) - {"Baker's"}
        actual, receipt = B.overlay_rows(parent, head, supported_targets=supported)
        self.assertEqual([row[B.ID] for row in actual], [row[B.ID] for row in parent])
        self.assertEqual(actual[0], parent[0])
        self.assertEqual(actual[-1], parent[-1])
        for name in ['MCL', 'Synovitis', "Baker's"]:
            self.assertEqual([row[name] for row in actual], [row[name] for row in parent])
        self.assertEqual(receipt['feature_successes'], 4)
        self.assertEqual(receipt['feature_failures'], 2)

    def test_zero_one_and_fp32_constant_successes_preserve_original(self):
        parent = parents(3)
        for count in [0, 1]:
            actual, receipt = B.overlay_rows(parent, heads(parent[:count]), supported_targets=B.TARGETS)
            self.assertEqual(actual, parent)
            self.assertEqual(receipt['active_cells'], 0)
        head = {row[B.ID]: [.5 + i * 1e-10] * 12 for i, row in enumerate(parent)}
        actual, receipt = B.overlay_rows(parent, head, supported_targets=B.TARGETS)
        self.assertEqual(actual, parent)
        self.assertEqual(receipt['active_cells'], 0)

    def test_rank_ties_are_identified_after_fp32_quantization(self):
        ranked = B.rank_fp32([.5, .1, .5 + 1e-10, 1.])
        self.assertEqual([float(x) for x in ranked], [.625, .25, .625, 1.])

    def test_bad_csv_alignment_or_probabilities_fail_closed(self):
        parent = parents(3)
        head = heads(parent)
        cases = []
        cases.append(([], head, B.TARGETS))
        cases.append(([parent[0], parent[0]], head, B.TARGETS))
        missing = copy.deepcopy(parent)
        del missing[0]['ACL']
        cases.append((missing, head, B.TARGETS))
        extra = copy.deepcopy(parent)
        extra[0]['Report'] = 'not accepted'
        cases.append((extra, head, B.TARGETS))
        bad_uid = dict(head)
        bad_uid['unknown'] = [0.] * 12
        cases.append((parent, bad_uid, B.TARGETS))
        short = dict(head)
        short['uid0'] = [0.] * 11
        cases.append((parent, short, B.TARGETS))
        cases.append((parent, head, ['unknown']))
        for bad in [math.nan, math.inf, -.1, 1.1, '', None, True]:
            altered = copy.deepcopy(head)
            altered['uid0'][0] = bad
            cases.append((parent, altered, B.TARGETS))
        for rows, scores, supported in cases:
            with self.subTest(rows=len(rows), supported=supported), self.assertRaises(ValueError):
                B.overlay_rows(rows, scores, supported_targets=supported)


if __name__ == '__main__':
    unittest.main()
