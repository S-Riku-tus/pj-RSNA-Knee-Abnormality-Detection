"""Analytical controls for paired replay and inert frozen-source transport."""
import copy
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_r6_pair_probe as B
import r6_pair_math as M


class Arithmetic(unittest.TestCase):
    def setUp(self):
        self.meta = {'labels': ['MCL', 'Medial OA', 'Lateral OA'], 'class_perm': [0, 2, 1],
                     'state_perm': [0, 2, 1], 'mcl_index': 0, 'mcl_state_cols': [0], 'fluid': {}, 'state': {}}
        self.record = {'z_plain': [[1., 2., 4.]], 'z_mirror_raw': [[8., 10., 20.]],
                       'state_plain': [[.2, .4, .6]], 'state_mirror_raw': [[.9, .8, .6]]}

    def test_swap_weight_and_mcl_negative_control(self):
        for weight, want in [(0., [1., 2., 4.]), (.25, [1., 6.5, 5.5]), (.5, [1., 11., 7.])]:
            z, state = M.mix(self.record, self.meta, weight)
            self.assertEqual(z.tolist(), want)
            self.assertAlmostEqual(state[0], .2, places=6)
        self.assertAlmostEqual(M.replay(self.record, {}, self.meta, .5)[1], 1/(1+math.exp(-11)), places=12)

    def test_missing_volume_differs_from_measured_zero(self):
        meta = copy.deepcopy(self.meta)
        meta['fluid'] = {'Medial OA': {'volume_column': 'volume', 'b0': 0., 'b_z': 1., 'b_lv': 2., 'b_miss': 3.}}
        a = M.replay(self.record, {}, meta, 0.)[1]
        b = M.replay(self.record, {'volume': 0.}, meta, 0.)[1]
        self.assertAlmostEqual(a, M.sigmoid(5.))
        self.assertAlmostEqual(b, M.sigmoid(2.))
        self.assertNotEqual(a, b)
        self.assertEqual(a, M.replay(self.record, {'volume': float('nan')}, meta, 0.)[1])

    def test_state_feature_order_and_disabled_calibration(self):
        meta = copy.deepcopy(self.meta)
        meta['state'] = {'Lateral OA': {'features': ['z', 'state:present', 'lv', 'miss'],
                                      'columns': [None, 2, None, None], 'coef': [2., 3., 4., 5.],
                                      'b0': -1., 'volume_column': 'volume'}}
        actual = M.replay(self.record, {'volume': math.e-1}, meta, 0.)[2]
        self.assertAlmostEqual(actual, M.sigmoid(-1 + 2*4 + 3*.6 + 4), places=8)
        self.assertAlmostEqual(M.replay(self.record, {}, meta, 0., False)[2], M.sigmoid(4.))

    def test_nonfinite_wrong_model_count_and_unregistered_weight_rejected(self):
        for key, value in [('z_plain', [[float('nan'), 2, 3]]), ('state_plain', [[-1., .2, .3]]),
                           ('z_mirror_raw', [[1, 2, 3], [1, 2, 3]])]:
            r = copy.deepcopy(self.record)
            r[key] = value
            with self.assertRaises(ValueError):
                M.mix(r, self.meta, .5)
        with self.assertRaises(ValueError):
            M.mix(self.record, self.meta, .1)

    def test_flat_state_permutation_has_independent_expected_map(self):
        labels = ['MCL', 'Medial OA', 'Lateral OA']
        columns = [{'class': c, 'categories': ['N', 'P'], 'start': 2*i, 'stop': 2*i+2} for i, c in enumerate(labels)]
        columns += [{'class': c, 'binary': 'acute', 'column': 6+i} for i, c in enumerate(labels)]
        perm, mcl = M.state_permutation(labels, {'columns': columns}, [0, 2, 1], [0, 2, 1])
        self.assertEqual(perm, [0, 1, 4, 5, 2, 3, 6, 8, 7])
        self.assertEqual(mcl, [0, 1, 6])
        columns[2]['categories'] = ['P', 'N']
        with self.assertRaises(ValueError):
            M.state_permutation(labels, {'columns': columns}, [0, 2, 1], [0, 2, 1])


class Transport(unittest.TestCase):
    def test_frozen_cells_inputs_and_reversible_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'probe'
            manifest = B.prepare(out, 'rsraki/test-r6-paired-probe')
            payload = json.loads((out / 'save-payload.json').read_bytes())
            book = json.loads(payload['text'])
            codes = [c['source'] for c in book['cells'] if c['cell_type'] == 'code']
            parent = B.F.read(B.F.P033 / 'source.ipynb')
            originals = [B.F.text(c) for c in parent['cells'] if c['cell_type'] == 'code']
            self.assertEqual(codes[1:3], [originals[0], originals[5]])
            self.assertEqual(len(payload['dataset_data_sources']), 1)
            self.assertEqual(len(payload['kernel_data_sources']), 1)
            self.assertEqual(len(payload['competition_data_sources']), 1)
            self.assertTrue(payload['is_private'])
            self.assertFalse(payload['enable_internet'])
            self.assertEqual(hashlib.sha256((out / 'source.ipynb').read_bytes()).hexdigest(), manifest['source_SHA'])
            # Restore only the data transport, stopping before any runtime execution.
            import ast
            import base64
            import zlib
            node = ast.parse(codes[-1]).body[0]
            packed = node.value.func.value.args[0].args[0].value
            self.assertEqual(hashlib.sha256(zlib.decompress(base64.b85decode(packed))).hexdigest(), B.SELECTION_SHA)
            for source in codes:
                ast.parse(source)
            with self.assertRaises(ValueError):
                B.prepare(out, 'rsraki/test-r6-paired-probe')

    def test_foreign_owner_refused_before_build(self):
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(ValueError):
            B.prepare(Path(temp)/'probe', 'other/test-r6-paired-probe')


if __name__ == '__main__':
    unittest.main()
