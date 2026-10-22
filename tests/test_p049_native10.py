"""P049内包runtimeを、独立FP32算術・失敗例・source保存で検証。"""
import ast
import importlib
import io
import json
import struct
import sys
import tempfile
import unittest
import zipfile
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_p049_native10 as builder  # noqa: E402


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def oracle(parent, values):
    values = [f32(v) for v in values]
    ranks = [Fraction(2 * sum(x < v for x in values) + sum(x == v for x in values) + 1,
                      2 * len(values)) for v in values]
    return [f32(f32(f32(.9) * f32(p)) + f32(f32(.1) * f32(r))) for p, r in zip(parent, ranks)]


class P049Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name) / 'candidate'
        cls.manifest = builder.prepare(cls.output, 'rsraki/p049-numeric-test-v1')
        cls.runtime = Path(cls.temp.name) / 'runtime'
        with zipfile.ZipFile(io.BytesIO((cls.output / 'runtime-sources.bin').read_bytes())) as archive:
            archive.extractall(cls.runtime)
        cls.saved = {k: v for k, v in sys.modules.items()
                     if k == 'p032blend' or k.startswith('p032blend.') or k == 'p048_parent_extension'}
        for key in cls.saved:
            del sys.modules[key]
        sys.path.insert(0, str(cls.runtime))
        cls.blend = importlib.import_module('p032blend.success20_blend')
        cls.extension = importlib.import_module('p048_parent_extension')
        cls.targets = cls.blend.original.TARGETS

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(cls.runtime))
        for key in list(sys.modules):
            if key == 'p032blend' or key.startswith('p032blend.') or key == 'p048_parent_extension':
                del sys.modules[key]
        sys.modules.update(cls.saved)
        cls.temp.cleanup()

    def rows(self):
        parent_values = ['0.00000011', '0.800000011', '.223456789', '1.000000', '.000130017', '.432111123']
        native_values = ['.2', '.200000001', '.9', '.6', '.4', '.6']
        parent = [dict(StudyInstanceUID=str(i), **{t: v for t in self.targets}) for i, v in enumerate(parent_values)]
        native = [dict(StudyInstanceUID=str(i), **{t: v for t in self.targets}) for i, v in enumerate(native_values)]
        return parent, native

    def test_independent_ties_fp32_and_original_order(self):
        parent, native = self.rows()
        result, receipt = self.blend.blend_rows(parent, list(reversed(native)))
        expected = oracle([float(r['ACL']) for r in parent], [float(r['ACL']) for r in native])
        self.assertEqual([float(r['ACL']) for r in result], expected)
        self.assertEqual([r['StudyInstanceUID'] for r in result], [r['StudyInstanceUID'] for r in parent])
        self.assertEqual(receipt['alpha_requested'], .1)
        self.assertIn('0.9 * parent_s + 0.1 *', receipt['formula'])
        self.assertFalse(receipt['arithmetic']['parent_reranked'])
        self.assertFalse(receipt['arithmetic']['output_reranked'])
        for t in ('MCL', 'Synovitis'):
            self.assertEqual([r[t] for r in result], [r[t] for r in parent])

    def test_partial_success_rank_denominator_and_failure_text(self):
        parent, native = self.rows()
        native[2]['ACL'] = ''
        result, receipt = self.blend.blend_rows(parent, native, failure_mask={'4': {'ACL': True}})
        valid = [0, 1, 3, 5]
        expected = oracle([float(parent[i]['ACL']) for i in valid], [float(native[i]['ACL']) for i in valid])
        self.assertEqual([float(result[i]['ACL']) for i in valid], expected)
        for i in (2, 4):
            self.assertEqual(result[i]['ACL'], parent[i]['ACL'])
        self.assertEqual(receipt['per_target']['ACL']['native_rank_population_count'], 4)
        self.assertEqual(receipt['per_target']['ACL']['effective_weights']['successful_cells'], {'parent_s': .9, 'native_rank': .1})

    def test_nonfinite_masked_and_wrong_alpha_refused(self):
        parent, native = self.rows()
        for value in (.2, .3, True):
            with self.assertRaises(ValueError):
                self.blend.blend_rows(parent, native, alpha=value)
        native[0]['ACL'] = 'nan'
        with self.assertRaises(ValueError):
            self.blend.blend_rows(parent, native, failure_mask={'0': True})

    def test_identity_and_constant_partial_preserve_text(self):
        parent, native = self.rows()
        result, _ = self.blend.blend_rows(parent, native, alpha=0)
        self.assertEqual(result, parent)
        for row in native:
            row['ACL'] = '.2'
        native[0]['ACL'] = ''
        result, _ = self.blend.blend_rows(parent, native)
        self.assertEqual([r['ACL'] for r in result], [r['ACL'] for r in parent])

    def test_extension_acceptance_and_nominal_receipt(self):
        result, receipt = self.blend.blend_rows(*self.rows())
        self.extension.p039_require_native_contribution(receipt)
        bound = self.extension.p039_bind_parent_metadata(receipt)
        weights = bound['per_target']['ACL']['conditional_P033_component_weights_nominal']['successful_cells']
        self.assertEqual(weights, {'Raptor384': .2925, 'Raptor224': .1575, 'CompactR6': .45,
                                   'F013_CLS_original_REG_native_rank': .1})
        receipt['alpha_requested'] = .2
        with self.assertRaises(RuntimeError):
            self.extension.p039_require_native_contribution(receipt)

    def test_only_three_runtime_boundaries_changed(self):
        book, files, original, _ = builder.variant_files()
        self.assertEqual({n for n in files if files[n] != original[n]}, builder.CHANGED)
        self.assertEqual(files['P033-source.ipynb'], original['P033-source.ipynb'])
        prepared = json.loads((self.output / 'save-payload.json').read_bytes())
        old = json.loads((builder.PARENT / 'save-payload.json').read_bytes())
        for key in old.keys() - {'text', 'slug', 'new_title'}:
            self.assertEqual(old[key], prepared[key], key)
        self.assertEqual(self.manifest['native_alpha'], .1)
        self.assertEqual(book['cells'][3]['source'].replace('_p048', '_p049'),
                         json.loads((self.output / 'source.ipynb').read_bytes())['cells'][3]['source'])

    def test_actual_frozen_restorer_extracts_entire_bundle(self):
        book = json.loads((self.output / 'source.ipynb').read_bytes())
        setup = ast.parse(book['cells'][2]['source'])
        setup.body[-1].value.args[-1] = ast.Constant(str(Path(self.temp.name) / 'strict-restored'))
        namespace = {}
        exec(compile(ast.fix_missing_locations(setup), 'actual-P048-restorer', 'exec'), namespace)
        restored = namespace['_p049_runtime']
        for name, expected in self.manifest['runtime_files'].items():
            self.assertEqual(builder.pin((restored / name).read_bytes()), expected)


if __name__ == '__main__':
    unittest.main()
