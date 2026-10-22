import ast
import copy
import hashlib
import itertools
import json
import struct
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_p046_dino20 as B


def rank(values):
    return [Fraction(2*sum(x<v for x in values)+sum(x==v for x in values)+1, 2*len(values)) for v in values]


class DINO20(unittest.TestCase):
    def test_fraction_oracle_all_failure_combinations_and_ties(self):
        ns = {'__name__': 'isolated_dino20'}
        exec(compile(B.runtime20((B.PARENT/'runtime.py').read_text(encoding='utf8')), '<isolated20>', 'exec'), ns)
        R = SimpleNamespace(**ns)
        patterns = list(itertools.product([False, True], repeat=4)) + [(False, False, False, True)]*5
        def rows(values):
            return [{R.ID: 'u%d'%i, **{t: str(value) for t in R.TARGETS}} for i, value in enumerate(values)]
        p = rows([((i*3)%11+1)/13 for i in range(len(patterns))])
        folds = [rows([((i*(k+2))%7+1)/9 for i in range(len(patterns))]) for k in range(5)]
        failed, success = [any(x[:3]) for x in patterns], [x[3] for x in patterns]
        original = copy.deepcopy((p, folds))
        out, receipt = R.fuse(p, folds, failed, success)
        indices = [i for i in range(len(p)) if not failed[i] and success[i]]
        for target in R.TARGETS:
            left = rank([float(p[i][target]) for i in indices])
            parts = [rank([float(f[i][target]) for i in indices]) for f in folds]
            right = rank([sum(v)/5 for v in zip(*parts)])
            for j, i in enumerate(indices):
                self.assertEqual(struct.pack('d', float(out[i][target])), struct.pack('d', .8*float(left[j])+.2*float(right[j])))
            for i in set(range(len(p)))-set(indices):
                self.assertEqual(out[i][target], p[i][target])
        self.assertEqual((p, folds), original)
        self.assertEqual(receipt['weight'], .2)

    def test_builder_preserves_transport_producer_inputs_and_parent(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'candidate'
            manifest = B.prepare(out, 'rsraki/test-dino20-clone')
            payload = json.loads((out/'save-payload.json').read_bytes())
            parent = B.F.read(B.PARENT/'save-payload.json')
            self.assertEqual({k:v for k,v in payload.items() if k not in ('slug','new_title','text')},
                             {k:v for k,v in parent.items() if k not in ('slug','new_title','text')})
            codes = [B.F.text(c) for c in json.loads(payload['text'])['cells'] if c['cell_type']=='code']
            oldcodes = [B.F.text(c) for c in json.loads(parent['text'])['cells'] if c['cell_type']=='code']
            for name in ('_p045_dino_source',):
                self.assertEqual(ast.literal_eval(B.F.assignment(codes[1],name)[1].value),
                                 ast.literal_eval(B.F.assignment(oldcodes[1],name)[1].value))
            restored = codes[1].replace("'weight': 0.2", "'weight': 0.1")
            self.assertEqual(ast.dump(ast.parse(restored),include_attributes=False),ast.dump(ast.parse(oldcodes[1]),include_attributes=False))
            self.assertEqual(hashlib.sha256((B.PARENT/'source.ipynb').read_bytes()).hexdigest(), B.PARENT_SHA)
            self.assertTrue(manifest['all8_science_byte_preserved'])
            with self.assertRaises(ValueError):
                B.prepare(out, 'rsraki/test-dino20-clone')

    def test_unexpected_runtime_or_foreign_slug_refused(self):
        with self.assertRaises(ValueError):
            B.runtime20((B.PARENT/'runtime.py').read_text(encoding='utf8').replace('DINO_WEIGHT = .10','DINO_WEIGHT = .15'))
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(ValueError):
            B.prepare(Path(temp)/'candidate', 'foreign/test-dino20-clone')


if __name__ == '__main__':
    unittest.main()
