"""Independent rank arithmetic and source/rollback contracts; no GPU or labels."""
import ast
import base64
import contextlib
import copy
import csv
import hashlib
import io
import itertools
import json
import lzma
import struct
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import p045_dino5_runtime as R
import prepare_p045_dino5 as B


def rows(values):
    return [{R.ID: 'u%d' % i, **{t: str(v) for t in R.TARGETS}} for i, v in enumerate(values)]


def pair_rank_fractions(values):
    return [Fraction(2 * sum(x < v for x in values) + sum(x == v for x in values) + 1,
                     2 * len(values)) for v in values]


def pair_ranks(values):
    return list(map(float, pair_rank_fractions(values)))


class Tests(unittest.TestCase):
    def test_all_failure_combinations_pair_count_oracle_and_input_immutability(self):
        patterns = list(itertools.product([False, True], repeat=4)) + [(False, False, False, True)] * 5
        n = len(patterns)
        parent = rows([format(((i * 3) % 11 + 1) / 13, '.9f') for i in range(n)])
        folds = [rows([((i * (k + 2)) % 7 + 1) / 9 for i in range(n)]) for k in range(5)]
        failed = [any(p[:3]) for p in patterns]
        success = [p[3] for p in patterns]
        before = copy.deepcopy((parent, folds))
        final, receipt = R.fuse(parent, folds, failed, success)
        eligible = [i for i in range(n) if not failed[i] and success[i]]
        for target in R.TARGETS:
            left = pair_ranks([float(parent[i][target]) for i in eligible])
            parts = [pair_rank_fractions([float(fold[i][target]) for i in eligible]) for fold in folds]
            right = pair_ranks([sum(values) / 5 for values in zip(*parts)])
            for j, i in enumerate(eligible):
                self.assertEqual(struct.pack('d', float(final[i][target])), struct.pack('d', .9 * left[j] + .1 * right[j]))
            for i in set(range(n)) - set(eligible):
                self.assertEqual(final[i][target], parent[i][target])
        self.assertEqual((parent, folds), before)
        self.assertEqual(receipt['eligible_rows'], len(eligible))

    def test_exact_fold_sum_ties_invariant_to_accumulation_order(self):
        codes = [[5, 1, 4, 2, 3], [5, 2, 1, 4, 3], [2, 3, 5, 4, 1],
                 [5, 3, 1, 2, 4], [1, 4, 5, 2, 3]]
        p = rows([.4] * 5)
        folds = [rows([x / 5 for x in fold]) for fold in codes]
        naive = [sum(fold[i] / 5 for fold in codes) / 5 for i in range(5)]
        self.assertNotEqual(naive[3], naive[4])
        out, _ = R.fuse(p, folds, [False] * 5, [True] * 5)
        reverse, _ = R.fuse(p, list(reversed(folds)), [False] * 5, [True] * 5)
        self.assertEqual(out, reverse)
        self.assertEqual(out[3]['ACL'], out[4]['ACL'])
        self.assertEqual(float(out[3]['ACL']), .9 * .6 + .1 * .5)

    def test_empty_singleton_ties_all_failure_and_uid_order(self):
        for values in [[], [.2], [.3, .3, .3]]:
            p = rows(values)
            folds = [list(reversed(p)) for _ in range(5)]
            out, report = R.fuse(p, folds, [False] * len(p), [True] * len(p))
            self.assertEqual(report['rows'], len(p))
            expected = (len(p) + 1) / (2 * len(p)) if p else None
            self.assertTrue(all(float(row['ACL']) == expected for row in out))
            out, _ = R.fuse(p, folds, [True] * len(p), [True] * len(p))
            self.assertEqual(out, p)

    def test_missing_duplicate_foreign_nonfinite_and_nonbool_refused(self):
        p = rows([.2, .8])
        for case in ['count', 'missing', 'duplicate', 'foreign', 'nan', 'mask']:
            folds, failed, success = [copy.deepcopy(p) for _ in range(5)], [False] * 2, [True] * 2
            if case == 'count':
                folds.pop()
            elif case == 'missing':
                folds[0].pop()
            elif case == 'duplicate':
                folds[0][1] = folds[0][0]
            elif case == 'foreign':
                folds[0][0][R.ID] = 'foreign'
            elif case == 'nan':
                folds[0][0]['ACL'] = 'nan'
            else:
                success = [1, 1]
            with self.assertRaises(ValueError):
                R.fuse(p, folds, failed, success)

    def test_transport_and_source_tamper_fail_before_runner(self):
        cells = ['# artificial %d' % i for i in range(8)]
        inventory = B.F.inventory(cells)
        raw = json.dumps(cells).encode()
        packed = lzma.compress(raw)
        self.assertEqual(R.restore_cells(base64.b85encode(packed), hashlib.sha256(packed).hexdigest(),
                                        hashlib.sha256(raw).hexdigest(), inventory), cells)
        with self.assertRaises(ValueError):
            R.restore_cells(base64.b85encode(packed), '0' * 64, hashlib.sha256(raw).hexdigest(), inventory)
        with tempfile.TemporaryDirectory() as tmp:
            for bad_cells, weight, sha in [(cells[:-1], .1, 'x'), (cells, .2, 'x'), (cells, .1, 'x')]:
                called = []
                with self.assertRaises(ValueError):
                    R.execute_pipeline(bad_cells, None, inventory, 'source', sha,
                                       {'candidate': 'P045', 'weight': weight}, work=tmp,
                                       runner=lambda *a, **k: called.append(1))
                self.assertFalse(called)
                self.assertFalse((Path(tmp) / 'P045').exists())

    def test_builder_exact8_inputs_policy_and_frozen_decorators(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'new'
            manifest = B.prepare(out, 'rsraki/p045-test-offline-source')
            payload = json.loads((out / 'save-payload.json').read_text(encoding='utf8'))
            science = json.loads((out / 'science-cells.json').read_text())
            original = [B.F.text(c) for c in B.F.read(B.F.P033 / 'source.ipynb')['cells'] if c['cell_type'] == 'code']
            self.assertEqual(science, original)
            self.assertEqual(payload['text'], (out / 'source.ipynb').read_text(encoding='utf8'))
            baseline = B.F.read(B.P041 / 'save-payload.json')
            self.assertEqual(payload['dataset_data_sources'], baseline['dataset_data_sources'] + ['mattiaangeli/knee-mri-fold-weights/2'])
            for k in ['kernel_data_sources', 'competition_data_sources', 'is_private', 'enable_gpu', 'enable_internet', 'machine_shape', 'docker_image']:
                self.assertEqual(payload[k], baseline[k])
            source = (out / 'dino-producer.py').read_text(encoding='utf8')
            tree = ast.parse(source)
            def literal(name):
                return ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
                                            and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)))
            frozen = literal('P045_FROZEN')
            predict = ast.parse(frozen['predict']['text'])
            micro = next(n for n in predict.body if isinstance(n, ast.FunctionDef) and n.name == '_micro')
            self.assertEqual(len(micro.decorator_list), 1)
            self.assertEqual(ast.unparse(micro.decorator_list[0]), 'torch.no_grad()')
            self.assertIn('strict=True', source)
            self.assertIn('weights_only=True', source)
            self.assertNotIn('train_series.csv', source)
            self.assertEqual(manifest['policy']['sha256'], B.POLICY_SHA)
            with self.assertRaises(ValueError):
                B.prepare(out, 'rsraki/p045-no-overwrite')
            with self.assertRaises(ValueError):
                B.prepare(Path(tmp) / 'foreign', 'elsewhere/p045-foreign-owner')

    def test_controller_requires_component_before_exact_parent_execution(self):
        cells = ['# synthetic %d' % i for i in range(8)]
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            work = Path(tmp)
            p = rows([.2, .8, .5])
            for name in ['submission.csv', 'raptor_raw.csv', 'raptor224_raw.csv', 'ours_sub.csv']:
                R.write_predictions(work / name, p)
            pred = work / 'predictions/worker0'
            pred.mkdir(parents=True)
            R.write_predictions(pred / 'predictions.csv', p)
            (work / 'blend_receipt.json').write_text('{}')
            ns = {'test_ids': [r[R.ID] for r in p], 'labels': list(R.TARGETS), 'W_RAPTOR_C': R.WR,
                  'W_RAPTOR': .5, 'W_OURS': .5, 'A224': .35, 'pred': pred.parent,
                  'r384_fail': [False] * 3, 'r224_fail': [False] * 3, 'our_fail': [False] * 3}
            # .5 across all targets means failure in the actual parent convention.
            ns['r384_fail'][2] = ns['r224_fail'][2] = True
            calls = []
            shell = SimpleNamespace(user_ns=ns, run_cell=lambda text, **kw:
                                    (calls.append(text) or SimpleNamespace(error_before_exec=None, error_in_exec=None)))
            def runner(command, **kwargs):
                d = Path(command[1]).parent / 'dino'
                d.mkdir()
                for i in range(5):
                    R.write_predictions(d / ('fold-%d.csv' % i), p)
                with (d / 'mask.csv').open('w', newline='') as stream:
                    w = csv.writer(stream)
                    w.writerow([R.ID, 'success'])
                    w.writerows([[r[R.ID], 1] for r in p])
                (d / 'receipt.json').write_text(json.dumps({'complete': True, 'folds': 5,
                    'models_strict_keys': True, 'label_order_verified': True, 'artificial_CPU_CUDA_pass': True,
                    'rows': 3, 'success_rows': 3}))
                return SimpleNamespace(returncode=0)
            result = R.execute_pipeline(cells, shell, B.F.inventory(cells), '# source',
                        hashlib.sha256(b'# source').hexdigest(), {'candidate': 'P045', 'weight': .1}, work=work, runner=runner)
            self.assertTrue(result['complete'])
            self.assertEqual(calls, cells)
            final = R.read_predictions(work / 'submission.csv')
            self.assertEqual(final[2], p[2])
            self.assertEqual(result['overlay']['eligible_rows'], 2)

    def test_failed_dino_cannot_fall_back_to_parent_only_qualification(self):
        cells = ['# synthetic %d' % i for i in range(8)]
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            shell = SimpleNamespace(run_cell=lambda *a, **k: calls.append(1))
            with self.assertRaisesRegex(ValueError, 'DINO process failed'):
                R.execute_pipeline(cells, shell, B.F.inventory(cells), '# source',
                    hashlib.sha256(b'# source').hexdigest(), {'candidate': 'P045', 'weight': .1}, work=tmp,
                    runner=lambda *a, **k: SimpleNamespace(returncode=1))
            self.assertFalse(calls)
            evidence = json.loads((Path(tmp) / 'P045/execution.json').read_text())
            self.assertFalse(evidence['complete'])
            self.assertFalse(evidence['qualified'])


if __name__ == '__main__':
    unittest.main()
