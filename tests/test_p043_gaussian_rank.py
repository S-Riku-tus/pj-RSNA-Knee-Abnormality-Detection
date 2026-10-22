"""Synthetic score/order and submission contracts; no models, labels or GPU."""
import contextlib
import hashlib
import io
import itertools
import json
import math
import random
import struct
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import p043_gaussian_rank_runtime as R
import prepare_p043_gaussian_rank as B


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def cdf(value):
    return math.erfc(-value / math.sqrt(2)) / 2


def inverse_by_bisection(probability):
    """Numerically invert erfc without statistics.NormalDist or its algorithm."""
    low, high = -12., 12.
    for _ in range(95):
        middle = (low + high) / 2
        if cdf(middle) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def ranks_by_pair_counts(values, valid, normal=False):
    cohort = [x for x, ok in zip(values, valid) if ok]
    result = []
    for value, ok in zip(values, valid):
        if not ok:
            result.append(float("nan"))
            continue
        lower = sum(x < value for x in cohort)
        equal = sum(x == value for x in cohort)
        q = float(Fraction(2 * lower + equal + (0 if normal else 1), 2 * len(cohort)))
        result.append(inverse_by_bisection(q) if normal else q)
    return result


def fixture(vectors, masks=None):
    n = len(vectors[0])
    masks = masks or [[False] * n for _ in range(3)]
    ids = ["1.2.3." + str(i + 1) for i in range(n)]
    raw = [[{R.ID: uid, **{target: repr(.5 if mask[i] else vector[i]) for target in R.TARGETS}}
            for i, uid in enumerate(ids)] for vector, mask in zip(vectors, masks)]
    valid = [[not x for x in mask] for mask in masks]
    rank384, rank224, rankown = [ranks_by_pair_counts(v, ok) for v, ok in zip(vectors, valid)]
    normals = [ranks_by_pair_counts(v, ok, normal=True) for v, ok in zip(vectors, valid)]
    parent = [{R.ID: uid} for uid in ids]
    expected = [{R.ID: uid} for uid in ids]
    for target in R.TARGETS:
        w = R.WR.get(target, .5)
        for i in range(n):
            rr = rank224[i] if masks[0][i] else rank384[i] if masks[1][i] else .65 * rank384[i] + .35 * rank224[i]
            value = w * rr + (1 - w) * rankown[i]
            if masks[0][i] and masks[1][i]:
                value = rankown[i]
            if masks[2][i]:
                value = rr
            if all(mask[i] for mask in masks):
                value = .5
            parent[i][target] = format(f32(value), ".9g")
            if not any(mask[i] for mask in masks):
                z = w * (.65 * normals[0][i] + .35 * normals[1][i]) + (1 - w) * normals[2][i]
                expected[i][target] = f32(cdf(z))
            else:
                expected[i][target] = parent[i][target]
    return parent, raw, masks, expected


class GaussianPooling(unittest.TestCase):
    def check_fixture(self, vectors, masks=None):
        parent, raw, masks, expected = fixture(vectors, masks)
        result, receipt = R.transform(parent, *raw, *masks)
        for i in range(len(parent)):
            for target in R.TARGETS:
                self.assertEqual(struct.pack("<f", float(result[i][target])),
                                 struct.pack("<f", float(expected[i][target])))
                if any(mask[i] for mask in masks):
                    self.assertEqual(result[i][target], parent[i][target])
        return parent, result, receipt

    def test_known_hazen_normal_quantiles_ties_singletons_and_empty(self):
        z = R.gaussian_on([.1, .2, .3], [True] * 3)
        for got, q in zip(z, [1/6, .5, 5/6]):
            self.assertAlmostEqual(cdf(got), q, places=14)
        self.assertEqual(R.gaussian_on([.2], [True]), [0.])
        self.assertEqual(R.gaussian_on([.2, .2], [True] * 2), [0., 0.])
        self.assertEqual(R.gaussian_on([], []), [])
        self.assertTrue(math.isnan(R.gaussian_on([.2], [False])[0]))
        z = R.gaussian_on([.1, .1, .4], [True] * 3)
        self.assertAlmostEqual(cdf(z[0]), 1/3, places=14)
        self.assertEqual(z[0], z[1])

    def test_erfc_bisection_oracle_all_failure_combinations_and_ties(self):
        n = 17
        a = [(i + 1) / 19 for i in range(n)]
        b = list(reversed(a))
        own = a[6:] + a[:6]
        a[3] = a[4]
        b[8] = b[9]
        masks = [[False] * n for _ in range(3)]
        for i, combo in enumerate(itertools.product([False, True], repeat=3)):
            for j in range(3):
                masks[j][i] = combo[j]
        _, _, receipt = self.check_fixture([a, b, own], masks)
        self.assertEqual(receipt['preserved_failure_rows'], 7)
        self.assertEqual(receipt['parent_FP32_values_verified'], n * 12)

    def test_random_permutations_and_equal_branch_order_invariant(self):
        rng = random.Random(20261020)
        a = [(i + 1) / 33 for i in range(31)]
        for _ in range(5):
            b, own = a[:], a[:]
            rng.shuffle(b)
            rng.shuffle(own)
            self.check_fixture([a, b, own])
        parent, result, receipt = self.check_fixture([a, a, a])
        self.assertEqual(receipt['changed_final_rank_cells'], 0)
        self.assertEqual(sorted(range(31), key=lambda i: float(parent[i]['ACL'])),
                         sorted(range(31), key=lambda i: float(result[i]['ACL'])))

    def test_three_model_pool_can_change_pair_order_not_only_final_scale(self):
        a = [.1, .2, .3, .4, .6]
        found = None
        for b in itertools.permutations(a):
            for own in (a, list(reversed(a)), a[2:] + a[:2]):
                parent, result, receipt = self.check_fixture([a, b, own])
                if receipt['changed_final_rank_cells']:
                    found = parent, result, receipt
                    break
            if found:
                break
        self.assertIsNotNone(found)
        self.assertFalse(found[2]['rr_rerank_added'])
        self.assertFalse(found[2]['new_final_percentile_rank_added'])

    def test_empty_and_singleton_have_defined_scores(self):
        self.check_fixture([[], [], []])
        _, result, _ = self.check_fixture([[.2], [.3], [.4]])
        self.assertEqual(float(result[0]['ACL']), .5)

    def test_realistic_cohort_endpoint_quantiles_remain_finite(self):
        z = R.gaussian_on([i / 1301 for i in range(1300)], [True] * 1300)
        self.assertTrue(all(math.isfinite(x) for x in z))
        self.assertAlmostEqual(z[0], -z[-1], places=11)
        self.assertLess(abs(z[-1]), 4)

    def test_malformed_UID_masks_probabilities_and_parent_rejected(self):
        for kind in ['duplicate', 'missing', 'foreign', 'parent', 'mask', 'nonfinite']:
            with self.subTest(kind=kind):
                parent, raw, masks, _ = fixture([[.1, .2], [.2, .1], [.1, .2]])
                if kind == 'duplicate':
                    raw[0][1][R.ID] = raw[0][0][R.ID]
                elif kind == 'missing':
                    raw[0].pop()
                elif kind == 'foreign':
                    raw[1][0][R.ID] = '1.2.foreign'
                elif kind == 'parent':
                    parent[0]['ACL'] = '.12345'
                elif kind == 'mask':
                    masks[0] = [0, 0]
                elif kind == 'nonfinite':
                    raw[0][0]['ACL'] = 'nan'
                with self.assertRaises(ValueError):
                    R.transform(parent, *raw, *masks)

    def test_csv_contract_roundtrip_and_wrong_header(self):
        parent, raw, masks, _ = fixture([[.1, .2], [.2, .1], [.1, .2]])
        result, _ = R.transform(parent, *raw, *masks)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'test.csv'
            R.write_predictions(p, result)
            self.assertEqual(R.read_predictions(p), result)
            p.write_text('ACL,StudyInstanceUID\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                R.read_predictions(p)

    def test_offline_builder_restores_all_parent_cells_and_exact_inputs(self):
        with tempfile.TemporaryDirectory(dir=B.ROOT) as tmp:
            out = Path(tmp) / 'new'
            manifest = B.prepare(out, 'rsraki/p043-test-frozen-source')
            nb = json.loads((out / 'source.ipynb').read_text(encoding='utf-8'))
            payload = json.loads((out / 'save-payload.json').read_text(encoding='utf-8'))
            parent = json.loads((B.PARENT / 'save-payload.json').read_text(encoding='utf-8'))
            for key in ['dataset_data_sources', 'kernel_data_sources', 'competition_data_sources',
                        'is_private', 'enable_gpu', 'enable_internet', 'machine_shape', 'docker_image']:
                self.assertEqual(payload[key], parent[key])
            self.assertEqual(payload['text'], (out / 'source.ipynb').read_text(encoding='utf-8'))
            parent_nb = json.loads((B.PARENT / 'source.ipynb').read_text(encoding='utf-8'))
            parent_notes = [B.text(c) for c in parent_nb['cells'] if c['cell_type'] == 'markdown']
            self.assertEqual([B.text(c) for c in nb['cells'][4:]], parent_notes)
            namespace = {}
            exec(B.text(nb['cells'][1]), namespace)
            # The following cell contains the restore expression and execution call.
            # Execute only its restore assignment, using AST, never the inference call.
            import ast
            tree = ast.parse(B.text(nb['cells'][2]))
            first = ast.Module(body=[tree.body[0]], type_ignores=[])
            exec(compile(first, '<restore-only>', 'exec'), namespace)
            cells = namespace['_p043_science']
            self.assertEqual(len(cells), 8)
            self.assertEqual([hashlib.sha256(x.encode()).hexdigest() for x in cells],
                             [x['sha256'] for x in manifest['science_inventory']])
            with self.assertRaises((ValueError, FileExistsError)):
                B.prepare(out, 'rsraki/p043-test-frozen-source')

    def test_controller_requires_all_cells_and_writes_auditable_outputs(self):
        parent, raw, masks, _ = fixture([[.1, .2, .3], [.3, .1, .2], [.2, .3, .1]])
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            pred = work / 'prediction'
            (pred / 'worker0').mkdir(parents=True)
            R.write_predictions(work / 'submission.csv', parent)
            (work / 'blend_receipt.json').write_text('{}', encoding='utf-8')
            for rows, name in zip(raw, ['raptor_raw.csv', 'raptor224_raw.csv', 'ours_sub.csv']):
                R.write_predictions(work / name, rows)
            R.write_predictions(pred / 'worker0' / 'predictions.csv', raw[2])
            ns = {'test_ids': [x[R.ID] for x in parent], 'labels': list(R.TARGETS),
                  'W_RAPTOR_C': R.WR, 'W_RAPTOR': .5, 'W_OURS': .5, 'A224': .35,
                  'r384_fail': masks[0], 'r224_fail': masks[1], 'our_fail': masks[2],
                  'rap_fail': [False] * 3, 'pred': pred}
            calls = []
            shell = SimpleNamespace(user_ns=ns, run_cell=lambda source, **kw:
                                    (calls.append(source) or SimpleNamespace(error_before_exec=None, error_in_exec=None)))
            cells = ['# synthetic cell ' + str(i) for i in range(8)]
            inventory = B.inventory(cells)
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                receipt = R.execute_cells(cells, shell, inventory, work=work)
            self.assertEqual(calls, cells)
            self.assertEqual(len(receipt['completed_cells']), 8)
            self.assertTrue(receipt['complete'])
            self.assertIn('P043_GAUSSIAN_RANK_COMPLETE', stdout.getvalue())
            self.assertEqual(R.read_predictions(work / 'submission.csv'),
                             R.read_predictions(work / 'P043/final-submission.csv'))
            self.assertTrue((work / 'P043/failure-masks.csv').exists())
            self.assertEqual(json.loads((work / 'P043/execution.json').read_text())['complete'], True)

    def test_controller_parent_error_never_qualifies_or_silently_falls_back(self):
        cells = ['# synthetic cell ' + str(i) for i in range(8)]
        calls = []

        def fail_second(source, **kw):
            calls.append(source)
            return SimpleNamespace(error_before_exec=None,
                                   error_in_exec=RuntimeError('synthetic failure') if len(calls) == 2 else None)

        with tempfile.TemporaryDirectory() as tmp:
            shell = SimpleNamespace(user_ns={}, run_cell=fail_second)
            with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                R.execute_cells(cells, shell, B.inventory(cells), work=tmp)
            receipt = json.loads((Path(tmp) / 'P043/execution.json').read_text())
            self.assertFalse(receipt['complete'])
            self.assertFalse(receipt['qualification_passed'])
            self.assertEqual(len(receipt['completed_cells']), 1)
            self.assertFalse((Path(tmp) / 'P043/final-submission.csv').exists())


if __name__ == '__main__':
    unittest.main()
