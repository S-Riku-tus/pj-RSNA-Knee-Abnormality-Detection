"""One-cell composition and inherited positive-Reader contracts; no MRI/GPU."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import p044_reader_mirror_runtime as R
import prepare_p044_reader_mirror as B

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prior_factorial_fixtures', ROOT / 'tests/test_public_factorial_candidates.py')
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)


class ReaderMirror(unittest.TestCase):
    def test_artificial_probe_selects_only_pure_helpers_and_gates_result(self):
        source = [B.F.text(c) for c in B.F.read(B.P036 / 'source.ipynb')['cells'] if c['cell_type'] == 'code'][3]
        probe = R.mirror_probe_source(source)
        ast.parse(probe)
        self.assertNotIn('torch.load', probe)
        self.assertNotIn('load_model', probe)
        self.assertIn('gpu_input', probe)
        with tempfile.TemporaryDirectory() as tmp:
            def failed(*args, **kwargs):
                from types import SimpleNamespace
                return SimpleNamespace(returncode=1, stdout='', stderr='synthetic failure')
            with self.assertRaisesRegex(ValueError, 'Artificial mirror CUDA forward failed'):
                R.mirror_artificial_qa(source, work=tmp, runner=failed)

    def test_exact_one_science_cell_diff_reader_and_inputs_unchanged(self):
        parent = B.F.read(B.P041 / 'science-cells.json')
        donor = B.F.read(B.P036 / 'source.ipynb')
        mirror = [B.F.text(c) for c in donor['cells'] if c['cell_type'] == 'code'][3]
        old_payload = B.F.read(B.P041 / 'save-payload.json')
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            out = Path(tmp) / 'new'
            manifest = B.prepare(out, 'rsraki/p044-test-source-composition')
            science = B.F.read(out / 'science-cells.json')
            self.assertEqual([i for i, pair in enumerate(zip(parent, science)) if pair[0] != pair[1]], [3])
            self.assertEqual(science[3], mirror)
            self.assertEqual(science[8], parent[8])
            payload = B.F.read(out / 'save-payload.json')
            for key in old_payload.keys() - {'slug', 'new_title', 'text'}:
                self.assertEqual(payload[key], old_payload[key], key)
            self.assertEqual(manifest['inputs']['total'], 7)
            self.assertFalse(manifest['descriptor']['R7_included'])
            self.assertFalse(manifest['descriptor']['Gaussian_pooling_included'])
            self.assertIsNone(manifest['own_Public'])
            notebook = B.F.read(out / 'source.ipynb')
            tree = ast.parse(B.F.text(notebook['cells'][1]))
            inherited = ast.literal_eval(B.literal(tree, '_factor_preflight_source').value)
            self.assertEqual(hashlib.sha256(inherited.encode()).hexdigest(),
                             manifest['inherited_preflight_source']['sha256'])
            config = ast.literal_eval(B.literal(tree, '_factor_preflight_config').value)
            self.assertEqual(config['producer_descriptor']['candidate'], 'P044')
            self.assertNotIn('parent8_byte_exact', config['producer_descriptor'])
            runtime = ast.literal_eval(B.literal(tree, '_factor_runtime_source').value)
            self.assertEqual(runtime, (ROOT / 'scripts/p044_reader_mirror_runtime.py').read_text(encoding='utf-8'))
            restore = ast.Module(body=[ast.parse(B.F.text(notebook['cells'][2])).body[0]], type_ignores=[])
            namespace = {'_factor_ns': {'restore_cells': R.restore_cells}}
            exec(compile(restore, '<restore-only-no-inference>', 'exec'), namespace)
            self.assertEqual(namespace['_factor_science'], science)
            notes = [B.F.text(c) for c in B.F.read(B.F.P033 / 'source.ipynb')['cells'] if c['cell_type'] == 'markdown']
            self.assertEqual([B.F.text(c) for c in notebook['cells'][4:]], notes)
            with self.assertRaises(ValueError):
                B.prepare(out, 'rsraki/p044-no-overwrite')

    def test_other_account_slug_rejected_before_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            out = Path(tmp) / 'invalid'
            with self.assertRaises(ValueError):
                B.prepare(out, 'another-account/not-ours')
            self.assertFalse(out.exists())

    @staticmethod
    def config():
        cells = ['synthetic cell %d' % i for i in range(9)]
        return cells, {'candidate': 'P044', 'science_inventory': B.F.inventory(cells),
                       'class_vector': B.F.OLD_VECTOR}

    def test_positive_reader_and_checkpoint_names_and_FP64_contract(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as stdout:
            cells, config = self.config()
            shell = H.FakeShell(tmp, B.F.OLD_VECTOR)
            receipt = R.execute_cells(cells, shell, config, work=tmp)
            self.assertTrue(receipt['complete'])
            self.assertEqual(shell.calls, 9)
            self.assertEqual(receipt['candidate'], 'P044')
            self.assertEqual(receipt['completed_cells'][8]['reader']['checked_values'], 36)
            self.assertEqual(len(receipt['completed_cells'][8]['reader']['checkpoint_names']), 3)
            self.assertIn('P044_EXACT_FACTORIAL_COMPLETE', stdout.getvalue())
            self.assertTrue((Path(tmp) / 'P044-execution.json').exists())

    def test_reader_rollback_or_wrong_numeric_never_qualifies(self):
        for mode in ['reader_failure', 'wrong_numeric']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                cells, config = self.config()
                with self.assertRaises(ValueError):
                    R.execute_cells(cells, H.FakeShell(tmp, B.F.OLD_VECTOR, mode), config, work=tmp)
                self.assertFalse(B.F.read(Path(tmp) / 'P044-execution.json')['complete'])

    def test_wrong_layout_and_source_hash_rejected_before_execution(self):
        for mode in ['layout', 'source', 'candidate']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                cells, config = self.config()
                shell = H.FakeShell(tmp, B.F.OLD_VECTOR)
                if mode == 'layout':
                    cells.pop()
                elif mode == 'source':
                    cells[3] += '\n# altered source'
                else:
                    config['candidate'] = 'P041'
                with self.assertRaises(ValueError):
                    R.execute_cells(cells, shell, config, work=tmp)
                self.assertEqual(shell.calls, 0)


if __name__ == '__main__':
    unittest.main()
