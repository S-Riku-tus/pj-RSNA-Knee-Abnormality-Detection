"""最終日の新規scopeが未完モデルやReader失敗を受理しないことを確認。"""
import ast
import importlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
TORCH_BEFORE_IMPORT = 'torch' in sys.modules
package_builder = importlib.import_module('prepare_f016_serving_package')
runtime = importlib.import_module('f016_cnn_serving_runtime')
reader_builder = importlib.import_module('prepare_p051_native10_reader')
cnn_builder = importlib.import_module('prepare_p053_f016_native10')


class FinalDayGuards(unittest.TestCase):
    def test_new_runtime_import_is_tensor_free(self):
        program = ('import sys,tempfile\n'
                   f'sys.path.insert(0,{str(ROOT / "scripts")!r})\n'
                   'import f016_cnn_serving_runtime as runtime\n'
                   'assert "torch" not in sys.modules\n'
                   'with tempfile.TemporaryDirectory() as tmp:\n'
                   '    try: runtime.prepare_package(tmp)\n'
                   '    except FileNotFoundError: pass\n'
                   '    else: raise AssertionError("missing binding accepted")\n'
                   'assert "torch" not in sys.modules\n')
        result = subprocess.run([sys.executable, '-c', program], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_incomplete_weak_audit_cannot_build_package(self):
        # generatorのworkspace制約を維持するためworkspaceの一時研究ディレクトリを使う。
        with tempfile.TemporaryDirectory(dir=ROOT / 'artifacts/research') as tmp:
            path = Path(tmp)
            audit = path / 'audit.json'
            audit.write_text(json.dumps({'schema': 'F016_fixedE12_current_family_proxy_audit_v1', '両fold候補条件': False}), encoding='utf8')
            with self.assertRaises(ValueError):
                package_builder.prepare(audit, path / 'package')
            self.assertFalse((path / 'package').exists())
        self.assertEqual('torch' in sys.modules, TORCH_BEFORE_IMPORT)

    def test_new_pair_preserves_REG_and_original_object(self):
        old = {'family_pairs': {'CLS': {'old': True}, 'REG': {'bindings': ['固定REG0', '固定REG1']}},
               'model_classes': {'CLS': 'old.CLS', 'REG': 'old.REG'}, 'pair_fingerprint_sha256': 'historical',
               'FP32_aggregation': {'fold': .5, 'family': .5}}
        before = json.dumps(old, sort_keys=True)
        manifest = {'config': {'固定': True}, 'checkpoints': {'0': {'sha256': '0' * 64}, '1': {'sha256': '1' * 64}}}
        new = runtime.new_pair(old, manifest, '2' * 64)
        self.assertEqual(json.dumps(old, sort_keys=True), before)
        self.assertEqual(new['family_pairs']['REG'], old['family_pairs']['REG'])
        self.assertEqual(new['FP32_aggregation'], old['FP32_aggregation'])
        self.assertEqual(new['family_pairs']['CLS']['bindings'][1]['selected_epoch'], 12)
        self.assertEqual(new['original_P039_pair_SHA_reference'], old['pair_fingerprint_sha256'])
        self.assertIn('F016 Tiny固定E12', new['experimental_factor'])

    def test_Reader_source_and_native_parent_are_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'candidate'
            manifest = reader_builder.prepare(path)
            book = json.loads((path / 'source.ipynb').read_bytes())
            native = json.loads((reader_builder.NATIVE / 'source.ipynb').read_bytes())
            cells = reader_builder.code(book)
            self.assertEqual(cells[1:5], reader_builder.code(native))
            overlay = json.loads((reader_builder.READER / 'science-cells.json').read_bytes())[-1]
            tree = ast.parse(cells[-1])
            value = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == '_p051_reader' for t in n.targets))
            self.assertEqual(value, overlay)
            self.assertTrue(manifest['P049_four_cells_byte_exact'])

    def test_Reader_failure_marker_cannot_be_parent_only_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            reader_builder.prepare(path / 'candidate')
            book = json.loads((path / 'candidate/source.ipynb').read_bytes())
            text = reader_builder.code(book)[-1].replace("'/kaggle/working'", repr(str(path / 'working')))
            (path / 'working').mkdir()
            helper = {'snapshot': lambda *args: {}, 'parent_evidence': lambda *args: {}}

            class FailedReader:
                def run_cell(self, source, store_history=False):
                    print('own leg FAILED, public stack submission kept: synthetic failure')
                    return SimpleNamespace(error_before_exec=None, error_in_exec=None)

            namespace = {'_factor_ns': helper, '_factor_hash': __import__('hashlib'),
                         'get_ipython': lambda: FailedReader()}
            with redirect_stdout(io.StringIO()), self.assertRaises(AssertionError):
                exec(compile(text, 'P051-failure-contract', 'exec'), namespace)
            self.assertFalse((path / 'working/P051/execution.json').exists())

    def test_CNN_adapter_binds_new_scope_and_keeps_native10(self):
        with zipfile.ZipFile(cnn_builder.PARENT / 'runtime-sources.bin') as bundle:
            original = {n: bundle.read(n) for n in bundle.namelist()}
        files = cnn_builder.derive_native_sources(original)
        child, extension = files['p048_native_child.py'].decode(), files['p048_parent_extension.py'].decode()
        self.assertIn('f016_cnn_serving_runtime.execute', child)
        self.assertNotIn('F013_current_constructor_qualification', child + extension)
        self.assertNotIn('F013_bridge_policy_SHA', child + extension)
        self.assertIn('F016_current_constructor_qualification', extension)
        self.assertIn(cnn_builder.A.SCOPE_SHA, extension)
        self.assertIn('alpha=0.1,', extension)
        self.assertIn('parent_factor, native_factor = (.9, .1)', extension)


if __name__ == '__main__':
    unittest.main()
