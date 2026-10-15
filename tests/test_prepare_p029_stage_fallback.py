"""Offline synthetic contracts; no notebook, model, MRI, or network execution."""
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    'prepare_p029_stage_fallback', Path(__file__).resolve().parents[1] / 'scripts/prepare_p029_stage_fallback.py'
)
P029 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P029)


class FallbackPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / 'new-run'
        self.slug = 'p029-test-stage-b2'
        never = "raise AssertionError('original science must not execute')\n"
        self.parent = {
            'nbformat': 4, 'nbformat_minor': 4,
            'metadata': {'papermill': {'status': 'old'}, 'kaggle': {'isGpuEnabled': False,
                'dataSources': [{'sourceId': 999999, 'sourceType': 'author-old-input'}],
                'authorRuntime': {'sessionId': 888888}, 'isInternetEnabled': True}},
            'cells': [P029.cell('markdown', 'Original author title\n'),
                      P029.cell('markdown', 'Original OAI credits\n'),
                      *[P029.cell('code', never) for _ in range(7)]]}
        for c in self.parent['cells'][2:]:
            c.update(outputs=[{'output_type': 'stream', 'text': 'Old author score'}], execution_count=99)
        self.parent['cells'][2]['source'].insert(0, '%%writefile /tmp/raptor_run.py\n')
        self.parent['cells'][4]['source'].insert(0, '%%writefile /tmp/raptor224_run.py\n')
        modules = {name: never for name in P029.embedded_sources.__globals__['MODULE_NAMES']}
        self.parent['cells'][6]['source'] = [
            "FLIP_TTA = True\nFOLDS = 4\nINPUT = None\n",
            "SOURCES = " + repr(modules) + '\n',
            "def find_mount(slug, required):\n    cands = []\n", P029.ANCHOR,
            "    return cands\n", never]
        sizes = {f'E_lrprior_s0_f{i}.pt': 10 for i in range(4)}
        sizes.update({'manifest.json': 30, 'side_cls.pt': 10, 'fluid_stacker.json': 10,
                      'state_spec.json': 10, 'state_stacker.json': 10})
        self.r6 = {'bundle_id': 'fe055406fad6a12f', 'models': [f'E_lrprior_s0_f{i}.pt' for i in range(4)],
                   'sha256': {name: '0' * 64 for name in sizes if name != 'manifest.json'}}
        self.parent_manifest = {'public_Input_metadata': {'R6_V1_file_listing': sizes},
            'fixed_inputs': [{'ref': 'competition'}, {'ref': 'public/ortho/1'}, {'ref': 'public/swa/1'}, {'ref': 'public/r6/1'}]}
        self.public = {label: ('public/' + str(index) + '.json', {'description': label + ' model terms', 'licenseName': 'Other',
                                                              'currentVersionNumber': 2 if 'V2' in label else 1})
                       for index, label in enumerate(P029.PUBLIC_EVIDENCE)}

    def fixture_pins(self):
        mapping = {
            'SOURCE_SHA': ('artifacts/kaggle/p029-public957-r6-v1-20261015/source.ipynb', self.parent),
            'PARENT_MANIFEST_SHA': ('artifacts/kaggle/p029-public957-r6-v1-20261015/manifest.json', self.parent_manifest),
            'R6_SHA': ('artifacts/research/20261015-r6-v1-manifest-only-v1/manifest.json', self.r6)}
        patches = {'ROOT': self.root}
        for name, (relative, value) in mapping.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = (json.dumps(value) + '\n').encode()
            path.write_bytes(raw)
            patches[name] = P029.sha(raw)
        evidence = {}
        for label, (relative, value) in self.public.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = json.dumps(value).encode()
            path.write_bytes(raw)
            evidence[label] = (relative, P029.sha(raw))
        patches['PUBLIC_EVIDENCE'] = evidence
        return patches

    def prepare(self, patches=None):
        pins = self.fixture_pins()
        pins.update(patches or {})
        with mock.patch.multiple(P029, **pins):
            return P029.prepare(self.output, self.slug)

    def test_01_preparation_never_executes_science_and_preserves_other_sources(self):
        result = self.prepare()
        book = json.loads((self.output / 'inference-mount-adapter.ipynb').read_bytes())
        for index in range(2, 9):
            before, after = P029.source_text(self.parent['cells'][index]), P029.source_text(book['cells'][index + 2])
            if index == 6:
                self.assertEqual(after.replace(P029.finder_branch(self.slug), '', 1), before)
                self.assertEqual(P029.embedded_sources(before), P029.embedded_sources(after))
                self.assertIn('FLIP_TTA = True\nFOLDS = 4', after)
            else:
                self.assertEqual(before, after)
        self.assertEqual(P029.source_text(book['cells'][3]), 'Original OAI credits\n')
        for c in book['cells']:
            if c['cell_type'] == 'code':
                self.assertEqual(c['outputs'], [])
                self.assertIsNone(c['execution_count'])
        self.assertFalse(result['verification_boundary']['upload'])
        self.assertIsNone(result['verification_boundary']['own_Public_score'])
        self.assertEqual(result['stage_settings_required']['ref'], 'rsraki/' + self.slug)
        for name, receipt in result['files'].items():
            raw = (self.output / name).read_bytes()
            self.assertEqual(P029.sha(raw), receipt['sha256'])

    def test_02_all_three_parent_evidence_hashes_fail_before_output(self):
        for key in ('SOURCE_SHA', 'PARENT_MANIFEST_SHA', 'R6_SHA'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Pinned source changed'):
                self.prepare({key: 'f' * 64})
            self.assertFalse(self.output.exists())

    def test_03_public_metadata_evidence_hashes_are_pinned(self):
        for label in self.public:
            pins = self.fixture_pins()
            relative, expected = pins['PUBLIC_EVIDENCE'][label]
            (self.root / relative).write_text('{}', encoding='utf-8')
            with mock.patch.multiple(P029, **pins), self.assertRaisesRegex(ValueError, 'Pinned source changed'):
                P029.prepare(self.output, self.slug)
            self.assertFalse(self.output.exists())

    def test_04_existing_run_rejected_before_any_source_read(self):
        self.output.mkdir()
        marker = self.output / 'keep.txt'
        marker.write_text('keep', encoding='utf-8')
        with mock.patch.object(P029, 'read_pinned') as read, self.assertRaises(FileExistsError):
            P029.prepare(self.output, self.slug)
        read.assert_not_called()
        self.assertEqual(marker.read_text(encoding='utf-8'), 'keep')

    def test_05_explicit_safe_slug_and_output_are_required(self):
        for slug in ('../escape', 'owner/slug', "slug'bad", 'UPPER', 'four', 'bad\\path', '-badslug', 'badslug-', 'a' * 51):
            with self.subTest(slug=slug), self.assertRaises(ValueError):
                P029.prepare(self.output, slug)
        for args in ([], ['--stage-slug', self.slug], ['--output-dir', str(self.output)]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                P029.main(args)
        self.assertFalse(self.output.exists())

    def test_06_finder_requires_unique_top_level_original_branch(self):
        source = P029.source_text(self.parent['cells'][6])
        after = P029.changed_finder(source, self.slug)
        branch = P029.finder_branch(self.slug)
        self.assertEqual(after.count(branch), 1)
        self.assertNotIn('rglob', branch)
        self.assertIn("/ 'rsraki' / '" + self.slug + "' / slug", branch)
        for bad in (source.replace(P029.ANCHOR, ''), source.replace(P029.ANCHOR, P029.ANCHOR * 2), source.replace('def find_mount(', 'def other(')):
            with self.subTest(bad=bad[:50]), self.assertRaises(ValueError):
                P029.changed_finder(bad, self.slug)

    def test_07_main_temp_setup_creates_missing_parent_and_rejects_redirected_parent(self):
        self.prepare()
        stage = json.loads((self.output / 'stage.ipynb').read_bytes())
        source = P029.source_text(stage['cells'][-1])
        helpers = {'__name__': 'synthetic_stage_helpers_main_disabled'}
        exec(compile(source, 'synthetic_stage_helpers_main_disabled', 'exec'), helpers)
        fake_kaggle = self.root / 'fake-kaggle'
        fake_kaggle.mkdir()
        fake_parent = fake_kaggle / 'temp'
        child = fake_parent / 'new-unverified'
        self.assertFalse(fake_parent.exists())
        def mapped_path(value):
            return fake_parent if value == '/kaggle/temp' else Path(value)
        with mock.patch.dict(helpers, {'Path': mapped_path}):
            helpers['make_temporary_directory'](child)
            self.assertTrue(child.is_dir())
            with self.assertRaises(FileExistsError):
                helpers['make_temporary_directory'](child)
            redirected = mock.Mock()
            redirected.parent.is_symlink.return_value = True
            with self.assertRaises(RuntimeError):
                helpers['make_temporary_directory'](redirected)
            redirected.mkdir.assert_not_called()
        main = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        self.assertTrue(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'make_temporary_directory' for n in ast.walk(main)))
        # Exercise main itself with all absolute Kaggle paths redirected to an
        # artificial root. Stop at a fake first transfer, before any HTTP/model.
        fake_main = self.root / 'fake-main-kaggle'
        (fake_main / 'working').mkdir(parents=True)
        (fake_main / 'input').mkdir()
        self.assertFalse((fake_main / 'temp').exists())
        class RedirectedPath:
            def __new__(cls, value):
                if value.startswith('/kaggle/'):
                    return fake_main / value[len('/kaggle/'):]
                return self.root / 'absent-device' if value == '/dev/nvidia0' else Path(value)
            @staticmethod
            def cwd():
                return fake_main / 'working'
        class FirstTransferReached(Exception):
            pass
        def fake_first_transfer(opener, name, destination):
            self.assertEqual(name, 'manifest.json')
            self.assertTrue(destination.parent.is_dir())
            self.assertEqual(destination.parent.parent, fake_main / 'temp')
            raise FirstTransferReached()
        with (mock.patch.dict(helpers, {'Path': RedirectedPath, 'download_verified': fake_first_transfer}),
              mock.patch.object(helpers['urllib'].request, 'build_opener', return_value=object()) as opener,
              self.assertRaises(FirstTransferReached)):
            helpers['main']()
        opener.assert_called_once()

    def test_08_generated_sources_remain_bounded_anonymous_and_all_member_guarded(self):
        result = self.prepare()
        stage = json.loads((self.output / 'stage.ipynb').read_bytes())
        source = P029.source_text(stage['cells'][-1])
        adapter = json.loads((self.output / 'inference-mount-adapter.ipynb').read_bytes())
        guard = P029.source_text(adapter['cells'][1])
        self.assertIn("'datasetVersionNumber': 1", source)
        self.assertIn("'raw': 'true'", source)
        self.assertIn('ProxyHandler({})', source)
        self.assertNotIn('authenticate(', source)
        self.assertNotIn('extractall', source)
        self.assertNotIn('torch.load', source)
        self.assertIn('for _p029_name, _p029_size in EXPECTED_SIZES.items()', guard)
        self.assertIn('is_relative_to', guard)
        self.assertIn('is_symlink', guard)
        self.assertFalse(stage['metadata']['kaggle']['isGpuEnabled'])
        self.assertTrue(stage['metadata']['kaggle']['isInternetEnabled'])
        self.assertEqual(adapter['metadata']['kaggle'], {'isGpuEnabled': True, 'isInternetEnabled': False})
        self.assertNotIn('dataSources', adapter['metadata']['kaggle'])
        self.assertNotIn('authorRuntime', adapter['metadata']['kaggle'])
        self.assertEqual(len(result['source_file_sha256']), 9)
        self.assertEqual(len(result['comparison']['embedded_modules']), 7)
        self.assertEqual(result['inference_settings_required']['kernel_sources'], ['rsraki/' + self.slug + '/1'])


if __name__ == '__main__':
    unittest.main()
