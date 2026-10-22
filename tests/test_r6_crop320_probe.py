import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_r6_crop320_probe as B


class CropProbe(unittest.TestCase):
    def test_only_one_model_argument_changes_without_mutating_manifest(self):
        ns = {'manifest': {'crop': 352, 'backbone': 'convnext_small'},
              'WORKER_ARGS': ['--crop', '352', '--flip-tta', '--compile']}
        original = ns['WORKER_ARGS']
        exec(compile(B.CROP_ONLY, '<artificial-args>', 'exec'), ns)
        self.assertEqual(ns['WORKER_ARGS'], ['--crop', '320', '--flip-tta', '--compile'])
        self.assertEqual(original, ['--crop', '352', '--flip-tta', '--compile'])
        self.assertEqual(ns['manifest']['crop'], 352)
        with self.assertRaises(AssertionError):
            exec(compile(B.CROP_ONLY, '<artificial-args>', 'exec'), ns)

    def test_neural_runtime_assets_inputs_preserved_and_only_provenance_rebound(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'probe'
            B.prepare(out, 'rsraki/test-r6-crop320-probe')
            payload = json.loads((out/'save-payload.json').read_bytes())
            old = B.F.read(B.PARENT/'save-payload.json')
            self.assertEqual({k:v for k,v in old.items() if k not in ('slug','new_title','text')},
                             {k:v for k,v in payload.items() if k not in ('slug','new_title','text')})
            before = [B.F.text(c) for c in json.loads(old['text'])['cells'] if c['cell_type']=='code']
            after = [B.F.text(c) for c in json.loads(payload['text'])['cells'] if c['cell_type']=='code']
            self.assertEqual(after[:3], before[:3])
            self.assertEqual(after[3], B.CROP_ONLY)
            self.assertEqual(after[4].replace(B.POLICY_SHA, B.OLD_POLICY_SHA), before[3])
            for source in after:
                ast.parse(source)
            with self.assertRaises(ValueError):
                B.prepare(out, 'rsraki/test-r6-crop320-probe')

    def test_foreign_owner_refused(self):
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(ValueError):
            B.prepare(Path(temp)/'probe', 'foreign/test-r6-crop320-probe')


if __name__ == '__main__':
    unittest.main()
