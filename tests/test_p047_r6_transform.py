import ast
import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import p047_r6_transform as T


class Transform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        book = T.F.read(T.F.P033/'source.ipynb')
        cls.stage = [T.F.text(c) for c in book['cells'] if c['cell_type']=='code'][5]

    def test_only_one_scalar_switch_or_crop_changes(self):
        for choice, old, new in [('tta_off', 'FLIP_TTA = True', 'FLIP_TTA = False'),
                                 ('crop320', "'--crop', str(manifest['crop'])", "'--crop', '320'")]:
            changed, info = T.transform_stage(self.stage, choice)
            self.assertEqual(changed.replace(new, old), self.stage)
            self.assertEqual(info['worker_SHA'], T.WORKER_SHA)

    def test_calibration_arguments_removed_while_asset_guards_unchanged(self):
        changed, info = T.transform_stage(self.stage, 'no_calibration')
        want = ast.parse(self.stage)
        changed_nodes = 0
        for node in ast.walk(want):
            if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id=='WORKER_ARGS':
                if isinstance(node.value, ast.List) and node.value.elts and isinstance(node.value.elts[0], ast.Constant) and node.value.elts[0].value in ('--fluid-stacker','--state-stacker'):
                    node.value = ast.List(elts=[], ctx=ast.Load())
                    changed_nodes += 1
        self.assertEqual(changed_nodes, 2)
        self.assertEqual(ast.dump(want, include_attributes=False), ast.dump(ast.parse(changed), include_attributes=False))
        self.assertFalse(info['calibration'])
        self.assertEqual(info['worker_SHA'], T.WORKER_SHA)

    def test_quarter_changes_only_registered_worker_bytes_and_corresponding_SHA(self):
        changed, info = T.transform_stage(self.stage, 'tta_quarter')
        before = ast.literal_eval(T.F.assignment(self.stage, 'SOURCES')[1].value)
        after = ast.literal_eval(T.F.assignment(changed, 'SOURCES')[1].value)
        for key in before.keys()-{'compact_infer.py'}:
            self.assertEqual(before[key], after[key])
        reverse = after['compact_infer.py']
        for a, b in T.QUARTER_EDITS:
            reverse = reverse.replace(b, a)
        self.assertEqual(reverse, before['compact_infer.py'])
        self.assertEqual(hashlib.sha256(after['compact_infer.py'].encode()).hexdigest(), info['worker_SHA'])
        self.assertEqual(info['mirror_weight'], .25)

    def test_unknown_choice_tampered_parent_and_double_transform_refused(self):
        for source, choice in [(self.stage, 'tta_probmean'), (self.stage+'\n', 'tta_off'),
                               (T.transform_stage(self.stage, 'crop320')[0], 'tta_quarter')]:
            with self.assertRaises(ValueError):
                T.transform_stage(source, choice)


if __name__ == '__main__':
    unittest.main()
