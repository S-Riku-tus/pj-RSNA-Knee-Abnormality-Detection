import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_p033", ROOT / "scripts/prepare_p033_public958_weights.py")
P033 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P033)


class VectorBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.before = "prefix = 17\nW_RAPTOR_C = " + repr(P033.PARENT_WEIGHTS) + "  # retain this comment\nsuffix = 19\n"

    def test_changes_only_literal_and_preserves_all_other_bytes(self):
        result = P033.replace_weights(self.before, P033.PUBLIC_WEIGHTS)
        self.assertEqual(result, self.before.replace(repr(P033.PARENT_WEIGHTS), repr(P033.PUBLIC_WEIGHTS)))
        self.assertIn("# retain this comment", result)

    def test_rejects_a_weight_vector_not_in_public_scored_source(self):
        weights = copy.deepcopy(P033.PUBLIC_WEIGHTS)
        weights["Synovitis"] = 0.81
        with self.assertRaises(ValueError):
            P033.replace_weights(self.before, weights)

    def test_rejects_parent_with_already_changed_weights(self):
        text = self.before.replace("'PF OA': 0.5", "'PF OA': 0.6")
        with self.assertRaises(ValueError):
            P033.replace_weights(text, P033.PUBLIC_WEIGHTS)

    def test_rejects_second_assignment(self):
        with self.assertRaises(ValueError):
            P033.replace_weights(self.before + "W_RAPTOR_C = {}\n", P033.PUBLIC_WEIGHTS)

    def test_rejects_a_nested_reassignment(self):
        with self.assertRaises(ValueError):
            P033.replace_weights(self.before + "def unrelated():\n    W_RAPTOR_C = {}\n", P033.PUBLIC_WEIGHTS)

    def test_does_not_execute_a_nonliteral_dictionary(self):
        text = "W_RAPTOR_C = dict(raise_if_executed())\n"
        with self.assertRaises((ValueError, TypeError)):
            P033.replace_weights(text, P033.PUBLIC_WEIGHTS)


@unittest.skipUnless((P033.RECOVERY / "P029M-source.ipynb").is_file() and (P033.PUBLIC / "public958-source.ipynb").is_file(),
                     "Recovered parent and public scored source required for preparation audit")
class RecoveredSourceTests(unittest.TestCase):
    def test_prepared_source_keeps_guard_inputs_and_parent_bytes(self):
        original = (P033.RECOVERY / "P029M-source.ipynb").read_bytes()
        parent = json.loads(original)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "new-run"
            manifest = P033.prepare(output)
            candidate = json.loads((output / "source.ipynb").read_bytes())
            payload = json.loads((output / "save-payload.json").read_bytes())
            self.assertEqual(payload["text"].encode(), (output / "source.ipynb").read_bytes())
            self.assertEqual(payload["kernel_data_sources"], [P033.STAGE])
            self.assertEqual(payload["dataset_data_sources"], P033.DATASETS)
            self.assertTrue(payload["is_private"])
            self.assertFalse(payload["enable_internet"])
            self.assertEqual(len(candidate["cells"]), 12)
            old_code = [P033.source(c) for c in parent["cells"] if c["cell_type"] == "code"]
            new_code = [P033.source(c) for c in candidate["cells"] if c["cell_type"] == "code"]
            self.assertEqual(new_code[:-1], old_code[:-1])
            self.assertEqual(P033.digest(new_code[0].encode()), P033.GUARD_SHA)
            self.assertEqual(manifest["candidate_weights"], P033.PUBLIC_WEIGHTS)
            self.assertFalse(manifest["F006_included"])
            self.assertIsNone(manifest["candidate_own_Public"])
        self.assertEqual((P033.RECOVERY / "P029M-source.ipynb").read_bytes(), original)

    def test_existing_output_is_refused_without_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            sentinel = output / "prior-result.txt"
            sentinel.write_text("keep")
            with self.assertRaises(FileExistsError):
                P033.prepare(output)
            self.assertEqual(sentinel.read_text(), "keep")

    def test_modified_parent_is_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            recovery = folder / "recovery"
            recovery.mkdir()
            (recovery / "P029M-source.ipynb").write_bytes((P033.RECOVERY / "P029M-source.ipynb").read_bytes() + b" ")
            output = folder / "candidate"
            with self.assertRaises(ValueError):
                P033.prepare(output, recovery=recovery)
            self.assertFalse(output.exists())

    def test_other_account_and_unbounded_slug_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for slug in ("heliosli/copy", "rsraki/../other", "rsraki/a" * 30):
                with self.subTest(slug=slug), self.assertRaises(ValueError):
                    P033.prepare(Path(folder) / "unused", slug=slug)


if __name__ == "__main__":
    unittest.main()
