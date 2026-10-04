import copy
import json
import tempfile
import unittest
from pathlib import Path

from rsna_knee.contracts import load_config


class CheckpointConfigTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.baseline = json.loads((root / "configs" / "baseline.json").read_text(encoding="utf-8"))

    def load(self, **settings):
        config = copy.deepcopy(self.baseline)
        config["train"].update(settings)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            return load_config(path)

    def test_legacy_config_contents_are_preserved(self):
        self.assertEqual(self.load(), self.baseline)

    def test_explicit_auc_and_milestones(self):
        result = self.load(checkpoint_selection="auc", checkpoint_milestones=[5, 10, 15, 20])
        self.assertEqual(result["train"]["checkpoint_selection"], "auc")
        self.assertEqual(result["train"]["checkpoint_milestones"], [5, 10, 15, 20])

    def test_invalid_selection_and_milestones_fail_before_training(self):
        for selection in (None, "gold", "accuracy", True):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                self.load(checkpoint_selection=selection)
        for milestones in (None, "5", [0], [-1], [True], [5.0], [5, 5]):
            with self.subTest(milestones=milestones), self.assertRaises(ValueError):
                self.load(checkpoint_milestones=milestones)


if __name__ == "__main__":
    unittest.main()
