"""Contracts for observation-only profiling, without MRI or checkpoint access."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load("p006_profile_builder", "build_p006_profile.py")
runtime = load("p006_profile_runtime", "p006_profile_runtime.py")


class ListTensor:
    def __init__(self, values, shape=None):
        self.values = values
        self.shape = shape

    def tolist(self):
        return self.values


class ProfileContracts(unittest.TestCase):
    def test_nested_fixed_cohorts(self):
        small, large = builder.select_uids(32), builder.select_uids(256)
        self.assertEqual(small, large[:32])
        self.assertEqual(len(set(large)), 256)

    def test_notebook_preserves_model_code_and_omits_diagnostic_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "profile.ipynb"
            receipt = builder.build(32, Path(tmp) / "package", target)
            original = json.loads(builder.CONTROL.read_text(encoding="utf-8"))
            result = json.loads(target.read_text(encoding="utf-8"))
            sources = [builder.source(c) for c in result["cells"] if c["cell_type"] == "code"]
            for index in (3, 5, 6, 7, 8):
                self.assertIn(builder.source(original["cells"][index]), sources)
            self.assertEqual(receipt["baseline_sha256"], builder.digest(builder.CONTROL))
            self.assertFalse(receipt["accuracy_evaluated"])
            self.assertNotIn("competition_submit", target.read_text(encoding="utf-8"))
            for cell in result["cells"]:
                if cell["cell_type"] == "code":
                    self.assertEqual(cell["outputs"], [])
                    self.assertIsNone(cell["execution_count"])

    def test_observer_returns_identical_tensors_and_restores_method(self):
        result = (ListTensor(None, (6, 24, 3, 384, 384)), ListTensor([True, False, True, False, False, False]),
                  ListTensor([[0.0, 0.5, 1.0]] * 6))

        class Dataset:
            studies = ["synthetic-study"]

            def __getitem__(self, index):
                return result

        old = Dataset.__getitem__
        with tempfile.TemporaryDirectory() as tmp:
            with runtime.reader_observer(Dataset, 24, tmp):
                self.assertIs(Dataset()[0], result)
            self.assertIs(Dataset.__getitem__, old)
            record = json.loads((Path(tmp) / "p006_cov_k24/000000.json").read_text())
            self.assertEqual(record["uid"], "synthetic-study")
            self.assertEqual(record["shape"], [6, 24, 3, 384, 384])
            self.assertEqual(record["unique_positions_per_slot"], [3] * 6)

    def test_observer_restores_method_after_failure(self):
        class BrokenDataset:
            def __getitem__(self, index):
                raise ValueError("synthetic failure")

        old = BrokenDataset.__getitem__
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                with runtime.reader_observer(BrokenDataset, 16, tmp):
                    BrokenDataset()[0]
            self.assertIs(BrokenDataset.__getitem__, old)


if __name__ == "__main__":
    unittest.main()
