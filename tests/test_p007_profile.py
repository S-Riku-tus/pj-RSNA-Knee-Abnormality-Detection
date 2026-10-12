"""Offline source/observer contracts for the paired plain/mirror resource profile."""

import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load("p007_profile_builder", ROOT / "scripts/build_p007_profile.py")
base_runtime = load("p006_metrics_test", ROOT / "scripts/p006_profile_runtime.py")
with mock.patch.dict(sys.modules, {"p006_metrics": base_runtime}):
    runtime = load("p007_metrics_test", ROOT / "scripts/p007_profile_runtime.py")


class ProfilePreparationTests(unittest.TestCase):
    def test_identical_uid_cohort_single_main_exact_reader_sources(self):
        book, inherited = builder.make_profile(32)
        self.assertEqual(inherited["uids"], builder.profile.select_uids(32))
        code = [builder.profile.source(c) for c in book["cells"] if c["cell_type"] == "code"]
        self.assertEqual(sum("class RaptorClassifier" in s for s in code), 1)
        live_main_calls = [n for source in code if not source.startswith("%%writefile ")
                           for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call)
                           and isinstance(n.func, ast.Name) and n.func.id == "main"]
        self.assertEqual(len(live_main_calls), 1)
        self.assertEqual(sum(s.startswith("%%writefile /kaggle/working/own_src/p006_metrics.py") for s in code), 1)
        self.assertEqual(sum(s.startswith("%%writefile /kaggle/working/own_src/p007_metrics.py") for s in code), 1)
        tail = ast.parse(code[-1])
        call = next(n for n in ast.walk(tail) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == "paired_mirror_readers")
        plain, mirrored = ast.literal_eval(call.args[2]), ast.literal_eval(call.args[3])
        baseline = builder.serving.control.read_json(builder.profile.CONTROL)
        serving = builder.serving.control.read_json(builder.serving.NOTEBOOK_PATH)
        self.assertEqual(plain, builder.serving.control.writefile_body(builder.serving.control.cell_source(baseline, 8)))
        self.assertEqual(mirrored, builder.serving.control.writefile_body(builder.serving.control.cell_source(serving, 8)))
        for index in (3, 5, 6, 7, 8):
            self.assertIn(builder.serving.control.cell_source(baseline, index), code)
        observed = runtime.observed_reader_source(mirrored, "mirror")
        self.assertEqual(observed.count("with inference_observer(globals(), 'mirror'):"), 1)
        self.assertIn(builder.serving.RUN_SIGNATURE, observed)
        self.assertIn("p = 0.5 * (p_plain + p_mirror)", observed)

    def test_immutable_package_and_no_submission_filename_at_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            paths = (temporary / "package", temporary / "book.ipynb", temporary / "audit.json")
            result = builder.build(32, *paths)
            self.assertEqual(result["main_coat_executions"], 1)
            self.assertEqual(result["reader_executions"], {"plain_k16": 1, "mirror2view_k16": 1})
            self.assertFalse(result["accuracy_evaluated"])
            package = builder.serving.control.read_json(paths[0] / "package.json")
            for name, record in package["files"].items():
                self.assertEqual(builder.profile.digest(paths[0] / name), record["sha256"])
            before = builder.profile.digest(paths[1])
            with self.assertRaises(FileExistsError):
                builder.build(32, *paths)
            self.assertEqual(builder.profile.digest(paths[1]), before)
        self.assertIn('Path("/kaggle/working/submission.csv").unlink()', builder.HELPER.read_text(encoding="utf-8"))

    def test_stale_csv_entry_reset_and_fallback_marker_gate(self):
        tree = ast.parse(builder.HELPER.read_text(encoding="utf-8"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "paired_mirror_readers")
        source = ast.unparse(fn)
        self.assertIn("shutil.copyfile('/kaggle/working/_sota_0949.csv', '/kaggle/working/submission.csv')", source)
        self.assertIn("Path('/kaggle/working/_own.csv').unlink(missing_ok=True)", source)
        self.assertIn("log.count('[Apex Fusion SUCCESS] Successfully fused') == 1", source)
        self.assertIn("'[Apex Fusion EXCEPTION]' not in log", source)
        self.assertIn("'[Apex Fusion FALLBACK]' not in log", source)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch unavailable")
class ObserverTests(unittest.TestCase):
    def test_actual_forward_and_mirror_counts_passthrough_restoration(self):
        import torch

        for mode in ("plain", "mirror"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                x = torch.zeros((1, 6, 16, 3, 384, 384), dtype=torch.uint8)
                mask = torch.tensor([[True, False, True, False, True, False]])
                pos = torch.linspace(0, 1, 16).view(1, 1, 16).expand(1, 6, 16).clone()

                class Dataset:
                    studies = ["synthetic-study"]

                    def __getitem__(self, index):
                        return x[0], mask[0], pos[0]

                class Model(torch.nn.Module):
                    def forward(self, values, masks, positions):
                        return torch.zeros((len(values), 12))

                models = [(Model().eval(), 256) for _ in range(3)]
                def original_load(*args, **kwargs):
                    return models
                ns = {"load_models": original_load, "DicomStudyDataset": Dataset, "torch": torch}
                if mode == "mirror":
                    exec(compile(builder.serving.MIRROR_HELPER, "synthetic_mirror", "exec"), ns)
                original_mirror = ns.get("anatomical_lr_mirror")
                original_item = Dataset.__getitem__
                with torch.no_grad(), runtime.inference_observer(ns, mode, temporary) as state:
                    loaded = ns["load_models"](["synthetic1", "synthetic2", "synthetic3"])
                    self.assertIs(loaded, models)
                    original_result = Dataset()[0]
                    self.assertIs(original_result[0]._base, x)
                    for model, _ in loaded:
                        model(x, mask, pos)
                    if mode == "mirror":
                        mirrored = ns["anatomical_lr_mirror"](x, mask, pos)
                        for model, _ in loaded:
                            model(*mirrored)
                self.assertIs(ns["load_models"], original_load)
                self.assertIs(ns.get("anatomical_lr_mirror"), original_mirror)
                self.assertIs(Dataset.__getitem__, original_item)
                gate = runtime.validate_observations(state, mode, 1)
                self.assertEqual(gate["model_forward_calls"], 3 if mode == "plain" else 6)
                self.assertEqual(gate["mirror_transform_calls"], 0 if mode == "plain" else 1)
                self.assertTrue(gate["grad_disabled_at_load_and_all_forwards"])
                self.assertTrue(all(not model._forward_hooks for model, _ in loaded))
                saved = json.loads((Path(temporary) / f"p007_observation_{mode}/inference.json").read_text())
                self.assertTrue(saved["complete"])

    def test_grad_enabled_model_load_is_rejected_before_loading(self):
        import torch

        class Dataset:
            def __getitem__(self, index):
                raise ValueError("not called")

        original = Dataset.__getitem__
        load = mock.Mock(return_value=[])
        ns = {"load_models": load, "DicomStudyDataset": Dataset}
        with tempfile.TemporaryDirectory() as temporary, torch.enable_grad():
            with self.assertRaisesRegex(AssertionError, "disable gradients before loading"):
                with runtime.inference_observer(ns, "plain", temporary):
                    ns["load_models"]([])
            load.assert_not_called()
            self.assertIs(ns["load_models"], load)
            self.assertIs(Dataset.__getitem__, original)

    def test_undecorated_production_reader_source_is_rejected(self):
        baseline = builder.serving.control.read_json(builder.profile.CONTROL)
        source = builder.serving.control.writefile_body(builder.serving.control.cell_source(baseline, 8))
        with self.assertRaisesRegex(AssertionError, "no_grad decorator"):
            runtime.observed_reader_source(source.replace("@torch.no_grad()\n", "", 1), "plain")

    def test_observers_and_hooks_restore_after_failure(self):
        class Dataset:
            def __getitem__(self, index):
                raise ValueError("not called")

        original = Dataset.__getitem__
        def original_load(*args):
            return []
        ns = {"load_models": original_load, "DicomStudyDataset": Dataset}
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "synthetic failure"):
                with runtime.inference_observer(ns, "plain", temporary):
                    raise ValueError("synthetic failure")
            self.assertIs(ns["load_models"], original_load)
            self.assertIs(Dataset.__getitem__, original)
            state = json.loads((Path(temporary) / "p007_observation_plain/inference.json").read_text())
            self.assertFalse(state["complete"])


if __name__ == "__main__":
    unittest.main()
