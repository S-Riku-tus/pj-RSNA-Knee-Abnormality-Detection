import ast
import csv
import hashlib
import importlib.util
import json
import multiprocessing
import pickle
import subprocess
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


builder = load("p009_profile_test_builder", ROOT / "scripts/build_p009_profile.py")
p006 = load("p006_metrics_test_p009", ROOT / "scripts/p006_profile_runtime.py")
with mock.patch.dict(sys.modules, {"p006_metrics": p006}):
    p007 = load("p007_metrics_test_p009", ROOT / "scripts/p007_profile_runtime.py")
    with mock.patch.dict(sys.modules, {"p007_metrics": p007}):
        runtime = load("p009_profile_test_runtime", ROOT / "scripts/p009_profile_runtime.py")


def functions(source):
    return {node.name: ast.get_source_segment(source, node) for node in ast.parse(source).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


class ProfileContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = builder.serving.control.read_json(builder.serving.BASELINE)
        cls.book, cls.inherited = builder.make_profile()
        cls.original = builder.serving.control.cell_source(cls.baseline, 4)
        cls.altered = builder.serving.modified_main(cls.original)

    def test_exact_sources_same_gold_excluded32cohort_original_code_and_single_reader(self):
        tail = builder.profile.source(self.book["cells"][-1])
        calls = [node for node in ast.walk(ast.parse(tail)) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "paired_fov"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(ast.literal_eval(calls[0].args[1]), self.original)
        self.assertEqual(ast.literal_eval(calls[0].args[2]), self.altered)
        expected_fusion = builder.serving.control.cell_source(self.baseline, 9).replace(
            '_o_data = _o_find("test_series.csv")', '_o_data = "/kaggle/working/p009_profile_data"')
        self.assertEqual(ast.literal_eval(calls[0].args[3]), expected_fusion)
        self.assertEqual(ast.literal_eval(calls[0].args[4]), builder.serving.control.writefile_body(
            builder.serving.control.cell_source(self.baseline, 8)))
        self.assertEqual(len(self.inherited["uids"]), 32)
        self.assertEqual(hashlib.sha256("\n".join(self.inherited["uids"]).encode()).hexdigest(), builder.UID_HASH)
        self.assertTrue(self.inherited["gold_excluded"])
        source_list = [builder.profile.source(cell) for cell in self.book["cells"]
                       if cell["cell_type"] == "code"]
        for index in (3, 5, 6, 7, 8):
            self.assertIn(builder.serving.control.cell_source(self.baseline, index), source_list)
        self.assertEqual(self.book["metadata"], self.baseline["metadata"])
        prelude = source_list[0]
        self.assertIn("KAGGLE_IS_COMPETITION_RERUN", prelude)
        self.assertNotIn("p006_profile_data", prelude)
        self.assertNotIn("train.csv", prelude)

    def test_original_prediction_functions_and_decorators_preserved(self):
        for source in (self.original, self.altered):
            observed = runtime.observed_main_source(source)
            original_functions, changed = functions(source), functions(observed)
            for name in original_functions:
                if name != "main":
                    self.assertEqual(changed[name], original_functions[name], name)
            expected_main = original_functions["main"].replace(
                "    ROOT = find_test_root()", "    ROOT = '/kaggle/working/p009_profile_data'"
            ).replace("            q.put((sid, vol, mask))",
                      "            _p009_masks[sid] = int(mask.sum())\n            q.put((sid, vol, mask))")
            self.assertEqual(changed["main"], expected_main)
            infer = next(node for node in ast.parse(observed).body if isinstance(node, ast.FunctionDef)
                         and node.name == "infer_probs")
            self.assertEqual([ast.unparse(item) for item in infer.decorator_list], ["torch.no_grad()"])
            compile(observed, "p009-observed-main", "exec")
        self.assertNotIn("p009 stale fastread source", runtime.observed_main_source(self.original))

    def test_original_pre_one_Pool_pickle_failure_reproduced_and_registration_fixes_identity(self):
        from multiprocessing.reduction import ForkingPickler

        import numpy as np

        node = next(node for node in ast.parse(self.original).body if isinstance(node, ast.FunctionDef)
                    and node.name == "_pre_one")
        original_code = compile(ast.Module(body=[node], type_ignores=[]), "actual-original-pre-one", "exec")
        previous = {"__name__": "__main__"}
        exec(original_code, previous)
        with self.assertRaises(pickle.PicklingError):
            ForkingPickler.dumps(previous["_pre_one"])
        volume, mask = np.full((3, 4, 4), 17, np.uint8), np.ones(3, np.uint8)
        name = "_p009_test_registered_original_pre_one"
        original_main = sys.modules["__main__"]
        with runtime.registered_namespace(name, {"np": np, "_SER": {}, "_TSDIR": "artificial",
                                                 "_READER": lambda sid, ser, tsdir: (volume, mask)}) as namespace:
            exec(original_code, namespace)
            task = namespace["_pre_one"]
            self.assertEqual(task.__module__, name)
            self.assertIs(pickle.loads(ForkingPickler.dumps(task)), task)
            uid, returned_volume, returned_mask = task("artificial_UID")
            self.assertEqual(uid, "artificial_UID")
            self.assertIs(returned_volume, volume)
            self.assertIs(returned_mask, mask)
        self.assertNotIn(name, sys.modules)
        self.assertIs(sys.modules["__main__"], original_main)
        with self.assertRaisesRegex(RuntimeError, "artificial exception"):
            with runtime.registered_namespace(name):
                raise RuntimeError("artificial exception")
        self.assertNotIn(name, sys.modules)
        with self.assertRaises(AssertionError):
            with runtime.registered_namespace("__main__"):
                pass

    @unittest.skipUnless("fork" in multiprocessing.get_all_start_methods(), "Windows has no Linuxfork; Kaggle runtime gate")
    def test_actual_fork_Pool_artificial_transport_probe(self):
        result = runtime.fork_transport_probe()
        self.assertTrue(result["task_roundtrip_passed"])
        self.assertEqual(result["artificial_tasks"], 3)
        self.assertTrue(result["registered_module_removed"])
        self.assertFalse(result["MRI_model_labels_or_GPU_consumed"])

    def test_source_cache_switch_and_real_import160_preserve_unrelated_files(self):
        allowed = {hashlib.sha256(runtime.embedded_source(source)).hexdigest()
                   for source in (self.original, self.altered)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = runtime.prepare_fastread(self.original, allowed, root)
            namespace = {"os": __import__("os"), "__file__": str(root / "entry.py"),
                         "_FASTREAD_B64": next(ast.literal_eval(node.value) for node in
                                               ast.parse(self.original).body if isinstance(node, ast.Assign)
                                               and any(isinstance(item, ast.Name) and item.id == "_FASTREAD_B64"
                                                       for item in node.targets))}
            loader = next(node for node in ast.parse(self.original).body if isinstance(node, ast.FunctionDef)
                          and node.name == "_load_fastread")
            exec(compile(ast.Module(body=[loader], type_ignores=[]), "original-loader", "exec"), namespace)
            module = namespace["_load_fastread"]()
            self.assertEqual(module.CROP_MM, 140)
            cache = Path(first["specific_cache_file"])
            self.assertTrue(cache.is_file(), "Real source import should create the stale cache fixture")
            unrelated = cache.parent / "unrelated.pyc"
            unrelated.write_bytes(b"preserve unrelated cache")
            second = runtime.prepare_fastread(self.altered, allowed, root)
            self.assertTrue(second["specific_cache_deleted"])
            self.assertEqual(unrelated.read_bytes(), b"preserve unrelated cache")
            self.assertFalse(cache.exists())
            namespace["_FASTREAD_B64"] = next(ast.literal_eval(node.value) for node in
                                            ast.parse(self.altered).body if isinstance(node, ast.Assign)
                                            and any(isinstance(item, ast.Name) and item.id == "_FASTREAD_B64"
                                                    for item in node.targets))
            module = namespace["_load_fastread"]()
            self.assertEqual(module.CROP_MM, 160)
            self.assertEqual(unrelated.read_bytes(), b"preserve unrelated cache")
            (root / "_fastread.py").write_bytes(b"unexpected code")
            with self.assertRaisesRegex(AssertionError, "Unexpected pre-existing"):
                runtime.prepare_fastread(self.altered, allowed, root)

    def test_actual_observed_worker_initializer_reads_global_fov_and_hash(self):
        source = runtime.observed_main_source(self.altered)
        observed_init = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
                             and node.name == "_p009_observe_init")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            module_path = path / "_fastread.py"
            module_path.write_bytes(runtime.embedded_source(self.altered))
            namespace = {
                "os": __import__("os"), "_p009_expected_fov": 160.,
                "_p009_expected_source_hash": p006.sha256(module_path),
                "_p009_worker_directory": str(path),
                "IMG": 384, "SLOTS": [("Sagittal", 1, 26), ("Sagittal", 0, 22),
                                     ("Coronal", 1, 18), ("Coronal", 0, 12), ("Axial", -1, 18)],
            }
            spec = importlib.util.spec_from_file_location("actual_test_fastread", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            def original_init():
                namespace["_READER"] = module.make_builder("study_lut", namespace["IMG"], namespace["SLOTS"],
                                                           hdr_threads=32, px_threads=16)
            namespace["_p009_original_init"] = original_init
            exec(compile(ast.Module(body=[observed_init], type_ignores=[]), "actual-worker-observer", "exec"),
                 namespace)
            namespace["_p009_observe_init"]()
            receipt = json.loads(next(path.glob("*.json")).read_text())
            self.assertEqual(receipt["actual_builder_CROP_MM"], 160)
            self.assertEqual(receipt["actual_builder_source_sha256"], p006.sha256(module_path))
            namespace["_p009_expected_fov"] = 140.
            with self.assertRaises(AssertionError):
                namespace["_p009_observe_init"]()

    def test_replay_only_exact_reader_command_once_stale_reject_and_restoration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cached = root / "cached.csv"
            cached.write_bytes(b"fixed first reader CSV")
            command = ["python", str(root / "own_src/infer.py"), str(root / "p009_profile_data"),
                       "test", str(root / "_own.csv"), "fold0.pt", "fold1.pt", "fold2.pt"]
            original = subprocess.run
            other = mock.Mock(return_value=subprocess.CompletedProcess(["pip"], 0))
            with mock.patch.object(subprocess, "run", other):
                with runtime.replay_reader_once(cached, p006.sha256(cached), root) as state:
                    result = subprocess.run(command, check=True, env={})
                    self.assertEqual(result.returncode, 0)
                    self.assertEqual((root / "_own.csv").read_bytes(), cached.read_bytes())
                    self.assertEqual(state["replay_calls"], 1)
                    subprocess.run(["pip"], check=False)
                    other.assert_called_once_with(["pip"], check=False)
                    with self.assertRaises(AssertionError):
                        subprocess.run(command, check=True, env={})
                self.assertIs(subprocess.run, other)
            self.assertIs(subprocess.run, original)
            with runtime.replay_reader_once(cached, p006.sha256(cached), root):
                with self.assertRaisesRegex(AssertionError, "Old reader CSV"):
                    subprocess.run(command, check=True, env={})
            self.assertIs(subprocess.run, original)
            (root / "_own.csv").unlink()
            with runtime.replay_reader_once(cached, p006.sha256(cached), root):
                with self.assertRaises(AssertionError):
                    subprocess.run(command[:-1], check=True, env={})

    def test_rank_formula_ties_and_wrong_csv_rejected(self):
        self.assertEqual(runtime.percentile_rank([.9, .1, .1, .5]), [1., .375, .375, .75])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uids = ["a", "b", "c", "d"]
            main, reader = [.9, .1, .1, .5], [.1, .5, .9, .5]
            first, second = runtime.percentile_rank(main), runtime.percentile_rank(reader)
            fused = runtime.percentile_rank([.75 * a + .25 * b for a, b in zip(first, second, strict=True)])
            for name, values in (("main", main), ("reader", reader), ("fused", fused)):
                with (root / name).open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["StudyInstanceUID", "ACL"])
                    writer.writerows(zip(uids, values, strict=True))
            self.assertEqual(runtime.fusion_maxdiff(root / "main", root / "reader", root / "fused",
                                                   uids, ["ACL"], {"ACL": .25}), 0)
            with self.assertRaises(AssertionError):
                runtime.fusion_maxdiff(root / "main", root / "reader", root / "fused",
                                       uids[::-1], ["ACL"], {"ACL": .25})

    def test_package_hashes_fresh_paths_and_all_cells_compile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = builder.build(root / "package", root / "profile.ipynb", root / "audit.json")
            self.assertEqual(receipt["main_coat_executions"], 2)
            self.assertEqual(receipt["reader_convnext_executions"], 1)
            self.assertEqual(receipt["notebook_sha256"], builder.profile.digest(root / "profile.ipynb"))
            package = json.loads((root / "package/package.json").read_text(encoding="utf-8"))
            for name, record in package["files"].items():
                self.assertEqual(record["sha256"], builder.profile.digest(root / "package" / name))
            self.assertEqual((root / "package/profile.ipynb").read_bytes(), (root / "profile.ipynb").read_bytes())
            with self.assertRaises(FileExistsError):
                builder.build(root / "package", root / "profile.ipynb", root / "audit.json")


if __name__ == "__main__":
    unittest.main()
