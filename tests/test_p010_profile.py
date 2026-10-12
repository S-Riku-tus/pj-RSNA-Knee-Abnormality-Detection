"""Profile source/transport/strict numerical tolerances without real MRI or models."""

import ast
import hashlib
import importlib.util
import json
import pickle
import sys
import unittest
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load("_p010_profile_test_builder", ROOT / "scripts/build_p010_profile.py")
p006 = load("_p010_profile_test_p006", ROOT / "scripts/p006_profile_runtime.py")
with mock.patch.dict(sys.modules, {"p006_metrics": p006}):
    p007 = load("_p010_profile_test_p007", ROOT / "scripts/p007_profile_runtime.py")
    with mock.patch.dict(sys.modules, {"p007_metrics": p007}):
        p009 = load("_p010_profile_test_p009", ROOT / "scripts/p009_profile_runtime.py")
        shared = load("_p010_profile_test_shared", ROOT / "scripts/p010_shared_runtime.py")
        author = load("_p010_profile_test_author", ROOT / "scripts/p010_author224_runtime.py")
        with mock.patch.dict(
            sys.modules, {"p009_metrics": p009, "p010_shared_metrics": shared, "p010_author224_metrics": author}
        ):
            runtime = load("_p010_profile_test_runtime", ROOT / "scripts/p010_profile_runtime.py")


class P010ProfileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.book, cls.inherited = builder.make_profile()
        cls.baseline = json.loads(builder.serving.CONTROL.read_text(encoding="utf-8"))

    def test_same_fixed32_gold_group_exclusion_and_literal_four_controls(self):
        receipt = self.inherited
        self.assertEqual(len(receipt["uids"]), 32)
        self.assertTrue(receipt["gold_excluded"] and receipt["gold_group_excluded"])
        self.assertEqual(hashlib.sha256("\n".join(receipt["uids"]).encode()).hexdigest(), builder.UID_SHA)
        tail = builder.serving.text(self.book["cells"][-1])
        call = next(
            node
            for node in ast.walk(ast.parse(tail))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "paired_shared224"
        )
        for index, name in enumerate(("original", "bounded", "shared", "fusion", "reader"), start=1):
            self.assertEqual(ast.literal_eval(call.args[index]), receipt[name])
        self.assertEqual(receipt["shared"], builder.serving.shared_main_source(receipt["original"]))
        sources = [builder.serving.text(cell) for cell in self.book["cells"]]
        for index in (3, 5, 6, 7, 8):
            self.assertEqual(sources.count(builder.serving.text(self.baseline["cells"][index])), 1)
        self.assertIn("KAGGLE_IS_COMPETITION_RERUN", sources[3])
        self.assertNotIn("train.csv", sources[3])

    def test_original_and_bounded_numerical_symbols_decorators_stay_literal(self):
        for name in ("original", "bounded", "shared"):
            source = self.inherited[name]
            observed = runtime.observed_source(source, name, original_transport=name == "original")

            def definitions(value):
                return {
                    node.name: ast.dump(node, include_attributes=False)
                    for node in ast.parse(value).body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name != "main"
                }

            originals, augmented = definitions(source), definitions(observed)
            self.assertEqual({key: augmented[key] for key in originals}, originals)
            self.assertTrue(observed.endswith("main()\n"))
            compile(observed, "<p010-profile-observer>", "exec")
            nodes = {node.name: node for node in ast.parse(observed).body if isinstance(node, ast.FunctionDef)}
            self.assertEqual([ast.unparse(d) for d in nodes["infer_probs"].decorator_list], ["torch.no_grad()"])
            self.assertIn("ForkingPickler", observed)
            self.assertIn("CROP_MM", observed)
            self.assertIn("source_sha256", observed)

    def test_preregistered_strict_raw_tolerance_and_CSV_identity_requirement(self):
        left = [[0.5] * 12 for _ in range(32)]
        self.assertTrue(runtime.raw_difference(left, left)["exact_zero"])
        right = [[0.5 + 5e-8] * 12 for _ in range(32)]
        self.assertLessEqual(runtime.raw_difference(left, right)["maximum_absolute_difference"], 1e-7)
        with self.assertRaises(AssertionError):
            runtime.raw_difference(left, [[0.5 + 1e-4] * 12 for _ in range(32)])
        with self.assertRaises(ValueError):
            runtime.raw_difference(left, left[:-1])
        source = (ROOT / "scripts/p010_profile_runtime.py").read_text(encoding="utf-8")
        self.assertIn('len({report["fused_csv"]["sha256"] for report in fusion_reports.values()}) == 1', source)
        self.assertIn('state["forwards"]', source)

    def test_registered_actual_pre_one_identity_and_cleanup(self):
        original = self.inherited["original"]
        task = next(
            node for node in ast.parse(original).body if isinstance(node, ast.FunctionDef) and node.name == "_pre_one"
        )
        source = compile(ast.Module(body=[task], type_ignores=[]), "<actual-original-pre-one>", "exec")
        name = "_p010_test_actual_pre_one"
        with runtime.registered_namespace(
            name,
            {
                "_READER": lambda sid, _series, _path: ("synthetic-volume", "synthetic-mask"),
                "_SER": {},
                "_TSDIR": "no-MRI",
            },
        ) as namespace:
            exec(source, namespace)
            function = namespace["_pre_one"]
            self.assertIs(pickle.loads(ForkingPickler.dumps(function)), function)
            self.assertEqual(function("uid"), ("uid", "synthetic-volume", "synthetic-mask"))
        self.assertNotIn(name, sys.modules)
        with self.assertRaises(pickle.PicklingError):
            ForkingPickler.dumps(function)

    def test_resource_gate_requires_both_T4_peaks_and_finite_elapsed(self):
        resource = {
            "sampled_gpu_peak_used_mib": [12000, 12000],
            "elapsed_seconds": 100,
            "sampled_peak_process_tree_rss_bytes": 1000000,
            "samples": 2,
        }
        self.assertTrue(runtime.resource_gate(resource, [11900, 11900])["all_passed"])
        for sampled, peak in (
            ([13000, 12000], [12000, 12000]),
            ([12000], [12000, 12000]),
            ([12000, 12000], [float("nan"), 12000]),
        ):
            with self.subTest(sampled=sampled, peak=peak), self.assertRaises(AssertionError):
                runtime.resource_gate({**resource, "sampled_gpu_peak_used_mib": sampled}, peak)


if __name__ == "__main__":
    unittest.main()
