"""Physical FOV routing, stale-source rejection and unchanged-model contracts."""

import ast
import copy
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p009_fov", ROOT / "scripts/build_p009_coat_fov160.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def loader_namespace(main_source, directory):
    tree = ast.parse(main_source)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in ("_load_fastread", "_pre_init", "main")]
    node, _ = builder.fastread(main_source)
    ns = {"__file__": str(Path(directory) / "artificial_entry.py"), "os": os,
          "_FASTREAD_B64": ast.literal_eval(node.value), "CROP_MM": 160.0,
          "FASTREAD_VARIANT": "study_lut", "IMG": 384, "SLOTS": [("Sagittal", 1, 26)],
          "FR_HDR": 32, "FR_PX": 16}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "actual_p009_loader_main", "exec"), ns)
    return ns


def function_segments(source):
    tree = ast.parse(source)
    return {node.name: ast.get_source_segment(source, node) for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


class FOVSourceTests(unittest.TestCase):
    def setUp(self):
        self.baseline = builder.control.read_json(builder.BASELINE)
        self.original = builder.control.cell_source(self.baseline, 4)
        self.candidate = builder.prepare_notebook(self.baseline)
        self.source = builder.control.cell_source(self.candidate, 4)

    def test_only_effective_FOV_and_declared_binding_changes(self):
        result = builder.verify_change(self.baseline, self.candidate)
        self.assertEqual(result["changed_code_cells"], [4])
        self.assertTrue(result["six_other_code_cells_byte_identical"])
        _, before = builder.fastread(self.original)
        _, after = builder.fastread(self.source)
        self.assertEqual(after, before.replace("CROP_MM = 140.0", "CROP_MM = 160.0", 1))
        old, new = function_segments(self.original), function_segments(self.source)
        self.assertEqual([name for name in old if old[name] != new[name]], ["_load_fastread", "main"])
        self.assertEqual(old["load_model"], new["load_model"])
        self.assertIn('model.load_state_dict(ck["model"], strict=True)', new["load_model"])
        self.assertEqual(old["build_backbone"], new["build_backbone"])
        self.assertEqual(old["RaptorClassifier"], new["RaptorClassifier"])
        self.assertEqual(old["_pre_init"], new["_pre_init"])
        self.assertEqual(len(builder.control.syntax_check(self.candidate)), 7)
        self.assertEqual(builder.control.verify_runtime_binding(self.candidate, 16)["effective_convnext_k_per_slot"], 16)

    def test_outer_only_or_extra_model_factor_rejected(self):
        outer_only = copy.deepcopy(self.baseline)
        outer_only["cells"][4]["source"] = self.original.replace("CROP_MM = 140.0", "CROP_MM = 160.0", 1).splitlines(keepends=True)
        with self.assertRaises(ValueError):
            builder.verify_change(self.baseline, outer_only)
        candidate = copy.deepcopy(self.candidate)
        candidate["cells"][4]["source"] = self.source.replace("INPUT_RES = 320", "INPUT_RES = 384", 1).splitlines(keepends=True)
        with self.assertRaises(ValueError):
            builder.verify_change(self.baseline, candidate)

    def test_stale_source_rejected_by_real_main_before_GPU_or_Pool(self):
        with tempfile.TemporaryDirectory() as directory:
            _, original = builder.fastread(self.original)
            (Path(directory) / "_fastread.py").write_text(original, encoding="utf-8")
            ns = loader_namespace(self.source, directory)
            with mock.patch.dict(sys.modules, {"pandas": types.SimpleNamespace()}):
                with self.assertRaisesRegex(AssertionError, "stale fastread source"):
                    ns["main"]()
            self.assertNotIn("_READER", ns)

    def test_each_initializer_reads_actual160_builder_globals(self):
        with tempfile.TemporaryDirectory() as directory:
            for _ in range(6):
                ns = loader_namespace(self.source, directory)
                ns["_pre_init"]()
                self.assertEqual(ns["_READER"].__globals__["CROP_MM"], 160.0)
                self.assertEqual(ns["_READER"].__globals__["LAYOUTS"]["d96"]["IMG"], 384)
                self.assertEqual(ns["_load_fastread"]().CROP_MM, 160.0)

    def test_package_immutable_and_all_hashes_match(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("package", "candidate.ipynb", "audit.json")]
            result = builder.prepare_package(*paths)
            self.assertIsNone(result["candidate_own_public_lb"])
            record = builder.control.read_json(paths[0] / "package.json")
            for name, value in record["files"].items():
                self.assertEqual(builder.control.digest(paths[0] / name), value["sha256"])
            with self.assertRaises(FileExistsError):
                builder.prepare_package(*paths)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy unavailable")
class ArtificialCropTests(unittest.TestCase):
    def test_actual_LUT_plain_and_plain_fallback_have160geometry_same384shape(self):
        import numpy as np

        book = builder.prepare_notebook(builder.control.read_json(builder.BASELINE))
        _, embedded = builder.fastread(builder.control.cell_source(book, 4))
        ns = {"__name__": "artificial_fastread"}
        exec(compile(embedded, "actual_embedded_p009", "exec"), ns)
        crops = []

        def resize(array, output, interpolation):
            crops.append(list(array.shape))
            ys = np.rint(np.linspace(0, array.shape[0] - 1, output[1])).astype(int)
            xs = np.rint(np.linspace(0, array.shape[1] - 1, output[0])).astype(int)
            return array[np.ix_(ys, xs)]

        fake_cv = types.SimpleNamespace(INTER_AREA=3, resize=resize)
        raw = (np.arange(512 * 512) % 4096).astype(np.uint16).reshape(512, 512)
        slice_ = ns["RawSlice"]()
        for name, value in {"raw": raw, "dt": "<u2", "shift": 0, "rs": 1.0, "ri": 0.0,
                            "mono1": False, "cnt": np.bincount(raw.ravel(), minlength=65536),
                            "shape": raw.shape, "size": raw.size}.items():
            setattr(slice_, name, value)
        with mock.patch.dict(sys.modules, {"cv2": fake_cv}):
            outputs = {}
            for fov in (140.0, 160.0):
                ns["CROP_MM"] = fov
                before = len(crops)
                lut = np.zeros((1, 384, 384), np.uint8)
                ns["_fill_lut"](lut, 0, 1, ([slice_], [0.5], 0.5, None), 384)
                plain = np.zeros_like(lut)
                ns["_fill_plain"](plain, 0, 1, [slice_.to_float()], [0.5], 0.5, 384, True, None)
                fallback = np.zeros_like(lut)
                ns["_fill_lut"](fallback, 0, 1, ([slice_.to_float()], [0.5], 0.5, None), 384)
                crop_fn = ns["_mm_crop_resize_fn"](384)
                crop_fn(slice_.to_float(), 0.5)
                self.assertEqual(crops[before:], [[int(fov / 0.5)] * 2] * 4)
                self.assertEqual(lut.shape, plain.shape)
                self.assertEqual(lut.dtype, np.uint8)
                self.assertTrue(np.isfinite(lut).all())
                outputs[fov] = lut
            self.assertFalse(np.array_equal(outputs[140.0], outputs[160.0]))
            before = len(crops)
            ns["CROP_MM"] = 160.0
            ns["_mm_crop_resize_fn"](384)(np.ones((100, 100), np.float32), 1.0)
            self.assertEqual(crops[before:], [[100, 100]])


if __name__ == "__main__":
    unittest.main()
