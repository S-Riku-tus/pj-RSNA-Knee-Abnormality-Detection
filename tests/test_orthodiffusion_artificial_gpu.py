"""Stdlib-only offline contract/negative tests; never import Torch/upstream code."""
import ast
import base64
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


builder = module("od_artificial_builder", ROOT / "scripts/prepare_orthodiffusion_artificial_gpu.py")
runtime = module("od_artificial_runtime", ROOT / "scripts/orthodiffusion_artificial_gpu_runtime.py")


class ArtificialQualification(unittest.TestCase):
    def test_scientific_import_zero(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("numpy", sys.modules)
        self.assertFalse(any(key.startswith("od_qual_pinned") for key in sys.modules))

    def test_prefix_normalization_and_collision(self):
        obj = object()
        normalized, changes = runtime.normalize_ema({"denoise_fn.module.a": obj, "betas": obj})
        self.assertIs(normalized["denoise_fn.a"], obj)
        self.assertEqual(changes, [["denoise_fn.module.a", "denoise_fn.a"]])
        for invalid in ({}, None, {1: obj}, {"denoise_fn.module.a": obj, "denoise_fn.a": obj}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                runtime.normalize_ema(invalid)
        interior, _ = runtime.normalize_ema({"x.denoise_fn.module.a": obj})
        self.assertEqual(set(interior), {"x.denoise_fn.module.a"})

    def test_complete_metadata_rejects_missing_extra_shape_dtype(self):
        expected = {"a": {"shape": [2, 3], "dtype": "torch.float32"}}
        runtime.validate_metadata(expected, expected)
        for bad in ({}, {**expected, "extra": expected["a"]},
                    {"a": {"shape": [3, 2], "dtype": "torch.float32"}},
                    {"a": {"shape": [2, 3], "dtype": "torch.float16"}}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                runtime.validate_metadata(bad, expected)

    def test_uid_scoped_seed_stable_and_unambiguous(self):
        seed = runtime.scoped_seed("uid", "series", [0, 16], 20261017)
        self.assertEqual(seed, runtime.scoped_seed("uid", "series", [0, 16], 20261017))
        seeds = [runtime.scoped_seed("uid2", "series", [0, 16], 20261017),
                 runtime.scoped_seed("uid", "series2", [0, 16], 20261017),
                 runtime.scoped_seed("uid", "series", [1, 17], 20261017),
                 runtime.scoped_seed("uid", "series", [0, 16], 20261018)]
        self.assertEqual(len({seed, *seeds}), 5)
        self.assertNotEqual(runtime.scoped_seed("ab", "c", [0, 16], 1), runtime.scoped_seed("a", "bc", [0, 16], 1))
        with self.assertRaises(ValueError):
            runtime.scoped_seed("u", "s", [], True)

    def test_actual_original_source_and_exact_trim(self):
        raw, manifest, texts = builder.source_package()
        self.assertEqual(len(manifest["assets"]), 9)
        self.assertEqual(json.loads(raw), manifest)
        vendor, audit = builder.vendor_package(texts)
        self.assertTrue(audit["trimmed_function_AST_exact"])
        self.assertEqual(vendor["unet.py"]["text"].encode(), (builder.OBSERVATION / "upstream/diffusion_model/unet.py").read_bytes())
        self.assertEqual(vendor["modules.py"]["text"].encode(), (builder.OBSERVATION / "upstream/diffusion_model/modules.py").read_bytes())
        self.assertNotIn("logger", audit["imports_by_AST_only"]["fp16_util.py"])

    def test_stage_guard_all_nine_and_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp)
            assets = []
            for index in range(9):
                path = "member" + str(index)
                raw = ("fake-not-a-model" + str(index)).encode()
                (stage / path).write_bytes(raw)
                assets.append({"path": path, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "git_blob_sha1": None})
            manifest = (json.dumps({"assets": assets}) + "\n").encode()
            (stage / "asset-manifest.json").write_bytes(manifest)
            config = {"assets": assets, "stage_manifest_pin": builder.pin(manifest)}
            self.assertEqual(len(runtime.guard_stage(stage, config)), 9)
            (stage / "member8").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                runtime.guard_stage(stage, config)

    def test_fresh_pack_transport_and_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "new"
            builder.prepare(out)
            notebook = json.loads((out / "source.ipynb").read_bytes())
            payload = json.loads((out / "save-payload.json").read_bytes())
            summary = json.loads((out / "manifest.json").read_bytes())
            self.assertEqual(payload["text"].encode(), (out / "source.ipynb").read_bytes())
            self.assertEqual(payload["kernel_data_sources"], [builder.STAGE])
            self.assertEqual(payload["dataset_data_sources"], [])
            self.assertEqual(payload["competition_data_sources"], [])
            self.assertTrue(payload["enable_gpu"] and payload["is_private"])
            self.assertFalse(payload["enable_internet"])
            source = notebook["cells"][1]["source"]
            tree = ast.parse(source)
            raw_assign = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_od_raw" for t in node.targets))
            encoded = ast.literal_eval(raw_assign.value.args[0].args[0])
            raw = zlib.decompress(base64.b85decode(encoded))
            bundle = json.loads(raw)
            self.assertEqual(builder.pin(raw), summary["bundle"])
            self.assertEqual(bundle["runtime"].encode(), (out / "runtime.py").read_bytes())
            self.assertEqual(bundle["config"], json.loads((out / "config.json").read_bytes()))
            self.assertEqual(bundle["vendor"], json.loads((out / "vendor.json").read_bytes()))
            self.assertEqual(summary["config"]["protocol"]["passes"], 3)
            with self.assertRaises(FileExistsError):
                builder.prepare(out)

    def test_runtime_static_cuda_strict_and_no_unsafe_fallback(self):
        tree = ast.parse(builder.RUNTIME.read_bytes())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        loads = [node for node in calls if isinstance(node.func, ast.Attribute) and ast.unparse(node.func) == "torch.load"]
        self.assertEqual(len(loads), 1)
        self.assertEqual({k.arg: ast.literal_eval(k.value) for k in loads[0].keywords}, {"map_location": "cpu", "weights_only": True})
        strict = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "load_state_dict")
        self.assertEqual({k.arg: ast.literal_eval(k.value) for k in strict.keywords}, {"strict": True})
        self.assertFalse(any(ast.unparse(node.func) == "torch.manual_seed" for node in calls))
        self.assertEqual(len(runtime.BUFFER_NAMES), 13)
        self.assertIn("torch.cuda.is_available()", builder.RUNTIME.read_text())


if __name__ == "__main__":
    unittest.main()
