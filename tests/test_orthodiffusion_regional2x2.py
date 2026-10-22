"""Artificial CPU arrays/metadata only; no MRI decoder, Torch, model or GPU."""
import ast
import base64
import copy
import hashlib
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "artifacts/kaggle/orthodiff-sag-regional2x2-20261017-v1"
PARENT = ROOT / "artifacts/kaggle/orthodiff-sag-fullweak4207-20261017-v2"


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


runtime = module("ortho_regional_cpu", "scripts/orthodiffusion_regional2x2_runtime.py")
full = module("ortho_regional_oldfull_cpu", "scripts/orthodiffusion_fullweak_runtime.py")
builder = module("ortho_regional_builder_cpu", "scripts/prepare_orthodiffusion_regional2x2.py")


class ArrayTensor:
    """Only the tensor operations used by fixed_regional_pool, on fake arrays."""
    def __init__(self, array):
        self.array = array

    def __getitem__(self, index):
        return ArrayTensor(self.array[index])

    def float(self):
        return ArrayTensor(self.array.astype(np.float32))

    def mean(self, dim):
        return ArrayTensor(self.array.mean(axis=dim, dtype=np.float32))

    def cpu(self):
        return self


def tensor_info(values):
    raw = struct.pack("<" + str(len(values)) + "f", *values)
    return {"shape": [1, len(values)], "dtype": "torch.float32", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def fake_success():
    values = [float(i) for i in range(256)]
    return {"StudyInstanceUID": "success", "status": "success", "metadata": {"SeriesInstanceUID": "series", "stratum": "PD_nonFS",
            "selected_sources": [{"SOPInstanceUID": "sop" + str(i), "sha256": str(i)} for i in range(16)]},
            "noise_seed": 123, "normalized_input": {"sha256": "input"}, "noisy_input": {"sha256": "noise"},
            "mid_2": {"sha256": "mid2"}, "pooled": tensor_info(values), "pooled_float32_values": values,
            "normalization_min_max_float32": [0., 1.], "forward_counts": {"denoiser": 1, "decoder_blocks": 0, "final_out": 0}, "all_RNG_restored": True}


class RegionalPooling(unittest.TestCase):
    def test_quadrant_order_channel_concat_and_all_depth_exact_FP32_oracle(self):
        for depth in (1, 2):
            data = np.fromfunction(lambda b, c, d, h, w: c * 256 + d * .25 + h * 16 + w, (1, 256, depth, 16, 16), dtype=np.float32).astype(np.float32)
            def cat(values, dim):
                return ArrayTensor(np.concatenate([value.array for value in values], axis=dim))
            actual = runtime.fixed_regional_pool(ArrayTensor(data), cat).array
            expected = []
            # Independent stdlib sum/divide/FP32 pack on exact artificial values.
            for row_start, col_start in ((0, 0), (0, 8), (8, 0), (8, 8)):
                for channel in range(256):
                    numbers = [float(data[0, channel, d, h, w]) for d in range(depth)
                               for h in range(row_start, row_start + 8) for w in range(col_start, col_start + 8)]
                    expected.append(sum(numbers) / len(numbers))
            self.assertEqual(actual.shape, (1, 1024))
            self.assertEqual(actual.dtype, np.float32)
            self.assertEqual(actual.astype("<f4").tobytes(), struct.pack("<1024f", *expected))
            self.assertNotEqual(actual[0, 0], actual[0, 256])

    def test_parent_scientific_mismatch_and_global_negative_zero_bits_are_fatal(self):
        parent = fake_success()
        current = copy.deepcopy(parent)
        current["data_seconds"] = 1000
        runtime.require_parent_success(current, parent)  # Runtime changes are allowed.
        for field in ("normalized_input", "noisy_input", "mid_2", "pooled"):
            changed = copy.deepcopy(current)
            changed[field]["sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "parity mismatch"):
                runtime.require_parent_success(changed, parent)
        changed = copy.deepcopy(current)
        changed["metadata"]["selected_sources"].reverse()
        with self.assertRaisesRegex(ValueError, "metadata"):
            runtime.require_parent_success(changed, parent)
        changed = copy.deepcopy(current)
        changed["pooled_float32_values"][0] = -0.0  # Python equality alone misses this.
        with self.assertRaisesRegex(ValueError, "FP32 bits"):
            runtime.require_parent_success(changed, parent)

    def test_journal_inherits_exact_failure_line_and_rejects_regional_fabrication(self):
        failure = {"StudyInstanceUID": "failure", "status": "data_failure", "selected_series": None,
                   "reason_type": "PilotDataError", "reason": "unknown_PD", "data_seconds": .125,
                   "feature_forward_calls": 0, "fallback_features": False}
        with tempfile.TemporaryDirectory() as temp:
            original = full.RecordJournal(Path(temp) / "records.jsonl", ["failure", "success"])
            journal = runtime.RegionalJournal(original)
            bad_failure = dict(failure, regional_2x2_float32_values=[0.] * 1024)
            with self.assertRaisesRegex(ValueError, "Inherited failure"):
                journal.append(bad_failure)
            journal.append(failure)
            self.assertEqual(original.path.read_bytes(), runtime.record_line(failure))
            success = fake_success()
            with self.assertRaisesRegex(ValueError, "Exactly1024"):
                journal.append(success)
            success.update(regional_2x2_float32_values=[float(i) for i in range(1024)], parent_scientific_parity_passed=True)
            success["regional_2x2"] = tensor_info(success["regional_2x2_float32_values"])
            journal.append(success)
            self.assertEqual(journal.coverage()["success"], 1)
            self.assertEqual(journal.coverage()["failure"], 1)
            expected = original.path.read_bytes()
            self.assertEqual(journal.finalize(), {"bytes": len(expected), "sha256": hashlib.sha256(expected).hexdigest()})

    def test_actual_failure_branch_precedes_images_and_pack_config_inputs_are_sealed(self):
        text = (PACK / "regional-runtime.py").read_bytes().decode()
        full_text = (PACK / "full-runtime.py").read_bytes().decode()
        self.assertTrue(builder.science_parity(full_text, text)["numeric_normalization_forward_global256_RNG_block_byte_identical"])
        tree = ast.parse(text)
        loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For) and ast.unparse(node.target) == "uid"
                    and ast.unparse(node.iter) == "config['uids']")
        parent_assign = next(node for node in loop.body if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "parent")
        branch = next(node for node in loop.body if isinstance(node, ast.If) and ast.unparse(node.test) == "parent['status'] == 'data_failure'")
        image_try = next(node for node in loop.body if isinstance(node, ast.Try))
        self.assertLess(loop.body.index(branch), loop.body.index(image_try))
        appended, image_calls = [], []
        sentinel = ast.parse("image_calls.append(uid)").body[0]
        actual_branch = ast.Module(body=[ast.For(target=copy.deepcopy(loop.target), iter=ast.Name(id="uids", ctx=ast.Load()),
            body=[copy.deepcopy(parent_assign), copy.deepcopy(branch), sentinel], orelse=[])], type_ignores=[])
        scope = {"uids": ["failed"], "parents": {"failed": {"status": "data_failure"}}, "journal": SimpleNamespace(append=appended.append),
                 "state": {"parent_failures_inherited": 0}, "progress": lambda _: None, "image_calls": image_calls}
        exec(compile(ast.fix_missing_locations(actual_branch), "<actual-inherit-branch>", "exec"), scope)
        self.assertEqual(appended, [{"status": "data_failure"}])
        self.assertEqual(image_calls, [])
        notebook = json.loads((PACK / "source.ipynb").read_bytes())
        carrier = ast.parse(notebook["cells"][1]["source"])
        assignment = next(node for node in carrier.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "_r2_raw" for target in node.targets))
        raw = zlib.decompress(base64.b85decode(ast.literal_eval(assignment.value.args[0].args[0])))
        bundle = json.loads(raw)
        for name in ("base", "pilot", "full"):
            self.assertEqual(bundle["runtimes"][name].encode(), (PARENT / (name + "-runtime.py")).read_bytes())
        self.assertEqual(bundle["runtimes"]["regional"], text)
        config, old = bundle["config"], json.loads((PARENT / "config.json").read_bytes())
        for key in ("regional_parent_input", "regional_source_bindings", "regional_protocol"):
            config.pop(key)
        self.assertEqual(config, old)
        payload, parent_payload = (json.loads((folder / "save-payload.json").read_bytes()) for folder in (PACK, PARENT))
        self.assertEqual(payload["kernel_data_sources"], parent_payload["kernel_data_sources"] + [builder.INPUT])
        payload["kernel_data_sources"].pop()
        for key in ("slug", "new_title", "text"):
            payload[key] = parent_payload[key]
        self.assertEqual(payload, parent_payload)
        self.assertEqual((PACK / "vendor.json").read_bytes(), (PARENT / "vendor.json").read_bytes())
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("pydicom", sys.modules)


if __name__ == "__main__":
    unittest.main()
