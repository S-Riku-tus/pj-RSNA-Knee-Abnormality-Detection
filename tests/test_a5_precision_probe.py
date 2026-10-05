"""Standard-library runner; use only synthetic arrays and no public models/MRI."""

import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import a5_precision_probe_runtime as runtime
from scripts import build_a5_precision_probe as builder


class PrecisionMetricsTests(unittest.TestCase):
    def test_selection_is_order_independent_and_seeded(self):
        values = [f"synthetic-{index}" for index in range(100)]
        first = runtime.select_uids(values, 64, 20261005)
        self.assertEqual(first, runtime.select_uids(list(reversed(values)) + values, 64, 20261005))
        self.assertNotEqual(first, runtime.select_uids(values, 64, 20261006))
        self.assertEqual(len(set(first)), 64)
        with self.assertRaises(ValueError):
            runtime.select_uids(["a", ""], 2, 1)

    def test_ties_inversions_and_probability_difference(self):
        self.assertEqual(runtime.average_ranks([3, 1, 1, 2]), [3, 0.5, 0.5, 2])
        same = runtime.compare_vectors([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])
        self.assertEqual(same["rank_spearman"], 1.0)
        self.assertEqual(same["pair_inversions"], 0)
        reverse = runtime.compare_vectors([0.1, 0.2, 0.3], [0.3, 0.2, 0.1])
        self.assertEqual(reverse["pair_inversions"], 3)
        self.assertEqual(reverse["rank_spearman"], -1.0)
        self.assertEqual(reverse["max_normalized_rank_difference"], 1.0)
        tied = runtime.compare_vectors([0.1, 0.1, 0.3], [0.1, 0.2, 0.3])
        self.assertEqual(tied["pair_tie_changes"], 1)
        self.assertIsNone(runtime.compare_vectors([0.5, 0.5], [0.1, 0.2])["rank_spearman"])

    def test_nonfinite_and_length_mismatch_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                runtime.compare_vectors([0, 1], [0, value])
        with self.assertRaises(ValueError):
            runtime.compare_vectors([0, 1], [0])

    def test_fixed_asset_hash_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "weights.bin").write_bytes(b"synthetic-model-placeholder")
            contract = {
                "inputs": [
                    {
                        "key": "synthetic",
                        "roots": [str(root)],
                        "files": [
                            {
                                "path": "weights.bin",
                                "bytes": 28,
                                "sha256": hashlib.sha256(b"synthetic-model-placeholder").hexdigest(),
                            }
                        ],
                    }
                ]
            }
            # The fixed size must be checked independently from the digest.
            with self.assertRaises(ValueError):
                runtime.resolve_inputs(contract)
            contract["inputs"][0]["files"][0]["bytes"] = 27
            roots, receipts = runtime.resolve_inputs(contract)
            self.assertEqual(roots["synthetic"], root.resolve())
            self.assertEqual(len(receipts), 1)
            (root / "weights.bin").write_bytes(b"x" * 27)
            with self.assertRaises(ValueError):
                runtime.resolve_inputs(contract)

    def test_runtime_refuses_local_execution_before_any_model_import(self):
        with patch.object(Path, "is_dir", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "restricted to Kaggle"):
                runtime.run_probe({}, "")


class PrecisionNotebookTests(unittest.TestCase):
    def test_definitions_hash_and_only_permitted_top_level_nodes(self):
        source = builder.DEFINITIONS.read_text(encoding="utf-8")
        provenance = json.loads(builder.DEFINITIONS.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(builder.sha256(source.encode()), provenance["definitions_sha256"])
        for node in ast.parse(source).body:
            self.assertIsInstance(
                node, (ast.Expr, ast.Import, ast.ImportFrom, ast.Assign, ast.FunctionDef, ast.ClassDef)
            )
            if isinstance(node, ast.Expr):
                self.assertIsInstance(node.value, ast.Constant)  # docstring, no executed calls
            if isinstance(node, ast.Assign):
                names = {n.id for target in node.targets for n in ast.walk(target) if isinstance(n, ast.Name)}
                self.assertTrue(names <= builder.CONSTANTS)
        self.assertIn("def _micro(images, masks):", source)
        self.assertIn("def build_study(args):", source)
        self.assertNotIn("sub_df = pd.read_csv", source)
        self.assertNotIn("models.append", source)

    def test_exact_extraction_matches_vendored_source_when_present(self):
        source = builder.ROOT / "notebooks/public/vendor/romantamrazov-dinosaur-v32.ipynb"
        if not source.is_file():
            self.skipTest("Vendored source is provided by the scored-source audit")
        definitions, provenance = builder.extract_definitions(source)
        self.assertEqual(definitions, builder.DEFINITIONS.read_text(encoding="utf-8"))
        self.assertEqual(provenance, json.loads(builder.DEFINITIONS.with_suffix(".json").read_text(encoding="utf-8")))

    def test_generator_reproducible_unexecuted_and_no_automatic_adoption(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "probe.ipynb"
            builder.build(path)
            generated = json.loads(path.read_text(encoding="utf-8"))
            for cell in generated["cells"]:
                if cell["cell_type"] == "code":
                    ast.parse("".join(cell["source"]))
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])
            self.assertEqual(
                generated,
                json.loads((builder.ROOT / "notebooks/public/03_a5_precision_probe.ipynb").read_text(encoding="utf-8")),
            )
            with self.assertRaises(FileExistsError):
                builder.build(path)
        contract = builder.probe_contract()
        self.assertEqual(contract["modes"], ["bf16", "fp16", "fp32"])
        checkpoints = next(item["files"] for item in contract["inputs"] if item["key"] == "consolidated")
        self.assertEqual(len(checkpoints), 5)

    def test_label_free_input_is_explicit(self):
        source = (builder.ROOT / "scripts/a5_precision_probe_runtime.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        reads = [
            ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "read_csv"
        ]
        self.assertEqual(reads, ["pd.read_csv(metadata, usecols=cols, dtype={cols[0]: str, cols[1]: str})"])
        self.assertNotIn(' / "train.csv"', source)
        self.assertNotIn(' / "test.csv"', source)
        self.assertNotIn(' / "submission.csv"', source)


class SyntheticForwardTests(unittest.TestCase):
    def test_original_micro_forward_with_synthetic_gpu_inputs(self):
        try:
            import numpy as np
            import torch
        except ImportError:
            self.skipTest("Optional NumPy/PyTorch not installed")
        if not torch.cuda.is_available():
            self.skipTest("Synthetic CUDA forward requires an available GPU")
        # Extract only the exact p002 batching and normalization functions: no timm/cv2/public model import.
        tree = ast.parse(builder.DEFINITIONS.read_text(encoding="utf-8"))
        nodes = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in {"_norm_", "_micro", "predict"}
        ]

        class SyntheticModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Linear(16, 12)

            def forward(self, im, slot, smeta, sidx, batch, vm=None):
                self.asserted_input = (tuple(im.shape), tuple(vm.shape), smeta.shape[1])
                each = self.head(im.mean(dim=(-2, -1)))
                result = torch.zeros(batch, 12, device=im.device, dtype=each.dtype)
                return result.index_add_(0, sidx, each)

        ns = {
            "torch": torch,
            "np": np,
            "DEV": "cuda:0",
            "CFG": {"norm": "none"},
            "LABELS": list(range(12)),
            "MICRO": 8,
            "models": [SyntheticModel().cuda().eval() for _ in range(5)],
        }
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "<synthetic-a5-micro>", "exec"), ns)
        rng = np.random.default_rng(20261005)
        images = rng.integers(1, 255, size=(3, 6, 16, 8, 8), dtype=np.uint8)
        masks = np.full((3, 6), 16, dtype=np.uint8)
        for mode in ("bf16", "fp16", "fp32"):
            result, logits, measured = runtime.measure_once(ns, images, masks, mode, torch, trace_dtypes=True)
            self.assertEqual(result.shape, (5, 3, 12))
            self.assertEqual(logits.shape, result.shape)
            self.assertEqual(measured["nonfinite_logits"], 0)
            self.assertTrue(runtime.check_probability_shape(result, (5, 3, 12))["valid"])
            self.assertGreater(measured["seconds"], 0)
            self.assertTrue(measured["observed_dtypes"])
        masks[-1] = 0
        result, _, _ = runtime.measure_once(ns, images, masks, "fp32", torch)
        self.assertTrue(np.isnan(result[:, -1]).all())
        self.assertFalse(runtime.check_probability_shape(result, (5, 3, 12))["valid"])


if __name__ == "__main__":
    unittest.main()
