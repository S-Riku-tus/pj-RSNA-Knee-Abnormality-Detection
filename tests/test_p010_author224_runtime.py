"""Author224 source/strict loading/shared-input contracts using CPU-only artificial inputs."""

import ast
import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = ROOT / "artifacts/research/20261008-public-candidates/nartaa-rsna-knee-0945-efficient-224crop-v2/source.ipynb"
spec = importlib.util.spec_from_file_location("p010_author224_cpu_tests", ROOT / "scripts/p010_author224_runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class TinyEncoder(torch.nn.Module):
    num_features = 4

    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(3, 4)

    def forward(self, x):
        return self.projection(x.mean(dim=(-1, -2)))


class TinyTimm:
    def __init__(self):
        self.calls = []

    def create_model(self, arch, **kwargs):
        self.calls.append((arch, kwargs))
        return TinyEncoder()


class Author224Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stock_source = runtime.build_stock_source(AUTHOR)
        cls.author = json.loads(AUTHOR.read_text(encoding="utf-8"))
        # Distinct slices and both image axes exercise crop/interpolation order.
        axis = np.arange(384, dtype=np.uint16)
        plane = (axis[:, None] * 5 + axis[None, :] * 7) % 256
        cls.vol = ((np.arange(96, dtype=np.uint16)[:, None, None] * 11 + plane) % 256).astype(np.uint8)
        cls.vol.flags.writeable = False

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="p010-artificial-cpu-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.timm = TinyTimm()
        self.ns = runtime.make_stock_namespace(self.stock_source, timm_module=self.timm, artificial_device="cpu")
        torch.manual_seed(20261008)
        bb = self.ns["build_backbone"](runtime.ARCH, pretrained=False)
        model = self.ns["RaptorClassifier"](bb, F_dim=bb.num_features).eval()
        self.ck = {"arch": runtime.ARCH, "res": 224, "model": model.state_dict()}
        self.path = self.root / "artificial.pt"
        torch.save(self.ck, self.path)
        self.sha = runtime.sha256_file(self.path)

    def load(self, ck=None):
        if ck is not None:
            torch.save(ck, self.path)
        return runtime.load_expert224(self.path, "cpu", stock_namespace=self.ns,
                                      artificial_expected_sha256=runtime.sha256_file(self.path))

    def test_exact_nine_literals_defaults_decorator_and_full_source_hash(self):
        self.assertEqual(hashlib.sha256(self.stock_source.encode()).hexdigest(), runtime.STOCK_SOURCE_SHA256)
        nodes = {node.name: node for node in ast.parse(self.stock_source).body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        self.assertEqual(set(nodes), set(runtime.SYMBOL_SHA256))
        for name, node in nodes.items():
            self.assertEqual(hashlib.sha256(ast.get_source_segment(self.stock_source, node).encode()).hexdigest(),
                             runtime.SYMBOL_SHA256[name], name)
        self.assertEqual([ast.unparse(d) for d in nodes["infer_probs"].decorator_list], ["torch.no_grad()"])
        self.assertEqual(ast.unparse(nodes["gpu_windows"].args.defaults[-1]), "NORM")
        self.assertIn("import numpy as np", self.stock_source)
        self.assertNotIn("import timm", self.stock_source)
        self.assertEqual(self.ns["LAB"], list(runtime.LABELS))
        self.assertEqual(self.ns["SLOTS"], [("Sagittal", 1, 26), ("Sagittal", 0, 22),
                                          ("Coronal", 1, 18), ("Coronal", 0, 12), ("Axial", -1, 18)])

    def test_source_metadata_numerics_constants_and_decorator_tampering_rejected(self):
        for path in ("metadata", "source"):
            book = copy.deepcopy(self.author)
            if path == "metadata":
                book["metadata"]["unapproved"] = True
            else:
                book["cells"][4]["source"] = self.ns["__name__"]
            with self.subTest(path=path), self.assertRaises(runtime.Expert224ContractError):
                runtime.build_stock_source(book)
        for altered in (self.stock_source.replace("CROP_FRAC = 0.8", "CROP_FRAC = 0.7"),
                        self.stock_source.replace("@torch.no_grad()", ""),
                        self.stock_source + "extra = 1\n"):
            with self.assertRaises(runtime.Expert224ContractError):
                runtime.make_stock_namespace(altered, timm_module=self.timm, artificial_device="cpu")

    def test_namespace_isolation_and_explicit_artificial_boundaries(self):
        other = runtime.make_stock_namespace(self.stock_source, timm_module=TinyTimm(), artificial_cpu=True)
        self.assertIsNot(other["_WLUT"], self.ns["_WLUT"])
        self.assertIsNot(other["_MEAN"], self.ns["_MEAN"])
        self.assertIsNot(other["RaptorClassifier"], self.ns["RaptorClassifier"])
        self.ns["_WLUT"]["test"] = object()
        self.assertNotIn("test", other["_WLUT"])
        with self.assertRaisesRegex(runtime.Expert224ContractError, "Injected"):
            runtime.make_stock_namespace(self.stock_source, timm_module=self.timm)
        with self.assertRaisesRegex(runtime.Expert224ContractError, "injected tiny"):
            runtime.make_stock_namespace(self.stock_source, artificial_device="cpu")
        with self.assertRaisesRegex(runtime.Expert224ContractError, "Conflicting"):
            runtime.make_stock_namespace(self.stock_source, timm_module=self.timm,
                                          artificial_cpu=True, artificial_device="cuda")
        cuda = runtime.make_stock_namespace(self.stock_source, timm_module=TinyTimm(), artificial_device="cuda")
        with self.assertRaisesRegex(runtime.Expert224ContractError, "explicit artificial device"):
            runtime.load_expert224(self.path, "cpu", stock_namespace=cuda, artificial_expected_sha256=self.sha)
        with mock.patch.object(torch.cuda, "is_available", return_value=False), mock.patch.object(torch, "load") as load:
            with self.assertRaisesRegex(runtime.Expert224ContractError, "available CUDA"):
                runtime.load_expert224(self.path, "cuda", stock_namespace=cuda, artificial_expected_sha256=self.sha)
            load.assert_not_called()

    def test_strict_load_receipt_keeps_gold_and_label_uncertainty_explicit(self):
        model, receipt = self.load()
        self.assertFalse(receipt["production_qualified"])
        self.assertEqual(receipt["status"], "artificial_cpu_passed")
        self.assertEqual(receipt["checkpoint_sha256"], self.sha)
        self.assertTrue(receipt["strict_load"])
        self.assertFalse(receipt["checkpoint_label_metadata_present"])
        self.assertTrue(receipt["author_gold_selection_reuse"])
        self.assertFalse(receipt["author_gold_training_exclusion_independently_verified"])
        self.assertFalse(receipt["own_gold_fit_or_selection"])
        self.assertTrue(all(not module.training for module in model.modules()))
        self.assertEqual(self.timm.calls[-1], (runtime.ARCH, {
            "pretrained": False, "num_classes": 0, "in_chans": 3, "global_pool": "avg"}))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, self.ck["model"][key], rtol=0, atol=0)

    def test_checkpoint_sha_explicit_arch_res_head_dtype_finite_labels_and_strict_keys(self):
        with mock.patch.object(torch, "load") as load:
            with self.assertRaisesRegex(runtime.Expert224ContractError, "SHA differs"):
                runtime.load_expert224(self.path, "cpu", stock_namespace=self.ns, artificial_expected_sha256="0" * 64)
            load.assert_not_called()
        with self.assertRaisesRegex(runtime.Expert224ContractError, "explicit test SHA"):
            runtime.load_expert224(self.path, "cpu", stock_namespace=self.ns)
        mutations = []
        for key, value in (("arch", "wrong"), ("res", 320), ("labels", list(reversed(runtime.LABELS)))):
            ck = copy.deepcopy(self.ck)
            ck[key] = value
            mutations.append((key, ck))
        for label, key, value in (
            ("head", "clsb", torch.zeros(11)),
            ("dtype", "clsW", self.ck["model"]["clsW"].double()),
            ("nonfinite", "clsb", torch.full((12,), float("nan"))),
        ):
            ck = copy.deepcopy(self.ck)
            ck["model"][key] = value
            mutations.append((label, ck))
        for label, ck in mutations:
            with self.subTest(label=label), self.assertRaises(runtime.Expert224ContractError):
                self.load(ck)
        ck = copy.deepcopy(self.ck)
        ck["model"]["unexpected.weight"] = torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
            self.load(ck)
        ck = copy.deepcopy(self.ck)
        ck["labels"] = list(runtime.LABELS)
        _, receipt = self.load(ck)
        self.assertTrue(receipt["checkpoint_label_metadata_present"])
        self.assertEqual(receipt["checkpoint_label_metadata_fields"], ["labels"])

    def test_stock_crop_geometry_and_exact_cpu_lut_vs_window_reference(self):
        image = torch.arange(384 * 384).reshape(1, 384, 384)
        cropped = self.ns["_crop"](image)
        self.assertEqual(tuple(cropped.shape), (1, 307, 307))
        self.assertEqual(cropped[0, 0, 0].item(), image[0, 38, 38].item())
        for positions in ([], [10, 90], [10, 48, 90]):
            mask = np.zeros(96, dtype=np.uint8)
            mask[positions] = 1
            mask.flags.writeable = False
            with self.subTest(positions=positions):
                centers = self.ns["_eval_centers"](mask, 96, 94)
                if len(positions) < 3:
                    self.assertEqual(centers, [1] * 94)
                else:
                    self.assertEqual((centers[0], centers[-1]), (11, 89))
                reference = self.ns["eval_windows"](self.vol, mask, 94, 224)
                actual = self.ns["gpu_windows"](self.vol, mask, 94, 224, "cpu")
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                self.assertEqual(tuple(actual.shape), (94, 3, 224, 224))

    def test_allmissing_single_stock_forward_no_grad_finite_and_shared_input_immutable(self):
        model, _ = self.load()
        mask = np.zeros(96, dtype=np.uint8)
        mask.flags.writeable = False
        before = hashlib.sha256(self.vol.tobytes()).hexdigest()
        raw, receipt = runtime.infer_expert224(self.ns, model, self.vol, mask, "cpu")
        self.assertEqual(raw.shape, (12,))
        self.assertTrue(np.isfinite(raw).all())
        self.assertTrue(receipt["allmissing"])
        self.assertEqual(receipt["mask_valid_count"], 0)
        self.assertEqual(receipt["stock_forward_calls"], 1)
        self.assertEqual(receipt["backbone_batch_windows"], 94)
        self.assertEqual(receipt["fallback_calls"], 0)
        self.assertFalse(receipt["cuda_fp16_observed"])
        self.assertFalse(receipt["production_qualified"])
        self.assertEqual(receipt["input_before"], receipt["input_after"])
        self.assertEqual(before, receipt["input_after"]["volume_sha256"])
        self.assertEqual([event["kind"] for event in receipt["forward_events"]], ["model", "backbone"])
        self.assertTrue(all(not event["grad_enabled"] for event in receipt["forward_events"]))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertFalse(model._forward_pre_hooks)
        self.assertFalse(model.backbone._forward_pre_hooks)

    def test_sparse_single_forward_matches_direct_literal_stock_raw12_exactly(self):
        model, _ = self.load()
        mask = np.zeros(96, dtype=np.uint8)
        mask[[10, 48, 90]] = 1
        reference = self.ns["infer_probs"](model, self.ns["gpu_windows"](self.vol, mask, 94, 224, "cpu"), "cpu")
        actual, receipt = runtime.infer_expert224(self.ns, model, self.vol, mask, "cpu")
        np.testing.assert_array_equal(actual, reference)
        self.assertEqual(receipt["mask_valid_count"], 3)
        self.assertFalse(receipt["allmissing"])

    def test_malformed_input_training_mode_and_other_namespace_rejected_before_forward(self):
        model, _ = self.load()
        mask = np.zeros(96, dtype=np.uint8)
        cases = [(self.vol[:95], mask), (self.vol, mask.astype(np.float32)), (self.vol, np.full(96, 2, dtype=np.uint8))]
        for vol, m in cases:
            with self.assertRaises(runtime.Expert224ContractError):
                runtime.infer_expert224(self.ns, model, vol, m, "cpu")
        model.train()
        with self.assertRaisesRegex(runtime.Expert224ContractError, "eval mode"):
            runtime.infer_expert224(self.ns, model, self.vol, mask, "cpu")
        model.eval()
        other = runtime.make_stock_namespace(self.stock_source, timm_module=TinyTimm(), artificial_device="cpu")
        with self.assertRaisesRegex(runtime.Expert224ContractError, "bound to this"):
            runtime.infer_expert224(other, model, self.vol, mask, "cpu")

    def test_unexpected_inference_exception_is_fatal_and_hooks_removed(self):
        model, _ = self.load()
        def fail(_x):
            raise RuntimeError("artificial unexpected backend failure")
        model.backbone.forward = fail
        with self.assertRaisesRegex(RuntimeError, "unexpected backend"):
            runtime.infer_expert224(self.ns, model, self.vol, np.zeros(96, dtype=np.uint8), "cpu")
        self.assertFalse(model._forward_pre_hooks)
        self.assertFalse(model.backbone._forward_pre_hooks)

    def test_input_mutation_and_nonfinite_probability_are_fatal(self):
        model, _ = self.load()
        vol = self.vol.copy()
        original_forward = model.backbone.forward
        def mutate(x):
            vol[0, 0, 0] ^= np.uint8(1)
            return original_forward(x)
        model.backbone.forward = mutate
        with self.assertRaisesRegex(runtime.Expert224ContractError, "mutated"):
            runtime.infer_expert224(self.ns, model, vol, np.zeros(96, dtype=np.uint8), "cpu")
        model, _ = self.load()
        with torch.no_grad():
            model.clsb.fill_(float("nan"))
        with self.assertRaisesRegex(runtime.Expert224ContractError, "nonfinite"):
            runtime.infer_expert224(self.ns, model, self.vol, np.zeros(96, dtype=np.uint8), "cpu")

    def test_source_callable_or_constant_replacement_rejected(self):
        model, _ = self.load()
        mask = np.zeros(96, dtype=np.uint8)
        original = self.ns["gpu_windows"]
        self.ns["gpu_windows"] = lambda *_args: None
        with self.assertRaisesRegex(runtime.Expert224ContractError, "callable was replaced"):
            runtime.infer_expert224(self.ns, model, self.vol, mask, "cpu")
        self.ns["gpu_windows"] = original
        self.ns["CROP_FRAC"] = 1
        with self.assertRaisesRegex(runtime.Expert224ContractError, "constants were changed"):
            runtime.infer_expert224(self.ns, model, self.vol, mask, "cpu")

    def test_LUT_reference_or_inplace_normalization_mutation_rejected(self):
        model, _ = self.load()
        mask = np.zeros(96, dtype=np.uint8)
        original = self.ns["_WLUT"]
        self.ns["_WLUT"] = {}
        with self.assertRaisesRegex(runtime.Expert224ContractError, "LUT dictionary"):
            runtime.infer_expert224(self.ns, model, self.vol, mask, "cpu")
        self.ns["_WLUT"] = original
        self.ns["_MEAN"].add_(1)
        with self.assertRaisesRegex(runtime.Expert224ContractError, "normalization tensors"):
            runtime.infer_expert224(self.ns, model, self.vol, mask, "cpu")


if __name__ == "__main__":
    unittest.main()
