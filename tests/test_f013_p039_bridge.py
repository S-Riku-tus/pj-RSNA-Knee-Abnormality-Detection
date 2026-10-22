"""Stdlib checks for constructor boundaries, source transport and refusal gates."""
import ast
import copy
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import f013_p039_bridge as F
import prepare_f009_p033_native20 as B
import prepare_p048_f013_native20 as P
from f013_cuda_evidence import validate_prior_cuda


class BridgeSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.control = B.inspect_control()
        cls.router = cls.control["components"]["serving/serving_router.py"]

    def test_router_only_three_declared_ast_boundaries(self):
        old = next(n for n in ast.parse(self.router).body if isinstance(n, ast.FunctionDef) and n.name == "execute")
        current = F.derive_execute(self.router).body[0]
        old_result = next(n for n in ast.walk(old) if isinstance(n, ast.Assign) and len(n.targets) == 1
                          and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "result")

        class Undo(ast.NodeTransformer):
            def visit_Assign(self, node):
                text = ast.dump(node)
                if text == ast.dump(ast.parse('mixed = plan["F013_mixed"]').body[0]):
                    return ast.parse("import isolated_mixed_runtime as mixed").body[0]
                if text == ast.dump(ast.parse('result = plan["F013_initial"](args, plan, determinism)').body[0]):
                    return copy.deepcopy(old_result)
                return self.generic_visit(node)

            def visit_Subscript(self, node):
                if ast.dump(node) == ast.dump(ast.parse('plan["F013_expected_fingerprints"]', mode="eval").body):
                    return ast.parse('plan["values"]["CPU_smoke"]["model_fingerprints"]', mode="eval").body
                return self.generic_visit(node)

        self.assertEqual(ast.dump(Undo().visit(current)), ast.dump(old))
        with self.assertRaises(ValueError):
            F.derive_execute(self.router + b"\n")

    def test_pair_preserves_REG_math_and_rejects_second_factor(self):
        config = {"seed": 1, "preprocess": {"image_size": 392}, "head": {"dropout": .2}, "augmentation": None}
        old = {"family_pairs": {"CLS": {"bindings": {n: {"config": copy.deepcopy(config)} for n in (0, 1)}},
                                "REG": {"unchanged": ["state", 123]}}, "pair_fingerprint_sha256": "old",
               "model_classes": {"CLS": "old.cls", "REG": "old.reg"}, "FP32_aggregation": "fixed",
               "targets": ["t%d" % n for n in range(12)], "family_fold_manifest_sha256": {"CLS": {}, "REG": {0: "x", 1: "y"}}}
        policy = {"fixed_checkpoints": {str(n): {"config": dict(config, augmentation={"minimum": .85, "maximum": 1.15}),
                        "epoch": 12, "fold": n, "sha256": str(n)} for n in (0, 1)}}
        saved = copy.deepcopy(old)
        result = F.new_pair(old, policy)
        self.assertEqual(old, saved)
        for key in ("REG",):
            self.assertEqual(result["family_pairs"][key], old["family_pairs"][key])
        self.assertEqual(result["FP32_aggregation"], old["FP32_aggregation"])
        self.assertEqual(result["targets"], old["targets"])
        policy["fixed_checkpoints"]["1"]["config"]["head"] = {"dropout": .3}
        with self.assertRaises(ValueError):
            F.new_pair(old, policy)

    def test_original_child_bootstrap_and_run_are_exact(self):
        old = ast.parse(self.control["outer_files"]["p032_native_child.py"])
        new = ast.parse(P.child_source(self.control["outer_files"]["p032_native_child.py"]))
        for name in ("BOOTSTRAP",):
            self.assertEqual(ast.dump(B.literal_assignment(old, name)), ast.dump(B.literal_assignment(new, name)))
        for name in ("run", "diagnostic_fixture", "pin_spawn_router_source"):
            left = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == name)
            right = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertEqual(ast.dump(left), ast.dump(right))

    def test_extension_numbers_and_blend_call_unchanged(self):
        old = ast.parse(self.control["outer_files"]["p032_parent_extension.py"])
        new = ast.parse(P.extension_source(self.control["outer_files"]["p032_parent_extension.py"]))
        def numbers(tree):
            return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and type(n.value) in (int, float)]

        self.assertEqual(numbers(old), numbers(new))

        def calls(tree):
            return [ast.dump(n) for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "blend_csv"]

        self.assertEqual(calls(old), calls(new))

    def test_source_only_package_cannot_enter_and_keeps_all_other_members(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "candidate"
            manifest = P.prepare(target, "rsraki/p048-f013-source-preview-20261020-v1")
            self.assertFalse(manifest["actual_entry_enabled"])
            cells = B.code_cells(json.loads((target / "source.ipynb").read_bytes()))
            with self.assertRaisesRegex(RuntimeError, "actual prior CUDA proof absent"):
                exec(cells[0], {})
            with zipfile.ZipFile(io.BytesIO((target / "runtime-sources.bin").read_bytes())) as z:
                for name, raw in self.control["outer_files"].items():
                    if name not in ("p032_native_child.py", "p032_parent_extension.py"):
                        self.assertEqual(z.read(name), raw)
                self.assertEqual(z.read("f013_cpu_qualifier.py"), (B.ROOT / "scripts/f013_cpu_qualifier.py").read_text(encoding="utf8").encode())
            payload = json.loads((target / "save-payload.json").read_bytes())
            self.assertEqual(payload["dataset_data_sources"][-1], "rsraki/f013-gamma-e12-qualification-20261020-v1/1")
            self.assertNotIn("torch", sys.modules)

    def test_incomplete_GPU_proof_refused_before_model_access(self):
        with tempfile.TemporaryDirectory(dir=B.ROOT / "artifacts/research", prefix="P048-invalid-proof-") as temporary:
            p = Path(temporary) / "proof.json"
            p.write_text(json.dumps({"schema_version": "F013_actual_artificial_CUDA_root_evidence_v1",
                         "source_SHA": "c801dbf33a0432aff56df1334d0d4341e0136d8867aff7c74fd0b5ffe88cf75d",
                         "all_pass": True, "checks": {}, "kernelId": 1, "saved_version": 1}))
            with self.assertRaises(ValueError):
                P.qualified_release(p)

    def test_one_nonpositive_weak_family_fold_refuses_even_source_eligibility(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary) / "weak.json"
            p.write_text(json.dumps({"complete": True, "bothfold_fixed10_positive": True,
                         "policy_SHA": F.NATIVE50_POLICY_SHA,
                         "folds": {"0": {"fixed10_delta": .001}, "1": {"fixed10_delta": 0.}}}))
            with patch.object(P, "NATIVE50", p), patch.object(P, "NATIVE50_SHA", B.raw_pin(p.read_bytes())["sha256"]):
                with self.assertRaises(ValueError):
                    P.qualified_release(None)

    def test_actual_local_CUDA_proof_enables_only_current_T4_gated_entry(self):
        path = B.ROOT / "artifacts/research/20261021-F013-local-CUDA-root-evidence-v1/root-evidence.json"
        release, raw = P.qualified_release(path)
        self.assertTrue(release["actual_entry_enabled"])
        self.assertEqual(release["prior_CUDA_hardware"], "NVIDIA GeForce RTX 4090")
        self.assertTrue(release["current_twoT4_GPU_gate_still_required"])
        self.assertEqual(validate_prior_cuda(raw)["kind"], "actual_local_RTX4090")

    def test_modified_GPU_proof_refused(self):
        path = B.ROOT / "artifacts/research/20261021-F013-local-CUDA-root-evidence-v1/root-evidence.json"
        raw = path.read_bytes()
        with self.assertRaises(ValueError):
            validate_prior_cuda(raw + b" ")

    def test_new_producer_import_follows_original_CPU_guard(self):
        events = []

        class ReachedNewProducer(Exception):
            pass

        def original_guard(*args, **kwargs):
            events.append("old_CPU_guard")
            return {}, {"model_fingerprints": {}}

        def new_import(name):
            self.assertEqual(events, ["old_CPU_guard"])
            self.assertEqual(name, "source14.native_training_fast16")
            raise ReachedNewProducer

        mixed = object.__new__(F.Mixed)
        mixed.original = SimpleNamespace(reconstruct_cpu_models=original_guard)
        mixed.old_pair, mixed.trainer = {}, None
        mixed.plan = {"values": {"CPU_smoke": {"model_fingerprints": {}}}}
        mixed.package = Path("artificial-source-package")
        with patch.dict(sys.modules, {"torch": ModuleType("torch")}), patch.object(sys, "path", list(sys.path)), \
                patch.object(F.importlib, "import_module", side_effect=new_import):
            with self.assertRaises(ReachedNewProducer):
                mixed.reconstruct_cpu_models({}, {}, {}, {}, {}, core=object())

    def test_package_preparation_cannot_import_Torch_producer(self):
        tree = ast.parse((B.ROOT / "scripts/f013_p039_bridge.py").read_text(encoding="utf8"))
        prepare = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "prepare_package")
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                             and n.func.attr == "import_module" for n in ast.walk(prepare)))


if __name__ == "__main__":
    unittest.main()
