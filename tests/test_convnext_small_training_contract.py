"""Pure F015 contract checks; no model construction, CUDA work or asset reads."""
import ast
import copy
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsna_knee import convnext_model as tiny_model
from rsna_knee import convnext_small_model as M
from rsna_knee import convnext_small_training as T
from rsna_knee import convnext_training as tiny_training
from rsna_knee import native_training_fast16 as C
from rsna_knee.contracts import TARGETS, sha256


class SmallTrainingContract(unittest.TestCase):
    def config(self):
        return json.loads((T.ROOT / "configs/experiments/f015-convnext-small-native-last2-late-cosine.json").read_bytes())

    def test_only_declared_encoder_fields_may_change(self):
        config = self.config()
        self.assertEqual(T.validate_native_config(copy.deepcopy(config)), config)
        for change in ("dropout", "gamma", "encoder"):
            bad = copy.deepcopy(config)
            if change == "dropout":
                bad["head"]["dropout"] = .3
            elif change == "gamma":
                bad["augmentation"] = {"gamma": True}
            else:
                bad["encoder"]["model_name"] = tiny_model.MODEL
            with self.assertRaises(ValueError):
                T.validate_native_config(bad)

    def test_old_DINO_and_Tiny_guards_are_not_patched_or_impersonated(self):
        self.assertIs(T.preflight_native_training.__globals__["validate_native_config"], T.validate_native_config)
        self.assertIs(C.preflight_native_training.__globals__["validate_native_config"], C.validate_native_config)
        self.assertIs(tiny_training.preflight_native_training.__globals__["validate_native_config"],
                      tiny_training.validate_native_config)
        for path in ("configs/experiments/f006-dinov2-native-last2-late-cosine.json",
                     "configs/experiments/f014-convnext-tiny-native-last2-late-cosine.json"):
            with self.assertRaises(ValueError):
                T.validate_native_config(json.loads((T.ROOT / path).read_bytes()))
        self.assertNotEqual(M.CHECKPOINT_SCHEMA, tiny_model.CHECKPOINT_SCHEMA)
        self.assertIsNot(M.ConvNeXtSmallTripletMIL, tiny_model.ConvNeXtTripletMIL)

    def test_same_data_audit_and_training_math_after_undoing_instrumentation(self):
        preflight, loop = T.training_core_ast().body
        self.assertEqual(ast.dump(preflight), ast.dump(ast.parse(inspect.getsource(C.preflight_native_training)).body[0]))
        loop.name = "train_native"

        class Undo(ast.NodeTransformer):
            def visit_Constant(self, node):
                if node.value == "F015_convnext_native392_training_run_v1":
                    return ast.Constant(value="native392_training_run_fast16_v1")
                return node

            def visit_Expr(self, node):
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id in (
                        "check_running_budget", "record_training_progress"):
                    return None
                return self.generic_visit(node)

            def visit_keyword(self, node):
                if (node.arg == "status" and isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Name) and node.value.func.id == "exception_status"):
                    node.value = ast.Constant(value="failed")
                return self.generic_visit(node)

        self.assertEqual(ast.dump(Undo().visit(loop)), ast.dump(ast.parse(inspect.getsource(C.train_native)).body[0]))

    def test_CUDA_guard_precedes_all_training_input_reads(self):
        class Guard(Exception):
            pass

        with patch.object(T, "ensure_cuda", side_effect=Guard), patch.object(T, "validate_native_config") as validator:
            with self.assertRaises(Guard):
                T.preflight_native_training(None, None, None, 0, None)
            validator.assert_not_called()

    def test_empty_or_Tiny_resource_checks_cannot_qualify(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gate.json"
            for schema in ("F015_native392_CNN_resource_gate_v1", "F014_native392_CNN_resource_gate_v1"):
                path.write_text(json.dumps({"schema_version": schema, "complete": True,
                                            "all_pass": True, "checks": {}}))
                with self.assertRaises(ValueError):
                    T.validate_resource_gate(path, {})

    def test_source18_preserves_all_protected_F014_source16(self):
        policy = json.loads(T.POLICY.read_bytes())
        self.assertEqual(sha256(T.POLICY), T.POLICY_SHA)
        self.assertEqual(len(M.SOURCE_FILES), 18)
        self.assertEqual(len(set(M.SOURCE_FILES)), 18)
        self.assertEqual(set(M.SOURCE_FILES) - set(tiny_model.SOURCE_FILES),
                         {"convnext_small_model.py", "convnext_small_training.py"})
        self.assertEqual(len(policy["protected_F014_source16_sha256"]), 16)
        for relative, expected in policy["protected_F014_source16_sha256"].items():
            self.assertEqual(sha256(T.ROOT / relative), expected, relative)

    def test_Small_factory_is_only_declared_Tiny_identity_depth_asset_changes(self):
        old = Path(tiny_model.__file__).read_text(encoding="utf-8")
        expected = (old.replace("F014", "F015").replace("f014", "f015")
                    .replace(tiny_model.MODEL, M.MODEL)
                    .replace(tiny_model.WEIGHTS_SHA, M.WEIGHTS_SHA)
                    .replace("114374272", "200928946")
                    .replace("[3, 3, 9, 3]", "[3, 3, 27, 3]")
                    .replace("Exact Tiny architecture", "Exact Small architecture")
                    .replace("ConvNeXtTripletMIL", "ConvNeXtSmallTripletMIL")
                    .replace('SOURCE_FILES = (\n',
                             'SOURCE_FILES = (\n    "convnext_small_model.py", "convnext_small_training.py",\n'))
        self.assertEqual(ast.dump(ast.parse(expected)), ast.dump(ast.parse(Path(M.__file__).read_text(encoding="utf-8"))))

    def test_artificial_and_Tiny_checkpoints_refused_without_constructing_a_model(self):
        value = {"schema_version": M.CHECKPOINT_SCHEMA, "targets": list(TARGETS),
                 "source_sha256": M.source_hashes(), "gold_used_for_training_or_selection": False,
                 "artificial": True}
        with patch.object(M.torch, "load", return_value=value):
            with self.assertRaises(ValueError):
                M.load_checkpoint("not-read.pt")
            self.assertIs(M.load_checkpoint("not-read.pt", allow_artificial=True), value)
            value["schema_version"] = tiny_model.CHECKPOINT_SCHEMA
            with self.assertRaises(ValueError):
                M.load_checkpoint("not-read.pt", allow_artificial=True)
        with self.assertRaises(ValueError):
            M.apply_checkpoint(object(), {}, {})


if __name__ == "__main__":
    unittest.main()
