"""Model-specific validator isolation and exact unchanged numerical loop checks."""
import ast
import copy
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsna_knee import convnext_training as T
from rsna_knee import native_training_fast16 as C


class TrainingContract(unittest.TestCase):
    def test_only_declared_encoder_fields_may_change(self):
        config = json.loads((T.ROOT / "configs/experiments/f014-convnext-tiny-native-last2-late-cosine.json").read_bytes())
        self.assertEqual(T.validate_native_config(copy.deepcopy(config)), config)
        bad = copy.deepcopy(config)
        bad["head"]["dropout"] = .3
        with self.assertRaises(ValueError):
            T.validate_native_config(bad)

    def test_old_model_guards_are_not_patched(self):
        self.assertIs(T.preflight_native_training.__globals__["validate_native_config"], T.validate_native_config)
        self.assertIs(C.preflight_native_training.__globals__["validate_native_config"], C.validate_native_config)
        self.assertIsNot(T.validate_native_config, C.validate_native_config)
        old = json.loads((T.ROOT / "configs/experiments/f006-dinov2-native-last2-late-cosine.json").read_bytes())
        with self.assertRaises(ValueError):
            T.validate_native_config(old)

    def test_same_data_audit_and_training_math_after_undoing_instrumentation(self):
        preflight, loop = T.training_core_ast().body
        self.assertEqual(ast.dump(preflight), ast.dump(ast.parse(inspect.getsource(C.preflight_native_training)).body[0]))
        loop.name = "train_native"

        class Undo(ast.NodeTransformer):
            def visit_Constant(self, node):
                if node.value == "F014_convnext_native392_training_run_v1":
                    return ast.Constant(value="native392_training_run_fast16_v1")
                return node

            def visit_Expr(self, node):
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id in (
                        "check_running_budget", "record_training_progress"):
                    return None
                return self.generic_visit(node)

            def visit_keyword(self, node):
                if node.arg == "status" and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "exception_status":
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

    def test_empty_resource_checks_cannot_qualify(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gate.json"
            path.write_text(json.dumps({"schema_version": "F014_native392_CNN_resource_gate_v1", "complete": True,
                                      "all_pass": True, "checks": {}}))
            with self.assertRaises(ValueError):
                T.validate_resource_gate(path, {})


if __name__ == "__main__":
    unittest.main()
