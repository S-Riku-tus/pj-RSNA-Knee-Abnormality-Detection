"""Static and artificial contracts for the prepared shared224 candidate."""

import ast
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p010_shared_builder", ROOT / "scripts/build_p010_shared224.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class P010SourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = json.loads(BUILDER.CONTROL.read_text(encoding="utf-8"))
        cls.original = BUILDER.text(cls.baseline["cells"][4])

    def test_other_six_code_cells_and_every_numerical_definition_are_literal(self):
        book, stock = BUILDER.make_notebook()
        sources = [BUILDER.text(cell) for cell in book["cells"]]
        for index in (3, 5, 6, 7, 8, 9):
            self.assertEqual(sources.count(BUILDER.text(self.baseline["cells"][index])), 1)
        altered = BUILDER.shared_main_source(self.original)
        BUILDER.assert_numerical_identity(self.original, altered)
        self.assertEqual(BUILDER.digest(BUILDER.CONTROL), BUILDER.CONTROL_SHA)
        self.assertEqual(stock.count("@torch.no_grad()"), 1)
        self.assertEqual(altered.count("queue_max=QUEUE_MAX"), 1)
        self.assertNotIn("pool.imap_unordered", altered)
        self.assertNotIn(
            "FALLBACK",
            ast.get_source_segment(
                altered,
                next(
                    node
                    for node in ast.parse(altered).body
                    if isinstance(node, ast.FunctionDef) and node.name == "main"
                ),
            ),
        )

    def test_transport_control_has_same_code_except_explicit_expert_boolean(self):
        full = BUILDER.shared_main_source(self.original, expert_enabled=True)
        control = BUILDER.shared_main_source(self.original, expert_enabled=False)
        self.assertEqual(full.replace("_p010_enabled = True and N >= 2", "_p010_enabled = False and N >= 2"), control)
        self.assertLess(full.index("infer_pair_probs(model, xw, xm, d)"), full.index("infer_expert224(_p010_stock"))
        self.assertIn("shared_input_receipt(vol, mask) == before", full)
        self.assertIn("global _p010_state", full)
        self.assertIn('write_submission_exclusive("/kaggle/working/p010_expert_raw.csv"', full)
        # The worker and its initializer are original source, including missing slots.
        for name in ("_pre_one", "_pre_init"):
            original_node = next(
                node
                for node in ast.parse(self.original).body
                if isinstance(node, ast.FunctionDef) and node.name == name
            )
            self.assertIn(ast.get_source_segment(self.original, original_node), full)

    def test_original_source_drift_is_rejected_and_decorators_cannot_be_dropped(self):
        with self.assertRaises(ValueError):
            BUILDER.shared_main_source(
                self.original.replace(
                    "    q = queue.Queue(maxsize=QUEUE_MAX); done_n = [0]; lock = threading.Lock()", "    drift = True"
                )
            )
        altered = BUILDER.shared_main_source(self.original)
        with self.assertRaises(AssertionError):
            BUILDER.assert_numerical_identity(self.original, altered.replace("@torch.no_grad()", "", 1))


if __name__ == "__main__":
    unittest.main()
