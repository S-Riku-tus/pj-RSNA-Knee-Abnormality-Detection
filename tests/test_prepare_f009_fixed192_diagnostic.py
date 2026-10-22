"""Artificial/source-only diagnostic composition tests; no Torch or real MRI."""
import copy
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_f009_fixed192_builder_tests", ROOT / "scripts/prepare_f009_fixed192_diagnostic.py")
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


class Fixed192Notebook(unittest.TestCase):
    def test_default_and_missing_actual_release_stop_before_input_bodies(self):
        with patch.object(D, "load_base", side_effect=AssertionError("Default input body read forbidden")), redirect_stdout(io.StringIO()):
            self.assertEqual(D.main([]), 0)
            with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "qualified actual metadata"):
                D.prepare(Path(tmp) / "actual", execute=True)

    def test_selection_is_exact_observed_ordered192_source_metadata(self):
        body = D.fixed_selection()
        self.assertEqual(D.raw_pin(body), {"bytes": D.SELECTION_BYTES, "sha256": D.SELECTION_SHA})
        values = json.loads(body)
        self.assertEqual(len(values["UIDs"]), 192)
        self.assertIs(values["train_csv_Report_labels_read"], False)
        changed = copy.deepcopy(values)
        changed["UIDs"][0:2] = changed["UIDs"][1::-1]
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "changed.json"
            target.write_bytes(D.json_bytes(changed))
            with patch.object(D, "SELECTION_PATH", target), self.assertRaisesRegex(ValueError, "Exact observed"):
                D.fixed_selection()

    def test_carrier_reuses_exact_restorer_cell_and_discards_parent_execution(self):
        base = D.load_base()
        control = base.inspect_control()
        files = {"P033-source.ipynb": control["parent_raw"], D.SELECTION_NAME: D.fixed_selection()}
        context = {"actual_entry_enabled": False}
        serving, _, _ = base.make_notebook(files, control, context, entry_enabled=False)
        diagnostic, archive, inventory = D.make_notebook(base, files, control, context, entry_enabled=False)
        serving_codes, diag_codes = base.code_cells(json.loads(serving)), base.code_cells(json.loads(diagnostic))
        self.assertEqual(diag_codes[:2], serving_codes[:2])
        self.assertEqual(len(diag_codes), 3)
        self.assertFalse(any("get_ipython" in code or "_f009_parent_code" in code for code in diag_codes))
        self.assertNotIn("_f009_extension_namespace", diag_codes[-1])
        self.assertIn("F009_FIXED192_DIAGNOSTIC/diagnostic-result.json", diag_codes[-1])
        with patch.object(Path, "read_bytes", side_effect=AssertionError("No model or metadata body before disabled entry")):
            with self.assertRaisesRegex(RuntimeError, "source-only"):
                exec(compile(diag_codes[0], "<disabled-diagnostic>", "exec"), {})
        self.assertEqual(base.checked_zip(archive, inventory), files)

    def test_generated_main_call_accepts_dict_without_SystemExit_and_creates_no_parent(self):
        base = D.load_base()
        control = base.inspect_control()
        raw, _, _ = D.make_notebook(base, {"P033-source.ipynb": control["parent_raw"]}, control,
                                    {"actual_entry_enabled": True}, entry_enabled=True)
        code = base.code_cells(json.loads(raw))[-1]
        calls = []
        module = types.ModuleType("_f009_diagnostic_support.f009_kaggle_diagnostic")
        module.main = lambda argv: calls.append(argv) or {
            "schema_version": "f009_fixed192_native_diagnostic_result_v1", "complete": True, "artificial_only": True}
        package = types.ModuleType("_f009_diagnostic_support")
        package.f009_kaggle_diagnostic = module
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "f009-context.sha256").write_text("a" * 64 + "\n", encoding="ascii")
            scope = {"_f009_runtime": root}
            with patch.dict(sys.modules, {package.__name__: package, module.__name__: module}), \
                    patch.object(sys, "path", list(sys.path)):
                exec(compile(code, "<artificial-diagnostic-main-call>", "exec"), scope)
                self.assertIs(scope["_f009_diagnostic_result"]["complete"], True)
                self.assertEqual(calls[0][calls[0].index("--output") + 1],
                                 "/kaggle/working/F009_FIXED192_DIAGNOSTIC/diagnostic-result.json")
                self.assertEqual(calls[0][calls[0].index("--selection-sha") + 1], D.SELECTION_SHA)
                self.assertEqual(calls[0][calls[0].index("--budget-seconds") + 1], "3600")
                self.assertIn("--execute", calls[0])
                module.main = lambda argv: {"schema_version": "wrong", "complete": True}
                with self.assertRaisesRegex(RuntimeError, "new contract"):
                    exec(compile(code, "<wrong-diagnostic-schema>", "exec"), scope)
            self.assertEqual({p.name for p in root.iterdir()}, {"f009-context.sha256"})

    def test_whole_source_pack_adds_only_four_and_emits_no_Save_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "source-only"
            manifest = D.prepare(output)
            self.assertEqual(manifest["unchanged_serving_source_count"], 108)
            self.assertEqual(manifest["source_member_count"], 112)
            self.assertIs(manifest["actual_entry_enabled"], False)
            self.assertIs(manifest["parent_eight_cells_executed"], False)
            self.assertIs(manifest["final_submission_csv_written"], False)
            self.assertIs(manifest["baseline_is_artificial_point5"], True)
            self.assertIs(manifest["actual192_runtime_qualified"], False)
            self.assertFalse((output / "save-payload.json").exists())
            base = D.load_base()
            files = base.checked_zip((output / "runtime-bundle.zip").read_bytes(), manifest["runtime_files"])
            self.assertEqual(len(files), 120)
            self.assertFalse(any(name.endswith((".pt", ".pth", ".dcm", ".npy", ".npz")) for name in files))
            for name, expected in manifest["unchanged_serving_sources"].items():
                self.assertEqual(D.raw_pin(files[name]), expected)
            original = set(manifest["unchanged_serving_sources"])
            inventory = json.loads(files["source-inventory.json"])
            self.assertEqual(set(inventory) - original, {D.SELECTION_NAME,
                D.DIAG_PACKAGE + "/__init__.py", D.DIAG_PACKAGE + "/f009_kaggle_diagnostic.py",
                D.DIAG_PACKAGE + "/f009_diagnostic_fixture.py"})
            self.assertEqual(len([n for n in inventory if n.startswith("_f009_kaggle_support/")]), 9)
            self.assertEqual(D.raw_pin(files[D.SELECTION_NAME])["sha256"], D.SELECTION_SHA)
            self.assertEqual(len(base.code_cells(json.loads((output / "source.ipynb").read_bytes()))), 3)
            self.assertLess(manifest["source"]["bytes"], 950001)


if __name__ == "__main__":
    unittest.main()
