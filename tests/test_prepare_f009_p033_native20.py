"""Source and artificial stdlib checks only; no checkpoint/MRI/Torch/API."""
import ast
import base64
import copy
import importlib.util
import io
import json
import lzma
import tempfile
import types
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_f009_notebook_builder_tests", ROOT / "scripts/prepare_f009_p033_native20.py")
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)
ADAPTER = "f009_P033_isolated_uid_native_adapter_v1"
RESULT = "f009_amended_CLS_P039_REG_owned_UID_native_result_v1"


class NotebookPreparation(unittest.TestCase):
    def test_default_and_explicit_actual_missing_release_stop_before_bodies(self):
        with patch.object(B, "inspect_control", side_effect=AssertionError("No input bodies")), redirect_stdout(io.StringIO()):
            self.assertEqual(B.main([]), 0)
            with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "qualified root release"):
                B.prepare(Path(tmp) / "actual", execute=True)

    def test_literal_sources_and_extension_ast_are_bound(self):
        values = B.inspect_control()
        self.assertEqual(len(values["components"]), len(values["component_inventory"]))
        self.assertEqual(B.raw_pin(values["parent_raw"])["sha256"], B.PARENT_SHA)
        self.assertEqual(len(values["parent_code_sha256"]), 8)
        self.assertEqual(set(values["reg_metadata"]), {
            "metadata/REG/fold0-permission.json", "metadata/REG/fold0-peer.json",
            "metadata/REG/fold1-permission.json", "metadata/REG/fold1-peer.json"})
        old = values["outer_files"]["p032_parent_extension.py"].decode()
        new = B.new_extension(old, adapter_schema=ADAPTER, result_schema=RESULT)
        reverse = {"f009_p033_CLS_pair_fixed20_execution_v1": "p039_p033_f006_fixed20_execution_v1",
                   "f009_native_child.py": "p032_native_child.py", ADAPTER: "p035_f006_isolated_uid_adapter_v1",
                   RESULT: "native_isolated_uid_operational_recovery_v1",
                   "F009_CLS_old_REG_native_rank": "F006_native_rank", "F009_P033_NATIVE20": "P032",
                   "F009_P033_final.tmp": "P032_final.tmp", "F009_P033_PARENT_PROGRESS": "P032_PARENT_PROGRESS",
                   "F009_P033_COMPLETE": "P032_COMPLETE"}

        class RestoreLabels(ast.NodeTransformer):
            def visit_Constant(self, node):
                if type(node.value) is str and node.value in reverse:
                    node.value = reverse[node.value]
                return node

        self.assertEqual(ast.dump(ast.parse(old)), ast.dump(RestoreLabels().visit(ast.parse(new))))
        # Bootstrap text is not returned as executable source.
        self.assertNotIn("bootstrap", values)
        self.assertNotIn("old_CPU", values["reg_metadata"])

    def test_atomic_source_restorer_roundtrip_and_prewrite_guards(self):
        namespace = {}
        exec(compile(B.RESTORER, "<source-restorer>", "exec"), namespace)
        restore = namespace["f009_restore_sources"]
        files = {"tiny/source.py": b"SENTINEL = 'artificial only'\n", "metadata/toy.json": b"{}\n"}
        archive = B.zip_sources(files)
        packed = lzma.compress(archive)
        inventory = {name: B.raw_pin(body) for name, body in files.items()}
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "fresh"
            actual = restore(base64.b85encode(packed).decode(), B.raw_pin(packed), B.raw_pin(archive), inventory, dest)
            self.assertEqual({name: (actual / name).read_bytes() for name in files}, files)
            self.assertFalse(dest.with_name("fresh.partial").exists())
            for changed in ("sha", "extra", "unsafe"):
                pins = copy.deepcopy(inventory)
                incoming, compressed = archive, packed
                if changed == "sha":
                    pins["tiny/source.py"]["sha256"] = "0" * 64
                elif changed == "extra":
                    pins["extra.py"] = B.raw_pin(b"extra")
                else:
                    pins["../escape.py"] = pins.pop("tiny/source.py")
                    bad = io.BytesIO()
                    with zipfile.ZipFile(bad, "w", compression=zipfile.ZIP_STORED) as book:
                        book.writestr("../escape.py", files["tiny/source.py"])
                        book.writestr("metadata/toy.json", files["metadata/toy.json"])
                    incoming = bad.getvalue()
                    compressed = lzma.compress(incoming)
                invalid = Path(tmp) / changed
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    restore(base64.b85encode(compressed).decode(), B.raw_pin(compressed), B.raw_pin(incoming), pins, invalid)
                self.assertFalse(invalid.exists())
                self.assertFalse(invalid.with_name(invalid.name + ".partial").exists())
            failed = Path(tmp) / "rename-failure"
            with patch.object(namespace["os"], "rename", side_effect=OSError("artificial publication failure")):
                with self.assertRaisesRegex(OSError, "publication"):
                    restore(base64.b85encode(packed).decode(), B.raw_pin(packed), B.raw_pin(archive), inventory, failed)
            self.assertFalse(failed.exists())
            failure = json.loads((Path(tmp) / "rename-failure.partial/RESTORE_FAILED.json").read_bytes())
            self.assertIs(failure["complete"], False)

    def test_actual_shim_translates_only_fixed_serving_argv(self):
        recorded = []
        driver = types.ModuleType("_f009_kaggle_support.f009_kaggle_driver")
        driver.main = lambda argv: recorded.append(argv) or {"complete": True, "artificial_only": True}
        package = types.ModuleType("_f009_kaggle_support")
        package.f009_kaggle_driver = driver
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f009_native_child.py"
            context = Path(tmp) / "f009-context.json"
            context.write_bytes(b"{}")
            (Path(tmp) / "f009-context.sha256").write_text("a" * 64 + "\n", encoding="ascii")
            scope = {"__name__": "_artificial_shim", "__file__": str(path)}
            exec(compile(B.native_shim(), "<artificial-shim>", "exec"), scope)
            argv = ["--mode", "serving", "--output", "result.json", "--baseline", "parent.csv",
                    "--budget-seconds", "120", "--diagnostic-count", "192", "--native-executor", "isolated-uid"]
            with patch.dict("sys.modules", {package.__name__: package, driver.__name__: driver}):
                self.assertEqual(scope["main"](argv), 0)
                self.assertEqual(recorded[0], ["--context", str(context), "--context-sha", "a" * 64,
                    "--output", "result.json", "--baseline", "parent.csv", "--budget-seconds", "120.0", "--execute"])
                with patch("sys.argv", [str(path), *argv]), self.assertRaises(SystemExit) as exit_state:
                    exec(compile(B.native_shim(), "<artificial-main-shim>", "exec"),
                         {"__name__": "__main__", "__file__": str(path)})
                self.assertEqual(exit_state.exception.code, 0)
                invalid = list(argv)
                invalid[invalid.index("192")] = "191"
                with self.assertRaisesRegex(ValueError, "Fixed serving"):
                    scope["main"](invalid)
                self.assertEqual(len(recorded), 2)

    def test_source_only_carrier_stops_before_parent_and_parent_cells_unchanged(self):
        control = B.inspect_control()
        files = {"P033-source.ipynb": control["parent_raw"], "artificial.py": b"raise RuntimeError('never execute')\n"}
        raw, archive, pins = B.make_notebook(files, control, {"actual_entry_enabled": False}, entry_enabled=False)
        book = json.loads(raw)
        first = B.code_cells(book)[0]
        with patch.object(Path, "read_bytes", side_effect=AssertionError("No body before source-only stop")):
            with self.assertRaisesRegex(RuntimeError, "source-only"):
                exec(compile(first, "<source-only-first-cell>", "exec"), {})
        self.assertEqual(B.checked_zip(archive, pins), files)
        self.assertNotIn("BOOTSTRAP", B.code_cells(book)[-1])
        self.assertEqual(B.raw_pin(files["P033-source.ipynb"])["sha256"], B.PARENT_SHA)

    def test_payload_rejects_artificial_or_old_CLS_and_has_no_API_execution(self):
        with self.assertRaisesRegex(ValueError, "Actual root"):
            B.manual_payload({"actual_entry_enabled": False, "artificial_fixture_only": False,
                              "transport_jobs": {}, "inputs": {}}, "toy", "rsraki/toy")
        plan = B.inspect_control()["plan"]
        jobs = {key: dict(value) for key, value in plan["intended_intervention"]["old_REG_transport_refs"].items()}
        jobs.update({"CLS:0": {"dataset_ref": "rsraki/new-test-cls0/1"},
                     "CLS:1": {"dataset_ref": "rsraki/new-test-cls1/1"}})
        context = {"actual_entry_enabled": True, "artificial_fixture_only": False,
                   "transport_jobs": jobs, "inputs": {"datasets": ["old-cls"], "kernels": [B.STAGE],
                                                       "competitions": [B.COMPETITION], "total": 9}}
        with self.assertRaisesRegex(ValueError, "Only new CLS2"):
            B.manual_payload(context, "toy", "rsraki/toy")

    def test_whole_source_only_preparation_has_no_payload_and_restores_exact_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "source-only"
            manifest = B.prepare(output)
            self.assertIs(manifest["actual_entry_enabled"], False)
            self.assertIs(manifest["actual_metadata_release_verified"], False)
            self.assertFalse((output / "save-payload.json").exists())
            self.assertFalse(manifest["old_source89_qualification_inherited"])
            self.assertEqual(manifest["old_component_source_count"], 89)
            self.assertLess(manifest["source"]["bytes"], 950001)
            files = B.checked_zip((output / "runtime-bundle.zip").read_bytes(), manifest["runtime_files"])
            inventory = json.loads(files["source-inventory.json"])
            self.assertNotIn("f009-context.json", inventory)
            self.assertNotIn("f009-context.sha256", inventory)
            self.assertEqual(inventory, {name: B.raw_pin(files[name]) for name in inventory})
            context = json.loads(files["f009-context.json"])
            self.assertIs(context["actual_entry_enabled"], False)
            self.assertEqual(files["f009-context.sha256"].decode().strip(), B.raw_pin(files["f009-context.json"])["sha256"])
            self.assertNotIn("p032_native_child.py", files)
            self.assertNotIn("old_CLS_qualification", context)
            # Complete source-only package itself still refuses actual metadata use.
            with B.new_support_modules() as (_, validator):
                root = Path(tmp) / "metadata-test"
                root.mkdir()
                (root / "f009-context.json").write_bytes(files["f009-context.json"])
                with self.assertRaisesRegex(ValueError, "Source preparation"):
                    validator.validate_context(root / "f009-context.json", B.raw_pin(files["f009-context.json"])["sha256"],
                                               runtime_root=root)


if __name__ == "__main__":
    unittest.main()
