"""Stdlib-only preview refusal, immutable source transport and interface checks."""
import ast
import copy
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import cnn_p039_bridge as C
import prepare_cnn_p039_preview as P
import prepare_f009_p033_native20 as B


class CNNPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.control = B.inspect_control()

    def test_one_candidate_only_and_returned_spec_cannot_modify_other_candidates(self):
        for invalid in (None, ["tiny"], ("tiny", "small"), "tiny,small", "F014", "", True):
            with self.assertRaises(ValueError):
                C.candidate_spec(invalid)
        value = C.candidate_spec("tiny")
        value["stage_depths"][2] = 27
        self.assertEqual(C.candidate_spec("tiny")["stage_depths"], [3, 3, 9, 3])

    def test_bridge_refuses_before_any_argument_or_proof_can_be_inspected(self):
        class Unreadable:
            def __getattribute__(self, key):
                raise AssertionError("Disabled entry inspected an argument")

        for enabled in (False, True):
            with patch.object(C, "ACTUAL_ENABLED", enabled):
                with self.assertRaises(C.PreviewOnlyError):
                    C.execute(Unreadable(), runtime=Unreadable(), model_package_proof=Unreadable())
                with self.assertRaises(C.PreviewOnlyError):
                    C.prepare_package(Unreadable(), model_package_proof=Unreadable())

    def test_disabled_stub_refuses_before_importing_or_reading_anything(self):
        tree = ast.parse(P.DISABLED_STUB)
        self.assertEqual(len(tree.body), 1)
        self.assertIsInstance(tree.body[0], ast.Raise)
        with self.assertRaisesRegex(RuntimeError, "unconditionally disabled"):
            exec(compile(P.DISABLED_STUB, "disabled_preview.py", "exec"), {})

    def test_both_original_producer_inventories_and_model_only_import_closure(self):
        for candidate, count in (("tiny", 16), ("small", 18)):
            value = C.inspect_sources(P.ROOT, candidate)
            self.assertEqual(len(value["sources"]), count)
            self.assertEqual(value["inference_import_closure"],
                             sorted([value["spec"]["model_module"] + ".py", "feature_model.py", "contracts.py"]))
            for name, expected in C.expected_sources(candidate).items():
                self.assertEqual(C.pin(value["sources"][name])["sha256"], expected)
            files = copy.deepcopy(value["sources"])
            files[value["spec"]["model_module"] + ".py"] += b"\nfrom .native_training_fast16 import tensor_study\n"
            with self.assertRaises(ValueError):
                C.source_import_closure(files, value["spec"]["model_module"])

    def test_preconditions_are_requirements_and_do_not_claim_folds_or_models_qualified(self):
        for candidate in ("tiny", "small"):
            value = C.future_preconditions(candidate)
            self.assertIs(value["actual_enabled"], False)
            self.assertIs(value["checkpoint_bindings_generated"], False)
            self.assertIs(value["runtime_constructor_implemented"], False)
            self.assertEqual(set(value["stage_order"]), set(value["required_evidence"]))
            self.assertEqual(value["invariants"]["fold_weights"], [.5, .5])
            self.assertEqual(value["invariants"]["family_weights"], {"CLS_role_CNN": .5, "REG": .5})
            self.assertEqual(value["invariants"]["native_weight"], .2)
            self.assertEqual(value["invariants"]["excluded_from_native"], ["MCL", "Synovitis"])
            self.assertIn("local_RTX4090_QA", value["not_qualified_by"])

    def test_real_Linux_lock_is_pinned_but_no_native_wheel_is_loaded(self):
        value = C.inspect_sources(P.ROOT, "small")
        lock = json.loads(value["metadata"]["Linux-wheel-lock.json"])
        self.assertEqual({p["filename"]: (p["bytes"], p["SHA"]) for p in lock["packages"]}, C.WHEELS)
        self.assertEqual(C.pin(value["metadata"]["Linux-wheel-lock.json"])["sha256"], C.WHEEL_LOCK_SHA)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("timm", sys.modules)
        self.assertNotIn("safetensors", sys.modules)
        self.assertFalse(any(n.startswith(("cnn_f014_source16", "cnn_f015_source18")) for n in sys.modules))

    def test_preview_both_candidates_preserve_every_original_reference_and_raw_source(self):
        originals = {**C.SOURCE16, **C.SMALL_EXTRA}
        before = {n: (P.ROOT / "src/rsna_knee" / n).read_bytes() for n in originals}
        with tempfile.TemporaryDirectory(dir=P.ROOT / "artifacts/research", prefix="cnn-preview-test-") as temporary:
            for candidate in ("tiny", "small"):
                output = Path(temporary) / candidate
                manifest = P.prepare(output, candidate)
                self.assertIs(manifest["actual_enabled"], False)
                self.assertIs(manifest["trained_weights_included"], False)
                self.assertIs(manifest["trained_checkpoint_bindings_generated"], False)
                self.assertIs(manifest["uploadable_Notebook_or_Savepayload_generated"], False)
                self.assertEqual({p.name for p in output.iterdir()},
                                 {"runtime-sources.bin", "pinmanifest.json", "interface-preconditions.json",
                                  "requirements.preview.txt", "disabled_preview.py"})
                self.assertEqual(manifest["P033_original_eight_cell_SHA"], self.control["parent_code_sha256"])
                with zipfile.ZipFile(io.BytesIO((output / "runtime-sources.bin").read_bytes())) as archive:
                    names = set(archive.namelist())
                    for name, raw in self.control["outer_files"].items():
                        self.assertEqual(archive.read("reference/P039/" + name), raw)
                    self.assertEqual(archive.read("reference/serving/serving_router.py"),
                                     self.control["components"]["serving/serving_router.py"])
                    namespace = C.candidate_spec(candidate)["namespace"]
                    for name, raw in before.items():
                        if name in C.expected_sources(candidate):
                            self.assertEqual(archive.read(namespace + "/" + name), raw)
                    self.assertFalse(any(n.endswith((".pt", ".pth", ".safetensors", ".whl", ".dcm", ".npz")) for n in names))
                    other = C.candidate_spec("small" if candidate == "tiny" else "tiny")["namespace"]
                    self.assertFalse(any(n.startswith(other + "/") for n in names))
                    self.assertNotIn("save-payload.json", names)
                    self.assertNotIn("source.ipynb", names)
                with self.assertRaises(ValueError):
                    P.prepare(output, candidate)
        self.assertEqual(before, {n: (P.ROOT / "src/rsna_knee" / n).read_bytes() for n in originals})

    def test_no_builder_or_bridge_flag_can_request_release_or_external_action(self):
        source = Path(P.__file__).read_text(encoding="utf-8")
        self.assertNotIn('--execute', source)
        self.assertNotIn('--slug', source)
        tree = ast.parse(Path(C.__file__).read_text(encoding="utf-8"))
        imports = {a.name.split(".")[0] for node in tree.body if isinstance(node, ast.Import) for a in node.names}
        self.assertEqual(imports, {"ast", "copy", "hashlib", "json"})
        for name in ("execute", "prepare_package"):
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertIsInstance(node.body[1], ast.Raise)
            self.assertEqual(len(node.body), 2)


if __name__ == "__main__":
    unittest.main()
