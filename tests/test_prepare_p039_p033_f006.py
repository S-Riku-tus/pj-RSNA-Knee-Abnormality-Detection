"""P039 source and synthetic contract checks; no actual Torch/model/MRI/API."""
import ast
import contextlib
import copy
import csv
import importlib.util
import io
import json
import sys
import tempfile
import time
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("p039_builder_tests", ROOT / "scripts/prepare_p039_p033_f006.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
TARGETS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
           "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")
IDS = ("1.2.3", "1.2.1", "1.2.2")


def csv_file(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


@contextlib.contextmanager
def packed(folder):
    out = folder / "source"
    manifest = B.prepare(out)
    package = folder / "runtime"
    package.mkdir()
    with zipfile.ZipFile(out / "runtime-bundle.zip") as archive:
        archive.extractall(package)
    namespace = {"__name__": "p039_artificial_extension_only"}
    exec(compile((out / "p039-parent-extension.py").read_bytes(), "p039-extension", "exec"), namespace)
    previous = {name: value for name, value in sys.modules.items() if name == "p032blend" or name.startswith("p032blend.")}
    for name in previous:
        del sys.modules[name]
    sys.path.insert(0, str(package))
    try:
        yield out, package, namespace, manifest
    finally:
        sys.path.remove(str(package))
        for name in list(sys.modules):
            if name == "p032blend" or name.startswith("p032blend."):
                del sys.modules[name]
        sys.modules.update(previous)


def synthetic_work(folder, namespace, weights=None):
    working, data = folder / "working", folder / "data"
    working.mkdir()
    data.mkdir()
    csv_file(working / "submission.csv", ["StudyInstanceUID", *TARGETS],
             [[uid, *[str((i + 1) / 4)] * 12] for i, uid in enumerate(IDS)])
    csv_file(data / "test.csv", ["StudyInstanceUID"], [[uid] for uid in IDS])
    csv_file(data / "sample_submission.csv", ["StudyInstanceUID", *TARGETS], [[uid, *["0.5"] * 12] for uid in IDS])
    (working / "blend_receipt.json").write_text(json.dumps({"weights": B.WEIGHTS if weights is None else weights}), encoding="utf-8")
    (working / "submission_receipt.json").write_text(json.dumps({"status": "PASS", "bundle_id": "fe055406fad6a12f",
        "worker_exit_codes": {"0": 0, "1": 0}}), encoding="utf-8")
    def mapped_path(value):
        text = str(value)
        if text == "/kaggle/working" or text.startswith("/kaggle/working/"):
            return working / text[len("/kaggle/working"):].lstrip("/")
        return Path(value)
    namespace["Path"] = mapped_path
    namespace["parent_progress"] = lambda *_args, **_kwargs: None
    return working, data


class P039Tests(unittest.TestCase):
    def test_exact_parent_cells_native_sources_inputs_and_release_guard(self):
        with tempfile.TemporaryDirectory() as temporary, packed(Path(temporary)) as (out, package, _ns, manifest):
            self.assertEqual((package / "P033-source.ipynb").read_bytes(), (B.PARENT / "source.ipynb").read_bytes())
            self.assertEqual(len(manifest["unchanged_runtime_members"]), 7)
            self.assertEqual(manifest["native"]["source_member_count"], 89)
            self.assertEqual(manifest["inputs"]["total"], 9)
            parent = B.codes(json.loads((package / "P033-source.ipynb").read_bytes()))
            self.assertEqual(len(parent), 8)
            book = json.loads((out / "source.ipynb").read_bytes())
            self.assertIn("parent-only success", B.codes(book)[-1])
            self.assertIn('get("native_contribution_used") is not True', B.codes(book)[-1])
            self.assertNotIn("P029M-source.ipynb", [path.name for path in package.iterdir()])
            payload = json.loads((out / "save-payload.json").read_bytes())
            old = B.read(B.PAYLOAD)
            for key in old:
                if key not in {"text", "slug", "new_title"}:
                    self.assertEqual(payload[key], old[key])

    def test_only_execute_binding_failure_and_receipt_metadata_changes(self):
        old_book, _manifest, files = B.unpack_old()
        old_tree = ast.parse(files["p032_parent_extension.py"].decode())
        old_functions = {n.name: n for n in old_tree.body if isinstance(n, ast.FunctionDef)}
        parent_shas = [B.raw_pin(text.encode())["sha256"] for text in B.codes(B.read(B.PARENT / "source.ipynb"))]
        extension = B.new_extension(files["p032_parent_extension.py"].decode(), parent_shas)
        new_functions = {n.name: n for n in ast.parse(extension).body if isinstance(n, ast.FunctionDef)}
        for name in old_functions.keys() - {"execute"}:
            self.assertEqual(ast.dump(old_functions[name], include_attributes=False), ast.dump(new_functions[name], include_attributes=False))
        old_execute = old_functions["execute"]
        new_execute = copy.deepcopy(new_functions["execute"])
        class RevertMetadata(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id == "P039_PARENT_WEIGHTS":
                    return ast.parse(B.OLD_WEIGHTS_LITERAL, mode="eval").body
                return node

            def visit_Dict(self, node):
                node = self.generic_visit(node)
                pairs = [(key, value) for key, value in zip(node.keys, node.values)
                         if not isinstance(key, ast.Constant) or key.value != "parent_source_sha256"]
                node.keys, node.values = [key for key, _ in pairs], [value for _, value in pairs]
                return node

            def visit_Constant(self, node):
                mapping = {"p039_p033_f006_fixed20_execution_v1": "p032_p029m_f006_fixed20_execution_v1",
                           "P033 must complete a regular submission.csv first": "P029M must complete a regular submission.csv first",
                           "Exact P033 parent blend weight receipt differs": "Original parent blend weight receipt differs",
                           "INVALID_EXTENSION_CANDIDATE_RUN_FAILED": "NOT_AN_ACCURACY_COMPARISON_NATIVE_DISABLED"}
                return ast.Constant(mapping.get(node.value, node.value)) if isinstance(node.value, str) else node

            def visit_Call(self, node):
                node = self.generic_visit(node)
                node.keywords = [kw for kw in node.keywords if kw.arg != "parent_only_submission_allowed"]
                for kw in node.keywords:
                    if kw.arg == "parent_artifact_restored":
                        kw.arg = "parent_fallback_preserved"
                return node

            def visit_Assign(self, node):
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "p039_bind_parent_metadata":
                    return None
                return self.generic_visit(node)

            def visit_Expr(self, node):
                if isinstance(node.value, ast.Call):
                    function = node.value.func
                    if isinstance(function, ast.Name) and function.id == "p039_require_native_contribution":
                        return None
                    if isinstance(function, ast.Attribute) and function.attr == "write_text":
                        return None
                return self.generic_visit(node)

            def visit_ExceptHandler(self, node):
                node = self.generic_visit(node)
                if isinstance(node.body[-1], ast.Raise) and node.body[-1].exc is None:
                    node.body[-1] = ast.Return(ast.Name(id="receipt", ctx=ast.Load()))
                return node
        reverted = RevertMetadata().visit(new_execute)
        self.assertEqual(ast.dump(old_execute, include_attributes=False), ast.dump(reverted, include_attributes=False))
        self.assertEqual(B.codes(old_book)[-1].count("execute("), 1)

    def test_actual_wrapper_receipt_mismatch_raises_and_preserves_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            with packed(folder) as (_out, package, namespace, _manifest):
                wrong = {"raptor": .5, "ours": .5, "raptor_per_class": {"Medial OA": .5, "Lateral OA": .5, "PF OA": .5}, "a224": .35}
                working, data = synthetic_work(folder, namespace, wrong)
                original = (working / "submission.csv").read_bytes()
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "Exact P033"):
                    namespace["execute"]({"DATA": data}, runtime=package, started=time.monotonic(),
                        blend_policy="success-subset", native_executor="isolated-uid")
                receipt = json.loads((working / "P032/execution-receipt.json").read_bytes())
                self.assertFalse(receipt["complete"])
                self.assertFalse(receipt["parent_only_submission_allowed"])
                self.assertEqual((working / "submission.csv").read_bytes(), original)
                self.assertNotIn("torch", sys.modules)

    def test_child_whole_failure_or_timeout_raises_no_parent_success(self):
        for timeout in (False, True):
            with self.subTest(timeout=timeout), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                with packed(folder) as (_out, package, namespace, _manifest):
                    working, data = synthetic_work(folder, namespace)
                    original = (working / "submission.csv").read_bytes()
                    calls = []
                    def fake_child(*_args, **_kwargs):
                        calls.append(1)
                        return {"complete": False, "timeout": timeout, "returncode": -9 if timeout else 1}
                    namespace["run_child"] = fake_child
                    cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
                        memory_allocated=lambda _i: 0, memory_reserved=lambda _i: 0,
                        device=lambda _i: contextlib.nullcontext(), empty_cache=lambda: None)
                    with mock.patch.dict(sys.modules, {"torch": types.SimpleNamespace(cuda=cuda)}), contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaisesRegex(RuntimeError, "Native child failed"):
                            namespace["execute"]({"DATA": data}, runtime=package, started=time.monotonic(),
                                blend_policy="success-subset", native_executor="isolated-uid")
                    self.assertEqual(len(calls), 1)
                    self.assertEqual((working / "submission.csv").read_bytes(), original)
                    receipt = json.loads((working / "P032/execution-receipt.json").read_bytes())
                    self.assertFalse(receipt["complete"])
                    self.assertFalse(receipt["native_contribution_used"])
                    self.assertFalse(receipt["parent_only_submission_allowed"])

    def test_fixed20_partial_cells_and_correct_P033_nominal_metadata(self):
        with tempfile.TemporaryDirectory() as temporary, packed(Path(temporary)) as (_out, _package, namespace, _manifest):
            from p032blend.success20_blend import blend_rows
            rows = [{"StudyInstanceUID": uid, **{target: str(.2 + i * .2) for target in TARGETS}} for i, uid in enumerate(IDS)]
            native = [{"StudyInstanceUID": uid, **{target: str(.8 - i * .3) for target in TARGETS}} for i, uid in enumerate(IDS)]
            final, receipt = blend_rows(rows, native, failure_mask={IDS[1]: True})
            before = json.dumps(final)
            bound = namespace["p039_bind_parent_metadata"](receipt)
            namespace["p039_require_native_contribution"](bound)
            self.assertEqual(json.dumps(final), before)
            self.assertEqual(final[1], rows[1])
            for old, new in zip(rows, final):
                self.assertEqual(old["MCL"], new["MCL"])
                self.assertEqual(old["Synovitis"], new["Synovitis"])
            active = bound["per_target"]["Lateral OA"]["conditional_P033_component_weights_nominal"]["successful_cells"]
            self.assertEqual(active, {"Raptor384": .104, "Raptor224": .056, "CompactR6": .64, "F006_native_rank": .2})
            syn = bound["per_target"]["Synovitis"]["conditional_P033_component_weights_nominal"]["successful_cells"]
            self.assertEqual(syn, {"Raptor384": .52, "Raptor224": .28, "CompactR6": .2, "F006_native_rank": 0.})
            self.assertTrue(all("conditional_P029M_component_weights_nominal" not in entry for entry in bound["per_target"].values()))
            _disabled_rows, disabled = blend_rows(rows, native, failure_mask={uid: True for uid in IDS})
            disabled = namespace["p039_bind_parent_metadata"](disabled)
            with self.assertRaisesRegex(RuntimeError, "no active"):
                namespace["p039_require_native_contribution"](disabled)


if __name__ == "__main__":
    unittest.main()
