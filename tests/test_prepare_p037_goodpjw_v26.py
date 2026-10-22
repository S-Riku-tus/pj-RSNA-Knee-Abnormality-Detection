"""Standard-library artificial QA; scientific cells/models/MRI are never run."""
import base64
import contextlib
import hashlib
import importlib.util
import io
import json
import lzma
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load("p037_builder", "prepare_p037_goodpjw_v26.py")
runtime = load("p037_runtime", "p037_fixed_recipe_runtime.py")


class FixedPackageTests(unittest.TestCase):
    def test_prepare_restore_fixed_source_and_requested_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidate"
            manifest = builder.prepare(output)
            payload = json.loads((output / "save-payload.json").read_bytes())
            self.assertEqual(payload["text"].encode(), (output / "source.ipynb").read_bytes())
            self.assertLess(manifest["source_bytes"], 950000)
            self.assertEqual(manifest["inputs"]["total"], 7)
            self.assertTrue(all(ref.endswith("/1") for ref in payload["dataset_data_sources"] + payload["kernel_data_sources"]))
            self.assertFalse(payload["enable_internet"])
            self.assertTrue(payload["is_private"])
            self.assertIn("37c64f7dd9c54116", payload["docker_image"])
            original = [path.read_bytes() for path in sorted((output / "author-cells").glob("*.py"))]
            adapted = [path.read_bytes() for path in sorted((output / "adapted-cells").glob("*.py"))]
            self.assertEqual(len(original), 15)
            self.assertEqual([i for i, (old, new) in enumerate(zip(original, adapted)) if old != new], [5])
            self.assertEqual(adapted[5].replace(builder.STAGE_LINE.encode(), b"", 1), original[5])
            raw = (output / "recipe-cells.json").read_bytes()
            compressed = lzma.compress(raw, preset=6)
            self.assertEqual(runtime.restore_cells(base64.b85encode(compressed), builder.digest(compressed),
                                                  builder.digest(raw), manifest["candidate_code_inventory"]),
                             [value.decode() for value in adapted])
            self.assertEqual([len(manifest["guard_config"][key]) for key in ("R6", "R7", "Reader")], [9, 9, 3])
            with self.assertRaises(FileExistsError):
                builder.prepare(output)

    def test_existing_own_synthetic_fixture_input_is_seventh(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = builder.prepare(Path(temporary) / "candidate")
            self.assertEqual(manifest["inputs"]["total"], 7)
            self.assertEqual(manifest["inputs"]["datasets"][-1], builder.DECODER_INPUT)
            self.assertEqual(manifest["guard_config"]["fixture_input_slug"], "rsna-dicom-decoders-py313-v1")
            self.assertEqual(len(manifest["guard_config"]["fixture_members"]), 6)
            self.assertFalse(manifest["decoder_preflight"]["author_fixture_found"])

    def test_restore_rejects_compressed_raw_cell_tampering(self):
        cells = ["x=1\n", "%%writefile /tmp/example.py\nx=2\n"]
        raw = json.dumps(cells).encode()
        compressed = lzma.compress(raw)
        encoded = base64.b85encode(compressed)
        self.assertEqual(runtime.restore_cells(encoded, builder.digest(compressed), builder.digest(raw), builder.inventory(cells)), cells)
        for sha_a, sha_b, inventory in (("0" * 64, builder.digest(raw), builder.inventory(cells)),
                                       (builder.digest(compressed), "0" * 64, builder.inventory(cells)),
                                       (builder.digest(compressed), builder.digest(raw), builder.inventory(["x=3\n", cells[1]]))):
            with self.subTest(sha_a=sha_a, sha_b=sha_b), self.assertRaises(ValueError):
                runtime.restore_cells(encoded, sha_a, sha_b, inventory)

    def test_stage_patch_exactly_once_and_only_for_r6(self):
        text = "def find_mount(slug, required):\n    cands = [INPUT / slug, INPUT / 'competitions' / slug]\n    return cands\n"
        changed = builder.patch_r6(text)
        self.assertEqual(changed.count(builder.STAGE_LINE), 1)
        self.assertIn("if slug == C_SLUG else []", changed)
        with self.assertRaises(ValueError):
            builder.patch_r6(changed)
        with self.assertRaises(ValueError):
            builder.patch_r6("unrelated=1\n")


class RuntimeGuardTests(unittest.TestCase):
    def test_artificial_member_guard_missing_size_and_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "test.bin"
            path.write_bytes(b"abc")
            pins = {"test.bin": {"bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}}
            self.assertEqual(runtime.verify_members(root, pins), pins)
            for value in (b"abd", b"ab"):
                path.write_bytes(value)
                with self.assertRaises(ValueError):
                    runtime.verify_members(root, pins)
            path.unlink()
            with self.assertRaises(ValueError):
                runtime.verify_members(root, pins)

    def test_dataset_mount_ambiguity_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "datasets/owner/slug"
            first.mkdir(parents=True)
            (first / "manifest.json").write_text("{}")
            self.assertEqual(runtime.dataset_root(root, "owner", "slug", "manifest.json"), first.resolve())
            other = root / "slug"
            other.mkdir()
            (other / "manifest.json").write_text("{}")
            with self.assertRaises(ValueError):
                runtime.dataset_root(root, "owner", "slug", "manifest.json")

    def test_all_decoder_fixtures_must_be_positive(self):
        positive = {"complete": True, "fixtures": [{"file": "a.dcm", "equal": True}, {"file": "b.dcm", "equal": True}]}
        self.assertEqual(runtime.validate_decoder_result(positive, ["a.dcm", "b.dcm"]), positive)
        for bad in ({"complete": False}, {"complete": True, "fixtures": []},
                    {"complete": True, "fixtures": [{"file": "a.dcm", "equal": True}]},
                    {"complete": True, "fixtures": [{"file": "a.dcm", "equal": True}, {"file": "b.dcm", "equal": False}]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                runtime.validate_decoder_result(bad, ["a.dcm", "b.dcm"])

    def test_strict_decoder_install_and_fixture_subprocess_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets, work, fixtures = root / "reader", root / "work", root / "phantoms"
            assets.mkdir()
            fixtures.mkdir()
            (work / "P037").mkdir(parents=True)
            ref = b"artificial expected placeholder; never loaded as NumPy"
            (fixtures / "expected.npy").write_bytes(ref)
            entries = []
            for i in range(1, 5):
                name = "synthetic-%d.dcm" % i
                (fixtures / name).write_bytes(b"synthetic marker; never decoded")
                entries.append({"filename": name, "sha256": runtime.sha(fixtures / name), "syntax": str(i)})
            (fixtures / "phantom-manifest.json").write_text(json.dumps({"expected_filename": "expected.npy",
                "expected_sha256": builder.digest(ref), "files": entries}))
            config = {"fixture_members": {path.name: {"bytes": path.stat().st_size, "sha256": runtime.sha(path)} for path in fixtures.iterdir()},
                      "fixture_syntaxes": {row["filename"]: row["syntax"] for row in entries},
                      "reader_wheel_sizes": {name: 8 for name in builder.READER_WHEEL_SIZES}}
            for name in config["reader_wheel_sizes"]:
                (assets / name).write_bytes(b"artificial")
                config["reader_wheel_sizes"][name] = len(b"artificial")
            positive = {"complete": True, "fixtures": [{"file": row["filename"], "equal": True, "syntax": row["syntax"],
                "sha256": row["sha256"], "shape_dtype_values_equal": True, "shape": [16, 16], "dtype": "uint16"}
                for row in entries], "reference_sha256": builder.digest(ref), "reference_shape": [16, 16], "reference_dtype": "uint16"}
            with mock.patch.object(runtime.sys, "version_info", (3, 12)), mock.patch.object(runtime.subprocess, "run", side_effect=[
                    types.SimpleNamespace(stdout="", stderr="", returncode=0),
                    types.SimpleNamespace(stdout=json.dumps(positive), returncode=0)]) as run:
                value = runtime.strict_decoder_preflight(assets, work, fixtures, config)
                self.assertTrue(value["complete"])
                self.assertTrue(all(call.kwargs["check"] for call in run.call_args_list))
                self.assertIn("--no-index", run.call_args_list[0].args[0])
                self.assertIn("--no-deps", run.call_args_list[0].args[0])
                self.assertEqual(len(value["wheel_files"]), 3)
            with mock.patch.object(runtime.sys, "version_info", (3, 12)), mock.patch.object(runtime.subprocess, "run", side_effect=RuntimeError("incompatible wheels")), self.assertRaises(RuntimeError):
                runtime.strict_decoder_preflight(assets, work, fixtures, config)
            (fixtures / "expected.npy").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                runtime.strict_decoder_preflight(assets, work, fixtures, config)

    def test_partial_member_receipts_survive_decoder_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, work = root / "input", root / "work"
            work.mkdir()
            groups = {"R6": inputs / "notebooks/rsraki/stage/frozen", "R7": inputs / "r7", "Reader": inputs / "reader"}
            config = {"stage_slug": "stage", "R6_folder": "frozen", "R7_slug": "r7", "Reader_slug": "reader",
                      "fixture_input_slug": "fixtures"}
            for family, folder in groups.items():
                folder.mkdir(parents=True)
                filename = "knee.py" if family == "Reader" else "manifest.json"
                (folder / filename).write_bytes(b"artificial guarded source")
                config[family] = {filename: {"bytes": len(b"artificial guarded source"), "sha256": runtime.sha(folder / filename)}}
            (inputs / "fixtures/phantoms").mkdir(parents=True)
            (inputs / "fixtures/phantoms/phantom-manifest.json").write_text("{}")
            with mock.patch.object(runtime, "strict_decoder_preflight", side_effect=RuntimeError("synthetic ABI failure")), self.assertRaises(RuntimeError):
                runtime.preflight(config, input_root=inputs, work=work)
            partial = json.loads((work / "P037/preflight.json").read_bytes())
            self.assertFalse(partial["complete"])
            self.assertEqual(sum(len(row["members"]) for row in partial["fixed_members"].values()), 3)
            self.assertEqual(partial["error_type"], "RuntimeError")

    def test_transport_order_observer_and_error_propagation(self):
        texts = ["synthetic_only_%d" % i for i in range(15)]
        indices = [0, 3, 4, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 20, 21]
        for fail_at in (None, 4):
            calls = []
            with tempfile.TemporaryDirectory() as temporary:
                work = Path(temporary)
                (work / "P037").mkdir()
                (work / "submission.csv").write_bytes(b"artificial exact parent bytes\r\n")
                def fake_run(text, store_history):
                    self.assertFalse(store_history)
                    calls.append(text)
                    return types.SimpleNamespace(error_before_exec=None,
                                                 error_in_exec=RuntimeError("fixture failure") if len(calls) == fail_at else None)
                shell = types.SimpleNamespace(run_cell=fake_run)
                with contextlib.redirect_stdout(io.StringIO()):
                    if fail_at is None:
                        proof = runtime.execute_cells(texts, indices, shell, work=work)
                        self.assertTrue(proof["complete"])
                        self.assertEqual(calls, texts)
                        self.assertEqual((work / "P037/audit/cell-15/submission.csv").read_bytes(), b"artificial exact parent bytes\r\n")
                    else:
                        with self.assertRaises(RuntimeError):
                            runtime.execute_cells(texts, indices, shell, work=work)
                        self.assertEqual(len(calls), fail_at)
                        self.assertFalse(json.loads((work / "P037/execution.json").read_bytes())["complete"])

    def test_operational_globals_stay_isolated_from_fake_IPython_user_namespace(self):
        isolated = {"__name__": "synthetic_p037_operational_runtime"}
        exec(compile(builder.RUNTIME.read_text(encoding="utf-8"), "<own-operational-definitions-only>", "exec"), isolated)
        self.assertIs(isolated["execute_cells"].__globals__, isolated)
        author_namespace = {}
        def fake_cell(text, store_history):
            author_namespace.update({key: "artificial author collision" for key in
                                     ("require", "sha", "time", "json", "Path", "hashlib", "observe", "write_json")})
            return types.SimpleNamespace(error_before_exec=None, error_in_exec=None)
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            work = Path(temporary)
            (work / "P037").mkdir()
            result = isolated["execute_cells"](["synthetic-only"] * 15,
                [0, 3, 4, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 20, 21],
                types.SimpleNamespace(run_cell=fake_cell), work=work)
            self.assertTrue(result["complete"])
            self.assertTrue(callable(isolated["sha"]))


if __name__ == "__main__":
    unittest.main()
