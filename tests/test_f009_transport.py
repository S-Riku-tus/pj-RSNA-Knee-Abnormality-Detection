"""Only generated non-model checkpoint strings and copied immutable source."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from rsna_knee import f009_cpu_proof_v2 as P
from rsna_knee import f009_pair_stage_v3 as S
from rsna_knee import f009_transport as T

ROOT = Path(__file__).resolve().parents[1]


def load_fixture():
    spec = importlib.util.spec_from_file_location("_f009_transport_artificial_pair_fixture", ROOT / "tests/test_f009_pair_stage_v3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    path.write_bytes((json.dumps(value, allow_nan=False) + "\n").encode())


def pair_fixture(root):
    fixture = load_fixture()
    spec, bindings = fixture.fixture(root)
    with patch.object(P, "AMENDED_BINDINGS", bindings):
        S._seal(root, spec, root / "sealed", fixture_only=True)
    return root / "sealed"


def packed_fixture(root):
    source = pair_fixture(root)
    T._pack(source, root / "packed", expected_pair_pin=T.pin(source / "pair-sidecar.json"),
            expected_complete_pin=T.pin(source / "BUILD_COMPLETE.json"), conditional_plan_path=ROOT / T.PLAN_REL, fixture_only=True)
    return source, root / "packed"


def rebind_manifest(folder, manifest):
    save(folder / "manifest.json", manifest)
    complete = T.read(folder / "TRANSPORT_COMPLETE.json")
    complete["fold_manifests"][str(manifest["fold"])] = T.pin(folder / "manifest.json")
    save(folder / "TRANSPORT_COMPLETE.json", complete)
    return T.pin(folder / "manifest.json")["sha256"]


class TransportTests(unittest.TestCase):
    def test_artificial_twofold_roundtrip_pins_and_scope_no_tensor_import(self):
        with tempfile.TemporaryDirectory(prefix="f009-transport-toy-") as directory:
            root = Path(directory)
            source, packed = packed_fixture(root)
            for fold in (0, 1):
                incoming = packed / ("fold" + str(fold))
                descriptor = T._unpack(incoming, root / ("unpacked" + str(fold)),
                    expected_manifest_sha256=T.pin(incoming / "manifest.json")["sha256"], fold=fold, fixture_only=True)
                self.assertFalse(descriptor["actual_transport"])
                self.assertFalse(descriptor["runtime_transport_qualified"])
                self.assertTrue(descriptor["checkpoint_body_verified"])
                self.assertEqual(T.pin(descriptor["checkpoint_path"]), T.pin(source / ("fold" + str(fold)) / "trained/epoch_012.pt"))
                self.assertEqual(len(list((descriptor["source14"] / "package").glob("*.py"))), 15)
                self.assertEqual(set(descriptor["qualification_pins"]), T.QUAL_KEYS)
                self.assertEqual(T.inventory(incoming), T.FILES)
                self.assertFalse(any(name.endswith(("weak.csv", "gold.csv", ".dcm")) for name in T.inventory(descriptor["bundle_root"])))
            self.assertEqual(T.expected_shapes(), S.expected_tensor_shapes())
            with self.assertRaisesRegex(ValueError, "Fresh"):
                T._pack(source, packed, expected_pair_pin=T.pin(source / "pair-sidecar.json"),
                    expected_complete_pin=T.pin(source / "BUILD_COMPLETE.json"), conditional_plan_path=ROOT / T.PLAN_REL, fixture_only=True)
        self.assertNotIn("torch", sys.modules)

    def test_public_actual_route_rejects_artificial_and_corrupted_actual_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, packed = packed_fixture(root)
            with self.assertRaises(ValueError):
                T.pack_f009_pair(source, root / "actual", expected_pair_pin=T.pin(source / "pair-sidecar.json"),
                    expected_complete_pin=T.pin(source / "BUILD_COMPLETE.json"), conditional_plan_path=ROOT / T.PLAN_REL)
            incoming = packed / "fold0"
            with self.assertRaises(ValueError):
                T.unpack_f009_transport(incoming, root / "actual-unpack", expected_manifest_sha256=T.pin(incoming / "manifest.json")["sha256"], fold=0)
            manifest = T.read(incoming / "manifest.json")
            manifest.update(artificial_fixture_only=False, actual_transport=True)
            digest = rebind_manifest(incoming, manifest)
            with self.assertRaises(ValueError):
                T.unpack_f009_transport(incoming, root / "forged", expected_manifest_sha256=digest, fold=0)
            self.assertFalse((root / "actual/TRANSPORT_COMPLETE.json").exists())

    def test_extra_unsafe_archive_and_incomplete_packets_fail_closed(self):
        for kind in ("extra", "unsafe", "incomplete"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, packed = packed_fixture(root)
                incoming = packed / "fold0"
                if kind == "incomplete":
                    (incoming / "TRANSPORT_COMPLETE.json").unlink()
                    expected = T.pin(incoming / "manifest.json")["sha256"]
                else:
                    with zipfile.ZipFile(incoming / "bundle.zip", "a", compression=zipfile.ZIP_STORED) as archive:
                        archive.writestr("../outside.txt" if kind == "unsafe" else "foreign.bin", b"ARTIFICIAL")
                    manifest = T.read(incoming / "manifest.json")
                    manifest["archive"] = T.pin(incoming / "bundle.zip", T.MAX_ARCHIVE)
                    expected = rebind_manifest(incoming, manifest)
                with self.assertRaises(ValueError):
                    T._unpack(incoming, root / "unpacked", expected_manifest_sha256=expected, fold=0, fixture_only=True)
                self.assertFalse((root / "outside.txt").exists())
                self.assertFalse((root / "unpacked/TRANSPORT_UNPACKED.json").exists())

    def test_onefold_changed_REG_or_entire_plan_and_invalid_qualification_before_checkpoint(self):
        for kind in ("onefold", "REG", "plan", "qualification"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = pair_fixture(root)
                pair = T.read(source / "pair-sidecar.json")
                plan = T.read(ROOT / T.PLAN_REL)
                folder = source / "fold0"
                if kind == "onefold":
                    pair["CLS_E12_bundles"].pop("1")
                elif kind == "REG":
                    pair["fixed_REG_references_only"]["REG:0"]["fingerprint"]["selected_epoch"] = 9
                elif kind == "plan":
                    plan["intended_intervention"]["native_alpha"] = .3
                else:
                    # Rehash all outer seals, but the copied statistical semantics must still reject.
                    changed = T.read(folder / "audit/result1.json")
                    changed["actual_comparison"] = False
                    save(folder / "audit/result1.json", changed)
                    bundle = T.read(folder / "bundle.json")
                    bundle["assets"]["audit/result1.json"] = T.pin(folder / "audit/result1.json")
                    save(folder / "bundle.json", bundle)
                    pair["CLS_E12_bundles"]["0"] = T.pin(folder / "bundle.json")
                original = T.pin
                def no_checkpoint(path, *args):
                    self.assertNotEqual(Path(path).suffix, ".pt", "Bad metadata reached checkpoint body")
                    return original(path, *args)
                with patch.object(T, "pin", side_effect=no_checkpoint), self.assertRaises(ValueError):
                    T._consume(folder, expected_bundle_pin=pair["CLS_E12_bundles"]["0"], pair=pair,
                               conditional_plan=plan, fold=0, fixture_only=True)

    def test_failure_marker_no_complete_and_default_CLI_opens_no_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = pair_fixture(root)
            original = T.copy_stream
            def interrupted(incoming, destination, expected):
                if "fold1" in str(destination):
                    raise OSError("Artificial interruption")
                return original(incoming, destination, expected)
            with patch.object(T, "copy_stream", side_effect=interrupted), self.assertRaises(OSError):
                T._pack(source, root / "interrupted", expected_pair_pin=T.pin(source / "pair-sidecar.json"),
                    expected_complete_pin=T.pin(source / "BUILD_COMPLETE.json"), conditional_plan_path=ROOT / T.PLAN_REL, fixture_only=True)
            self.assertFalse((root / "interrupted").exists())
            self.assertFalse(T.read(root / "interrupted.partial/TRANSPORT_FAILED.json")["complete"])
            self.assertFalse((root / "interrupted.partial/fold0/TRANSPORT_COMPLETE.json").exists())
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/prepare_f009_transport.py"),
                                 "--input", "NEVER_OPEN_THIS"], env=env, capture_output=True, text=True, check=True)
        self.assertFalse(json.loads(result.stdout)["input_model_or_run_opened"])
        self.assertFalse(json.loads(result.stdout)["actual_transport_created"])

    def test_late_second_fold_marker_failure_never_publishes_or_accepts_first_fold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = pair_fixture(root)
            original = T.copy_stream
            first_marker_copied = []
            def fail_final_marker(incoming, destination, expected):
                if destination.name == "TRANSPORT_COMPLETE.json":
                    if destination.parent.name == "fold1":
                        self.assertTrue(first_marker_copied)
                        raise OSError("Artificial failure after first complete fold marker")
                    result = original(incoming, destination, expected)
                    first_marker_copied.append(True)
                    return result
                return original(incoming, destination, expected)
            with patch.object(T, "copy_stream", side_effect=fail_final_marker), self.assertRaisesRegex(OSError, "first complete"):
                T._pack(source, root / "late", expected_pair_pin=T.pin(source / "pair-sidecar.json"),
                    expected_complete_pin=T.pin(source / "BUILD_COMPLETE.json"), conditional_plan_path=ROOT / T.PLAN_REL, fixture_only=True)
            self.assertFalse((root / "late").exists())
            partial = root / "late.partial"
            self.assertFalse(T.read(partial / "TRANSPORT_FAILED.json")["complete"])
            for fold in (0, 1):
                incoming = partial / ("fold" + str(fold))
                self.assertFalse((incoming / "TRANSPORT_COMPLETE.json").exists())
                with self.assertRaisesRegex(ValueError, "Exact transport files"):
                    T._unpack(incoming, root / ("must-not-unpack" + str(fold)),
                        expected_manifest_sha256=T.pin(incoming / "manifest.json")["sha256"], fold=fold, fixture_only=True)
                self.assertFalse((root / ("must-not-unpack" + str(fold))).exists())


if __name__ == "__main__":
    unittest.main()
