"""Synthetic metadata orchestration only; no actual qualification is produced."""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsna_knee import f009_release as R

ROOT = Path(__file__).resolve().parents[1]


def write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return {"path": str(path), **R.raw_pin(body)}


def save(path, value):
    return write(path, R.json_bytes(value))


def fixture(root):
    """Intentionally forged scope flags: only a mocked CPU consumer may accept."""
    control, paths, parents = R._frozen_control()
    inventory = {paths[key]: R.raw_pin(b"# synthetic integration source\n") for key in ("router", "child", "extension")}
    inventory[paths["parent"]] = R.raw_pin(control["parent_raw"])
    for directory in paths["source_roots"].values():
        inventory[directory + "/__init__.py"] = R.raw_pin(b"# synthetic package\n")
    for name in (*R.context.FIXED_SUPPORT, "f009_kaggle_context.py", "f009_kaggle_driver.py"):
        inventory[paths["source_roots"]["support"] + "/" + name] = R.raw_pin((ROOT / "src/rsna_knee" / name).read_bytes())
    pair_root = root / "synthetic-sealed"
    bundles = {}
    for fold in (0, 1):
        folder = pair_root / ("fold" + str(fold))
        assets = {}
        for name in sorted(R.transport.asset_names({"synthetic_root_spec": True})):
            if name == R.CHECKPOINT_MEMBER:
                # A real suffix/head checkpoint can exceed the metadata cap.
                # Its declared pin is accepted without creating or opening it.
                assets[name] = {"bytes": 64 * 1024**2, "sha256": hashlib.sha256(b"NO CHECKPOINT BODY").hexdigest()}
            else:
                body = R.json_bytes({"deliberately_incomplete_synthetic_proof": True}) if name.endswith(".json") else b"# synthetic source only\n"
                assets[name] = R.raw_pin(body)
                write(folder / name, body)
        b = {"schema_version": R.transport.BUNDLE_SCHEMA, "fold": fold,
             "artificial_fixture_only": False, "actual_strict_CPU_accepted": True, "assets": assets}
        bundles[str(fold)] = R._pin_pair(save(folder / "bundle.json", b))
    pair = {"schema_version": R.transport.PAIR_SCHEMA, "local_pair_complete": True,
            "artificial_fixture_only": False, "actual_CLS_stage_constructed": True,
            "four_model_transport_runtime_qualified": False, "old_v1_joint_screen": "HOLD",
            "root_spec_pin": {"synthetic_root_spec": True}, "CLS_E12_bundles": bundles}
    pair_ref = save(pair_root / "pair-sidecar.json", pair)
    complete_ref = save(pair_root / "BUILD_COMPLETE.json", {"complete": True, "artificial_fixture_only": False,
        "pair_sidecar": R._pin_pair(pair_ref), "four_model_transport_runtime_qualified": False})
    manifests, manifest_refs = {}, {}
    for fold in (0, 1):
        key = str(fold)
        bundle = json.loads((pair_root / ("fold" + key) / "bundle.json").read_bytes())
        manifest = {"schema_version": R.transport.SCHEMA, "fold": fold, "actual_transport": True,
            "artificial_fixture_only": False, "transport_source_sha256": R.TRANSPORT_SHA,
            "runtime_or_Public_qualified": False, "pair": R._pin_pair(pair_ref), "complete": R._pin_pair(complete_ref),
            "plan": R.raw_pin(control["plan_raw"]), "bundle": bundles[key], "archive": R.raw_pin(b"ARCHIVE MUST NOT BE OPENED"),
            "members": {**bundle["assets"], "bundle.json": bundles[key]}}
        manifest_refs[key] = save(root / "synthetic-transport" / ("fold" + key) / "manifest.json", manifest)
        manifests[key] = R._pin_pair(manifest_refs[key])
    completion = {"complete": True, "actual_transport": True, "artificial_fixture_only": False,
                  "pair": R._pin_pair(pair_ref), "fold_manifests": manifests, "runtime_or_Public_qualified": False}
    transports = {str(f): {"manifest": manifest_refs[str(f)], "complete": save(root / "synthetic-transport" / ("fold" + str(f)) / "TRANSPORT_COMPLETE.json", completion)} for f in (0, 1)}
    spec = {"schema_version": R.SPEC_SCHEMA, "pair_directory": str(pair_root), "pair": pair_ref, "complete": complete_ref,
            "transports": transports, "CLS_datasets": {"0": "rsraki/artificial-release-cls-fold0/1", "1": "rsraki/artificial-release-cls-fold1/1"},
            "source_inventory": save(root / "source-inventory.json", inventory), "runtime_paths": save(root / "runtime-paths.json", paths)}
    return spec, control, paths, parents


def assemble(root, spec, *, output_name="assembled"):
    path = root / "input-spec.json"
    ref = save(path, spec)
    return R.assemble_f009_release(path, ref["sha256"], root / output_name,
        expected_assembler_sha256=hashlib.sha256(Path(R.__file__).read_bytes()).hexdigest(), execute=True)


class ReleaseTests(unittest.TestCase):
    def test_default_no_body_and_required_explicit_source(self):
        with patch.object(Path, "read_bytes", side_effect=AssertionError("No body read")):
            self.assertFalse(R.assemble_f009_release()["actual_release_assembled"])
            with self.assertRaisesRegex(ValueError, "Explicit assembler"):
                R.assemble_f009_release(execute=True)
        self.assertNotIn("torch", sys.modules)

    def test_mocked_positive_stable_paths_and_no_checkpoint_or_archive_access(self):
        with tempfile.TemporaryDirectory(prefix="f009-release-metadata-qa-") as directory:
            root = Path(directory)
            spec, _, _, _ = fixture(root)
            original = Path.read_bytes
            def guarded(path):
                self.assertNotIn(path.suffix, {".pt", ".zip", ".dcm", ".bin"})
                return original(path)
            with patch.object(Path, "read_bytes", guarded), patch.object(R.context.transport, "_consume",
                    return_value={"checkpoint_body_verified": False, "actual_transport": False}) as consume:
                result = assemble(root, spec)
            self.assertTrue(result["complete"])
            self.assertEqual(consume.call_count, 2)
            self.assertTrue(all(c.kwargs["metadata_only"] is True and c.kwargs["fixture_only"] is False for c in consume.call_args_list))
            output = root / "assembled"
            release = json.loads((output / "release.json").read_bytes())
            self.assertEqual(set(release["context"]), R.context.CLAIMS)
            self.assertEqual(release["context"]["inputs"]["total"], 9)
            self.assertFalse(release["private_Inputs_exist_or_mounted_verified"])
            for target, ref in release["metadata_files"].items():
                self.assertEqual(Path(ref["path"]), output / target)
                self.assertEqual(R.raw_pin(Path(ref["path"]).read_bytes()), R._pin_pair(ref))
            self.assertFalse((output / ".release-candidate.json").exists())
            self.assertTrue((output / "ASSEMBLY_COMPLETE.json").is_file())
            self.assertFalse(any(p.suffix in {".pt", ".zip", ".dcm"} for p in output.rglob("*")))
            with self.assertRaisesRegex(ValueError, "Fresh"):
                assemble(root, spec)

    def test_public_actual_refuses_artificial_and_forged_cpu_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, _, _, _ = fixture(root)
            with self.assertRaises((KeyError, ValueError)):
                assemble(root, spec)
            output = root / "assembled"
            self.assertFalse((output / "release.json").exists())
            self.assertFalse((output / ".release-candidate.json").exists())
            self.assertFalse((output / "metadata/root-release.json").exists())
            self.assertFalse((output / "f009-context.json").exists())
            self.assertFalse((output / "ASSEMBLY_COMPLETE.json").exists())
            self.assertFalse(json.loads((output / "ASSEMBLY_FAILED.json").read_bytes())["complete"])
            pair = json.loads(Path(spec["pair"]["path"]).read_bytes())
            pair["artificial_fixture_only"] = True
            spec["pair"] = save(Path(spec["pair"]["path"]), pair)
            spec["complete"] = save(Path(spec["complete"]["path"]), {"complete": True, "artificial_fixture_only": True,
                "pair_sidecar": R._pin_pair(spec["pair"])})
            with self.assertRaisesRegex(ValueError, "Actual amended"):
                assemble(root, spec, output_name="artificial-rejected")
            self.assertFalse((root / "artificial-rejected").exists())

    def test_input_mutations_and_forbidden_metadata_suffix_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, _, _, _ = fixture(root)
            for kind in ("same-dataset", "version2", "transport-complete-pin", "source-inventory-pin"):
                changed = copy.deepcopy(spec)
                if kind == "same-dataset":
                    changed["CLS_datasets"]["1"] = changed["CLS_datasets"]["0"]
                elif kind == "version2":
                    changed["CLS_datasets"]["1"] = changed["CLS_datasets"]["1"][:-1] + "2"
                elif kind == "transport-complete-pin":
                    changed["transports"]["1"]["complete"]["sha256"] = "0" * 64
                else:
                    changed["source_inventory"]["sha256"] = "0" * 64
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    assemble(root, changed, output_name=kind)
                self.assertFalse((root / kind).exists())
            with patch.object(Path, "read_bytes", side_effect=AssertionError("No forbidden body read")):
                with self.assertRaisesRegex(ValueError, "suffixes"):
                    R._reference({"path": str(root / "metadata.pt"), "bytes": 1, "sha256": "0" * 64})
            # Simulate the OS symlink flag without requiring Windows privilege.
            alias = root / "synthetic-output-link"
            original_is_symlink = Path.is_symlink
            with patch.object(Path, "is_symlink", lambda path: path == alias or original_is_symlink(path)):
                with self.assertRaisesRegex(ValueError, "Output symlink ancestor"):
                    assemble(root, spec, output_name="synthetic-output-link/output")
            self.assertFalse(alias.exists())

    def test_late_failure_withdraws_every_actual_release_marker(self):
        for target in ("f009-context.json", "ASSEMBLY_COMPLETE.json"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                spec, _, _, _ = fixture(root)
                original_replace = R.os.replace
                def fault(source, destination):
                    if Path(destination).name == target:
                        self.assertTrue(Path(source).is_file())
                        self.assertEqual(Path(source).name, target + ".partial")
                        if target == "ASSEMBLY_COMPLETE.json":
                            self.assertTrue((root / "assembled/release.json").is_file())
                        raise OSError("Artificial partial-to-final publication fault")
                    return original_replace(source, destination)
                with patch.object(R.context.transport, "_consume", return_value={"checkpoint_body_verified": False, "actual_transport": False}), patch.object(R.os, "replace", fault):
                    with self.assertRaisesRegex(OSError, "publication fault"):
                        assemble(root, spec)
                output = root / "assembled"
                gates = ("release.json", ".release-candidate.json", "metadata/root-release.json", "f009-context.json", "ASSEMBLY_COMPLETE.json", "assembly-receipt.json")
                for name in (*gates, *(name + ".partial" for name in gates)):
                    self.assertFalse((output / name).exists())
                self.assertFalse(json.loads((output / "ASSEMBLY_FAILED.json").read_bytes())["actual_qualified_release"])


if __name__ == "__main__":
    unittest.main()
