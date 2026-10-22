"""Generated metadata plumbing only; no real model, MRI, tensor or API access.

The positive envelope test mocks the already independently tested real CPU
metadata consumer. The same envelope must fail with the real consumer.
"""
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsna_knee import f009_kaggle_context as C

ROOT = Path(__file__).resolve().parents[1]


def builder():
    spec = importlib.util.spec_from_file_location("_context_test_source_builder", ROOT / "scripts/prepare_f009_p033_native20.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(root, name, body):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return {"path": name, **C.raw_pin(body)}


def fixture(root):
    """Use actual immutable source and generated, deliberately incomplete proof metadata."""
    b = builder()
    control, paths = b.inspect_control(), b.runtime_paths()
    files = {paths[key]: b"# artificial source plumbing\n" for key in ("router", "child", "extension")}
    files[paths["parent"]] = control["parent_raw"]
    for role, directory in paths["source_roots"].items():
        files[directory + "/__init__.py"] = b"# artificial import root\n"
    for name in (*C.FIXED_SUPPORT, "f009_kaggle_driver.py", "f009_kaggle_context.py"):
        files[paths["source_roots"]["support"] + "/" + name] = (ROOT / "src/rsna_knee" / name).read_bytes()
    for name, body in files.items():
        write(root, name, body)
    inventory = {name: C.raw_pin(body) for name, body in files.items()}
    meta = dict(control["reg_metadata"])
    meta["metadata/conditional-plan.json"] = control["plan_raw"]
    def pin_meta(name):
        return {"path": name, **C.raw_pin(meta[name])}
    pair = {"CLS_E12_bundles": {}, "schema_version": "DELIBERATELY_INCOMPLETE_SYNTHETIC_PROOF"}
    bundles = {}
    for f in (0, 1):
        bundles[str(f)] = {"assets": {"trained/epoch_012.pt": C.raw_pin(b"NO REAL CHECKPOINT")}}
        name = "metadata/CLS/fold" + str(f) + "/bundle/bundle.json"
        meta[name] = C.json_bytes(bundles[str(f)])
        pair["CLS_E12_bundles"][str(f)] = C.raw_pin(meta[name])
    meta["metadata/CLS/pair-sidecar.json"] = C.json_bytes(pair)
    pair_ref = pin_meta("metadata/CLS/pair-sidecar.json")
    complete = {"complete": True, "pair_sidecar": C._pin_spec(pair_ref, path=True), "artificial_fixture_only": False}
    meta["metadata/CLS/BUILD_COMPLETE.json"] = C.json_bytes(complete)
    complete_ref = pin_meta("metadata/CLS/BUILD_COMPLETE.json")
    plan_ref = pin_meta("metadata/conditional-plan.json")
    jobs = copy.deepcopy(control["plan"]["intended_intervention"]["old_REG_transport_refs"])
    rows, manifests = {}, {}
    for f in (0, 1):
        key = str(f)
        prefix = "metadata/CLS/fold" + key
        manifest = {"schema_version": C.transport.SCHEMA, "fold": f, "actual_transport": True,
                    "artificial_fixture_only": False, "runtime_or_Public_qualified": False,
                    "transport_source_sha256": C.FIXED_SUPPORT["f009_transport.py"],
                    "pair": C._pin_spec(pair_ref, path=True), "complete": C._pin_spec(complete_ref, path=True),
                    "plan": C._pin_spec(plan_ref, path=True), "bundle": pair["CLS_E12_bundles"][key],
                    "archive": C.raw_pin(b"NO REAL ARCHIVE")}
        manifest["members"] = {**bundles[key]["assets"], "bundle.json": manifest["bundle"]}
        meta[prefix + "/transport-manifest.json"] = C.json_bytes(manifest)
        ref = pin_meta(prefix + "/transport-manifest.json")
        manifests[key] = C._pin_spec(ref, path=True)
        rows[key] = {"bundle_root": prefix + "/bundle", "transport_manifest": ref}
        dataset = "rsraki/artificial-context-cls-fold" + key + "/1"
        jobs["CLS:" + key] = {"input_root": "/kaggle/input/datasets/" + dataset.rsplit("/", 1)[0],
                             "dataset_ref": dataset, "manifest_sha": ref["sha256"],
                             "archive_sha": manifest["archive"]["sha256"], "bundle_sha": manifest["bundle"]["sha256"]}
        jobs["REG:" + key].update(root_permission=paths["REG_metadata"]["REG:" + key]["permission"],
                                 peer_review=paths["REG_metadata"]["REG:" + key]["peer"])
    tc = {"complete": True, "actual_transport": True, "artificial_fixture_only": False,
          "pair": C._pin_spec(pair_ref, path=True), "fold_manifests": manifests}
    for f in (0, 1):
        name = "metadata/CLS/fold" + str(f) + "/TRANSPORT_COMPLETE.json"
        meta[name] = C.json_bytes(tc)
        rows[str(f)]["transport_complete"] = pin_meta(name)
    claims = {"conditional_plan": plan_ref, "CLS_pair": pair_ref, "CLS_complete": complete_ref,
              "CLS_metadata": rows, "parent": control["plan"]["frozen_control"],
              "integration": control["plan"]["intended_intervention"]["integration_contract"],
              "inputs": {"datasets": sorted(C.FIXED_INPUTS | {j["dataset_ref"] for j in jobs.values()}),
                         "kernels": ["rsraki/p029-r6v1-stage-20261015-b2/1"],
                         "competitions": ["rsna-knee-abnormality-detection"], "total": 9},
              "decoder_root": C.DECODER_ROOT, "transport_jobs": jobs}
    metadata_pins = {}
    for name, body in meta.items():
        origin = root / "origin" / name
        origin.parent.mkdir(parents=True, exist_ok=True)
        origin.write_bytes(body)
        metadata_pins[name] = {"path": str(origin), **C.raw_pin(body)}
    release = {"schema_version": C.RELEASE_SCHEMA, "actual_qualified_release": True,
               "artificial_fixture_only": False, "root_authorized_CPU_permission_derivation": True,
               "context": claims, "metadata_files": metadata_pins,
               "fixture_scope": "Incomplete synthetic envelope; real CPU consumer must reject"}
    release_path = root / "origin/root-release.json"
    release_path.write_bytes(C.json_bytes(release))
    return inventory, paths, release_path


class ContextTests(unittest.TestCase):
    def test_template_is_unready_and_does_not_read_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory, paths, _ = fixture(root)
            with patch.object(C, "_body", side_effect=AssertionError("No body read")):
                context, files = C.context_template(source_inventory=inventory, runtime_paths=paths)
            self.assertFalse(context["actual_entry_enabled"])
            path = root / "f009-context.json"
            path.write_bytes(C.json_bytes(context))
            with self.assertRaisesRegex(ValueError, "Source preparation"):
                C.validate_context(path, C.raw_pin(path.read_bytes())["sha256"], runtime_root=root)
            self.assertEqual(set(files), {"source-inventory.json"})

    def test_generated_envelope_roundtrip_no_checkpoint_and_real_proof_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory, paths, release_path = fixture(root)
            kwargs = dict(source_inventory=inventory, runtime_paths=paths)
            digest = C.raw_pin(release_path.read_bytes())["sha256"]
            with self.assertRaises((ValueError, KeyError)):
                C.prepare_context(release_path, digest, **kwargs)
            with patch.object(C.transport, "_consume", return_value={"checkpoint_body_verified": False, "actual_transport": False}) as consumer:
                context, files = C.prepare_context(release_path, digest, **kwargs)
                for name, body in files.items():
                    write(root, name, body)
                path = root / "f009-context.json"
                path.write_bytes(C.json_bytes(context))
                result = C.validate_context(path, C.raw_pin(path.read_bytes())["sha256"], runtime_root=root)
                self.assertEqual(consumer.call_count, 4)
                self.assertTrue(all(call.kwargs["metadata_only"] is True and call.kwargs["fixture_only"] is False for call in consumer.call_args_list))
            self.assertEqual(set(result), {"context", "source_root", "source_roots", "source_inventory", "conditional_plan", "CLS_pair", "root_release", "transport_jobs"})
            self.assertTrue(Path(result["transport_jobs"]["REG:0"]["root_permission"]).is_absolute())
            self.assertFalse(any(name.endswith((".pt", ".zip", ".dcm", ".bin")) for name in files))
            with self.assertRaises((ValueError, KeyError)):
                C.validate_context(path, C.raw_pin(path.read_bytes())["sha256"], runtime_root=root)
            self.assertNotIn("torch", sys.modules)

    def test_source_pin_release_claim_and_input_mutations_fail_before_transport(self):
        for kind in ("source", "parent", "inputs", "onefold", "authority", "context-sha"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                inventory, paths, release_path = fixture(root)
                with patch.object(C.transport, "_consume", return_value={"checkpoint_body_verified": False, "actual_transport": False}):
                    context, files = C.prepare_context(release_path, C.raw_pin(release_path.read_bytes())["sha256"],
                                                       source_inventory=inventory, runtime_paths=paths)
                for name, body in files.items():
                    write(root, name, body)
                if kind == "source":
                    (root / paths["child"]).write_bytes(b"changed\n")
                elif kind == "parent":
                    context["parent"]["version"] = 2
                elif kind == "inputs":
                    context["inputs"]["total"] = 8
                elif kind == "onefold":
                    context["transport_jobs"].pop("CLS:1")
                elif kind == "authority":
                    release = json.loads(files["metadata/root-release.json"])
                    release["root_authorized_CPU_permission_derivation"] = False
                    body = C.json_bytes(release)
                    write(root, "metadata/root-release.json", body)
                    context["root_release"] = {"path": "metadata/root-release.json", **C.raw_pin(body)}
                path = root / "f009-context.json"
                body = C.json_bytes(context)
                path.write_bytes(body)
                with patch.object(C.transport, "_consume", side_effect=AssertionError("Must reject before CPU metadata")):
                    with self.assertRaises(ValueError):
                        C.validate_context(path, "0" * 64 if kind == "context-sha" else C.raw_pin(body)["sha256"], runtime_root=root)

    def test_unsafe_inventory_and_metadata_checkpoint_disguise_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory, paths, release_path = fixture(root)
            for name in ("../outside.py", "bad.pt", "metadata/cycle.json", "CON.py"):
                with self.subTest(name=name):
                    changed = {**inventory, name: C.raw_pin(b"x")}
                    with self.assertRaises(ValueError):
                        C.context_template(source_inventory=changed, runtime_paths=paths)
            release = json.loads(release_path.read_bytes())
            fake_cp = root / "small-synthetic.pt"
            fake_cp.write_bytes(b"DO NOT OPEN AS METADATA")
            release["metadata_files"]["metadata/disguised.json"] = {"path": str(fake_cp), **C.raw_pin(fake_cp.read_bytes())}
            release_path.write_bytes(C.json_bytes(release))
            original_body = C._body
            def guard(path, expected=None):
                self.assertNotEqual(Path(path), fake_cp)
                return original_body(path, expected)
            with patch.object(C, "_body", side_effect=guard), self.assertRaisesRegex(ValueError, "suffix"):
                C.prepare_context(release_path, C.raw_pin(release_path.read_bytes())["sha256"], source_inventory=inventory, runtime_paths=paths)

    def test_two_versions_of_same_dataset_cannot_claim_two_mounts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory, paths, release_path = fixture(root)
            release = json.loads(release_path.read_bytes())
            claims = release["context"]
            jobs = claims["transport_jobs"]
            old_ref = jobs["CLS:1"]["dataset_ref"]
            new_ref = jobs["CLS:0"]["dataset_ref"].rsplit("/", 1)[0] + "/2"
            jobs["CLS:1"].update(dataset_ref=new_ref, input_root=jobs["CLS:0"]["input_root"])
            claims["inputs"]["datasets"] = [new_ref if ref == old_ref else ref for ref in claims["inputs"]["datasets"]]
            release_path.write_bytes(C.json_bytes(release))
            with patch.object(C.transport, "_consume", return_value={"checkpoint_body_verified": False, "actual_transport": False}):
                with self.assertRaisesRegex(ValueError, "Exact seven"):
                    C.prepare_context(release_path, C.raw_pin(release_path.read_bytes())["sha256"],
                                      source_inventory=inventory, runtime_paths=paths)

    def test_checkpoint_named_entry_points_fail_before_open(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "synthetic.pt"
            path.write_bytes(b"THIS IS NOT A CHECKPOINT")
            with patch.object(Path, "read_bytes", side_effect=AssertionError("No data body may open")):
                with self.assertRaisesRegex(ValueError, "bodies prohibited"):
                    C.validate_context(path, "0" * 64, runtime_root=root)
                with self.assertRaisesRegex(ValueError, "bodies prohibited"):
                    C._body(path)


if __name__ == "__main__":
    unittest.main()
