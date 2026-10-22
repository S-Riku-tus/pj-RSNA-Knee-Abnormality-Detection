"""Artificial CPU contracts only: no real MRI/model/labels or Torch import."""
import copy
import csv
import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = load("innerC_runtime_artificial", "scripts/orthodiffusion_inner_c_test_runtime.py")
B = load("innerC_builder_artificial", "scripts/prepare_orthodiffusion_inner_c_test.py")


def fake_success(uid, value=.25):
    values = [value] * 1024
    raw = struct.pack("<1024f", *values)
    return {R.ID: uid, "status": "success", "regional_2x2_float32_values": values,
            "regional_2x2": {"shape": [1, 1024], "dtype": "torch.float32", "bytes": 4096,
                             "sha256": hashlib.sha256(raw).hexdigest()},
            "forward_counts": {"denoiser": 1, "decoder_blocks": 0, "final_out": 0},
            "all_RNG_restored": True}


def dump(path, value):
    path.write_bytes((json.dumps(value, allow_nan=False) + "\n").encode())
    return R.pin(path)["sha256"]


class InnerCTestController(unittest.TestCase):
    def test_import_is_stdlib_only_and_bundle_preserves_science_without_cohort(self):
        code = ("import importlib.util,sys;"
                "s=importlib.util.spec_from_file_location('r','scripts/orthodiffusion_inner_c_test_runtime.py');"
                "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                "assert 'torch' not in sys.modules and 'numpy' not in sys.modules")
        self.assertEqual(subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True).returncode, 0)
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "pack"
            manifest = B.build(out)
            bundle = R.load_bundle(out / "bundle.json", manifest["bundle"]["sha256"])
            self.assertFalse(bundle["actual_entry_enabled"])
            self.assertNotIn("uids", bundle)
            self.assertNotIn("regional_parent_input", bundle)
            self.assertNotIn("protocol", bundle["assets"])
            self.assertEqual(manifest["normalization_forward_assignments_AST_identical"], 9)
            self.assertEqual(manifest["helper_members_AST_identical"], 24)
            spaces = R.helper_namespaces(bundle)
            self.assertTrue(all("run" not in value and "main" not in value for value in spaces.values()))
            self.assertTrue(spaces["pilot"]["classify_pd"]("PD_FS", "") ["eligible"])
            self.assertFalse(spaces["pilot"]["classify_pd"]("T2 TSE", "") ["eligible"])
            self.assertFalse(spaces["pilot"]["classify_pd"]("DP", "") ["eligible"])
            self.assertTrue(manifest["pinned_old_sources_unchanged"])

    def test_dynamic_UID_projection_order_and_missing_series(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "test.csv").write_text("Report,StudyInstanceUID,ACL\nPOISON,u3,POISON\nPOISON,u1,POISON\n", encoding="utf-8")
            (folder / "test_series.csv").write_text(
                "StudyInstanceUID,SeriesInstanceUID,Anatomical_Plane,Fat_Suppression,Fluid_Sensitive,Report\n"
                "u1,s1,sagittal,0,0,POISON\nother,s99,sagittal,0,0,POISON\n", encoding="utf-8")
            uids, by_uid, metadata = R.read_test_metadata(folder / "test.csv", folder / "test_series.csv")
            self.assertEqual(uids, ["u3", "u1"])
            self.assertEqual(by_uid["u3"], [])
            self.assertEqual(set(by_uid["u1"][0]), {R.ID, "SeriesInstanceUID", "Anatomical_Plane", "Fat_Suppression", "Fluid_Sensitive"})
            self.assertEqual(metadata["requested_count"], 2)
            (folder / "test.csv").write_text("StudyInstanceUID\nu1\nu1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                R.read_test_metadata(folder / "test.csv", folder / "test_series.csv")
            for value in ("../u", "/u", "u\\x", "..", ""):
                with self.assertRaises(ValueError):
                    R.safe_uid(value)

    def test_selection_is_original_PD_and_chosen_geometry_failure_does_not_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            manifest = B.build(folder / "pack")
            bundle = R.load_bundle(folder / "pack/bundle.json", manifest["bundle"]["sha256"])
            pilot = R.helper_namespaces(bundle)["pilot"]
            for sid in ("a", "b"):
                target = folder / "u" / sid
                target.mkdir(parents=True)
                (target / "fake.dcm").write_bytes(b"ARTIFICIAL_HEADER_STUB_NOT_DICOM")
            def header(path, **_kwargs):
                return types.SimpleNamespace(StudyInstanceUID="u", SeriesInstanceUID=path.parent.name,
                                             SeriesDescription="PD", SequenceName="")
            fake_dicom = types.SimpleNamespace(dcmread=header, errors=types.SimpleNamespace(InvalidDicomError=ValueError))
            rows = [{R.ID: "u", "SeriesInstanceUID": sid, "Anatomical_Plane": "sagittal",
                     "Fat_Suppression": "0", "Fluid_Sensitive": "0"} for sid in ("b", "a")]
            chosen = []
            def prepare(uid, series):
                selected, _paths = pilot["choose_series"](uid, series, folder, fake_dicom)
                chosen.append(selected["SeriesInstanceUID"])
                error = pilot["PilotDataError"]("artificial_chosen_geometry_failure")
                error.selected_series = selected
                raise error
            records = R.process_uids(["u"], {"u": rows}, prepare,
                                     lambda *_: self.fail("No forward on data failure"), (pilot["PilotDataError"],))
            self.assertEqual(chosen, ["a"])
            self.assertEqual(records[0]["selected_series"]["SeriesInstanceUID"], "a")
            self.assertEqual(R.validate_partition(["u"], records), ([], ["u"]))

    def test_partition_allfail_and_forward_clock_errors_are_fatal(self):
        def no_data(*_args):
            raise ValueError("artificial_missing_PD")
        failed = R.process_uids(["c", "a", "b"], {}, no_data, lambda *_: self.fail("forward"), (ValueError,))
        self.assertEqual(R.validate_partition(["c", "a", "b"], failed), ([], ["c", "a", "b"]))
        for operation in (lambda: R.process_uids(["u"], {}, lambda *_: 1,
                          lambda *_: (_ for _ in ()).throw(RuntimeError("model-fatal")), (RuntimeError,)),
                          lambda: R.process_uids(["u"], {}, lambda *_: 1, lambda *_: fake_success("u"),
                          (ValueError,), boundary=lambda: (_ for _ in ()).throw(ValueError("clock-fatal")))):
            with self.assertRaises((RuntimeError, ValueError)):
                operation()
        record = fake_success("u")
        for mutate in (lambda x: x.update(StudyInstanceUID="other"),
                       lambda x: x["regional_2x2_float32_values"].pop(),
                       lambda x: x["regional_2x2"].update(sha256="0" * 64),
                       lambda x: x.update(all_RNG_restored=False)):
            bad = copy.deepcopy(record)
            mutate(bad)
            with self.assertRaises(ValueError):
                R.validate_partition(["u"], [bad])
        bad_failure = copy.deepcopy(failed[0])
        bad_failure["regional_2x2_float32_values"] = [0.] * 1024
        with self.assertRaises(ValueError):
            R.validate_partition(["c"], [bad_failure])

    def test_handoff_roundtrip_success_only_allfail_and_hash_shape_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            failure = R.make_failure("b", None, ValueError("fake_PD_missing"), 0.)
            records = [fake_success("z"), failure, fake_success("a", .5)]
            R.write_feature_handoff(folder, ["z", "b", "a"], records, np, "f" * 64, {})
            sha = R.pin(folder / "feature-receipt.json")["sha256"]
            receipt, values = R.load_feature_handoff(folder, sha, np, "f" * 64)
            self.assertEqual(receipt["success_uids"], ["z", "a"])
            self.assertEqual(values.shape, (2, 1024))
            self.assertEqual(values[:, 0].tolist(), [.25, .5])
            (folder / "success-features.npz").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                R.load_feature_handoff(folder, sha, np, "f" * 64)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            R.write_feature_handoff(folder, ["b"], [failure], np, "f" * 64, {})
            sha = R.pin(folder / "feature-receipt.json")["sha256"]
            _receipt, values = R.load_feature_handoff(folder, sha, np, "f" * 64)
            self.assertEqual(values.shape, (0, 1024))
            with (folder / "success-features.npz").open("wb") as stream:
                np.savez(stream, success_uids=np.asarray([], dtype=str), features=np.empty((0, 256), dtype=np.float32))
            receipt = json.loads((folder / "feature-receipt.json").read_bytes())
            receipt["matrix"].update(R.pin(folder / "success-features.npz"))
            sha = dump(folder / "feature-receipt.json", receipt)
            with self.assertRaises(ValueError):
                R.load_feature_handoff(folder, sha, np, "f" * 64)

    def test_qualification_and_source_are_failclosed_before_data_or_Torch(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            manifest = B.build(folder / "disabled")
            with self.assertRaisesRegex(ValueError, "disabled"):
                R.actual_entry(folder / "disabled/bundle.json", manifest["bundle"]["sha256"],
                               "DO_NOT_OPEN_MODEL", R.MODEL_SHA, "DO_NOT_OPEN_QUALIFICATION", "a" * 64)
            with self.assertRaisesRegex(ValueError, "gate required"):
                R.extract_test_features({}, "DO_NOT_OPEN_TEST_METADATA", folder / "DO_NOT_CREATE", 1.)
            self.assertFalse((folder / "DO_NOT_CREATE").exists())
            enabled = B.build(folder / "enabled", True)
            bad_q = folder / "bad-root-qualification.json"
            sha = dump(bad_q, {"feature_transport_bundle_sha256": "0" * 64})
            with self.assertRaisesRegex(ValueError, "bind this new feature transport"):
                R.actual_entry(folder / "enabled/bundle.json", enabled["bundle"]["sha256"],
                               "DO_NOT_OPEN_MODEL", R.MODEL_SHA, bad_q, sha)
            bundle = json.loads((folder / "disabled/bundle.json").read_bytes())
            bundle["helper_closures"]["pilot"]["source"] += "\n# changed\n"
            altered = folder / "changed-bundle.json"
            sha = dump(altered, bundle)
            with self.assertRaisesRegex(ValueError, "closure changed"):
                R.load_bundle(altered, sha)
            self.assertNotIn("torch", sys.modules)

    def test_parent_header_is_exact_and_original_row_text_order_survives(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "parent.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow([R.ID, *R.TARGETS])
                writer.writerows([["b", *["1e-3"] * 12], ["a", *["0.50000"] * 12]])
            rows = R.read_parent_rows(path, ["a", "b"])
            self.assertEqual([row[R.ID] for row in rows], ["b", "a"])
            self.assertEqual(rows[0]["ACL"], "1e-3")
            raw = path.read_text(encoding="utf-8")
            path.write_text(raw.replace("ACL,MCL", "MCL,ACL"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "header order"):
                R.read_parent_rows(path, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
