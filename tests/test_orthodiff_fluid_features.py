"""Four bounded synthetic transport guards, no fit/model/MRI/child execution."""
import copy
import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from rsna_knee import orthodiff_fluid_features as C  # noqa: E402

spec = importlib.util.spec_from_file_location("fd_actual_validator_test", ROOT / "scripts/validate_orthodiffusion_fluid_fallback.py")
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def success(uid):
    values = [i / 128 for i in range(256)]
    return {"StudyInstanceUID": uid, "status": "success", "metadata": {"route": "fluid_sensitive_fallback", "confirmed_PD": False, "stratum": "fluid_sensitive_FS"},
            "pooled_float32_values": values, "pooled": {"shape": [1, 256], "dtype": "torch.float32", "bytes": 1024,
                "sha256": hashlib.sha256(struct.pack("<256f", *values)).hexdigest()},
            "all_RNG_restored": True, "forward_counts": {"denoiser": 1, "decoder_blocks": 0, "final_out": 0}}


def failure(uid):
    return {"StudyInstanceUID": uid, "status": "data_failure", "route": "fluid_sensitive_fallback", "confirmed_PD": False,
            "feature_forward_calls": 0, "fallback_features": False}


class FluidFeatureTransport(unittest.TestCase):
    def test_merge_only_success_keeps_parent_bytes_and_unrecovered_reason(self):
        parent = {"a": [-0.0, *[i / 64 for i in range(255)]]}
        failures = {"b": {"reason": "parent unknown b"}, "c": {"reason": "parent unknown c"}}
        original = struct.pack("<256f", *parent["a"])
        features, missing, strata = C.merge_child_records(parent, failures, {"a": "PD_nonFS"}, ["a", "b", "c"], [success("b"), failure("c")], ["b", "c"])
        self.assertEqual(struct.pack("<256f", *features["a"]), original)
        self.assertEqual(missing, {"c": failures["c"]})
        self.assertEqual(strata, {"a": "PD_nonFS", "b": "fluid_sensitive_FS"})
        self.assertEqual(set(parent), {"a"})
        self.assertEqual(set(failures), {"b", "c"})

    def test_unknown_duplicate_overwrite_and_fake_failure_rejected(self):
        parent, failures = {"a": [0.0] * 256}, {"b": {"reason": "unknown"}, "c": {"reason": "unknown"}}
        cases = [([success("a")], ["a"]), ([success("unknown")], ["unknown"]),
                 ([success("b"), success("b")], ["b", "b"]),
                 ([{**failure("b"), "pooled_float32_values": [0.] * 256}], ["b"]), ([], ["b"])]
        for records, uids in cases:
            with self.assertRaises(ValueError):
                C.merge_child_records(parent, failures, {"a": "PD_nonFS"}, ["a", "b", "c"], records, uids)

    def test_new_route_validates_numeric_record_without_PD_name_gate(self):
        prior = json.loads((ROOT / "artifacts/qa/odm1/20261017T034429492163Z/files/ORTHODIFF_MRI_PILOT.json").read_bytes())
        row = copy.deepcopy(next(r for r in prior["records"] if r["status"] == "success"))
        detail = row["metadata"]
        detail.update(route="fluid_sensitive_fallback", confirmed_PD=False, stratum="fluid_sensitive_FS",
                      Anatomical_Plane="Sagittal", retry_another_series_after_chosen_failure=False)
        detail["SeriesDescription"] = "synthetic unknown name; not confirmed PD"
        request = {"StudyInstanceUID": row["StudyInstanceUID"], "selected_series": {"SeriesInstanceUID": detail["SeriesInstanceUID"], "Fat_Suppression": True,
                    "metadata": {key: detail[key] for key in ("Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression")}}}
        validator.validate_record(row, request, prior["environment"]["visible_GPUs"])
        for field in ("noise_seed", "confirmed_PD", "pooled_sha"):
            bad = copy.deepcopy(row)
            if field == "noise_seed":
                bad[field] += 1
            elif field == "confirmed_PD":
                bad["metadata"][field] = True
            else:
                bad["pooled"]["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                validator.validate_record(bad, request, prior["environment"]["visible_GPUs"])

    def test_missing_actual_child_never_qualifies_or_seals(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            false_audit = temp / "child.json"
            false_audit.write_text('{"schema_version":"orthodiff_fluid1080_actual_qualification_v1","passed":false,"child_features_qualified":false}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Actual complete"):
                C.qualified_child(false_audit, C.pin(false_audit)["sha256"], {"bytes": 1993, "sha256": C.PARENT_BUNDLE_SHA})
            obs = temp / "obs"
            obs.mkdir()
            (obs / "status.json").write_text('{"status":"RUNNING"}', encoding="utf-8")
            output = temp / "qualification.json"
            command = [sys.executable, str(ROOT / "scripts/validate_orthodiffusion_fluid_fallback.py"),
                "--observation", str(obs), "--pack", str(ROOT / "artifacts/kaggle/orthodiff-sag-fluid1080-20261017-v3"),
                "--launch", str(ROOT / "artifacts/research/20261017-orthodiff-fluid-execution-v1/launch-v3"),
                "--parent-bundle", str(ROOT / "artifacts/research/20261017-orthodiff-weak-head-preparation-v1/qualified-feature-bundle.json"),
                "--output", str(output)]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            value = json.loads(output.read_bytes())
            self.assertFalse(value["passed"])
            self.assertFalse(value["child_features_qualified"])
            self.assertIn("COMPLETE", value["error"])


if __name__ == "__main__":
    unittest.main()
