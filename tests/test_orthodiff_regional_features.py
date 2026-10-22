"""Limited artificial checks for regional transport; no MRI/model imports."""
import copy
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rsna_knee import orthodiff_regional_features as R  # noqa: E402


def info(values):
    raw = struct.pack("<" + str(len(values)) + "f", *values)
    return {"shape": [1, len(values)], "dtype": "torch.float32", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def success():
    values = [-0.0, *([0.5] * 255)]
    return {"StudyInstanceUID": "fake-one", "status": "success", "metadata": {"SeriesInstanceUID": "fake-series", "stratum": "PD_nonFS"},
            "noise_seed": 17, "normalized_input": {"sha256": "a" * 64}, "normalization_min_max_float32": [0., 1.],
            "noisy_input": {"sha256": "b" * 64}, "mid_2": {"sha256": "c" * 64}, "pooled": info(values), "pooled_float32_values": values,
            "forward_counts": {"denoiser": 1, "decoder_blocks": 0, "final_out": 0}, "all_RNG_restored": True,
            "data_seconds": 1., "feature_seconds": 2.}


def regional(parent):
    row = copy.deepcopy(parent)
    values = [float(i % 8) / 8 for i in range(1024)]
    row.update(regional_2x2=info(values), regional_2x2_float32_values=values, regional_pooling_seconds=.1,
               parent_record_line_sha256=hashlib.sha256(R.line_bytes(parent)).hexdigest(), parent_scientific_parity_passed=True)
    return row


class RegionalTransport(unittest.TestCase):
    def test_parent_science_and_global_signed_zero_are_preserved(self):
        old = success()
        row = regional(old)
        row["data_seconds"] = 99.
        R.compare_parent(row, old, R.line_bytes(old))
        for key, value in (("noise_seed", 18), ("normalized_input", {"sha256": "d" * 64}), ("parent_record_line_sha256", "0" * 64)):
            bad = copy.deepcopy(row)
            bad[key] = value
            with self.assertRaises(ValueError):
                R.compare_parent(bad, old, R.line_bytes(old))
        bad = copy.deepcopy(row)
        bad["pooled_float32_values"][0] = 0.0
        bad["pooled"] = info(bad["pooled_float32_values"])
        with self.assertRaises(ValueError):
            R.compare_parent(bad, old, R.line_bytes(old))

    def test_regional_1024_fp32_shape_hash_and_finite(self):
        row = regional(success())
        self.assertEqual(len(R.vector(row)), 1024)
        for values in ([0.] * 256, [float("nan")] * 1024, [True] * 1024, [0.1] * 1024):
            bad = copy.deepcopy(row)
            bad["regional_2x2_float32_values"] = values
            with self.assertRaises(ValueError):
                R.vector(bad)
        row["regional_2x2"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            R.vector(row)

    def test_failure_is_byte_exact_featureless_and_no_new_status(self):
        old = {"StudyInstanceUID": "fake-one", "status": "data_failure", "feature_forward_calls": 0,
               "fallback_features": False, "reason_type": "PilotDataError", "reason": "unknown_PD", "data_seconds": 1.}
        R.compare_parent(copy.deepcopy(old), old, R.line_bytes(old))
        for key, value in (("reason", "other"), ("regional_2x2_float32_values", [0.] * 1024), ("feature_forward_calls", 1)):
            row = copy.deepcopy(old)
            row[key] = value
            with self.assertRaises(ValueError):
                R.compare_parent(row, old, R.line_bytes(old))

    def test_noncomplete_actual_wrapper_and_false_audit_cannot_qualify(self):
        launch = ROOT / "artifacts/research/20261017-orthodiff-regional2x2-execution-v1/launch-v1"
        self.assertTrue((launch / "get-saved-raw.json").is_file(), "Root Save/Get evidence required for bounded incomplete-wrapper QA")
        with tempfile.TemporaryDirectory(prefix="od-regional-qa-") as tmp:
            tmp = Path(tmp)
            (tmp / "status.json").write_text('{"status":"RUNNING"}', encoding="utf-8")
            output = tmp / "not-qualified.json"
            call = subprocess.run([sys.executable, str(ROOT / "scripts/validate_orthodiffusion_regional2x2.py"),
                "--observation", str(tmp), "--pack", str(ROOT / "artifacts/kaggle/orthodiff-sag-regional2x2-20261017-v1"),
                "--launch", str(launch), "--parent-bundle", str(ROOT / "artifacts/research/20261017-orthodiff-weak-head-preparation-v1/qualified-feature-bundle.json"),
                "--kernel-id", "139160976", "--output", str(output)], capture_output=True, text=True, timeout=60)
            self.assertEqual(call.returncode, 1, call.stderr)
            value = json.loads(output.read_bytes())
            self.assertFalse(value["passed"])
            self.assertFalse(value["head_features_qualified"])
            self.assertIn("Official regional COMPLETE required", value["error"])
            with self.assertRaises(ValueError):
                R.audit_evidence(output, R.pin(output)["sha256"], tmp / "missing-parent.json")


if __name__ == "__main__":
    unittest.main()
