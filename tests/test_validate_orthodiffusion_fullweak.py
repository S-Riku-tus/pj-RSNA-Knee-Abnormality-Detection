"""Three bounded receipt-tampering cases; stdlib only, no MRI/model/fit."""
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_orthodiffusion_fullweak.py"
spec = importlib.util.spec_from_file_location("ortho_full_qualification_test", SCRIPT)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)
PILOT = ROOT / "artifacts/qa/odm1/20261017T034429492163Z/files/ORTHODIFF_MRI_PILOT.json"


class QualificationGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior = json.loads(PILOT.read_bytes())

    def test_terminal_api_COMPLETE_cannot_qualify_incomplete_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            obs = temp / "observation"
            files = obs / "files"
            records = []
            for suffix in ("SUMMARY.json", "STATUS.json", "RECORDS.jsonl"):
                name = validator.STEM + suffix
                path = files / name
                path.parent.mkdir(parents=True, exist_ok=True)
                body = b'{"complete":false}\n' if suffix.endswith(".json") else b'{}\n'
                path.write_bytes(body)
                records.append({"file": name, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(), "http_status": 200})
            (obs / "status.json").write_text('{"status":"COMPLETE"}', encoding="utf-8")
            (obs / "receipts.json").write_text(json.dumps(records), encoding="utf-8")
            output = temp / "qualification.json"
            result = subprocess.run([sys.executable, str(SCRIPT), "--observation", str(obs),
                "--pack", str(ROOT / "artifacts/kaggle/orthodiff-sag-fullweak4207-20261017-v2"),
                "--launch", str(ROOT / "artifacts/research/20261017-orthodiff-fullweak-execution-v1/launch-v2"),
                "--project", str(ROOT), "--output", str(output)], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            value = json.loads(output.read_bytes())
            self.assertFalse(value["passed"])
            self.assertFalse(value["head_features_qualified"])
            self.assertIn("Incomplete/fatal", value["error"])

    def test_finite_changed_FP32_or_decoder_forward_rejects_success(self):
        good = next(row for row in self.prior["records"] if row["status"] == "success")
        detail = good["metadata"]
        official = {row["SeriesInstanceUID"]: {key: row[key] for key in ("Fat_Suppression", "Fluid_Sensitive")}
                    for row in [detail, *detail["rejected_candidates"]]}
        validator.validate_success(good, official, self.prior["environment"]["visible_GPUs"])
        altered = copy.deepcopy(good)
        altered["pooled_float32_values"][0] = 0.0
        with self.assertRaisesRegex(ValueError, "FP32 representation/SHA"):
            validator.validate_success(altered, official, self.prior["environment"]["visible_GPUs"])
        altered = copy.deepcopy(good)
        altered["forward_counts"]["decoder_blocks"] = 1
        with self.assertRaisesRegex(ValueError, "feature-only/RNG"):
            validator.validate_success(altered, official, self.prior["environment"]["visible_GPUs"])

    def test_fabricated_failure_and_prior_scientific_mismatch_reject(self):
        failed = copy.deepcopy(next(row for row in self.prior["records"] if row["status"] == "data_failure"))
        failed["pooled_float32_values"] = [0.0] * 256
        with self.assertRaisesRegex(ValueError, "fabrication"):
            validator.validate_failure(failed, {})
        previous = next(row for row in self.prior["records"] if row["status"] == "success")
        changed = copy.deepcopy(previous)
        changed["normalized_input"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Prior32"):
            validator.compare_prior(changed, previous)
        changed = copy.deepcopy(previous)
        changed["StudyInstanceUID"] += "0"
        with self.assertRaisesRegex(ValueError, "Prior32"):
            validator.compare_prior(changed, previous)


if __name__ == "__main__":
    unittest.main()
