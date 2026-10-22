"""New consumer binding on artificial bytes; actual CPU/encoder qualification0."""
import copy
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsna_knee import f009_cpu_proof as P
from rsna_knee import f009_pair_stage_v2 as V

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("f009_v2_artificial_stage_fixture", ROOT / "tests/test_f009_pair_stage.py")
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)


def fixture(root):
    value = fixture_module.fixture(root)
    for name in ("src/rsna_knee/f009_cpu_proof.py", "scripts/check_f009_E12_cpu.py"):
        shutil.copyfile(ROOT / name, root / name)
    for fold in (0, 1):
        item = value["folds"][str(fold)]
        receipt = V.read(root / item["CPU"]["path"])
        receipt.update(complete=True, producer_sha256=V.CPU_PRODUCER_SHA,
            producer_CLI_sha256=V.CPU_PRODUCER_CLI_SHA, eval_gamma_RNG_unchanged=True,
            strict_apply_identity_and_tensor_bytes_exact=True,
            checkpoint_loader={"source_sha256": V.sources.CORE_SHA["native_training_fast16.py"],
                               "map_location": "cpu", "weights_only": True, "allow_artificial": False})
        item["CPU"] = fixture_module.save(root, item["CPU"]["path"], receipt)
        item["expected_CPU_producer_sha256"] = V.CPU_PRODUCER_SHA
    return value


class PairStageV2(unittest.TestCase):
    def test_frozen_producer_exact_fixture_copy_and_default_actual0(self):
        self.assertEqual(V.APPROVED_ACTUAL_CPU_PRODUCERS, frozenset({P.sha(P.__file__)}))
        self.assertEqual(V.CPU_PRODUCER_CLI_SHA, P.sha(ROOT / "scripts/check_f009_E12_cpu.py"))
        self.assertEqual(P.sha(ROOT / "src/rsna_knee/f009_pair_stage.py"), P.SNAPSHOT_SHA)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = fixture(root)
            pair = V._seal(root, value, root / "toy-stage-v2", fixture_only=True)
            self.assertFalse(pair["actual_CLS_stage_constructed"])
            self.assertFalse(pair["four_model_transport_runtime_qualified"])
            self.assertFalse(pair["REG_checkpoint_body_read_or_copied"])
            for fold in (0, 1):
                folder = root / "toy-stage-v2" / ("fold" + str(fold))
                self.assertEqual(V.pin(folder / "audit/CPU_producer.py")["sha256"], V.CPU_PRODUCER_SHA)
                self.assertEqual(V.pin(folder / "audit/CPU_producer_CLI.py")["sha256"], V.CPU_PRODUCER_CLI_SHA)
                self.assertFalse(V.read(folder / "bundle.json")["actual_strict_CPU_accepted"])
            self.assertTrue(V.read(root / "toy-stage-v2/BUILD_COMPLETE.json")["complete"])
        run = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/seal_f009_E12_pair_v2.py")],
                             capture_output=True, text=True, check=True)
        self.assertFalse(json.loads(run.stdout)["actual_stage_constructed"])

    def test_no_actual_receipt_or_source_or_flag_tamper_stops_before_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = fixture(root)
            original = V.pin

            def no_checkpoint(path):
                self.assertNotEqual(Path(path).suffix, ".pt", "Actual receipt must pass before checkpoint read")
                return original(path)

            with patch.object(V, "pin", side_effect=no_checkpoint):
                with self.assertRaisesRegex(ValueError, "Actual F009 strict CPU producer"):
                    V._seal(root, value, root / "actual")
                self.assertFalse((root / "actual").exists())
                bad = copy.deepcopy(value)
                receipt = V.read(root / bad["folds"]["0"]["CPU"]["path"])
                receipt["checkpoint_loader"]["weights_only"] = False
                bad["folds"]["0"]["CPU"] = fixture_module.save(root, "CPU/wrong-loader.json", receipt)
                with self.assertRaisesRegex(ValueError, "strict loader/eval evidence"):
                    V._seal(root, bad, root / "bad-flags", fixture_only=True)
                path = root / "src/rsna_knee/f009_cpu_proof.py"
                path.write_bytes(path.read_bytes() + b"\n# ARTIFICIAL_SOURCE_TAMPER\n")
                with self.assertRaisesRegex(ValueError, "Exact reviewed CPU producer"):
                    V._seal(root, value, root / "bad-source", fixture_only=True)


if __name__ == "__main__":
    unittest.main()
