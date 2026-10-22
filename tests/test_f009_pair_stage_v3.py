"""Artificial metadata/non-model bytes only; no real CPU/model/MRI/GPU work."""
import ast
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

from rsna_knee import f009_cpu_proof_v2 as P
from rsna_knee import f009_pair_stage_v3 as S

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("amended_sealer_qa_metadata_fixture", ROOT / "tests/test_f009_cpu_proof_v2.py")
F = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(F)


def save(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, allow_nan=False) + "\n").encode("utf-8"))
    return {"path": relative, **S.pin(path)}


def fixture(root):
    producer_spec, bindings = F.fixture(root)
    for relative in ["src/rsna_knee/f009_cpu_proof_v2.py", "scripts/check_f009_E12_cpu_v2.py",
                     "src/rsna_knee/f009_cpu_proof.py", P.SNAPSHOT,
                     "src/rsna_knee/f009_pair_stage_v2.py", "src/rsna_knee/f009_pair_stage_v3.py",
                     "scripts/seal_f009_E12_pair_v3.py", P.AMENDED_WRAPPER]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, path)
    with patch.object(P, "AMENDED_BINDINGS", bindings):
        gate = P.validate_amended_joint_metadata(root, producer_spec["qualification"])
    spec = {"schema_version": S.SPEC_SCHEMA, "qualification": producer_spec["qualification"], "folds": {}}
    for fold in (0, 1):
        context = gate["candidate_contexts"][fold]
        cpu = S.read(root / ("CPU/fold" + str(fold) + ".json"))
        cpu.update(schema_version=S.CPU_SCHEMA, complete=True, producer_sha256=S.CPU_PRODUCER_SHA,
            producer_CLI_sha256=S.CPU_PRODUCER_CLI_SHA, run_pin=context["run_pin"],
            retained_index_pin=context["retained_index_pin"], checkpoint_pin=context["checkpoint_pin"],
            qualification_schema=S.QUALIFICATION_SCHEMA, qualification_pins=spec["qualification"],
            qualification_binding_pins=gate["binding_pins"], eval_gamma_RNG_unchanged=True,
            strict_apply_identity_and_tensor_bytes_exact=True, runtime_or_adoption_qualified=False,
            checkpoint_loader={"source_sha256": S.sources.CORE_SHA["native_training_fast16.py"],
                "map_location": "cpu", "weights_only": True, "allow_artificial": False})
        run = context["run_path"].relative_to(root).as_posix()
        index = context["index_path"].relative_to(root).as_posix()
        spec["folds"][str(fold)] = {
            "run": {"path": run, **S.pin(root / run)}, "index": {"path": index, **S.pin(root / index)},
            "CPU": save(root, "new-CPU/fold" + str(fold) + ".json", cpu),
            "expected_CPU_producer_sha256": S.CPU_PRODUCER_SHA}
    return spec, bindings


class AmendedPairSealer(unittest.TestCase):
    def test_frozen_registry_and_cpu_tensor_contract_AST_unchanged(self):
        self.assertEqual(S.CPU_PRODUCER_SHA, S.pin(ROOT / "src/rsna_knee/f009_cpu_proof_v2.py")["sha256"])
        self.assertEqual(S.CPU_PRODUCER_CLI_SHA, S.pin(ROOT / "scripts/check_f009_E12_cpu_v2.py")["sha256"])
        before = ast.parse((ROOT / "src/rsna_knee/f009_pair_stage_v2.py").read_bytes())
        after = ast.parse(Path(S.__file__).read_bytes())
        old = next(node for node in before.body if isinstance(node, ast.FunctionDef) and node.name == "validate_cpu")
        new = next(node for node in after.body if isinstance(node, ast.FunctionDef) and node.name == "_validate_cpu_base")
        new.name = old.name
        self.assertEqual(ast.dump(old, include_attributes=False), ast.dump(new, include_attributes=False))

    def test_exact_artificial_pair_old_HOLD_and_all_six_evidence_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec, bindings = fixture(root)
            originals = {key: S.pin(root / value["path"]) for key, value in spec["qualification"].items()}
            with patch.object(P, "AMENDED_BINDINGS", bindings):
                pair = S._seal(root, spec, root / "stage", fixture_only=True)
            self.assertTrue(pair["artificial_fixture_only"])
            self.assertFalse(pair["actual_CLS_stage_constructed"])
            self.assertEqual(pair["old_v1_joint_screen"], "HOLD")
            self.assertFalse(pair["four_model_transport_runtime_qualified"])
            base = S.read(root / S.previous.BASE)
            for key in ("REG:0", "REG:1"):
                self.assertEqual(pair["fixed_REG_references_only"][key]["fingerprint"], base["native"]["model_fingerprints"][key])
            for fold in (0, 1):
                folder = root / "stage" / ("fold" + str(fold))
                manifest = S.read(folder / "bundle.json")
                self.assertFalse(manifest["actual_strict_CPU_accepted"])
                self.assertFalse(S.read(folder / "audit/legacy_guard1.json")["machine_engineering_passed"])
                self.assertFalse(S.read(folder / "audit/legacy_comparison1.json")["joint_candidate_screen_passed"])
                self.assertEqual(manifest["qualification_pins"], spec["qualification"])
                for key, observed in originals.items():
                    self.assertEqual(S.pin(folder / "audit" / (key + ".json")), observed)
                self.assertEqual(S.pin(folder / "audit/CPU_producer_v2.py")["sha256"], S.CPU_PRODUCER_SHA)
                self.assertEqual(S.pin(folder / "audit/amended-policy2.json"), S.pin(root / P._amended_wrapper().POLICY_REL))
                self.assertFalse(any(path.name in {"weak.csv", "gold.csv"} for path in folder.rglob("*")))
            self.assertTrue(S.read(root / "stage/BUILD_COMPLETE.json")["complete"])
            self.assertEqual(originals, {key: S.pin(root / value["path"]) for key, value in spec["qualification"].items()})
            with patch.object(P, "AMENDED_BINDINGS", bindings), self.assertRaisesRegex(ValueError, "Fresh"):
                S._seal(root, spec, root / "stage", fixture_only=True)

    def test_qualification_CPU_dimension_schema_or_cross_pin_tamper_before_checkpoint(self):
        for kind in ("legacy_schema", "missing_six", "CPU_qualification", "CPU_binding", "CPU_schema", "CPU_shape", "CPU_loader", "CPU_registry", "actual_false_comparison"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                spec, bindings = fixture(root)
                bad = copy.deepcopy(spec)
                if kind == "legacy_schema":
                    bad["schema_version"] = "f009_pair_stage_root_pins_v1"
                elif kind == "missing_six":
                    bad["qualification"].pop("legacy_guard1")
                elif kind == "actual_false_comparison":
                    ref = bad["qualification"]["result1"]
                    value = S.read(root / ref["path"])
                    value["actual_comparison"] = False
                    bad["qualification"]["result1"] = save(root, "bad-result.json", value)
                else:
                    item = bad["folds"]["0"]
                    value = S.read(root / item["CPU"]["path"])
                    if kind == "CPU_qualification":
                        value["qualification_pins"]["legacy_guard1"]["sha256"] = "e" * 64
                    elif kind == "CPU_binding":
                        value["qualification_binding_pins"]["policy2"]["sha256"] = "e" * 64
                    elif kind == "CPU_schema":
                        value["schema_version"] = "f009_fixed_E12_CPU_strict_reconstruction_v1"
                    elif kind == "CPU_shape":
                        value["tensor_metadata"]["head"][0]["shape"] = [383]
                    elif kind == "CPU_loader":
                        value["checkpoint_loader"]["weights_only"] = False
                    else:
                        value["producer_sha256"] = "a" * 64
                    item["CPU"] = save(root, "bad-CPU.json", value)
                original_pin = S.pin
                def no_checkpoint(path):
                    self.assertNotEqual(Path(path).suffix, ".pt", "Invalid metadata reached checkpoint body")
                    return original_pin(path)
                with patch.object(P, "AMENDED_BINDINGS", bindings), patch.object(S, "pin", side_effect=no_checkpoint):
                    with self.assertRaises(ValueError):
                        S._seal(root, bad, root / "invalid", fixture_only=True)
                self.assertFalse((root / "invalid").exists())

    def test_artificial_CPU_cannot_be_actual_and_metadata_pt_path_is_refused_early(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec, bindings = fixture(root)
            with patch.object(P, "AMENDED_BINDINGS", bindings):
                base, plan, gate, _registry = S.preflight(root, spec, fixture_only=True)
            item = plan[0]
            with self.assertRaisesRegex(ValueError, "Actual F009 strict CPU producer"):
                S.validate_cpu(item["CPU"], fold=0, run_pin=item["metadata_pins"]["run"], index_pin=item["metadata_pins"]["index"],
                    checkpoint=item["checkpoint_pin"], prefix=base["native"]["model_fingerprints"]["REG:0"]["frozen_prefix_sha256"],
                    fixture_only=False, qualification=spec["qualification"], qualification_binding_pins=gate["binding_pins"])
            with patch.object(S, "pin", side_effect=AssertionError("Do not open the .pt path")), self.assertRaises(ValueError):
                S.bound(root, {"path": "x.pt", "bytes": 1, "sha256": "a" * 64})

    def test_partial_copy_cannot_publish_and_default_CLI_stays_disabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec, bindings = fixture(root)
            original = S.copy_exact
            def fail_second(source, destination, expected):
                if Path(source).suffix == ".pt" and "fold1" in str(source):
                    raise OSError("Artificial interrupted copy")
                return original(source, destination, expected)
            with patch.object(P, "AMENDED_BINDINGS", bindings), patch.object(S, "copy_exact", side_effect=fail_second), self.assertRaises(OSError):
                S._seal(root, spec, root / "partial", fixture_only=True)
            self.assertFalse((root / "partial/BUILD_COMPLETE.json").exists())
            self.assertFalse((root / "partial/pair-sidecar.json").exists())
            self.assertFalse(S.read(root / "partial/BUILD_FAILED.json")["complete"])
        result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/seal_f009_E12_pair_v3.py")],
                                capture_output=True, text=True, check=True)
        self.assertFalse(json.loads(result.stdout)["actual_stage_constructed"])
        self.assertFalse(json.loads(result.stdout)["weights_or_run_opened"])


if __name__ == "__main__":
    unittest.main()
