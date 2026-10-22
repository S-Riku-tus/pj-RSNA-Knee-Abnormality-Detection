"""Small artificial CSV-only checks; no real E12/REG scores or label values."""
import copy
import csv
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/diagnose_f009_reg50_fold0.py"
spec = importlib.util.spec_from_file_location("REG50_secondary_tested", SCRIPT)
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


def primary_fixture(policy):
    completed, inputs = {}, {}
    for label in ("control", "candidate"):
        pred = {k: policy["fixed_control_E12"][k] for k in ("bytes", "sha256")} if label == "control" else {"bytes": 123, "sha256": "a" * 64}
        completed[label] = {"run": {"sha256": "run" + label}, "index": {"sha256": "index" + label},
                            "E12_prediction": pred, "E12_checkpoint_body_stream_hash": {"sha256": "checkpoint" + label}}
        inputs[label] = {"run_sha256": "run" + label, "epoch_index_sha256": "index" + label,
                         "prediction_sha256": pred["sha256"], "checkpoint_sha256_reference_only_not_loaded": "checkpoint" + label}
    result_pin = {"bytes": 333, "sha256": "b" * 64}
    result = {"status": "F009_E12_WEAK_COMPARISON_ONLY_POLICY_V2", "candidate_epoch": 12, "control_epoch": 12,
              "policy_sha256": policy["parent_policy"]["sha256"],
              "ordered_UID_sha256": policy["original_v2_sorted_UID_json_sha256"], "fold1_automatic_start": False,
              "root_review_still_required": True, "fold1_gate_conditions_met": False, "inputs": inputs}
    guard = {"schema_version": "F009_fixed_E12_guarded_evaluation_receipt_v1", "complete": True,
             "original_evaluate_calls": 1, "original_policy_thresholds_CI_BCE_output_modified": False,
             "original_result_pin": result_pin, "wrapper_pin": {k: policy["guard_wrapper"][k] for k in ("bytes", "sha256")},
             "fold1_gate_conditions_met": False, "result_status": result["status"],
             "fold1_automatic_start": False, "root_review_still_required": True,
             "guards": {"schema_version": "F009_fixed_E12_input_guard_v1", "complete": True,
                        "Gold_body_opened_or_hashed": False, "model_deserialized": False,
                        "target_projection_or_statistics_before_legacy_evaluate": False,
                        "completed_E12_control_candidate": completed}}
    return guard, result, {"sha256": "c" * 64}, result_pin


class PoisonTokens(list):
    def __init__(self, values, forbidden):
        super().__init__(values)
        self.forbidden = set(forbidden)

    def __getitem__(self, key):
        if key in self.forbidden:
            raise AssertionError("Otherfold target/Report read")
        return super().__getitem__(key)


class REG50(unittest.TestCase):
    def test_FP32_add_then_divide_two_and_CSV_roundtrip(self):
        pairs = [(0.1, 0.3), (0., 1.), (.333333333, .99999999), (2 ** -149, 2 ** -149)]
        for a, b in pairs:
            qa, qb = D.f32(a), D.f32(b)
            exact_sum = Fraction.from_float(qa) + Fraction.from_float(qb)
            rounded_sum = struct.unpack("<f", struct.pack("<f", float(exact_sum)))[0]
            expected = struct.unpack("<f", struct.pack("<f", float(Fraction.from_float(rounded_sum) / 2)))[0]
            self.assertEqual(struct.pack("<f", D.mean2_fp32(a, b)), struct.pack("<f", expected))
        self.assertEqual(D.mean2_fp32(2 ** -149, 2 ** -149), 2 ** -149)
        self.assertEqual(D.f32(D.f32(2 ** -149 * .5) + D.f32(2 ** -149 * .5)), 0.)
        for bad in (float("nan"), float("inf"), -.01, 1.01):
            with self.assertRaises(ValueError):
                D.mean2_fp32(bad, .5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mix.csv"
            values = {"fake": {D.ID: "fake", **{t: D.mean2_fp32(.1, .3) for t in D.TARGETS}}}
            D.write_scores(path, values)
            with path.open(newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], [D.ID, *D.TARGETS])
            self.assertTrue(all(struct.pack("<f", float(x)) == struct.pack("<f", values["fake"][D.TARGETS[0]]) for x in rows[1][1:]))

    def test_completed_primary_guard_and_same_REG_metadata_only(self):
        policy = D.read(D.HERE / "policy.json")
        guard, result, gp, rp = primary_fixture(policy)
        before = copy.deepcopy(result)
        D.validate_primary(guard, result, gp, rp, policy)  # A negative primary is allowed for report-only diagnostic.
        self.assertEqual(result, before)
        for mutate in (lambda g, r: g.update(complete=False),
                       lambda g, r: g["original_result_pin"].update(sha256="wrong"),
                       lambda g, r: r["inputs"]["candidate"].update(prediction_sha256="wrong"),
                       lambda g, r: r.update(candidate_epoch=10)):
            g, r = copy.deepcopy(guard), copy.deepcopy(result)
            mutate(g, r)
            with self.assertRaises(ValueError):
                D.validate_primary(g, r, gp, rp, policy)
        fixed = policy["fixed_REG0"]
        regional = {"status": "complete", "fold": 0, "seed": 20261002, "targets": list(D.TARGETS),
                    "gold_used_for_training_or_selection": False, "gold_evaluated": False,
                    "selected": {"epoch": 8, "prediction_sha256": fixed["prediction"]["sha256"],
                                 "checkpoint_sha256": fixed["checkpoint_SHA_reference_only_not_loaded"]}}
        D.validate_REG0(regional, fixed)
        regional["fold"] = 1
        with self.assertRaises(ValueError):
            D.validate_REG0(regional, fixed)

    def test_incomplete_primary_refused_before_predictions_targets_or_paired(self):
        policy = D.read(D.HERE / "policy.json")
        guard, result, _, _ = primary_fixture(policy)
        guard["complete"] = False
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gpath, rpath = root / "guard.json", root / "primary.json"
            gpath.write_text(json.dumps(guard), encoding="utf-8")
            rpath.write_text(json.dumps(result), encoding="utf-8")
            args = SimpleNamespace(expected_cli_sha256=D.pin(SCRIPT)["sha256"], guard_receipt=gpath,
                primary_result=rpath, expected_guard_receipt_sha256=D.pin(gpath)["sha256"],
                expected_primary_result_sha256=D.pin(rpath)["sha256"], output_dir=root / "out")
            with (patch.object(D, "project_fold0", side_effect=AssertionError("Must not project")) as projector,
                 patch.object(D, "predictions", side_effect=AssertionError("Must not read predictions")) as predicted,
                 patch.object(D, "load_original_paired", side_effect=AssertionError("Must not calculate"))):
                with self.assertRaises(ValueError):
                    D.run(args)
                projector.assert_not_called()
                predicted.assert_not_called()
            self.assertFalse(args.output_dir.exists())

    def test_otherfold_poison_and_original_paired_1000_report_only(self):
        header = [D.ID, *D.TARGETS, "group_id", "fold", "source", "Report"]
        metadata, tokens = [], []
        for i in range(5):
            fold = "0" if i < 4 else "1"
            row = {D.ID: "u%d" % i, "group_id": "g%d" % (i // 2), "fold": fold, "source": "report_weak"}
            metadata.append(row)
            cells = [row[D.ID], *([str(i % 2)] * 12), row["group_id"], fold, "report_weak", "ignore"]
            if i == 0:
                cells[1] = ""
            tokens.append(PoisonTokens(cells, [len(header) - 1] if fold == "0" else [*range(1, 13), len(header) - 1]))
        truth = D.project_fold0(header, tokens, metadata)
        self.assertEqual(len(truth), 4)
        self.assertEqual(truth["u0"]["ACL"], "")
        policy = D.read(D.HERE / "policy.json")
        paired = D.load_original_paired(policy)
        control = {u: {t: .2 + .2 * i for t in D.TARGETS} for i, u in enumerate(truth)}
        candidate = {u: dict(row) for u, row in control.items()}
        REG0 = {u: {t: .3 for t in D.TARGETS} for u in truth}
        left, right, values = D.compare_secondary(truth, control, candidate, REG0, paired)
        self.assertEqual(left, right)
        self.assertEqual(values["fixed10_delta"], 0.)
        self.assertEqual(values["per_target"]["ACL"]["support"]["missing"], 1)
        self.assertEqual(values["bootstrap"]["repeats"], 1000)
        self.assertEqual(values["bootstrap"]["seed"], 20261012)
        self.assertFalse(values["bootstrap"]["CI_used_for_gate"])
        with self.assertRaises(ValueError):
            D.compare_secondary(truth, {k: v for k, v in control.items() if k != "u0"}, candidate, REG0, paired)
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
