"""Synthetic standard-library checks; no real run diagnostic or Torch import."""
from __future__ import annotations

import ast
import copy
import csv
import hashlib
import importlib.util
import io
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("f014_csv_audit_test_target", ROOT / "scripts/check_f014_fixed12_weak.py")
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def encoded_csv(header, rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode()


def history_fixture():
    identity = {"targets": list(C.TARGETS), "gold_used_for_training_or_selection": False, "gold_evaluated": False,
                "patient_independence_verified": False, "fold": 0, "source_sha256": {"historic.py": "a" * 64}}
    run = {**identity, "status": "complete", "train_studies": 17, "valid_studies": 4, "elapsed_seconds": 121., "history": []}
    entries = []
    for epoch in range(1, 13):
        multiplier = 1. if epoch <= 6 else .1 + .9 * (1 + math.cos(math.pi * (epoch - 6) / 6)) / 2
        item = {"epoch": epoch, "optimizer_steps": 2, "optimizer_step_opportunities": 2, "amp_skipped_steps": 0,
                "elapsed_seconds": epoch * 10., "train_bce": .5, "weak_bce": .5, "gpu_peak_reserved_bytes": 1024,
                "learning_rate_multiplier": multiplier, "learning_rates": {"head": .0003 * multiplier, "last2": .00001 * multiplier}}
        run["history"].append(item)
        entries.append({**item, "checkpoint": f"epoch_{epoch:03d}.pt", "predictions_csv": f"epoch_{epoch:03d}_weak.csv"})
    index = {"schema_version": "native392_epoch_suffix_head_retention_v1", "checkpoint_schema": "synthetic_schema",
             "fold": 0, "targets": list(C.TARGETS), "source_sha256": identity["source_sha256"], "selection_data": "weak_validation",
             "selection_rule": "minimum observed weak BCE; earliest exact tie", "resume_supported": False, "checkpoints": entries}
    return run, index, identity, {"train": 17, "valid": 4}


class CSVContract(unittest.TestCase):
    def test_prediction_columns_ids_finiteness_are_strict(self):
        raw = encoded_csv(C.COLUMNS, [["u1", *([".2"] * 12)], ["u2", *([".8"] * 12)]])
        self.assertEqual(len(C.prediction_rows(raw, ["u1", "u2"])), 2)
        variants = [encoded_csv(C.COLUMNS[::-1], [[".2"] * 12 + ["u1"]]),
                    encoded_csv(C.COLUMNS, [["u1", *([".2"] * 12)], ["u1", *([".8"] * 12)]]),
                    encoded_csv(C.COLUMNS, [["u2", *([".2"] * 12)], ["u1", *([".8"] * 12)]])]
        variants.extend(encoded_csv(C.COLUMNS, [["u1", value, *([".2"] * 11)], ["u2", *([".8"] * 12)]])
                        for value in ("nan", "inf", "", "-0.01", "1.01"))
        for bad in variants:
            with self.subTest(raw=bad[:50]), self.assertRaises(ValueError):
                C.prediction_rows(bad, ["u1", "u2"])

    def test_metadata_projection_ignores_gold_numeric_cells(self):
        raw = encoded_csv([C.ID, "ACL", "group_id"], [["gold1", "not-a-number-never-converted", "g1"]])
        self.assertEqual(C.metadata_rows(raw, (C.ID, "group_id")), [{C.ID: "gold1", "group_id": "g1"}])

    def test_group_closure_and_gold_exclusion(self):
        weak = [{C.ID: "a", "source": "report_weak", "group_id": "g", "fold": "0"},
                {C.ID: "b", "source": "report_weak", "group_id": "h", "fold": "1"}]
        gold = [{C.ID: "z", "group_id": "gold"}]
        C.validate_groups(weak, gold)
        bad = copy.deepcopy(weak)
        bad[1]["group_id"] = "g"
        with self.assertRaises(ValueError):
            C.validate_groups(bad, gold)
        for gold_row in ({C.ID: "a", "group_id": "gold"}, {C.ID: "z", "group_id": "h"}):
            with self.assertRaises(ValueError):
                C.validate_groups(weak, [gold_row])


class MetricsContract(unittest.TestCase):
    def test_average_tie_auc_and_undefined_classes(self):
        self.assertEqual(C.binary_auc([1, 0, 1, 0], [.5, .5, .9, .1]), .875)
        self.assertEqual(C.binary_auc([0, 1], [.4, .4]), .5)
        self.assertEqual(C.binary_auc([1, 0], [.1, .9]), 0.)
        self.assertIsNone(C.binary_auc([1, 1], [.1, .9]))
        with self.assertRaises(ValueError):
            C.binary_auc([0, .2], [.1, .9])

    def test_missing_and_soft_labels_excluded_from_auc_not_imputed(self):
        truth, predicted = [], []
        for uid, y, p in (("a", "0", ".1"), ("b", "1", ".9"), ("c", "", ".99"), ("d", ".2", ".4")):
            truth.append({C.ID: uid, **{t: y for t in C.TARGETS}})
            predicted.append({C.ID: uid, **{t: p for t in C.TARGETS}})
        result = C.evaluate_weak(truth, predicted)
        self.assertEqual(result["observed_cells"], 36)
        self.assertEqual(result["per_target"]["ACL"], {"auc": 1., "binary_observed": 2, "positive": 1.,
                                                    "soft_excluded": 1, "missing_excluded": 1})
        expected = (-math.log(.9) * 2 - .2 * math.log(.4) - .8 * math.log(.6)) / 3
        self.assertAlmostEqual(result["CSV_reconstructed_BCE"], expected, places=8)

    def test_missing_teacher_tokens_match_shared_contract_predictions_stay_finite(self):
        for value in ("", "nan", "NaN", " NA ", "null"):
            self.assertIsNone(C.probability(value, missing=True))
            with self.assertRaises(ValueError):
                C.probability(value)

    def test_fixed10_is_the_only_primary_no_rescue_or_denominator_change(self):
        control = {"per_target": {t: {"auc": .7} for t in C.TARGETS}}
        candidate = {"per_target": {t: {"auc": .69 if t in C.FIXED10 else 1.} for t in C.TARGETS}}
        self.assertFalse(C.primary_delta(control, candidate)["strictly_positive"])
        self.assertFalse(C.primary_delta(control, copy.deepcopy(control))["strictly_positive"])
        candidate["per_target"]["ACL"]["auc"] = None
        with self.assertRaises(ValueError):
            C.primary_delta(control, candidate)

    def test_writer_metrics_cannot_be_forged(self):
        truth = [{C.ID: "a", **{t: "0" for t in C.TARGETS}}, {C.ID: "b", **{t: "1" for t in C.TARGETS}}]
        predicted = [{C.ID: "a", **{t: ".1" for t in C.TARGETS}}, {C.ID: "b", **{t: ".9" for t in C.TARGETS}}]
        calculated = C.evaluate_weak(truth, predicted)
        metrics = {k: copy.deepcopy(calculated[k]) for k in ("per_target", "macro_auc_12", "available_class_macro_auc", "defined_classes")}
        item = {"metrics": metrics, "weak_macro_auc_12": 1., "weak_bce": calculated["CSV_reconstructed_BCE"]}
        tolerances = {"AUC_abs": 1e-12, "BCE_abs": 2e-6}
        C.verify_writer(calculated, item, tolerances)
        item["metrics"]["per_target"]["ACL"]["positive"] = 2
        with self.assertRaises(ValueError):
            C.verify_writer(calculated, item, tolerances)

    def test_teacher_mask_counts_match_fold_audit(self):
        weak = [{C.ID: "a", "fold": "0", **{t: "1" for t in C.TARGETS}},
                {C.ID: "b", "fold": "1", **{t: "" for t in C.TARGETS}}]
        audit = {"per_fold": {"0": {"valid": {t: {"observed": 1, "missing": 0, "exact_zero": 0, "exact_one": 1, "soft": 0} for t in C.TARGETS},
                                   "train": {t: {"observed": 0, "missing": 1, "exact_zero": 0, "exact_one": 0, "soft": 0} for t in C.TARGETS}}}}
        C.verify_teacher_coverage(weak, 0, audit)
        weak[1]["ACL"] = "0"
        with self.assertRaises(ValueError):
            C.verify_teacher_coverage(weak, 0, audit)


class FixedArtifactContract(unittest.TestCase):
    def test_complete_fixed12_and_actual_amp_updates(self):
        run, index, identity, counts = history_fixture()
        self.assertEqual(C.validate_history(run, index, identity, counts, "synthetic_schema"),
                         {"actual_updates": 24, "opportunities": 24, "AMP_skips": 0})
        # A legitimate skip is disclosed, never counted as an actual update.
        for row in (run["history"][9], index["checkpoints"][9]):
            row.update(optimizer_steps=1, amp_skipped_steps=1)
        self.assertEqual(C.validate_history(run, index, identity, counts, "synthetic_schema")["actual_updates"], 23)

    def test_partial_reordered_alias_and_zero_update_epochs_are_rejected(self):
        mutations = [lambda r, i: r.update(status="running"), lambda r, i: r["history"].pop(),
                     lambda r, i: i["checkpoints"].reverse(),
                     lambda r, i: i["checkpoints"][11].update(checkpoint="best.pt"),
                     lambda r, i: i["checkpoints"][11].update(predictions_csv="weak_valid_predictions.csv"),
                     lambda r, i: r["history"][0].update(epoch=True),
                     lambda r, i: r["history"][0].update(optimizer_steps=0),
                     lambda r, i: r.update(gold_used_for_training_or_selection=True),
                     lambda r, i: r["source_sha256"].update(extra="forged")]
        for change in mutations:
            run, index, identity, counts = history_fixture()
            identity = copy.deepcopy(identity)
            change(run, index)
            with self.subTest(change=change), self.assertRaises(ValueError):
                C.validate_history(run, index, identity, counts, "synthetic_schema")

    def test_checkpoint_is_hash_only_and_content_reader_rejects_pt(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "epoch_012.pt"
            body = b"not-a-deserializable-checkpoint"
            path.write_bytes(body)
            snapshot = C.Snapshot(temporary)
            self.assertEqual(snapshot.pin(path.name), {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()})
            with self.assertRaises(ValueError):
                snapshot.raw(path.name)
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                snapshot.recheck()

    def test_full_run_retained_byte_correspondence_without_tensor_reads(self):
        run, index, identity, counts = history_fixture()
        identity = {**identity, "config": {"synthetic": True}, "environment": {"torch": "recorded-only"}}
        run.update(identity)
        fold_audit, manifest, gate = {"synthetic_fold_audit": True}, {"synthetic_manifest": True}, {"synthetic_gate": True}
        hashes = {"weak.csv": "teacher-byte-pin"}
        run["audit"] = {"manifest_sha256": hashes, "fold_audit": fold_audit, "pixel_index_sha256": "pixel-pin",
                        "resource_gate_sha256": "gate-pin", "resource_gate": gate, "manifest": manifest}
        for entry in index["checkpoints"]:
            entry.update(checkpoint_bytes=3, checkpoint_sha256="checkpoint-pin", prediction_sha256="prediction-pin")
        policy = {"pixel_index": {"sha256": "pixel-pin"}, "candidate": {
            "directories": {"0": "synthetic-run"}, "run_identity": identity, "checkpoint_schema": "synthetic_schema",
            "counts": {"0": counts}, "resource_gate": {"sha256": "gate-pin", "object_sha256": C.object_sha(gate)},
            "shared_manifest_object_sha256": C.object_sha(manifest), "environment_object_sha256": C.object_sha(run["environment"])}}

        class FakeSnapshot:
            def __init__(self):
                self.bad_prediction = False
                self.opened = []

            def json(self, name):
                self.opened.append(name)
                return {"run.json": run, "epoch_checkpoint_index.json": index, "config.json": identity["config"]}[Path(name).name]

            def pin(self, name, expected=None):
                self.opened.append(name)
                value = ({"bytes": 3, "sha256": "checkpoint-pin"} if name.endswith(".pt") else
                         {"bytes": 3, "sha256": "wrong" if self.bad_prediction else "prediction-pin"})
                if expected is not None and expected != value:
                    raise ValueError("synthetic bytes differ")
                return value

        snapshot = FakeSnapshot()
        result = C.verify_run(snapshot, policy, 0, "candidate", fold_audit, hashes)
        self.assertEqual(result["checkpoint"], "checkpoint-pin")
        self.assertEqual(result["source_scope"], "current_exact_source16")
        self.assertFalse(any("best.pt" in p or "weak_valid_predictions" in p for p in snapshot.opened))
        snapshot.bad_prediction = True
        with self.assertRaises(ValueError):
            C.verify_run(snapshot, policy, 0, "candidate", fold_audit, hashes)

    def test_history_identity_does_not_relabel_historical_source_as_current(self):
        run, index, identity, counts = history_fixture()
        C.validate_history(run, index, identity, counts, "synthetic_schema")
        current = copy.deepcopy(identity)
        current["source_sha256"] = {"new.py": "b" * 64}
        with self.assertRaises(ValueError):
            C.validate_history(run, index, current, counts, "synthetic_schema")

    def test_actual_prospective_policy_pin_and_no_alternate_policy(self):
        # Only the inert policy is read; no real run/teacher/prediction is opened.
        policy = C.load_policy(C.Snapshot(ROOT))
        self.assertEqual(policy["fixed_epoch"], 12)
        original = (ROOT / C.POLICY).read_bytes()

        class FakeSnapshot:
            def raw(self, _):
                return original + b" "

        with self.assertRaises(ValueError):
            C.load_policy(FakeSnapshot())

    def test_json_rejects_duplicate_and_nonfinite_tokens(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'):
            with self.assertRaises(ValueError):
                C.strict_json(raw)

    def test_cli_has_no_tensor_gpu_or_arbitrary_endpoint_imports(self):
        tree = ast.parse((ROOT / "scripts/check_f014_fixed12_weak.py").read_text(encoding="utf-8"))
        modules = {name.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for name in node.names}
        modules |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        self.assertFalse(modules & {"torch", "numpy", "kaggle", "requests", "rsna_knee", "pickle"})
        flags = [node.args[0].value for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"]
        self.assertEqual(set(flags), {"--fold", "--output"})
        with tempfile.TemporaryDirectory() as temporary, patch.object(C, "ROOT", Path(temporary)):
            existing = Path(temporary) / "artifacts/research/already-exists"
            existing.mkdir(parents=True)
            with self.assertRaises(ValueError):
                C.audit(0, existing)


if __name__ == "__main__":
    unittest.main()
