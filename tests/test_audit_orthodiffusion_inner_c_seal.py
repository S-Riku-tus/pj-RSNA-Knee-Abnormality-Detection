"""Small stdlib-only artificial checks; no real run/targets/features read."""
import copy
import csv
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_orthodiffusion_inner_c_seal.py"
spec = importlib.util.spec_from_file_location("innerC_independent_auditor", SCRIPT)
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def fake_data():
    training, outer, features = [], [], {}
    for fold in A.FOLDS:
        for label in (0, 1):
            uid = "f%d-y%d" % (fold, label)
            training.append({A.ID: uid, "group_id": "g" + uid, "fold": str(fold),
                             "source": "report_weak", "targets": [float(label)] * 12})
            features[uid] = [float(fold), float(label), 2.]
    for n in range(3):
        uid = "out%d" % n
        outer.append({A.ID: uid, "group_id": "g" + uid, "fold": "2", "source": "report_weak"})
        features[uid] = [float(n) + .25, .75, 2.]
    return training, outer, features


def fake_selection(training, features):
    support = []
    for fold in A.FOLDS:
        train = [r for r in training if int(r["fold"]) != fold and r[A.ID] in features]
        valid = [r for r in training if int(r["fold"]) == fold and r[A.ID] in features]
        fields = {}
        for i, target in enumerate(A.TARGETS):
            if target in A.FIXED10:
                observed = [r for r in valid if r["targets"][i] is not None]
                fields[target] = {"train_observed": sum(r["targets"][i] is not None for r in train),
                                  "valid_observed": len(observed), "valid_observed_UID_hash": A.uid_hash(observed)}
        support.append({"inner_fold": fold, "train_count": len(train), "valid_count": len(valid),
                        "train_UID_hash": A.uid_hash(train), "valid_UID_hash": A.uid_hash(valid),
                        "fixed10_support": fields})
    table = []
    for C, score in zip(A.GRID, (.6, .8, .8, .7)):
        table.append({"C": C, "eligible": True, "score": score, "failure_reason": None,
                      "inner": [{"fold": fold, "AUC": {target: score for target in A.FIXED10}, "macro10": score,
                                 "numeric_roundtrip": {"JSON_model_roundtrip": True,
                                     "sklearn_NumPy_FP32_bitwise_equal": True, "CPU_threads": 1, "pickle_joblib_used": False},
                                 "auxiliary_status": {"MCL": "qualified", "Synovitis": "qualified"}} for fold in A.FOLDS]})
    return {"schema_version": "orthodiff_regional_inner4_C_selection_v1", "common_C_all12": True,
            "outer2_target_or_prediction_used_for_selection": False,
            "inner_validation_labels_used_for_C_selection": True, "tie_tolerance": 1e-12,
            "tie_preference": "smallest_C", "support": support, "table": table, "selected_C": .01}


def fake_model(training, features):
    vectors = [features[r[A.ID]] for r in training]
    columns = list(zip(*vectors))
    mean = [math.fsum(c) / len(c) for c in columns]
    variance = [math.fsum((x - m) ** 2 for x in c) / len(c) for c, m in zip(columns, mean)]
    scale = [math.sqrt(v) if v else 1. for v in variance]
    heads = []
    for i, target in enumerate(A.TARGETS):
        heads.append({"target": target, "status": "qualified", "coef": [.13 * (i + 1), -.21, .07],
                      "intercept": -.2, "train_observed": 8, "train_missing": 0,
                      "train_positive": 4, "train_negative": 4, "warnings": []})
    return {"schema_version": "orthodiff_weak_head_numeric_model_v1", "targets": list(A.TARGETS),
            "feature_dim": 3, "dtype": "float64", "seed": 20261002, "heads": heads,
            "scaler": {"mean": mean, "variance": variance, "scale": scale, "training_samples": 8}}


class TargetFreeAuditor(unittest.TestCase):
    def test_view_rejects_outer2_targets_group_leakage_and_soft_training_labels(self):
        training, outer, _ = fake_data()
        cohort = [{k: r[k] for k in A.META} for r in training] + outer
        view = {"schema_version": "orthodiff_regional_inner4_training_only_labels_v1", "target_order": list(A.TARGETS),
                "outer2_target_cells_projected_converted_or_retained": False,
                "training": training, "outer2_metadata": outer}
        self.assertEqual(A.validate_view(view, cohort), (training, outer))
        altered = copy.deepcopy(view)
        altered["outer2_metadata"][0]["targets"] = ["FORBIDDEN"] * 12
        with self.assertRaises(ValueError):
            A.validate_view(altered, cohort)
        altered = copy.deepcopy(view)
        altered["training"][0]["targets"][0] = .3
        with self.assertRaises(ValueError):
            A.validate_view(altered, cohort)
        altered = copy.deepcopy(view)
        altered["training"][0][A.ID] = altered["training"][1][A.ID]
        with self.assertRaises(ValueError):
            A.validate_view(altered, cohort)

    def test_table_arithmetic_masks_and_tie_without_claiming_inner_replay(self):
        training, _, features = fake_data()
        selection = fake_selection(training, features)
        result = A.validate_selection(selection, training, features)
        self.assertEqual(result["selected_C"], .01)
        self.assertFalse(result["inner_predictions_or_models_independently_replayed"])
        for action in (lambda d: d.update(selected_C=.1),
                       lambda d: d["table"][0].update(score=.61),
                       lambda d: d["table"][0]["inner"][0]["AUC"].pop("ACL"),
                       lambda d: d["support"][0].update(valid_count=999)):
            altered = copy.deepcopy(selection)
            action(altered)
            with self.assertRaises(ValueError):
                A.validate_selection(altered, training, features)

    def test_scaler_fsum_prediction_twoCSVs_and_corruption_rejection(self):
        training, outer, features = fake_data()
        model = fake_model(training, features)
        report = A.verify_scaler_and_heads(model, training, features, 3)
        self.assertEqual(report["mean_max_abs_difference"], 0.)
        predictions = A.independent_predictions(model, outer, features)
        # A separate direct three-term expression provides a tiny numeric oracle.
        z = sum((features[outer[0][A.ID]][i] - model["scaler"]["mean"][i]) / model["scaler"]["scale"][i]
                * model["heads"][0]["coef"][i] for i in range(3)) - .2
        self.assertAlmostEqual(predictions[0][0], 1 / (1 + math.exp(-z)), places=15)
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / name for name in ("f64.csv", "f32.csv")]
            for path, is32 in zip(paths, (False, True)):
                with path.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow([A.ID, *A.TARGETS])
                    for row, values in zip(outer, predictions):
                        writer.writerow([row[A.ID]] + [str(A.struct.unpack("<f", A.f32bits(x))[0] if is32 else x) for x in values])
            result = A.compare_csvs(*paths, outer, predictions)
            self.assertEqual(result["finite_FP32_cells_bitwise_equal"], 36)
            changed = copy.deepcopy(predictions)
            changed[0][0] += .001
            with self.assertRaises(ValueError):
                A.compare_csvs(*paths, outer, changed)
        model["scaler"]["mean"][0] += .1
        with self.assertRaises(ValueError):
            A.verify_scaler_and_heads(model, training, features, 3)

    def test_running_or_unfinished_run_cannot_issue_actual_qualification(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "run.json").write_text(json.dumps({"schema_version": A.RUN_SCHEMA, "status": "RUNNING"}))
            target = directory / "rejected.json"
            result = subprocess.run([sys.executable, str(SCRIPT), "--run-dir", str(directory),
                "--expected-run-sha256", A.pin(directory / "run.json")["sha256"],
                "--expected-auditor-sha256", A.pin(SCRIPT)["sha256"], "--output", str(target)],
                cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            receipt = json.loads(target.read_bytes())
            self.assertFalse(receipt["passed"])
            self.assertFalse(receipt["target_free_prediction_seal_engineering_qualified"])
            self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
