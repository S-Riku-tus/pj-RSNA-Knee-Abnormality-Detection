"""Artificial selected-model math and deployment boundary tests; no real data."""
import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rsna_knee import orthodiff_inner_c_numpy as N


def fixture_model():
    coefficients = [0.] * 1024
    coefficients[:3] = [2., -1., .5]
    return {"schema_version": "orthodiff_weak_head_numeric_model_v1", "targets": list(N.TARGETS),
            "feature_dim": 1024, "dtype": "float64", "seed": 20261002, "selected_common_C": .01,
            "scaler": {"mean": [.25] * 1024, "scale": [2.] * 1024, "variance": [4.] * 1024,
                       "training_samples": 2409},
            "fit_context": {"recipe_schema": "orthodiff_regional_inner4_C_outer2_sealed_run_v1",
                "outer_heldout_fold": 2, "inner_original_folds": [0, 1, 3, 4],
                "fixed_C_grid": [.001, .01, .1, 1.], "selected_common_C": .01,
                "successful_train": 2409, "successful_outer2": 654,
                "inner_validation_labels_used_for_C_selection": True, "outer2_targets_read_or_used": False,
                "selection": {"sha256": N.SELECTION_SHA}, "inputs": {
                    "policy": {"sha256": N.POLICY_SHA},
                    "qualified_regional_bundle": {"sha256": N.BUNDLE_SHA},
                    "training_only_label_view": {"sha256": N.VIEW_SHA},
                    "source_pins": {name: {"sha256": digest} for name, digest in N.SOURCE_PINS.items()}}},
            "heads": [{"target": name, "status": "qualified", "warnings": [], "coef": coefficients[:],
                       "intercept": i / 10. - .5, "iterations": 2, "train_observed": 2000,
                       "train_positive": 500, "train_negative": 1500, "train_missing": 409}
                      for i, name in enumerate(N.TARGETS)]}


def save(path, value):
    raw = (json.dumps(value, allow_nan=False) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def qualified_fixture(folder):
    model = folder / "model.json"
    digest = save(model, fixture_model())
    qualification = folder / "qualification.json"
    q_sha = save(qualification, {"schema_version": "orthodiff_innerC_numpy_transport_qualification_v1",
                 "passed": True, "bindings": N.bindings(digest), "scope": "artificial_only",
                 "actual_model_qualified": False, "root_candidate_approved": False})
    return model, digest, qualification, q_sha


class InnerCNumpyTests(unittest.TestCase):
    def test_sparse_fixture_matches_independent_scalar_sigmoid(self):
        with tempfile.TemporaryDirectory() as temp:
            model, digest, qualification, q_sha = qualified_fixture(Path(temp))
            head = N.InnerCNumpyHead(model, expected_model_sha256=digest, qualification_path=qualification,
                                    expected_qualification_sha256=q_sha, allow_artificial=True)
            values = np.zeros((7, 1024), dtype=np.float32)
            values[:, :3] = [[-3, 1, 2], [0, 0, 0], [2, 4, 8], [1, 2, 1], [9, -8, 3], [-4, -2, 1], [0, 0, 0]]
            expected = []
            for row in values:
                base = math.fsum([(float(row[0]) - .25), -(float(row[1]) - .25) / 2,
                                  (float(row[2]) - .25) / 4])
                expected.append([1 / (1 + math.exp(-(base + i / 10. - .5))) for i in range(12)])
            actual = head.predict(values)
            self.assertTrue(np.array_equal(actual.astype(np.float32).view(np.uint32),
                                           np.asarray(expected, dtype=np.float32).view(np.uint32)))
            self.assertFalse(head.actual_model_qualified)
            self.assertFalse(head.public_submission_contract_qualified)
            self.assertEqual(head.predict(np.empty((0, 1024))).shape, (0, 12))

    def test_seal_provenance_and_missing_masks_are_not_interchangeable(self):
        model = fixture_model()
        mutations = [lambda m: m.update(selected_common_C=1.),
                     lambda m: m["fit_context"].update(outer_heldout_fold=0),
                     lambda m: m["fit_context"]["inputs"]["policy"].update(sha256="0" * 64),
                     lambda m: m["fit_context"]["inputs"]["source_pins"].pop("contracts.py"),
                     lambda m: m["heads"][0].update(train_missing=0),
                     lambda m: m["heads"][0].update(status="nonconverged_or_nonfinite"),
                     lambda m: m["heads"][0].update(warnings=["not converged"]),
                     lambda m: m["scaler"]["scale"].__setitem__(0, 0.)]
        for change in mutations:
            altered = copy.deepcopy(model)
            change(altered)
            with self.subTest(change=change), self.assertRaises(ValueError):
                N.predict_numeric(altered, np.zeros((1, 1024)))

    def test_artificial_cannot_satisfy_actual_gate_and_runtime_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            model, digest, qualification, q_sha = qualified_fixture(Path(temp))
            with self.assertRaises(ValueError):
                N.InnerCNumpyHead(model, expected_model_sha256=digest, qualification_path=qualification,
                                 expected_qualification_sha256=q_sha)
        expected = N.bindings(N.MODEL_SHA)
        q = {"schema_version": "orthodiff_innerC_numpy_transport_qualification_v1", "passed": True,
             "bindings": expected, "scope": "actual_sealed_single_model", "actual_model_qualified": True,
             "runtime_numerical_audit_passed": True, "qualified_numpy_version": np.__version__,
             "outer2_engineering_and_fixed10_positive": True, "outer2_result_sha256": "a" * 64,
             "root_candidate_review_sha256": "b" * 64, "root_candidate_approved": True}
        N.validate_qualification(q, expected, False)
        for key, value in [("qualified_numpy_version", "wrong"), ("outer2_engineering_and_fixed10_positive", False),
                           ("outer2_result_sha256", None), ("root_candidate_approved", False)]:
            altered = dict(q)
            altered[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                N.validate_qualification(altered, expected, False)

    def test_model_and_qualification_hash_and_loaded_bytes_are_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            model, digest, qualification, q_sha = qualified_fixture(Path(temp))
            head = N.InnerCNumpyHead(model, expected_model_sha256=digest, qualification_path=qualification,
                                    expected_qualification_sha256=q_sha, allow_artificial=True)
            head._model_raw += b" "
            with self.assertRaises(ValueError):
                head.predict(np.zeros((1, 1024)))
            model.write_bytes(model.read_bytes() + b" ")
            with self.assertRaises(ValueError):
                N.InnerCNumpyHead(model, expected_model_sha256=digest, qualification_path=qualification,
                                 expected_qualification_sha256=q_sha, allow_artificial=True)
        for raw in ['{"a":1,"a":2}', '{"a":NaN}']:
            with self.assertRaises(ValueError):
                N.parse_json(raw)

    def test_nonfinite_wrong_shape_or_missing_features_rejected(self):
        model = fixture_model()
        for values in [np.zeros((2, 1023)), np.zeros(1024), np.full((1, 1024), np.nan),
                       np.full((1, 1024), np.inf), np.full((1, 1024), "0"),
                       np.zeros((1, 1024), dtype=bool), np.full((1, 1024), None)]:
            with self.subTest(shape=values.shape, dtype=values.dtype), self.assertRaises(ValueError):
                N.predict_numeric(model, values)
        extreme = model
        extreme["heads"][0]["intercept"] = -1000.
        self.assertEqual(N.predict_numeric(extreme, np.zeros((1, 1024)))[0, 0], 0.)


if __name__ == "__main__":
    unittest.main()
