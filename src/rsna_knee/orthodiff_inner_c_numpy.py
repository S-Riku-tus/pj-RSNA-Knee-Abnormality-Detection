"""NumPy transport for the sealed regional1024 C=.01 outer2 model.

This module predicts from numeric JSON only. Actual deployment additionally
requires a separately pinned root qualification after the outer2 screen and a
runtime numerical audit. It neither chooses a Public parent nor blends scores.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

TARGETS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")
MODEL_SHA = "c9b5ae26d0ad5895903002ff6bfd2c4947bed152c070c749588be8c76710a714"
POLICY_SHA = "dd26d15c5b27c624c08eeee84e65d828b4c0df572854dc301af149954a1c2c6f"
BUNDLE_SHA = "d892f0f94a6b98a172c6644564e8769ad8395b3a2223093313b95062123340d5"
SELECTION_SHA = "4535e443da59c81dec19a9ddf4f542225a26659a34383b0c269bbd6dfd5f42b7"
SEAL_SHA = "11cbdcf1752960a9bb96edbaefc55d61c1325e5eaaef5e07d8be3cbf9f536934"
AUDIT_SHA = "644ecd0657d1b200b3b3fb39aace5f1eaef709d413808c945905780878b94879"
VIEW_SHA = "c674e098d561f2503ca2d5fc8489c09598fc50d079c0ae91c9833dd19a17a1d7"
SOURCE_PINS = {
    "fit_orthodiffusion_regional_inner_c.py": "160cc6391522dc1a811aa9d071b00b91dfaba3f66c50a08db66fadc42352c215",
    "orthodiff_regional_inner_c.py": "fa4cfb206b90db9a3c5b8e23badadb251f500d22fa56a3a45106fb85e42e079f",
    "orthodiff_weak_head.py": "e1149f8e88eeba0f32418e8ee4986cd41ef5aae208a31142cd907689958d78e7",
    "contracts.py": "aa246fd1a49ecbe88f1795f49904ef9675f119c935e99b9ffd5eee8220cbc71b",
    "orthodiff_regional_features.py": "1363055c2d9493fd9018b8d9c0f7f0f1da52dba6277a30bbc9c865e12add855d",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def parse_json(raw):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON field")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON token")))


def checked_json(path, expected_sha):
    require(is_digest(expected_sha), "Explicit full SHA256 required")
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= 2 * 1024**2,
            "Bounded regular JSON artifact required")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha, "JSON artifact SHA256 differs")
    return raw, parse_json(raw)


def vector(values, name):
    require(isinstance(values, list) and len(values) == 1024 and all(
        type(value) in (int, float) and math.isfinite(value) for value in values),
        "Finite width1024 vector required: " + name)
    return np.asarray(values, dtype=np.float64)


def validate_model(model):
    require(isinstance(model, dict) and model.get("schema_version") == "orthodiff_weak_head_numeric_model_v1"
            and model.get("targets") == list(TARGETS) and model.get("feature_dim") == 1024
            and model.get("dtype") == "float64" and model.get("seed") == 20261002
            and type(model.get("selected_common_C")) is float and model["selected_common_C"] == .01,
            "Only the sealed regional1024 C=.01 model schema is supported")
    context = model.get("fit_context", {})
    require(context.get("recipe_schema") == "orthodiff_regional_inner4_C_outer2_sealed_run_v1"
            and type(context.get("outer_heldout_fold")) is int and context["outer_heldout_fold"] == 2
            and context.get("inner_original_folds") == [0, 1, 3, 4]
            and context.get("fixed_C_grid") == [.001, .01, .1, 1.]
            and context.get("selected_common_C") == .01 and context.get("successful_train") == 2409
            and context.get("successful_outer2") == 654
            and context.get("inner_validation_labels_used_for_C_selection") is True
            and context.get("outer2_targets_read_or_used") is False,
            "Inner4 selection and outer2 seal context differ")
    inputs = context.get("inputs", {})
    require(inputs.get("policy", {}).get("sha256") == POLICY_SHA
            and inputs.get("qualified_regional_bundle", {}).get("sha256") == BUNDLE_SHA
            and inputs.get("training_only_label_view", {}).get("sha256") == VIEW_SHA
            and context.get("selection", {}).get("sha256") == SELECTION_SHA,
            "Sealed policy/feature bundle/selection/training view differ")
    pins = inputs.get("source_pins", {})
    require(set(pins) == set(SOURCE_PINS) and all(
        pins[name].get("sha256") == digest for name, digest in SOURCE_PINS.items()),
        "Selected-model producer/source provenance differs")
    scaler = model.get("scaler", {})
    vector(scaler.get("mean"), "mean")
    scale, variance = vector(scaler.get("scale"), "scale"), vector(scaler.get("variance"), "variance")
    require(np.all(scale > 0) and np.all(variance >= 0)
            and type(scaler.get("training_samples")) is int and scaler["training_samples"] == 2409,
            "Training-only scaler population/scale/variance differ")
    heads = model.get("heads")
    require(isinstance(heads, list) and len(heads) == 12, "Exactly twelve ordered heads required")
    for target, head in zip(TARGETS, heads):
        require(isinstance(head, dict) and head.get("target") == target and head.get("status") == "qualified"
                and head.get("warnings") == [], "Sealed twelve qualified warning-free heads required")
        vector(head.get("coef"), target + ".coef")
        require(type(head.get("intercept")) in (int, float) and math.isfinite(head["intercept"])
                and type(head.get("iterations")) is int and 0 < head["iterations"] <= 5000,
                "Finite intercept and converged iteration count required")
        counts = [head.get(key) for key in ("train_observed", "train_positive", "train_negative", "train_missing")]
        require(all(type(value) is int and 0 <= value <= 2409 for value in counts)
                and counts[1] > 0 and counts[2] > 0 and counts[1] + counts[2] == counts[0]
                and counts[0] + counts[3] == 2409, "Training-only missing/observed class counts differ")
    return model


def bindings(model_sha):
    require(is_digest(model_sha), "Explicit model SHA required")
    return {"model_sha256": model_sha, "policy_sha256": POLICY_SHA, "bundle_sha256": BUNDLE_SHA,
            "selection_sha256": SELECTION_SHA, "seal_sha256": SEAL_SHA,
            "target_free_actual_audit_sha256": AUDIT_SHA, "source_sha256": SOURCE_PINS,
            "dimension": 1024, "C": .01, "outer_holdout": 2, "successful_training": 2409,
            "targets": list(TARGETS)}


def validate_qualification(qualification, expected, artificial):
    require(isinstance(qualification, dict)
            and qualification.get("schema_version") == "orthodiff_innerC_numpy_transport_qualification_v1"
            and qualification.get("passed") is True and qualification.get("bindings") == expected,
            "Separate source/model/runtime-bound qualification required")
    if artificial:
        require(qualification.get("scope") == "artificial_only"
                and qualification.get("actual_model_qualified") is False
                and qualification.get("root_candidate_approved") is False,
                "Artificial mode cannot qualify actual deployment")
    else:
        require(expected["model_sha256"] == MODEL_SHA
                and qualification.get("scope") == "actual_sealed_single_model"
                and qualification.get("actual_model_qualified") is True
                and qualification.get("runtime_numerical_audit_passed") is True
                and qualification.get("qualified_numpy_version") == np.__version__
                and qualification.get("outer2_engineering_and_fixed10_positive") is True
                and is_digest(qualification.get("outer2_result_sha256"))
                and is_digest(qualification.get("root_candidate_review_sha256"))
                and qualification.get("root_candidate_approved") is True,
                "Actual model, this NumPy runtime, positive outer2 and root review required")


def predict_numeric(model, features):
    """Low-level numerical audit helper; does not grant deployment qualification."""
    validate_model(model)
    values = np.asarray(features)
    require(values.dtype.kind in {"i", "u", "f"} and values.ndim == 2 and values.shape[1] == 1024
            and np.isfinite(values).all(), "Finite numeric N by1024 features required")
    x = values.astype(np.float64)
    mean = np.asarray(model["scaler"]["mean"], dtype=np.float64)
    scale = np.asarray(model["scaler"]["scale"], dtype=np.float64)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        standardized = (x - mean) / scale
        out = np.empty((len(x), 12), dtype=np.float64)
        for i, head in enumerate(model["heads"]):
            coefficient = np.asarray(head["coef"], dtype=np.float64)
            z = (standardized @ coefficient.reshape(-1, 1)).ravel() + head["intercept"]
            require(np.isfinite(z).all(), "Nonfinite selected-model logits")
            positive = z >= 0
            result = np.empty_like(z)
            result[positive] = 1 / (1 + np.exp(-z[positive]))
            exp_value = np.exp(z[~positive])
            result[~positive] = exp_value / (1 + exp_value)
            out[:, i] = result
    require(np.isfinite(out).all() and np.all((out >= 0) & (out <= 1)), "Invalid model probabilities")
    return out


class InnerCNumpyHead:
    """One frozen selected model, without fitting, averaging or Public blending."""

    def __init__(self, model_path, *, expected_model_sha256, qualification_path,
                 expected_qualification_sha256, allow_artificial=False):
        require(type(allow_artificial) is bool, "Explicit Boolean artificial mode required")
        expected = bindings(expected_model_sha256)
        model_raw, model = checked_json(model_path, expected_model_sha256)
        qualification_raw, qualification = checked_json(qualification_path, expected_qualification_sha256)
        validate_model(model)
        validate_qualification(qualification, expected, allow_artificial)
        self._model_raw = model_raw
        self._qualification_raw = qualification_raw
        self._model_sha = expected_model_sha256
        self._qualification_sha = expected_qualification_sha256
        self._artificial = allow_artificial
        self.actual_model_qualified = not allow_artificial
        self.public_submission_contract_qualified = False

    def predict(self, features):
        require(hashlib.sha256(self._model_raw).hexdigest() == self._model_sha
                and hashlib.sha256(self._qualification_raw).hexdigest() == self._qualification_sha,
                "Loaded model/qualification bytes changed")
        validate_qualification(parse_json(self._qualification_raw), bindings(self._model_sha), self._artificial)
        return predict_numeric(parse_json(self._model_raw), features)
