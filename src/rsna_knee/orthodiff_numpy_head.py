"""Single-fold0 numeric head transport; stdlib plus NumPy, no estimator loading."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

TARGETS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")
POLICY_SHA = "2b9704b70ab4197129e87db5e77fd08fd5a39c3c37ef950eac12a51cf0162b21"
TRAINER_MODULE_SHA = "e1149f8e88eeba0f32418e8ee4986cd41ef5aae208a31142cd907689958d78e7"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def finite_vector(value, width, name):
    require(isinstance(value, list) and len(value) == width and all(
        type(item) in (int, float) and math.isfinite(item) for item in value
    ), "Expected finite numeric vector: " + name)
    return np.asarray(value, dtype=np.float64)


def validate_model(model):
    require(isinstance(model, dict) and model.get("schema_version") == "orthodiff_weak_head_numeric_model_v1"
            and model.get("targets") == list(TARGETS) and model.get("feature_dim") == 256
            and model.get("dtype") == "float64" and model.get("seed") == 20261002,
            "Expected fixed256/12target numeric head schema")
    context = model.get("fit_context", {})
    require(context.get("fold") == 0 and context.get("seed") == 20261002, "Only a single original fold0 artifact is supported")
    inputs = context.get("inputs", {})
    require(inputs.get("policy", {}).get("sha256") == POLICY_SHA, "Fixed prospective policy-v2 provenance required")
    require(inputs.get("source_pins", {}).get("orthodiff_weak_head.py", {}).get("sha256") == TRAINER_MODULE_SHA,
            "Pinned numeric head trainer source provenance required")
    require(re.fullmatch(r"[0-9a-f]{64}", str(inputs.get("qualified_bundle", {}).get("sha256", ""))) is not None,
            "Qualified feature bundle hash provenance required")
    scaler = model["scaler"]
    finite_vector(scaler["mean"], 256, "scaler.mean")
    scale = finite_vector(scaler["scale"], 256, "scaler.scale")
    variance = finite_vector(scaler["variance"], 256, "scaler.variance")
    require(np.all(scale > 0) and np.all(variance >= 0), "Invalid scaler scale/variance")
    require(type(scaler["training_samples"]) is int and 0 < scaler["training_samples"] <= 3386,
            "Fold0 scaler must use successful training samples only")
    heads = model["heads"]
    require(isinstance(heads, list) and len(heads) == 12, "Exactly12 ordered head entries required")
    for target, entry in zip(TARGETS, heads):
        require(isinstance(entry, dict) and entry.get("target") == target, "Head target order differs")
        if entry.get("status") == "qualified":
            finite_vector(entry["coef"], 256, target + ".coef")
            require(type(entry["intercept"]) in (float, int) and math.isfinite(entry["intercept"]), "Finite scalar intercept required")
        else:
            require(entry.get("status") in {"insufficient_training_classes", "nonconverged_or_nonfinite"}
                    and entry.get("coef") is None and entry.get("intercept") is None,
                    "Unsupported target must remain explicitly unavailable, without fabricated coefficients")
    return model


def predict_numeric(model, features):
    """Independent NumPy-only predictor; no estimator deserialization/import."""
    require(model.get("schema_version") == "orthodiff_weak_head_numeric_model_v1" and model["targets"] == list(TARGETS), "Numeric model contract differs")
    x = np.asarray(features, dtype=np.float64)
    mean, scale = np.asarray(model["scaler"]["mean"], dtype=np.float64), np.asarray(model["scaler"]["scale"], dtype=np.float64)
    require(x.ndim == 2 and x.shape[1] == model["feature_dim"] and mean.shape == scale.shape == (model["feature_dim"],)
            and np.isfinite(x).all() and np.isfinite(mean).all() and np.isfinite(scale).all() and np.all(scale > 0), "Numeric scaler/input invalid")
    standardized = (x - mean) / scale
    out = np.full((len(x), len(TARGETS)), np.nan, dtype=np.float64)
    for i, entry in enumerate(model["heads"]):
        require(entry["target"] == TARGETS[i], "Numeric model target order differs")
        if entry["status"] != "qualified":
            continue
        coefficient = np.asarray(entry["coef"], dtype=np.float64)
        require(coefficient.shape == (model["feature_dim"],) and np.isfinite(coefficient).all()
                and math.isfinite(entry["intercept"]), "Numeric coefficient invalid")
        z = (standardized @ coefficient.reshape(-1, 1)).ravel() + entry["intercept"]
        positive = z >= 0
        result = np.empty_like(z)
        result[positive] = 1 / (1 + np.exp(-z[positive]))
        exp_value = np.exp(z[~positive])
        result[~positive] = exp_value / (1 + exp_value)
        out[:, i] = result
    return out


class Fold0NumpyHead:
    """Hash-checked JSON adapter; callers supply complete finite features only."""

    def __init__(self, model_path, expected_sha256):
        require(isinstance(expected_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None,
                "Full expected model SHA256 is mandatory")
        path = Path(model_path)
        require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= 2 * 1024**2,
                "Bounded regular numeric JSON model required")
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected_sha256, "Numeric model SHA256 differs")
        model = validate_model(json.loads(raw))
        self._raw_model = raw
        self.model_pin = {"bytes": len(raw), "sha256": expected_sha256}
        self.supported_targets = tuple(entry["status"] == "qualified" for entry in model["heads"])

    def predict(self, features):
        # No feature imputation. Absent/None/NaN/infinite or wrong-width rows fail.
        require(hashlib.sha256(self._raw_model).hexdigest() == self.model_pin["sha256"], "Loaded model bytes changed")
        model = validate_model(json.loads(self._raw_model))
        require(np.asarray(features).dtype.kind in {"f", "i", "u"}, "Numeric feature array required; no missing-value conversion")
        result = predict_numeric(model, features)
        supported = np.asarray([entry["status"] == "qualified" for entry in model["heads"]], dtype=bool)
        require(np.isfinite(result[:, supported]).all() and np.all(np.isnan(result[:, ~supported])),
                "Qualified/unavailable output mask differs")
        return result
