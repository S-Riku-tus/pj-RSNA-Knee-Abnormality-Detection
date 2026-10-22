"""Pinned single-fold FD256/regional1024 transport, stdlib and NumPy only.

The copied rank_blend kernel is for the existing full-holdout CV calculation.
It does not qualify re-ranking a Public parent, averaging heads, or a submission.
Actual model use requires a separately pinned root/independent qualification.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

TARGETS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")
FIXED10 = tuple(x for x in TARGETS if x not in ("MCL", "Synovitis"))
TRAINER_SHA = "e1149f8e88eeba0f32418e8ee4986cd41ef5aae208a31142cd907689958d78e7"
RECIPES = {
    "FD256": {"dimension": 256, "producer": "fit_orthodiffusion_fluid_composite_head.py",
              "producer_sha256": "e1947983134687a52d1be7c338b48900276365c7b7bf8f2327bb0a43db0ef99a",
              "policy_sha256": "105bda1d86b677925356249d913a69abd9c03d4c950c86fc96589e8a79787abe",
              "bundle_key": "qualified_composite_bundle"},
    "REGIONAL1024": {"dimension": 1024, "producer": "fit_orthodiffusion_regional_head.py",
                     "producer_sha256": "8982a65b7d2f4fe0c19196c1cbae8632a427513433a5de40fd2bd4e23f7066cd",
                     "policy_sha256": "c6796180a208c1b56f7aeee28154d537081e34d7927874978cf4d255c4590ff7",
                     "bundle_key": "qualified_regional_bundle"}}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def parse_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON field")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON token")))


def checked_json(path, expected_sha256):
    require(digest(expected_sha256), "Explicit full SHA256 required")
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= 2 * 1024**2,
            "Bounded regular numeric JSON artifact required")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, "Artifact SHA256 differs")
    return raw, parse_json(raw)


def finite_vector(values, width, name):
    require(isinstance(values, list) and len(values) == width and all(
        type(value) in (int, float) and math.isfinite(value) for value in values), "Finite numeric vector required: " + name)
    return np.asarray(values, dtype=np.float64)


def contract(recipe, fold, model_sha256, producer_sha256, policy_sha256, bundle_sha256):
    require(recipe in RECIPES and type(fold) is int and fold in (0, 1), "Explicit supported recipe and original fold0/1 required")
    spec = RECIPES[recipe]
    require(all(digest(value) for value in (model_sha256, producer_sha256, policy_sha256, bundle_sha256)), "All explicit provenance SHA256 pins required")
    require(producer_sha256 == spec["producer_sha256"] and policy_sha256 == spec["policy_sha256"],
            "Only the fixed new recipe producer and policy are supported; old PD producers are excluded")
    return {"recipe": recipe, "fold": fold, "feature_dimension": spec["dimension"], "targets": list(TARGETS),
            "model_sha256": model_sha256, "producer_sha256": producer_sha256,
            "policy_sha256": policy_sha256, "bundle_sha256": bundle_sha256, "trainer_sha256": TRAINER_SHA}


def validate_model(model, expected):
    width, fold = expected["feature_dimension"], expected["fold"]
    spec = RECIPES[expected["recipe"]]
    require(isinstance(model, dict) and model.get("schema_version") == "orthodiff_weak_head_numeric_model_v1"
            and model.get("feature_dim") == width and model.get("targets") == list(TARGETS)
            and model.get("dtype") == "float64" and model.get("seed") == 20261002, "Pinned dimension/target numeric model schema required")
    context = model.get("fit_context", {})
    require(type(context.get("fold")) is int and context["fold"] == fold and context.get("seed") == 20261002,
            "Single original-fold model provenance differs")
    inputs = context.get("inputs", {})
    require(inputs.get("policy", {}).get("sha256") == expected["policy_sha256"]
            and inputs.get(spec["bundle_key"], {}).get("sha256") == expected["bundle_sha256"], "Model policy/qualified feature bundle differs")
    sources = inputs.get("source_pins", {})
    require(sources.get(spec["producer"], {}).get("sha256") == expected["producer_sha256"]
            and sources.get("orthodiff_weak_head.py", {}).get("sha256") == TRAINER_SHA,
            "Model producer/numerical trainer provenance differs")
    scaler = model["scaler"]
    finite_vector(scaler["mean"], width, "scaler.mean")
    scale = finite_vector(scaler["scale"], width, "scaler.scale")
    variance = finite_vector(scaler["variance"], width, "scaler.variance")
    require(np.all(scale > 0) and np.all(variance >= 0), "Scaler scale/variance invalid")
    training = scaler["training_samples"]
    require(type(training) is int and 0 < training <= (3386 if fold == 0 else 3316), "Original training-fold sample bound differs")
    if expected["recipe"] == "REGIONAL1024":
        require(training == (2469 if fold == 0 else 2426), "Same qualified parent successful training population required")
    heads = model["heads"]
    require(isinstance(heads, list) and len(heads) == len(TARGETS), "Exactly12 ordered head entries required")
    for target, entry in zip(TARGETS, heads):
        require(isinstance(entry, dict) and entry.get("target") == target, "Head target order differs")
        if entry.get("status") == "qualified":
            finite_vector(entry["coef"], width, target + ".coef")
            require(type(entry["intercept"]) in (int, float) and math.isfinite(entry["intercept"]), "Finite scalar intercept required")
        else:
            require(target not in FIXED10 and entry.get("status") in {"insufficient_training_classes", "nonconverged_or_nonfinite"}
                    and entry.get("coef") is None and entry.get("intercept") is None,
                    "Fixed10 must qualify; unavailable auxiliary heads must not fabricate coefficients")
    return model


def validate_qualification(qualification, expected, allow_artificial):
    require(isinstance(qualification, dict)
            and qualification.get("schema_version") == "orthodiff_fd_regional_numpy_model_qualification_v1"
            and qualification.get("passed") is True and qualification.get("bindings") == expected,
            "Pinned single-fold model qualification and exact bindings required")
    if allow_artificial:
        require(qualification.get("scope") == "artificial_only" and qualification.get("actual_model_qualified") is False
                and qualification.get("root_candidate_approved") is False,
                "Artificial mode cannot claim actual model or candidate qualification")
    else:
        require(qualification.get("scope") == "actual_single_fold"
                and qualification.get("actual_model_qualified") is True
                and qualification.get("independent_actual_audit_passed") is True
                and qualification.get("both_folds_gate_passed") is True
                and qualification.get("root_candidate_approved") is True
                and qualification.get("qualified_numpy_version") == np.__version__,
                "Actual model/runtime numerical audit, twofold gate and explicit root candidate approval required")


class CandidateNumpyHead:
    """One explicitly bound model; no averaging, imputation or Public score overlay."""

    def __init__(self, model_path, *, expected_model_sha256, recipe, fold, expected_producer_sha256,
                 expected_policy_sha256, expected_bundle_sha256, qualification_path,
                 expected_qualification_sha256, allow_artificial=False):
        require(type(allow_artificial) is bool, "Explicit Boolean artificial mode required")
        expected = contract(recipe, fold, expected_model_sha256, expected_producer_sha256,
                            expected_policy_sha256, expected_bundle_sha256)
        model_raw, model = checked_json(model_path, expected_model_sha256)
        require(qualification_path is not None, "Separately pinned model qualification is mandatory")
        qualification_raw, qualification = checked_json(qualification_path, expected_qualification_sha256)
        validate_model(model, expected)
        validate_qualification(qualification, expected, allow_artificial)
        self._model_raw, self._qualification_raw = model_raw, qualification_raw
        self._expected_raw = json.dumps(expected, sort_keys=True).encode()
        self._qualification_sha256 = expected_qualification_sha256
        self._allow_artificial = allow_artificial
        self.actual_model_qualified = not allow_artificial
        self.public_submission_contract_qualified = False
        self.supported_targets = tuple(entry["status"] == "qualified" for entry in model["heads"])

    def predict(self, features):
        expected = parse_json(self._expected_raw)
        require(hashlib.sha256(self._model_raw).hexdigest() == expected["model_sha256"]
                and hashlib.sha256(self._qualification_raw).hexdigest() == self._qualification_sha256, "Loaded artifact bytes changed")
        model = validate_model(parse_json(self._model_raw), expected)
        validate_qualification(parse_json(self._qualification_raw), expected, self._allow_artificial)
        require(np.asarray(features).dtype.kind in {"f", "i", "u"}, "Finite numeric features only; no missing-value conversion")
        result = predict_numeric(model, features)
        supported = np.asarray(self.supported_targets, dtype=bool)
        require(np.isfinite(result[:, supported]).all() and np.isnan(result[:, ~supported]).all(), "Qualified/unavailable prediction mask differs")
        return result


def average_rank_fp32(values):
    values = np.asarray(values)
    require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(), "Finite nonempty rank vector required")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    out = np.empty(len(values), dtype=np.float64)
    for start, end in zip(starts, ends):
        out[order[start:end]] = (float(start) + 1 + float(end)) / (2 * len(values))
    return out.astype(np.float32)


def rank_blend(raw_control, head_probabilities, success_mask, supported):
    raw = np.asarray(raw_control, dtype=np.float64)
    head = np.asarray(head_probabilities, dtype=np.float64)
    success_mask = np.asarray(success_mask, dtype=bool)
    require(raw.ndim == 2 and raw.shape == head.shape and raw.shape[1] == len(TARGETS)
            and success_mask.shape == (len(raw),) and np.isfinite(raw).all(), "Aligned full-holdout shapes required")
    baseline = np.column_stack([average_rank_fp32(raw[:, i]) for i in range(len(TARGETS))])
    blend = baseline.copy()
    details = {}
    for i, name in enumerate(TARGETS):
        order = np.argsort(raw[:, i], kind="mergesort")
        require(np.array_equal(raw[order[1:], i] == raw[order[:-1], i], baseline[order[1:], i] == baseline[order[:-1], i])
                and np.all(np.diff(baseline[order, i]) >= 0), "FP32 baseline does not preserve all raw ties/order")
        if name not in FIXED10 or not supported.get(name, False):
            details[name] = {"active": False, "reason": "unsupported_or_MCL_Synovitis", "N_valid": int(success_mask.sum())}
            continue
        candidate = head[success_mask, i].astype(np.float32)
        require(np.isfinite(candidate).all(), "Nonfinite supported head probability")
        if len(candidate) < 2 or len(np.unique(candidate)) < 2:
            details[name] = {"active": False, "reason": "N_valid_lt2_or_degenerateFP32", "N_valid": len(candidate)}
            continue
        ranks = average_rank_fp32(candidate)
        left = np.multiply(np.float32(.8), baseline[success_mask, i], dtype=np.float32)
        right = np.multiply(np.float32(.2), ranks, dtype=np.float32)
        blend[success_mask, i] = np.add(left, right, dtype=np.float32)
        details[name] = {"active": True, "N_valid": len(candidate), "distinct_head_FP32": len(np.unique(candidate))}
    require(np.array_equal(blend[~success_mask].view(np.uint32), baseline[~success_mask].view(np.uint32)), "Failed feature baseline cells changed")
    return baseline, blend, details


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
