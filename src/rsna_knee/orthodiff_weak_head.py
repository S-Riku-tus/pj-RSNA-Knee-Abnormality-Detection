"""CPU-only fixed OrthoDiffusion linear probe and rank-complementarity evaluation."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import random
import struct
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

from .contracts import ID, TARGETS, probability

FIXED10 = tuple(x for x in TARGETS if x not in ("MCL", "Synovitis"))
POLICY_SHA = "2b9704b70ab4197129e87db5e77fd08fd5a39c3c37ef950eac12a51cf0162b21"
SOURCE_SHA = "15bb2d6c7c224021f95a304b08eb9e3644164ea8373f4ae6ec1c7afc77582e1f"
UID_SHA = "911f91fbc914213db045780675d6518a8cc4a49808599bbb74f2f247bc30eb82"
CONTROL_SHA = "8f75feced2e02584b6581c0ed124e1bdc79d370a312966fd0c976ce302f0b964"
ENVIRONMENT = {"python": "3.9.18", "numpy": "1.22.3", "scipy": "1.7.3", "scikit-learn": "1.6.1"}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def pin(path):
    digest, length = hashlib.sha256(), 0
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            length += len(block)
            digest.update(block)
    return {"bytes": length, "sha256": digest.hexdigest()}


def check_pin(path, expected):
    require(pin(path) == {key: expected[key] for key in ("bytes", "sha256")}, "File pin differs: " + str(path))


def read_json(path):
    return json.loads(Path(path).read_bytes())


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")


def environment(require_fixed=True):
    values = {"python": sys.version.split()[0]}
    values.update({name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")})
    if require_fixed:
        require(values == ENVIRONMENT, "Use the prequalified existing CPU Python; do not install or silently change versions")
    return {"versions": values, "executable": sys.executable, "platform": sys.platform,
            "Torch_imported": "torch" in sys.modules, "new_packages_installed": False}


def seal_bundle(audit_path, expected_audit_sha, records, summary, status, output):
    """Only hash/bind an independently qualified extraction. No labels or fitting."""
    require(not Path(output).exists(), "Use a new feature bundle path")
    require(pin(audit_path)["sha256"] == expected_audit_sha, "Independent qualification SHA differs")
    audit = read_json(audit_path)
    require(audit.get("schema_version") == "orthodiff_fullweak4207_actual_qualification_v1"
            and audit.get("passed") is True and audit.get("head_features_qualified") is True,
            "Actual full4207 independent qualification required")
    require(audit.get("source_sha256") == SOURCE_SHA and audit.get("UID_order_sha256") == UID_SHA
            and audit.get("requested_count") == 4207, "Qualified source/cohort differs")
    require(audit.get("prior32_comparison", {}).get("passed") is True, "Original32 scientific parity qualification required")
    coverage = audit["coverage"]
    require(coverage["requested"] == coverage["recorded"] == 4207
            and coverage["success"] + coverage["failure"] == 4207 and coverage["success"] > 0,
            "Qualified coverage differs")
    files = {}
    for name, path in (("records", records), ("summary", summary), ("status", status)):
        require(Path(path).name == "ORTHODIFF_FULLWEAK_" + {"records": "RECORDS.jsonl", "summary": "SUMMARY.json", "status": "STATUS.json"}[name], "Unexpected feature file name")
        check_pin(path, audit[name + "_pin"])
        files[name] = {"path": str(Path(path).resolve()), **pin(path)}
    summary_value = read_json(summary)
    status_value = read_json(status)
    require(summary_value["complete"] is True and status_value["complete"] is True,
            "Terminal complete summary/status required")
    require(summary_value["coverage"] == coverage == status_value["coverage"], "Qualified coverage/terminal receipts differ")
    require(summary_value["journal"]["sha256"] == files["records"]["sha256"] == status_value["journal"]["sha256"], "Journal/summary/status hash differs")
    bundle = {"schema_version": "orthodiff_fullweak4207_qualified_feature_bundle_v1", "qualified": True,
              "source_sha256": SOURCE_SHA, "UID_order_sha256": UID_SHA, "requested_count": 4207,
              "coverage": coverage, "files": files,
              "qualification": {"path": str(Path(audit_path).resolve()), **pin(audit_path)},
              "scope": "Numerical features/metadata only; no MRI, labels, model loading or fitting during sealing"}
    save_json(output, bundle)
    return pin(output)


def load_qualified_features(bundle_path, expected_bundle_sha):
    require(pin(bundle_path)["sha256"] == expected_bundle_sha, "Expected qualified feature bundle SHA is mandatory")
    bundle = read_json(bundle_path)
    require(bundle.get("schema_version") == "orthodiff_fullweak4207_qualified_feature_bundle_v1"
            and bundle.get("qualified") is True and bundle.get("source_sha256") == SOURCE_SHA
            and bundle.get("UID_order_sha256") == UID_SHA and bundle.get("requested_count") == 4207,
            "Qualified full4207 feature bundle required")
    audit_file = bundle["qualification"]
    check_pin(audit_file["path"], audit_file)
    audit = read_json(audit_file["path"])
    require(audit.get("schema_version") == "orthodiff_fullweak4207_actual_qualification_v1"
            and audit.get("passed") is True and audit.get("head_features_qualified") is True
            and audit.get("source_sha256") == SOURCE_SHA and audit.get("UID_order_sha256") == UID_SHA
            and audit.get("prior32_comparison", {}).get("passed") is True,
            "Independent qualification no longer agrees")
    require(audit["coverage"] == bundle["coverage"] and audit["requested_count"] == 4207, "Bundle qualification coverage differs")
    for name, value in bundle["files"].items():
        check_pin(value["path"], value)
        require({key: value[key] for key in ("bytes", "sha256")} == {key: audit[name + "_pin"][key] for key in ("bytes", "sha256")}, "Bundle/raw/audit pin mismatch")
    summary = read_json(bundle["files"]["summary"]["path"])
    status = read_json(bundle["files"]["status"]["path"])
    uids = summary["requested_UIDs"]
    require(summary["complete"] is True and status["complete"] is True and len(uids) == len(set(uids)) == 4207,
            "Complete unique4207 requests required")
    require(summary["cohort_UID_hash"] == UID_SHA == hashlib.sha256("".join(uid + "\n" for uid in uids).encode()).hexdigest(), "Full cohort UID order differs")
    require(summary["coverage"] == status["coverage"] == bundle["coverage"], "Terminal coverage differs")
    features, failures, strata = {}, {}, {}
    with Path(bundle["files"]["records"]["path"]).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            require(index < 4207 and len(line) <= 2 * 1024**2, "Journal count/line bound exceeded")
            row = json.loads(line)
            uid = row[ID]
            require(uid == uids[index], "Journal must follow exact unique4207 UID order")
            if row["status"] == "success":
                values = row["pooled_float32_values"]
                require(len(values) == 256 and all(type(x) in (float, int) and math.isfinite(x) for x in values), "Finite256 feature values required")
                raw = struct.pack("<256f", *values)
                require(list(struct.unpack("<256f", raw)) == values and row["pooled"] == {
                    "shape": [1, 256], "dtype": "torch.float32", "bytes": 1024, "sha256": hashlib.sha256(raw).hexdigest()}, "Feature FP32 numerical pin differs")
                require(row["all_RNG_restored"] is True and row["forward_counts"] == {"denoiser": 1, "decoder_blocks": 0, "final_out": 0}, "Success contract differs")
                features[uid] = np.asarray(values, dtype=np.float64)
                strata[uid] = row["metadata"]["stratum"]
            else:
                require(row["status"] == "data_failure" and row["feature_forward_calls"] == 0
                        and row["fallback_features"] is False and "pooled_float32_values" not in row and "pooled" not in row,
                        "Failure must have no feature or forward")
                failures[uid] = {"reason_type": row["reason_type"], "reason": row["reason"]}
    require(len(features) + len(failures) == 4207 and len(features) == bundle["coverage"]["success"]
            and len(failures) == bundle["coverage"]["failure"], "Full journal coverage differs")
    return bundle, features, failures, strata, uids


def labels_from_cells(cells):
    return np.asarray([np.nan if probability(value, missing=True, binary=True) is None
                       else probability(value, missing=True, binary=True) for value in cells], dtype=np.float64)


def index_rows(rows):
    indexed = {}
    for row in rows:
        require(row[ID] not in indexed, "Duplicate UID")
        indexed[row[ID]] = row
    return indexed


def align_controls(ordered_uids, controls):
    require(len(ordered_uids) == len(set(ordered_uids)) == len(controls)
            and set(ordered_uids) == set(controls), "Control UID set must match exactly before alignment")
    return np.asarray([[probability(controls[uid][name]) for name in TARGETS] for uid in ordered_uids], dtype=np.float64)


def load_weak_inputs(manifest_dir, cohort_path, control_path, policy):
    """Called only after qualified-feature guards. Gold targets are never read."""
    require(pin(control_path)["sha256"] == CONTROL_SHA == policy["control"]["prediction_sha256_required"], "Fixed F006 E12 prediction SHA differs")
    check_pin(cohort_path, policy["original_metadata_cohort"])
    cohort = read_json(cohort_path)
    for name, expected in cohort["manifest_pins"].items():
        check_pin(Path(manifest_dir) / name, expected)
    with (Path(manifest_dir) / "weak.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(set((ID, "group_id", "fold", "source", *TARGETS)).issubset(reader.fieldnames), "Weak columns missing")
        # Drop any Report or unrelated fields at the boundary.
        rows = [{key: row[key] for key in (ID, "group_id", "fold", "source", *TARGETS)} for row in reader]
    require(len(rows) == 4207 and len(index_rows(rows)) == 4207, "Exactly4207 weak rows required")
    projected = [{key: row[key] for key in (ID, "group_id", "fold", "source")} for row in rows]
    require(projected == cohort["requested"], "Original UID/group/fold/source mapping differs")
    require(Counter(row["fold"] for row in rows) == Counter({"0": 821, "1": 891, "2": 888, "3": 778, "4": 829}), "Fixed original5fold sizes differ")
    groups = {}
    for row in rows:
        require(row["source"] == "report_weak", "Unexpected label source")
        previous = groups.setdefault(row["group_id"], row["fold"])
        require(previous == row["fold"], "Supplied/report group crosses folds")
        labels_from_cells([row[name] for name in TARGETS])  # Enforce binary/missing contract; no soft-label coercion.
    with Path(control_path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == [ID, *TARGETS], "Fixed F006 prediction column order differs")
        controls = index_rows(list(reader))
    valid = [row for row in rows if row["fold"] == "0"]
    require(len(valid) == len(controls) == 821 and set(controls) == {row[ID] for row in valid}, "F006/control exact821 UID set differs")
    raw = align_controls([row[ID] for row in valid], controls)
    return rows, valid, raw


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


def fit_numeric(train_x, train_y, valid_x):
    """One fixed CPU fit per supported target; no validation selection or retries."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    train_x, train_y, valid_x = (np.asarray(value, dtype=np.float64) for value in (train_x, train_y, valid_x))
    require(train_x.ndim == valid_x.ndim == 2 and train_x.shape[1] == valid_x.shape[1]
            and train_y.shape == (len(train_x), len(TARGETS)) and len(train_x) > 0
            and np.isfinite(train_x).all() and np.isfinite(valid_x).all(), "Finite train/valid feature arrays required")
    require(np.all(np.isnan(train_y) | np.isin(train_y, [0, 1])), "Train labels must be observed0/1 or missing")
    with threadpool_limits(limits=1):
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(train_x)
        transformed_train, transformed_valid = scaler.transform(train_x), scaler.transform(valid_x)
        model = {"schema_version": "orthodiff_weak_head_numeric_model_v1", "targets": list(TARGETS),
                 "feature_dim": train_x.shape[1], "dtype": "float64", "seed": 20261002,
                 "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                            "variance": scaler.var_.tolist(), "training_samples": int(scaler.n_samples_seen_),
                            "fit_scope": "successful unchanged training-fold vectors only"}, "heads": []}
        sklearn_predictions = np.full((len(valid_x), len(TARGETS)), np.nan, dtype=np.float64)
        for i, target in enumerate(TARGETS):
            mask = np.isfinite(train_y[:, i])
            labels = train_y[mask, i]
            entry = {"target": target, "train_observed": int(mask.sum()),
                     "train_positive": int(np.count_nonzero(labels == 1)), "train_negative": int(np.count_nonzero(labels == 0)),
                     "train_missing": int((~mask).sum())}
            if len(np.unique(labels)) != 2:
                model["heads"].append({**entry, "status": "insufficient_training_classes", "coef": None, "intercept": None})
                continue
            estimator = LogisticRegression(solver="lbfgs", C=1.0, penalty="l2", fit_intercept=True,
                                           class_weight=None, max_iter=2000, tol=1e-4, random_state=20261002)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                estimator.fit(transformed_train[mask], labels)
            messages = [{"category": row.category.__name__, "message": str(row.message)} for row in caught]
            converged = not any(row["category"] == "ConvergenceWarning" for row in messages)
            finite = np.isfinite(estimator.coef_).all() and np.isfinite(estimator.intercept_).all()
            if not converged or not finite:
                model["heads"].append({**entry, "status": "nonconverged_or_nonfinite", "warnings": messages, "coef": None, "intercept": None})
                continue
            require(np.array_equal(estimator.classes_, [0., 1.]), "Binary class order differs")
            prediction = estimator.predict_proba(transformed_valid)[:, 1] if len(valid_x) else np.empty(0)
            require(np.isfinite(prediction).all(), "Nonfinite estimator probability")
            sklearn_predictions[:, i] = prediction
            model["heads"].append({**entry, "status": "qualified", "coef": estimator.coef_[0].tolist(),
                                   "intercept": float(estimator.intercept_[0]), "iterations": int(estimator.n_iter_[0]), "warnings": messages})
    # JSON roundtrip is the deployed model representation, independent of sklearn objects.
    reloaded = json.loads(json.dumps(model, allow_nan=False))
    independent = predict_numeric(reloaded, valid_x)
    supported = np.isfinite(sklearn_predictions)
    require(np.array_equal(np.isfinite(independent), supported), "Numeric model supported-mask changed")
    require(np.allclose(independent[supported], sklearn_predictions[supported], rtol=1e-12, atol=2e-15), "Numeric predictor differs from sklearn")
    require(np.array_equal(independent[supported].astype(np.float32).view(np.uint32),
                           sklearn_predictions[supported].astype(np.float32).view(np.uint32)), "Numeric predictor/sklearn differ after FP32 cast")
    roundtrip = {"JSON_model_roundtrip": True, "sklearn_NumPy_max_abs_difference":
                 float(np.max(np.abs(independent[supported] - sklearn_predictions[supported]))) if supported.any() else 0.,
                 "sklearn_NumPy_FP32_bitwise_equal": True, "pickle_joblib_used": False,
                 "CPU_threads": 1, "validation_labels_used_for_fit_or_selection": False}
    return reloaded, independent, roundtrip


def auc_plan(labels, scores, group_indices=None):
    y, s = np.asarray(labels), np.asarray(scores)
    mask = np.isfinite(y)
    require(np.isfinite(s[mask]).all() and np.all(np.isin(y[mask], [0, 1])), "AUC binary/missing contract differs")
    y, s = y[mask], s[mask]
    order = np.argsort(s, kind="mergesort")
    y, s = y[order], s[order]
    groups = np.arange(len(mask))[mask][order] if group_indices is None else np.asarray(group_indices)[mask][order]
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1] if len(s) else np.empty(0, dtype=int)
    return y, groups, starts


def auc_from_plan(plan, multiplicities=None):
    labels, groups, starts = plan
    if len(labels) == 0:
        return None
    weights = np.ones(len(labels), dtype=np.float64) if multiplicities is None else np.asarray(multiplicities, dtype=np.float64)[groups]
    positive = np.add.reduceat(weights * labels, starts)
    negative = np.add.reduceat(weights * (1 - labels), starts)
    total_positive, total_negative = positive.sum(), negative.sum()
    if not total_positive or not total_negative:
        return None
    favorable = np.sum(positive * (np.cumsum(negative) - negative + .5 * negative))
    return float(favorable / (total_positive * total_negative))


def metric(labels, scores):
    y, s = np.asarray(labels), np.asarray(scores)
    observed = np.isfinite(y)
    require(np.isfinite(s[observed]).all(), "Metric score nonfinite")
    positive, negative = int(np.count_nonzero(y[observed] == 1)), int(np.count_nonzero(y[observed] == 0))
    clipped = np.clip(s[observed], 1e-7, 1 - 1e-7)
    bce = -float(np.mean(y[observed] * np.log(clipped) + (1 - y[observed]) * np.log(1 - clipped))) if observed.any() else None
    return {"auc": auc_from_plan(auc_plan(y, s)), "BCE": bce, "BCE_clip_epsilon": 1e-7,
            "positive": positive, "negative": negative, "missing": int((~observed).sum()), "observed": int(observed.sum())}


def fixed_mean(values, selected):
    return sum(values[name] for name in selected) / len(selected) if all(values[name] is not None for name in selected) else None


def paired_group_bootstrap(labels, left, right, group_ids, repeats=1000, seed=20261012):
    require(repeats == 1000 and seed == 20261012, "Prospective group bootstrap settings fixed")
    groups = sorted(set(group_ids))
    mapping = {group: i for i, group in enumerate(groups)}
    indices = np.asarray([mapping[x] for x in group_ids])
    plans = {name: (auc_plan(labels[:, i], left[:, i], indices), auc_plan(labels[:, i], right[:, i], indices)) for i, name in enumerate(TARGETS)}
    generator = random.Random(seed)
    macro10, macro12 = [], []
    target_draws = {name: [] for name in TARGETS}
    for _ in range(repeats):
        draw = Counter(generator.choices(range(len(groups)), k=len(groups)))
        weights = [draw[i] for i in range(len(groups))]
        deltas = {}
        for name, pair in plans.items():
            a, b = (auc_from_plan(plan, weights) for plan in pair)
            deltas[name] = b - a if a is not None and b is not None else None
            if deltas[name] is not None:
                target_draws[name].append(deltas[name])
        for selected, values in ((FIXED10, macro10), (TARGETS, macro12)):
            value = fixed_mean(deltas, selected)
            if value is not None:
                values.append(value)

    def interval(values):
        return {"low_2.5_percent": float(np.percentile(values, 2.5)) if values else None,
                "high_97.5_percent": float(np.percentile(values, 97.5)) if values else None,
                "defined_draws": len(values), "requested_draws": repeats}

    return {"fixed10": interval(macro10), "all12_auxiliary": interval(macro12),
            "per_target": {name: interval(values) for name, values in target_draws.items()},
            "groups": len(groups), "unit": "supplied/report group; patient independence unverified", "seed": seed,
            "repeats": repeats, "generator": "Python random.Random choices", "report_only_not_a_gate": True}


def evaluate(labels, raw_control, baseline, blend, head, success_mask, group_ids, model, details):
    per_target, deltas = {}, {}
    for i, target in enumerate(TARGETS):
        raw, control, candidate = (metric(labels[:, i], scores[:, i]) for scores in (raw_control, baseline, blend))
        require(raw["auc"] == control["auc"], "Raw/rank CV baseline masked AUC must match exactly")
        left, right = control["auc"], candidate["auc"]
        deltas[target] = right - left if left is not None and right is not None else None
        paired_mask = np.asarray(success_mask) & np.isfinite(head[:, i])
        head_metric = metric(labels[paired_mask, i], head[paired_mask, i])
        paired_raw_control = metric(labels[paired_mask, i], raw_control[paired_mask, i])
        paired_control = metric(labels[paired_mask, i], baseline[paired_mask, i])
        require(paired_raw_control["auc"] == paired_control["auc"], "Paired-success raw/rank masked AUC must match exactly")
        per_target[target] = {"raw_control": raw, "CV_rank_baseline": control, "fixed20_rank_mixture": candidate,
                              "delta_auc": deltas[target], "paired_success_head": head_metric,
                              "paired_success_raw_control": paired_raw_control,
                              "paired_success_control": paired_control, "overlay": details[target], "head_status": model["heads"][i]["status"]}
    primary = fixed_mean(deltas, FIXED10)
    qualified_heads = all(entry["status"] == "qualified" for entry in model["heads"] if entry["target"] in FIXED10)
    return {"fixed10_delta": primary, "fixed10_control_auc": fixed_mean({x: per_target[x]["CV_rank_baseline"]["auc"] for x in TARGETS}, FIXED10),
            "fixed10_candidate_auc": fixed_mean({x: per_target[x]["fixed20_rank_mixture"]["auc"] for x in TARGETS}, FIXED10),
            "all12_auxiliary_delta": fixed_mean(deltas, TARGETS), "raw_rank_masked_AUC_exactly_equal_all_targets": True,
            "paired_success_raw_rank_masked_AUC_exactly_equal_all_targets": True,
            "per_target": per_target, "fixed10_heads_qualified": qualified_heads,
            "continuation_screen_passed": bool(qualified_heads and primary is not None and primary > 0),
            "continuation_automatic_start": False, "root_review_required": True,
            "BCE_and_CI_report_only": True, "rank_score_BCE_is_not_calibrated_accuracy": True,
            "paired_group_bootstrap": paired_group_bootstrap(labels, baseline, blend, group_ids),
            "limitations": ["reused weak holdout, not public-parent independent OOF", "feature coverage may be selected nonrandomly",
                            "supplied/report groups are not verified patient identity", "no Public/hidden/clinical or0.960 guarantee"]}
