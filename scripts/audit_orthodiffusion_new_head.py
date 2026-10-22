"""Independent FD256/regional1024 output replay. Never imports a fit producer."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import struct
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")
FIXED = tuple(name for name in TARGETS if name not in {"MCL", "Synovitis"})
ID = "StudyInstanceUID"
UID_SHA = "911f91fbc914213db045780675d6518a8cc4a49808599bbb74f2f247bc30eb82"
PARENT_SOURCE = "15bb2d6c7c224021f95a304b08eb9e3644164ea8373f4ae6ec1c7afc77582e1f"
PARENT_BUNDLE = "525094b583c7cd821769fac56b9e64a524d5ef185a4a9eb689f91480b31ce83c"
PARENT_AUDIT = "7fbc160b8715e69dc506b64e9187f55a2c2b67925bf6a5b0a4a95adf943c8270"
H_SHA = "e1149f8e88eeba0f32418e8ee4986cd41ef5aae208a31142cd907689958d78e7"
COHORT_SHA = "86d976f55bb67bd3a738679625c9281aead06b0c7f01d7c894ffe870ead65246"
CONTROL = {0: "8f75feced2e02584b6581c0ed124e1bdc79d370a312966fd0c976ce302f0b964",
           1: "7c052b7f16a86c9f8ee3bf0ef5f2807057acc7677f050e1337eac74b91274e7c"}
PRODUCERS = {
    "fluid": {"dimension": 256, "CLI": "fit_orthodiffusion_fluid_composite_head.py", "CLI_sha": "e1947983134687a52d1be7c338b48900276365c7b7bf8f2327bb0a43db0ef99a",
              "policy_sha": "105bda1d86b677925356249d913a69abd9c03d4c950c86fc96589e8a79787abe", "loader": "orthodiff_fluid_features.py",
              "loader_sha": "732b3349d585c9024518c36bdf08b49f86470429650aec35ae7535b8592c85ad", "run_schema": "orthodiff_fluid_availability_composite_head_run_v1",
              "bundle_key": "qualified_composite_bundle", "receipt_key": "qualified_composite_receipt", "gate_key": "approved_new_composite_fold0_gate",
              "prior_key": "strict_PD_fixed20_recipe", "new_source": "4b8237aa96f44e1a4365180bba5fd2b65782549ad0f0833a6d7e4213a01209b0"},
    "regional": {"dimension": 1024, "CLI": "fit_orthodiffusion_regional_head.py", "CLI_sha": "8982a65b7d2f4fe0c19196c1cbae8632a427513433a5de40fd2bd4e23f7066cd",
                 "policy_sha": "c6796180a208c1b56f7aeee28154d537081e34d7927874978cf4d255c4590ff7", "loader": "orthodiff_regional_features.py",
                 "loader_sha": "1363055c2d9493fd9018b8d9c0f7f0f1da52dba6277a30bbc9c865e12add855d", "run_schema": "orthodiff_regional2x2_head_run_v1",
                 "bundle_key": "qualified_regional_bundle", "receipt_key": "qualified_regional_receipt", "gate_key": "approved_new_regional_fold0_gate",
                 "prior_key": "prior_global256_fixed20_recipe", "new_source": "52842d703bcf05c762a6bf44a2c520fbdd47589239f553cc5aaabbaa7220024e"},
}
PINS = {}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def pin(path, maximum=128 * 1024**2):
    require(not Path(path).is_symlink(), "Symlink feature audit input refused")
    path = Path(path).resolve()
    require(ROOT in path.parents and path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= maximum, "Bounded regular input within this project required")
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            size += len(block)
            require(size <= maximum, "Input exceeded bound during read")
            digest.update(block)
    value = {"bytes": size, "sha256": digest.hexdigest()}
    PINS[str(path)] = value
    return value


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "Duplicate audit JSON field")
        result[key] = value
    return result


def parse(raw):
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON token")))


def data(path):
    pin(path, 4 * 1024**2)
    return parse(Path(path).read_bytes())


def checked(item):
    path = Path(item["path"])
    require(pin(path) == {key: item[key] for key in ("bytes", "sha256")}, "Positive input pin mismatch")
    return path


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def ranks(values):
    require(values and all(math.isfinite(x) for x in values), "Nonempty finite rank population required")
    counts, below, result = Counter(values), 0, {}
    for value in sorted(counts):
        result[value] = f32((below + (counts[value] + 1) / 2) / len(values))
        below += counts[value]
    return [result[value] for value in values]


def vector(record, dimension, values_key, info_key):
    values = record.get(values_key)
    require(isinstance(values, list) and len(values) == dimension and all(type(x) in (float, int) and math.isfinite(x) for x in values), "Finite exact-dimension feature values required")
    raw = struct.pack("<" + str(dimension) + "f", *values)
    require(list(struct.unpack("<" + str(dimension) + "f", raw)) == values and record.get(info_key) == {
        "shape": [1, dimension], "dtype": "torch.float32", "bytes": 4 * dimension, "sha256": hashlib.sha256(raw).hexdigest()}, "Independent feature FP32 vector/SHA mismatch")
    return values


def records(path, uids):
    pin(path)
    count = 0
    with Path(path).open("rb") as handle:
        for count, line in enumerate(handle, 1):
            require(count <= len(uids) and line.endswith(b"\n") and len(line) <= 256 * 1024, "Bounded complete feature journal required")
            record = parse(line)
            require(record.get(ID) == uids[count - 1], "Exact unique feature UID order required")
            yield record, line
    require(count == len(uids), "Feature journal incomplete")


def raw_files(bundle, audit, phase):
    require(set(bundle["files"]) == {"records", "summary", "status"}, "Feature raw3 required")
    for name, item in bundle["files"].items():
        checked(item)
        require({key: item[key] for key in ("bytes", "sha256")} == audit[name + "_pin"], "Bundle/raw/audit pin mismatch")
    summary, status = (data(bundle["files"][name]["path"]) for name in ("summary", "status"))
    require(summary.get("complete") is status.get("complete") is True and "error" not in summary
            and summary.get("phase") == status.get("phase") == phase and summary["coverage"] == status["coverage"] == audit["coverage"]
            and summary["journal"]["sha256"] == status["journal"]["sha256"] == bundle["files"]["records"]["sha256"], "Complete feature terminal/raw3 qualification required")
    return summary


def feature_inputs(inputs, spec):
    bundle_ref = inputs[spec["bundle_key"]]
    bundle = data(checked(bundle_ref))
    require(bundle == inputs[spec["receipt_key"]] and bundle.get("qualified") is True and bundle.get("requested_count") == 4207
            and bundle.get("UID_order_sha256") == UID_SHA, "Same qualified feature bundle as fit required")
    parent_ref = bundle["parent"]
    require(parent_ref["sha256"] == PARENT_BUNDLE, "Immutable parent bundle required")
    parent = data(checked(parent_ref))
    audit = data(checked(parent["qualification"]))
    require(parent["qualification"]["sha256"] == PARENT_AUDIT and parent.get("schema_version") == "orthodiff_fullweak4207_qualified_feature_bundle_v1"
            and parent.get("qualified") is True and parent.get("source_sha256") == PARENT_SOURCE
            and audit.get("passed") is audit.get("head_features_qualified") is True and audit["prior32_comparison"]["passed"] is True,
            "Exact actual-qualified parent source required")
    parent_summary = raw_files(parent, audit, "full4207_finished")
    uids = parent_summary["requested_UIDs"]
    require(len(uids) == len(set(uids)) == 4207 and hashlib.sha256("".join(uid + "\n" for uid in uids).encode()).hexdigest() == UID_SHA, "Original4207 UID order required")
    features, failures, strata, parent_canonical = {}, {}, {}, {}
    for record, _ in records(parent["files"]["records"]["path"], uids):
        uid = record[ID]
        if record.get("status") == "success":
            features[uid] = vector(record, 256, "pooled_float32_values", "pooled")
            strata[uid] = record["metadata"]["stratum"]
        else:
            require(record.get("status") == "data_failure" and record.get("feature_forward_calls") == 0 and record.get("fallback_features") is False
                    and not any(key in record for key in ("pooled", "pooled_float32_values", "mid_2", "normalized_input")), "Parent failed features fabricated")
            failures[uid] = {"reason_type": record["reason_type"], "reason": record["reason"]}
            parent_canonical[uid] = hashlib.sha256((json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()
    require(len(features) == 3063 and len(failures) == 1144, "Parent3063/1144 required")
    if spec["dimension"] == 256:
        require(bundle.get("schema_version") == "orthodiff_parent_fluid_composite_bundle_v1" and bundle["child_source_sha256"] == spec["new_source"]
                and bundle["parent_source_sha256"] == PARENT_SOURCE and bundle["parent_qualification"] == parent["qualification"]
                and bundle["merge_proof"]["parent_success_features_byte_exact"] is True and bundle["merge_proof"]["parent_success_overwrite_count"] == 0,
                "New fluid composite schema/source required")
        child = data(checked(bundle["child_qualification"]))
        require(child.get("schema_version") == "orthodiff_fluid1080_actual_qualification_v1" and child.get("passed") is child.get("child_features_qualified") is True
                and child["source_sha256"] == spec["new_source"] and child["parent_bundle_pin"] == pin(parent_ref["path"])
                and child["parent_qualification_pin"]["sha256"] == PARENT_AUDIT
                and child["parent_success_unchanged"] is True and child["parent_success_overwrite_count"] == 0
                and child["confirmed_PD"] is False and child["route"] == "fluid_sensitive_fallback", "Actual non-overwriting child qualification required")
        summary = raw_files({"files": bundle["child_files"]}, child, "fluid1080_finished")
        child_uids = summary["requested_UIDs"]
        require(len(child_uids) == len(set(child_uids)) == 1080 and set(child_uids).issubset(failures)
                and hashlib.sha256("".join(uid + "\n" for uid in child_uids).encode()).hexdigest() == child["UID_order_sha256"] == "2fb8f009f2f2372ea0e460c174bfb373172f2477a356cd8bee783e0d0792bf7e", "Fixed disjoint child1080 UID order required")
        added = 0
        for record, _ in records(bundle["child_files"]["records"]["path"], child_uids):
            uid = record[ID]
            require(uid not in features, "Parent successful feature overwritten")
            if record.get("status") == "success":
                detail = record["metadata"]
                require(detail["route"] == "fluid_sensitive_fallback" and detail["confirmed_PD"] is False
                        and detail["stratum"] in {"fluid_sensitive_FS", "fluid_sensitive_nonFS"}
                        and record["all_RNG_restored"] is True and record["forward_counts"] == {"denoiser": 1, "decoder_blocks": 0, "final_out": 0}, "Child success route/forward contract")
                features[uid] = vector(record, 256, "pooled_float32_values", "pooled")
                strata[uid] = detail["stratum"]
                del failures[uid]
                added += 1
            else:
                require(record.get("status") == "data_failure" and record.get("route") == "fluid_sensitive_fallback" and record.get("confirmed_PD") is False
                        and record.get("feature_forward_calls") == 0 and record.get("fallback_features") is False
                        and record.get("parent_record_canonical_sha256") == parent_canonical[uid]
                        and not any(key in record for key in ("pooled", "pooled_float32_values", "mid_2", "normalized_input")), "Child failure fabricated features/lost parent identity")
        require(len(features) == bundle["combined_success_count"] == 3063 + added and added == child["coverage"]["success"]
                and len(failures) == bundle["combined_failure_count"], "Composite final success/failure count differs")
    else:
        require(bundle.get("schema_version") == "orthodiff_regional2x2_qualified_feature_bundle_v1" and bundle.get("source_sha256") == spec["new_source"]
                and bundle.get("feature_dimension") == 1024 and bundle.get("FD_availability_mixed") is False, "Standalone regional1024 qualification required")
        regional = data(checked(bundle["qualification"]))
        require(regional.get("schema_version") == "orthodiff_regional2x2_actual_qualification_v1"
                and regional.get("passed") is regional.get("head_features_qualified") is True and regional["source_sha256"] == spec["new_source"]
                and regional["parent_comparison"]["passed"] is True and regional["parent_comparison"]["successful_scientific_parity"] == 3063
                and regional["parent_comparison"]["failure_records_byte_exact"] == 1144 and regional["FD_availability_mixed"] is False
                and regional["parent_bundle_pin"] == pin(parent_ref["path"]), "Actual regional parent parity required")
        summary = raw_files(bundle, regional, "regional4207_finished")
        require(summary["requested_UIDs"] == uids, "Regional4207 order differs")
        science = ("metadata", "noise_seed", "normalized_input", "normalization_min_max_float32", "noisy_input", "mid_2", "pooled", "pooled_float32_values", "forward_counts", "all_RNG_restored")
        old_records = records(parent["files"]["records"]["path"], uids)
        for previous, current in zip_longest(old_records, records(bundle["files"]["records"]["path"], uids)):
            require(previous is not None and current is not None, "Regional/parent journal lengths differ")
            old, old_line = previous
            record, line = current
            uid = record[ID]
            require(old[ID] == uid and record["status"] == old["status"], "Regional success mask differs")
            if uid in features:
                require(all(record[k] == old[k] for k in science)
                        and struct.pack("<256f", *record["pooled_float32_values"]) == struct.pack("<256f", *features[uid])
                        and record["parent_scientific_parity_passed"] is True and record["parent_record_line_sha256"] == hashlib.sha256(old_line).hexdigest(), "Regional parent scientific/global bits differ")
                features[uid] = vector(record, 1024, "regional_2x2_float32_values", "regional_2x2")
            else:
                require(line == old_line, "Inherited regional failure byte mismatch")
        require(len(features) == 3063 and len(failures) == 1144, "Regional same3063/1144 partition required")
    require(set(features).isdisjoint(failures) and set(features) | set(failures) == set(uids), "Full feature partition differs")
    return bundle, features, failures, strata, uids


def read_scores(path, ordered=None, missing=False):
    pin(path)
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        require(next(reader) == [ID, *TARGETS], "Prediction CSV columns mismatch")
        rows = list(reader)
    uids = [row[0] for row in rows]
    require(len(uids) == len(set(uids)) and all(len(row) == 13 for row in rows), "Prediction row shape/UID duplicate")
    require(ordered is None or uids == ordered, "Prediction CSV UID order differs")
    values = np.asarray([[float(value) if value != "" else float("nan") for value in row[1:]] for row in rows], dtype=np.float64)
    require(np.all(np.isfinite(values) | (np.isnan(values) if missing else False)) and np.all((values[np.isfinite(values)] >= 0) & (values[np.isfinite(values)] <= 1)), "Prediction probability/missing contract")
    return uids, values


def label(value):
    if value.strip().casefold() in {"", "nan", "na", "null"}:
        return float("nan")
    parsed = float(value)
    require(parsed in (0., 1.), "Weak observed binary labels only; missing/soft coercion forbidden")
    return parsed


def auc_plan(labels, scores, groups):
    buckets = defaultdict(list)
    for y, s, group in zip(labels, scores, groups):
        if not math.isnan(y):
            require(y in (0, 1) and math.isfinite(s), "Masked AUC contract")
            buckets[float(s)].append((int(y), group))
    return [buckets[key] for key in sorted(buckets)]


def auc(plan, multiplicities=None):
    below_negative = favorable = positive_total = 0.
    for bucket in plan:
        positive = negative = 0.
        for y, group in bucket:
            weight = 1 if multiplicities is None else multiplicities[group]
            if y:
                positive += weight
            else:
                negative += weight
        favorable += positive * (below_negative + .5 * negative)
        below_negative += negative
        positive_total += positive
    return favorable / (positive_total * below_negative) if positive_total and below_negative else None


def metric(y, scores):
    scores, y = np.asarray(scores), np.asarray(y)
    observed = [(float(a), min(max(float(b), 1e-7), 1 - 1e-7)) for a, b in zip(y, scores) if not math.isnan(a)]
    bce = None
    if observed:
        if scores.dtype == np.float32:
            clipped = np.clip(scores[np.isfinite(y)], 1e-7, 1 - 1e-7)
            bce = -math.fsum(a * float(b) + (1 - a) * float(c) for (a, _), b, c in zip(observed, np.log(clipped), np.log(1 - clipped))) / len(observed)
        else:
            bce = -math.fsum(a * math.log(b) + (1 - a) * math.log(1 - b) for a, b in observed) / len(observed)
    return {"auc": auc(auc_plan(y, scores, list(range(len(y))))), "positive": sum(a == 1 for a, _ in observed), "negative": sum(a == 0 for a, _ in observed),
            "observed": len(observed), "missing": len(y) - len(observed), "BCE": bce}


def compare_metric(actual, saved):
    for key in ("auc", "positive", "negative", "observed", "missing"):
        require(actual[key] == saved[key], "Metric/mask/support differs: " + key)
    require(actual["BCE"] is saved["BCE"] if actual["BCE"] is None or saved["BCE"] is None else abs(actual["BCE"] - saved["BCE"]) <= 2e-12, "Secondary BCE differs")


def percentile(values, percent):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * percent / 100
    lower, upper = math.floor(index), math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def scalar_predict(model, valid_x):
    dimension = model["feature_dim"]
    require(valid_x.ndim == 2 and valid_x.shape[1] == dimension and np.isfinite(valid_x).all(), "Prediction vector shape/finite contract")
    scaler = model["scaler"]
    standardized = (valid_x - np.asarray(scaler["mean"], dtype=np.float64)) / np.asarray(scaler["scale"], dtype=np.float64)
    predictions = np.full((len(valid_x), 12), np.nan, dtype=np.float64)
    for column, entry in enumerate(model["heads"]):
        if entry["status"] != "qualified":
            require(entry["coef"] is entry["intercept"] is None, "Unqualified head must have no numeric coefficients")
            continue
        require(len(entry["coef"]) == dimension and all(type(v) in (float, int) and math.isfinite(v) for v in [entry["intercept"], *entry["coef"]]), "Finite full-dimensional head coefficients required")
        for index, row in enumerate(standardized):
            z = math.fsum(float(a) * float(b) for a, b in zip(row, entry["coef"])) + entry["intercept"]
            predictions[index, column] = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
    return predictions


def training_moments(train_x):
    mean = train_x.sum(axis=0, dtype=np.float64) / len(train_x)
    centered = train_x - mean
    variance = ((centered * centered).sum(axis=0, dtype=np.float64) - centered.sum(axis=0, dtype=np.float64) ** 2 / len(train_x)) / len(train_x)
    epsilon = np.finfo(np.float64).eps
    constant_error_bound = len(train_x) * epsilon * variance + (len(train_x) * mean * epsilon) ** 2
    scale = np.where(variance <= constant_error_bound, 1., np.sqrt(variance))
    return mean, variance, scale


def rank_mix(raw, predictions, success_mask, heads):
    mask = np.asarray(success_mask, dtype=bool)
    require(raw.shape == (len(mask), 12) and predictions.shape == (int(mask.sum()), 12), "Rank mixture UID/target shapes differ")
    baseline = np.asarray([ranks(raw[:, column].tolist()) for column in range(12)], dtype=np.float32).T.copy()
    mixture = baseline.copy()
    for column, target in enumerate(TARGETS):
        values = predictions[:, column].astype(np.float32)
        if target not in FIXED or heads[column]["status"] != "qualified" or len(values) < 2 or len(set(values.tolist())) < 2:
            continue
        require(np.isfinite(values).all(), "Supported head is nonfinite")
        for index, value in zip(np.flatnonzero(mask), ranks(values.tolist())):
            mixture[index, column] = f32(f32(f32(.8) * float(baseline[index, column])) + f32(f32(.2) * value))
    require(np.array_equal(mixture[~mask].view(np.uint32), baseline[~mask].view(np.uint32))
            and np.array_equal(mixture[:, [1, 8]].view(np.uint32), baseline[:, [1, 8]].view(np.uint32)), "Missing/MCL/Synovitis baseline bits changed")
    return baseline, mixture


def initial_gate(run, recipe, fold):
    spec = PRODUCERS[recipe]
    require(run.get("status") == "complete" and run.get("schema_version") == spec["run_schema"] and run.get("fold") == fold
            and run.get("seed") == 20261002 and run.get("engineering_qualified") is True,
            "Complete engineering-qualified exact new-recipe run required before weak labels")
    return spec


def source_inputs(run, recipe, fold):
    spec = initial_gate(run, recipe, fold)
    inputs = run["inputs"]
    require(inputs["policy"]["sha256"] == spec["policy_sha"] and inputs["F006_E12_control"]["sha256"] == CONTROL[fold]
            and inputs["source_pins"][spec["CLI"]]["sha256"] == spec["CLI_sha"]
            and inputs["source_pins"]["orthodiff_weak_head.py"]["sha256"] == H_SHA
            and inputs["source_pins"][spec["loader"]]["sha256"] == spec["loader_sha"], "Fixed producer/H/loader/policy/control pins differ")
    expected_names = {spec["CLI"], spec["loader"], "orthodiff_weak_head.py", "contracts.py", "fit_orthodiffusion_weak_head.py", "fit_orthodiffusion_weak_head_fold1.py"}
    if recipe == "fluid":
        expected_names.add("orthodiff_fluid_fallback.py")
    require(set(inputs["source_pins"]) == expected_names, "Exact fitting/loader source closure required")
    for name, expected in inputs["source_pins"].items():
        relative = ("scripts/" if name.startswith("fit_") else "src/rsna_knee/") + name
        require(pin(ROOT / relative) == expected, "Fitting/loader source changed since run")
    for key in ("policy", spec["bundle_key"], "cohort", "F006_E12_control"):
        checked(inputs[key])
    policy = data(inputs["policy"]["path"])
    references = policy[spec["prior_key"]]["observed_results"][str(fold)]
    require(inputs["strict_PD_comparison_pins"] == {key: references[key] for key in ("full_rankmix20", "success_head_predictions")}, "Secondary immutable score provenance differs")
    for relative, expected in policy["immutable_old_source_pins"].items():
        require(pin(ROOT / relative) == {key: expected[key] for key in ("bytes", "sha256")}, "Old frozen source changed")
    require(inputs["cohort"]["sha256"] == COHORT_SHA and inputs["environment"]["Torch_imported"] is False
            and inputs["environment"]["versions"] == {"python": "3.9.18", "numpy": "1.22.3", "scipy": "1.7.3", "scikit-learn": "1.6.1"}, "Fixed weak metadata/environment differs")
    gate = None
    if fold == 1:
        gate = inputs[spec["gate_key"]]
        require(gate["approved"] is True and gate["fixed10_delta"] > 0, "New recipe fold0 approved positive gate required")
        review = data(checked(gate["root_review"]))
        prior = data(checked(gate["run"]))
        prior_eval = data(checked(gate["evaluation"]))
        initial_gate(prior, recipe, 0)
        require(review.get("root_review_passed") is True and prior["fixed10_delta"] == prior_eval["fixed10_delta"] == gate["fixed10_delta"]
                and prior["continuation_screen_passed"] is True
                and prior["inputs"]["policy"]["sha256"] == spec["policy_sha"]
                and prior["inputs"][spec["bundle_key"]] == inputs[spec["bundle_key"]]
                and prior["inputs"]["source_pins"] == inputs["source_pins"], "Other recipe/bundle/source approval cannot authorize this fold1")
        for name in ("run.json", "evaluation.json", "numeric-model.json", "cv-rankmix20.csv", "head-valid-success.csv"):
            expected = review["inputs"][name]
            require(pin(Path(gate["run"]["path"]).parent / name) == {key: expected[key] for key in ("bytes", "sha256")}, "New fold0 reviewed output pin differs")
    return spec, inputs, policy, gate


def bootstrap(y, baseline, mixture, group_ids, saved):
    groups = sorted(set(group_ids))
    mapping = {group: i for i, group in enumerate(groups)}
    indices = [mapping[group] for group in group_ids]
    plans = {name: (auc_plan(y[:, column], baseline[:, column], indices), auc_plan(y[:, column], mixture[:, column], indices)) for column, name in enumerate(TARGETS)}
    rng, per_target, macro10, macro12 = random.Random(20261012), {name: [] for name in TARGETS}, [], []
    for _ in range(1000):
        multiplicities = Counter(rng.choices(range(len(groups)), k=len(groups)))
        delta = {}
        for name, (left, right) in plans.items():
            a, b = auc(left, multiplicities), auc(right, multiplicities)
            delta[name] = b - a if a is not None and b is not None else None
            if delta[name] is not None:
                per_target[name].append(delta[name])
        for names, values in ((FIXED, macro10), (TARGETS, macro12)):
            if all(delta[name] is not None for name in names):
                values.append(sum(delta[name] for name in names) / len(names))
    for values, item in [(macro10, saved["fixed10"]), (macro12, saved["all12_auxiliary"]), *[(values, saved["per_target"][name]) for name, values in per_target.items()]]:
        require(item["defined_draws"] == len(values) and item["requested_draws"] == 1000, "Bootstrap defined/missing draws differ")
        for percent, key in ((2.5, "low_2.5_percent"), (97.5, "high_97.5_percent")):
            expected = percentile(values, percent)
            require(expected is item[key] if expected is None or item[key] is None else abs(expected - item[key]) <= 1e-15, "Paired supplied-group bootstrap interval differs")
    require(saved["groups"] == len(groups) and saved["seed"] == 20261012 and saved["repeats"] == 1000 and saved["report_only_not_a_gate"] is True,
            "Bootstrap group/seed/report-only scope differs")


def secondary_audit(run_dir, policy, spec, fold, valid_uids, y, raw_baseline, mixture, head_full, success_mask):
    references = policy[spec["prior_key"]]["observed_results"][str(fold)]
    old_uids, old_mix = read_scores(checked({**references["full_rankmix20"], "path": str(ROOT / references["full_rankmix20"]["path"])}), valid_uids)
    head_uids, old_values = read_scores(checked({**references["success_head_predictions"], "path": str(ROOT / references["success_head_predictions"]["path"])}), missing=True)
    del old_uids
    old_head = np.full_like(head_full, np.nan)
    index = {uid: i for i, uid in enumerate(valid_uids)}
    require(set(head_uids).issubset(index), "Secondary original head UID mismatch")
    for uid, values in zip(head_uids, old_values):
        old_head[index[uid]] = values
    old_success = np.asarray([uid in set(head_uids) for uid in valid_uids])
    require(not np.any(old_success & ~success_mask), "Old successful feature UID lost")
    saved = data(run_dir / "secondary-strict-PD-comparison.json")
    new_name = "composite" if spec["dimension"] == 256 else "regional"
    deltas = {}
    for column, target in enumerate(TARGETS):
        item = saved["per_target"][target]
        left, right = metric(y[:, column], old_mix[:, column]), metric(y[:, column], mixture[:, column])
        compare_metric(left, item["full_original_holdout_strict_PD_rankmix20"])
        compare_metric(right, item["full_original_holdout_" + new_name + "_rankmix20"])
        deltas[target] = right["auc"] - left["auc"] if left["auc"] is not None and right["auc"] is not None else None
        require(deltas[target] == item["full_holdout_delta_auc_vs_strict_PD"], "Secondary full-mask delta differs")
        common = success_mask & old_success & np.isfinite(head_full[:, column]) & np.isfinite(old_head[:, column])
        require(item["common_success_UID_count"] == int(common.sum()), "Secondary common-success UID count differs")
        for name, values in (("F006_rank_baseline", raw_baseline), ("strict_PD_head", old_head), (new_name + "_head", head_full),
                             ("strict_PD_rankmix20", old_mix), (new_name + "_rankmix20", mixture)):
            compare_metric(metric(y[common, column], values[common, column]), item["common_success_same_mask_metrics"][name])
    fixed = sum(deltas[name] for name in FIXED) / 10 if all(deltas[name] is not None for name in FIXED) else None
    require(fixed == saved["fixed10_full_holdout_delta_auc_vs_strict_PD"] and saved["report_only_not_a_gate"] is True, "Secondary macro/scope differs")


def audit(run_dir, expected_run_sha, recipe, fold):
    require(pin(run_dir / "run.json")["sha256"] == expected_run_sha, "Explicit expected actual run SHA required")
    run = data(run_dir / "run.json")
    spec, inputs, policy, gate = source_inputs(run, recipe, fold)
    required_outputs = {"numeric-model.json", "evaluation.json", "cv-rank-baseline.csv", "cv-rankmix20.csv", "head-valid-success.csv", "coverage.json", "feature-failures.json", "secondary-strict-PD-comparison.json"}
    require(required_outputs.issubset(run["outputs"]), "Actual head output set incomplete")
    for name, expected in run["outputs"].items():
        require(Path(name).name == name and pin(run_dir / name) == expected, "Actual output digest differs")
    require(run["numeric_model_roundtrip"]["on_disk_JSON_reload_exact_numeric_predictions"] is True
            and run["numeric_model_roundtrip"]["sklearn_NumPy_FP32_bitwise_equal"] is True
            and run["numeric_model_roundtrip"]["CPU_threads"] == 1, "Completed numeric roundtrip/CPU engineering receipts required")
    _, vectors, failures, strata, uids = feature_inputs(inputs, spec)
    cohort = data(inputs["cohort"]["path"])
    require(cohort["Gold_UID_overlap"] == cohort["Gold_supplied_group_overlap"] == 0, "Reserved Gold structural overlap")
    weak = ROOT / "data/manifests/v1-vmohitrao-research/weak.csv"
    require(pin(weak) == cohort["manifest_pins"]["weak.csv"], "Actual weak-label input pin differs")
    # Only explicitly approved weak fields are retained. Gold.csv/Report not read.
    with weak.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        names = [ID, "group_id", "fold", "source", *TARGETS]
        selected = [header.index(name) for name in names]
        rows = [{name: values[index] for name, index in zip(names, selected)} for values in reader]
    require([{key: row[key] for key in (ID, "group_id", "fold", "source")} for row in rows] == cohort["requested"]
            and [row[ID] for row in rows] == uids, "Original weak UID/group/fold mapping differs")
    train = [row for row in rows if row["fold"] != str(fold) and row[ID] in vectors]
    valid = [row for row in rows if row["fold"] == str(fold)]
    valid_success = [row for row in valid if row[ID] in vectors]
    expected_population = policy["folds"][str(fold)]
    require(len(valid) == expected_population["original_valid"] == (821 if fold == 0 else 891)
            and 4207 - len(valid) == expected_population["original_train"]
            and run["original_valid_count"] == len(valid) and run["original_train_count"] == 4207 - len(valid)
            and len(train) == run["successful_train_count"] and len(valid_success) == run["successful_valid_count"]
            and len(failures) == run["failed_feature_count"], "Actual train/valid success population differs")
    require(not {row["group_id"] for row in rows if row["fold"] != str(fold)} & {row["group_id"] for row in valid}, "Whole supplied-group boundary leakage")
    model = data(run_dir / "numeric-model.json")
    require(model.get("schema_version") == "orthodiff_weak_head_numeric_model_v1" and model["feature_dim"] == spec["dimension"]
            and model["dtype"] == "float64" and model["seed"] == 20261002 and model["targets"] == list(TARGETS) and len(model["heads"]) == 12
            and model["fit_context"]["inputs"] == inputs and model["fit_context"]["fold"] == fold and model["fit_context"]["seed"] == 20261002,
            "Actual numeric model/provenance differs")
    require(model["fit_context"]["train_UID_order_sha256"] == hashlib.sha256("".join(row[ID] + "\n" for row in train).encode()).hexdigest()
            and model["fit_context"]["successful_training_groups"] == len({row["group_id"] for row in train}), "Training vector/group provenance differs")
    train_x = np.asarray([vectors[row[ID]] for row in train], dtype=np.float64)
    mean, variance, scale = training_moments(train_x)
    scaler = model["scaler"]
    require(scaler["training_samples"] == len(train) and len(scaler["mean"]) == len(scaler["scale"]) == len(scaler["variance"]) == spec["dimension"]
            and np.allclose(mean, scaler["mean"], rtol=1e-13, atol=1e-14) and np.allclose(variance, scaler["variance"], rtol=1e-13, atol=1e-14)
            and np.allclose(scale, scaler["scale"], rtol=1e-13, atol=1e-14), "Training-only scaler population moments differ")
    train_y = np.asarray([[label(row[target]) for target in TARGETS] for row in train])
    for column, entry in enumerate(model["heads"]):
        y = train_y[:, column]
        require(entry["target"] == TARGETS[column] and (entry["train_positive"], entry["train_negative"], entry["train_missing"], entry["train_observed"])
                == (int((y == 1).sum()), int((y == 0).sum()), int(np.isnan(y).sum()), int(np.isfinite(y).sum())), "Per-target observed/soft/missing training mask differs")
        if entry["status"] == "qualified":
            require(entry["warnings"] == [] and 0 < entry["iterations"] <= 2000 and entry["train_positive"] > 0 and entry["train_negative"] > 0, "Qualified fit warning/iterations/support differ")
        else:
            require(entry["target"] not in FIXED and entry["coef"] is entry["intercept"] is None, "Engineering fixed10 target unqualified")
    require(run["head_status_counts"] == dict(Counter(entry["status"] for entry in model["heads"])), "Run head qualification counts differ")
    valid_x = np.asarray([vectors[row[ID]] for row in valid_success], dtype=np.float64)
    predictions = scalar_predict(model, valid_x)
    _, saved_head = read_scores(run_dir / "head-valid-success.csv", [row[ID] for row in valid_success], missing=True)
    require(np.array_equal(np.isfinite(predictions), np.isfinite(saved_head)) and np.array_equal(predictions[np.isfinite(predictions)].astype(np.float32).view(np.uint32),
            saved_head[np.isfinite(saved_head)].astype(np.float32).view(np.uint32)), "Independent fsum/sigmoid head FP32 prediction bits differ")
    valid_uids = [row[ID] for row in valid]
    control_uids, control_values = read_scores(inputs["F006_E12_control"]["path"])
    require(set(control_uids) == set(valid_uids) and len(control_uids) == len(valid), "Fold-specific F006 control UID alignment differs")
    control = dict(zip(control_uids, control_values))
    raw = np.asarray([control[uid] for uid in valid_uids])
    success_mask = np.asarray([uid in vectors for uid in valid_uids])
    baseline, mixture = rank_mix(raw, predictions, success_mask, model["heads"])
    for filename, expected in (("cv-rank-baseline.csv", baseline), ("cv-rankmix20.csv", mixture)):
        _, actual = read_scores(run_dir / filename, valid_uids)
        require(np.array_equal(actual.astype(np.float32).view(np.uint32), expected.view(np.uint32)), "Independent full-N rank/FP32 blend bits differ")
    head_full = np.full((len(valid), 12), np.nan, dtype=np.float64)
    head_full[success_mask] = predictions
    evaluation = data(run_dir / "evaluation.json")
    y = np.asarray([[label(row[target]) for target in TARGETS] for row in valid])
    deltas, per_target = {}, {}
    for column, target in enumerate(TARGETS):
        saved = evaluation["per_target"][target]
        left, right = metric(y[:, column], baseline[:, column]), metric(y[:, column], mixture[:, column])
        require(left["auc"] == metric(y[:, column], raw[:, column])["auc"], "Masked raw/rank AUC equivalence differs")
        for values, key in ((raw, "raw_control"), (baseline, "CV_rank_baseline"), (mixture, "fixed20_rank_mixture")):
            compare_metric(metric(y[:, column], values[:, column]), saved[key])
        pair = success_mask & np.isfinite(head_full[:, column])
        for values, key in ((head_full, "paired_success_head"), (raw, "paired_success_raw_control"), (baseline, "paired_success_control")):
            compare_metric(metric(y[pair, column], values[pair, column]), saved[key])
        deltas[target] = right["auc"] - left["auc"] if left["auc"] is not None and right["auc"] is not None else None
        require(deltas[target] == saved["delta_auc"], "Per-target AUC delta differs")
        per_target[target] = {"control_auc": left["auc"], "candidate_auc": right["auc"], "delta": deltas[target], "observed": left["observed"], "missing": left["missing"]}
    primary = sum(deltas[name] for name in FIXED) / 10 if all(deltas[name] is not None for name in FIXED) else None
    require(primary == evaluation["fixed10_delta"] == run["fixed10_delta"]
            and evaluation["fixed10_control_auc"] == sum(per_target[name]["control_auc"] for name in FIXED) / 10
            and evaluation["fixed10_candidate_auc"] == sum(per_target[name]["candidate_auc"] for name in FIXED) / 10
            and evaluation["fixed10_heads_qualified"] is True
            and evaluation["continuation_screen_passed"] is run["continuation_screen_passed"] is bool(primary is not None and primary > 0), "Primary fixed10 mean/screen differs")
    joint = bool(gate["fixed10_delta"] > 0 and primary is not None and primary > 0) if gate else None
    require(run["joint_new_twofold_screen_passed"] is joint, "New separate-fold joint positive gate differs")
    bootstrap(y, baseline, mixture, [row["group_id"] for row in valid], evaluation["paired_group_bootstrap"])
    secondary_audit(run_dir, policy, spec, fold, valid_uids, y, baseline, mixture, head_full, success_mask)
    require(data(run_dir / "feature-failures.json") == {"count": len(failures), "failures": failures}, "Typed feature failure/missing report differs")
    coverage = data(run_dir / "coverage.json")
    require(coverage["parent_success_lost_in_this_holdout"] == 0 and coverage["domains_remain_separate"] is True, "Coverage parent preservation/domain scope differs")
    expected_folds, expected_groups = {}, {}
    for row in rows:
        uid, fold_id, group = row[ID], row["fold"], row["group_id"]
        for store, key in ((expected_folds, fold_id), (expected_groups, group)):
            entry = store.setdefault(key, {"requested": 0, "success": 0, "failure": 0, "success_strata": {}})
            entry["requested"] += 1
            entry["success"] += uid in vectors
            entry["failure"] += uid not in vectors
            if uid in vectors:
                entry["success_strata"][strata[uid]] = entry["success_strata"].get(strata[uid], 0) + 1
        expected_groups[group]["fold"] = fold_id
    require(coverage["by_original_fold"] == expected_folds and coverage["by_supplied_report_group"] == expected_groups
            and coverage["patient_independence_verified"] is False, "Whole-fold/group coverage/strata counts differ")
    for fold_id in ("0", "1", "2", "3", "4"):
        selected = [row for row in rows if row["fold"] == fold_id and row[ID] in vectors]
        for target in TARGETS:
            labels = [label(row[target]) for row in selected]
            require(coverage["success_target_support_by_fold"][fold_id][target] == {
                "positive": sum(value == 1 for value in labels), "negative": sum(value == 0 for value in labels), "missing": sum(math.isnan(value) for value in labels)}, "Original fold target support/missing counts differ")
    for domain, saved in coverage["success_target_support_by_domain"].items():
        selected = [row for row in rows if row[ID] in vectors and strata[row[ID]] == domain]
        require(saved["successful_UIDs"] == len(selected) and saved["original_fold_counts"] == dict(Counter(row["fold"] for row in selected)), "Domain support UID/fold counts differ")
        for target in TARGETS:
            labels = [label(row[target]) for row in selected]
            require(saved["targets"][target] == {"positive": sum(v == 1 for v in labels), "negative": sum(v == 0 for v in labels), "missing": sum(math.isnan(v) for v in labels)}, "Domain observed/negative/missing support differs")
    return {"schema_version": "orthodiff_new_head_actual_independent_audit_v1", "passed": True, "recipe": recipe, "fold": fold,
            "at_UTC": datetime.now(timezone.utc).isoformat(), "run_pin": pin(run_dir / "run.json"), "model_pin": pin(run_dir / "numeric-model.json"),
            "producer_CLI_sha256": spec["CLI_sha"], "policy_sha256": spec["policy_sha"], "feature_dimension": spec["dimension"],
            "train_success": len(train), "valid_success": len(valid_success), "valid_full": len(valid), "feature_failures": len(failures),
            "scaler_train_only_mean_max_abs": float(np.max(np.abs(mean - scaler["mean"]))),
            "scaler_train_only_variance_max_abs": float(np.max(np.abs(variance - scaler["variance"]))),
            "independent_JSON_prediction_FP32_values_bitmatched": int(np.isfinite(predictions).sum()),
            "baseline_FP32_cells_bitmatched": len(valid) * 12, "blend_FP32_cells_bitmatched": len(valid) * 12,
            "fixed10_control_auc": sum(per_target[name]["control_auc"] for name in FIXED) / 10,
            "fixed10_candidate_auc": sum(per_target[name]["candidate_auc"] for name in FIXED) / 10, "fixed10_delta": primary,
            "per_target": per_target, "paired_group_bootstrap1000_recomputed": True, "fixed10_CI": evaluation["paired_group_bootstrap"]["fixed10"],
            "secondary_same_full_and_common_observed_masks_recomputed": True, "new_joint_screen_passed": joint,
            "accuracy_Public_or_adoption_claimed": False, "input_pins": PINS,
            "limits": ["No estimator refit: coefficients re-predicted independently; fixed fit source and warning/iteration/support receipts audited.",
                       "Reused weak holdout, selected feature coverage and supplied/report groups do not establish patient independence or Public/clinical gain."],
            "local_activity": {"approved_weak_columns_read": True, "Gold_Report_read": False, "parent_H_fit_predict_rank_evaluate_called": False,
                               "refit_hyperparameter_search_MRI_pretrained_model_GPU_network_Save_Submit": 0}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--expected-run-sha256", required=True)
    parser.add_argument("--recipe", choices=tuple(PRODUCERS), required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "New independent output audit path required")
    pin(Path(__file__))
    try:
        result = audit(args.run, args.expected_run_sha256, args.recipe, args.fold)
    except Exception as error:
        result = {"schema_version": "orthodiff_new_head_actual_independent_audit_v1", "passed": False,
                  "recipe": args.recipe, "fold": args.fold, "error_type": type(error).__name__, "error": str(error), "input_pins": PINS}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result.get(key) for key in ("passed", "recipe", "fold", "fixed10_delta", "error")}))
    raise SystemExit(not result["passed"])


if __name__ == "__main__":
    main()
