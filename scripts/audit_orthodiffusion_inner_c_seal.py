"""Independent target-free engineering audit; no fitting or outer2 metrics.

Inner models are not exported. This audit can check selection-table arithmetic
and recorded eligibility but cannot independently recompute its inner AUCs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ID = "StudyInstanceUID"
TARGETS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")
FIXED10 = tuple(x for x in TARGETS if x not in ("MCL", "Synovitis"))
GRID = (0.001, 0.01, 0.1, 1.)
FOLDS = (0, 1, 3, 4)
META = (ID, "group_id", "fold", "source")
RUN_SCHEMA = "orthodiff_regional_inner4_C_outer2_sealed_run_v1"
POLICY_SHA = "dd26d15c5b27c624c08eeee84e65d828b4c0df572854dc301af149954a1c2c6f"
COHORT_SHA = "86d976f55bb67bd3a738679625c9281aead06b0c7f01d7c894ffe870ead65246"
BUNDLE_SHA = "d892f0f94a6b98a172c6644564e8769ad8395b3a2223093313b95062123340d5"
H_SHA = "e1149f8e88eeba0f32418e8ee4986cd41ef5aae208a31142cd907689958d78e7"
LOADER_SHA = "1363055c2d9493fd9018b8d9c0f7f0f1da52dba6277a30bbc9c865e12add855d"
CONTRACTS_SHA = "aa246fd1a49ecbe88f1795f49904ef9675f119c935e99b9ffd5eee8220cbc71b"
PRODUCER_SHA = "fa4cfb206b90db9a3c5b8e23badadb251f500d22fa56a3a45106fb85e42e079f"
CLI_SHA = "160cc6391522dc1a811aa9d071b00b91dfaba3f66c50a08db66fadc42352c215"
REVIEW_SHA = "b496cdd2363b9d708fa5a1570833fa40473a0be7b8f2f58d13a21f289de33ea5"
ENV = {"python": "3.9.18", "numpy": "1.22.3", "scipy": "1.7.3", "scikit-learn": "1.6.1"}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path):
    digest, length = hashlib.sha256(), 0
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            length += len(block)
            digest.update(block)
    return {"bytes": length, "sha256": digest.hexdigest()}


def bound(path, expected):
    need(pin(path) == {k: expected[k] for k in ("bytes", "sha256")}, "File SHA/size differs: " + str(path))


def read(path):
    return json.loads(Path(path).read_bytes())


def uid_hash(rows):
    return hashlib.sha256("".join(r[ID] + "\n" for r in rows).encode()).hexdigest()


def f32bits(value):
    return struct.pack("<f", float(value))


def validate_header(run):
    need(run.get("schema_version") == RUN_SCHEMA
         and run.get("status") == "SEALED_OUTER2_PREDICTIONS_NO_TARGET_EVALUATION",
         "Only completed target-free prediction seal is auditable")
    need(run.get("outer2_targets_projected_converted_retained_or_scored") is False
         and "new_outer2_AUC" in run and run["new_outer2_AUC"] is None
         and run.get("Public_candidate_adoption_qualified") is False
         and run.get("inner_validation_labels_used_for_C_selection") is True
         and run.get("original_train") == 3319 and run.get("successful_train") == 2409
         and run.get("original_outer2") == 888 and run.get("successful_outer2") == 654
         and run.get("actual_training_only_CPU_weak_fit_performed") is True
         and run.get("Gold_Report_MRI_Torch_GPU_network_Save_Submit") == 0
         and run.get("old_C1_recipes_remain_HOLD") is True and run.get("automatic_followup") is False,
         "Target-free/adaptive-selection scope differs")
    numerical = run["numeric_roundtrip"]
    need(numerical["JSON_model_roundtrip"] is True and numerical["sklearn_NumPy_FP32_bitwise_equal"] is True
         and numerical["CPU_threads"] == 1 and numerical["pickle_joblib_used"] is False,
         "Recorded final export qualification differs")


def validate_view(view, cohort_rows):
    need(view["schema_version"] == "orthodiff_regional_inner4_training_only_labels_v1"
         and view["target_order"] == list(TARGETS)
         and view["outer2_target_cells_projected_converted_or_retained"] is False,
         "Training-only view contract differs")
    train = [r for r in cohort_rows if r["fold"] != "2"]
    outer = [r for r in cohort_rows if r["fold"] == "2"]
    actual = view["training"]
    need(len(actual) == len(train) and [{k: r[k] for k in META} for r in actual] == train
         and view["outer2_metadata"] == outer
         and all(set(r) == set(META) for r in outer), "View UID/group/fold partition differs")
    for row in actual:
        need(set(row) == {*META, "targets"} and len(row["targets"]) == 12
             and all(x is None or (type(x) in (int, float) and x in (0, 1)) for x in row["targets"]),
             "Only observed0/1 or missing training targets allowed")
    need(not {r["group_id"] for r in actual} & {r["group_id"] for r in outer}, "Outer2 group leakage")
    return actual, outer


def validate_selection(selection, training, features):
    table, expected_support = selection["table"], []
    need(selection["schema_version"] == "orthodiff_regional_inner4_C_selection_v1"
         and selection["common_C_all12"] is True
         and selection["outer2_target_or_prediction_used_for_selection"] is False
         and selection["inner_validation_labels_used_for_C_selection"] is True
         and selection["tie_tolerance"] == 1e-12 and selection["tie_preference"] == "smallest_C",
         "Selection algorithm/scope differs")
    need([r["C"] for r in table] == list(GRID), "Exact four-C table required")
    success = [r for r in training if r[ID] in features]
    for fold in FOLDS:
        train = [r for r in success if int(r["fold"]) != fold]
        valid = [r for r in success if int(r["fold"]) == fold]
        support = {}
        for i, target in enumerate(TARGETS):
            if target not in FIXED10:
                continue
            a = [r["targets"][i] for r in train if r["targets"][i] is not None]
            b = [r["targets"][i] for r in valid if r["targets"][i] is not None]
            need(set(a) == set(b) == {0., 1.}, "Undefined fixed10 inner support")
            support[target] = {"train_observed": len(a), "valid_observed": len(b),
                               "valid_observed_UID_hash": uid_hash([r for r in valid if r["targets"][i] is not None])}
        expected_support.append({"inner_fold": fold, "train_count": len(train), "valid_count": len(valid),
                                 "train_UID_hash": uid_hash(train), "valid_UID_hash": uid_hash(valid),
                                 "fixed10_support": support})
    need(selection["support"] == expected_support, "C selection masks/support differ")
    eligible = []
    for entry in table:
        inner = entry["inner"]
        need([r["fold"] for r in inner] == list(FOLDS[:len(inner)]), "Recorded innerfold order differs")
        for row in inner:
            auc = row["AUC"]
            need(set(auc) == set(FIXED10)
                 and all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in auc.values()),
                 "Ten finite standalone AUCs required")
            macro = math.fsum(auc[t] for t in FIXED10) / 10
            need(abs(macro - row["macro10"]) <= 2e-15, "Inner macro10 arithmetic differs")
            numerical = row["numeric_roundtrip"]
            need(numerical["JSON_model_roundtrip"] is True
                 and numerical["sklearn_NumPy_FP32_bitwise_equal"] is True
                 and numerical["CPU_threads"] == 1
                 and numerical["pickle_joblib_used"] is False,
                 "Recorded inner export qualification differs")
            need(set(row["auxiliary_status"]) == {"MCL", "Synovitis"}, "Auxiliary head receipt differs")
        if entry["eligible"] is True:
            need(len(inner) == 4 and entry["failure_reason"] is None, "Eligible C missing four innerfolds")
            score = math.fsum(r["macro10"] for r in inner) / 4
            need(type(entry["score"]) in (int, float) and math.isfinite(entry["score"])
                 and abs(score - entry["score"]) <= 2e-15, "Forty-AUC objective arithmetic differs")
            eligible.append(entry)
        else:
            reason = entry["failure_reason"]
            need(entry["eligible"] is False and entry["score"] is None and len(inner) < 4
                 and reason["inner_fold"] == FOLDS[len(inner)]
                 and reason["type"] in ("ValueError", "FloatingPointError", "HoldRecipe")
                 and isinstance(reason["message"], str) and reason["message"],
                 "Recorded C ineligibility receipt differs")
    need(eligible, "No eligible C can produce a qualified final seal")
    maximum = max(r["score"] for r in eligible)
    chosen = min(r["C"] for r in eligible if maximum - r["score"] <= 1e-12)
    need(selection["selected_C"] == chosen, "Maximum-score/tie-selected C differs")
    return {"selected_C": chosen, "eligible_C_count": len(eligible),
            "arithmetic_and_recorded_masks_qualified": True,
            "inner_predictions_or_models_independently_replayed": False,
            "recorded_inner_AUCs_and_fit_eligibility_independently_verified": False}


def verify_scaler_and_heads(model, training, features, dimension):
    need(model["schema_version"] == "orthodiff_weak_head_numeric_model_v1"
         and model["targets"] == list(TARGETS) and model["feature_dim"] == dimension
         and model["dtype"] == "float64" and model["seed"] == 20261002,
         "Final numeric model schema/dimension differs")
    rows = [r for r in training if r[ID] in features]
    scaler = model["scaler"]
    mean, scale, variance = (scaler[k] for k in ("mean", "scale", "variance"))
    need(all(len(x) == dimension for x in (mean, scale, variance))
         and scaler["training_samples"] == len(rows)
         and all(type(x) in (float, int) and math.isfinite(x) for v in (mean, scale, variance) for x in v)
         and all(x > 0 for x in scale) and all(x >= 0 for x in variance), "Final scaler contract differs")
    vectors = [features[r[ID]] for r in rows]
    max_mean, max_variance, max_scale = 0., 0., 0.
    n, eps = len(rows), sys.float_info.epsilon
    for j, column in enumerate(zip(*vectors)):
        expected_mean = math.fsum(float(x) for x in column) / n
        expected_variance = math.fsum((float(x) - expected_mean) ** 2 for x in column) / n
        constant = expected_variance <= n * eps * expected_variance + (n * expected_mean * eps) ** 2
        expected_scale = 1. if constant else math.sqrt(expected_variance)
        for actual, expected, label in ((mean[j], expected_mean, "mean"),
                                        (variance[j], expected_variance, "variance"),
                                        (scale[j], expected_scale, "scale")):
            need(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), "Training-only scaler " + label + " differs")
        max_mean = max(max_mean, abs(mean[j] - expected_mean))
        max_variance = max(max_variance, abs(variance[j] - expected_variance))
        max_scale = max(max_scale, abs(scale[j] - expected_scale))
    need([e["target"] for e in model["heads"]] == list(TARGETS), "Final12 head order differs")
    qualified = []
    for i, head in enumerate(model["heads"]):
        observed = [r["targets"][i] for r in rows if r["targets"][i] is not None]
        need(head["train_observed"] == len(observed)
             and head["train_missing"] == len(rows) - len(observed)
             and head["train_positive"] == observed.count(1.)
             and head["train_negative"] == observed.count(0.), "Final target-specific training mask differs")
        if head["status"] == "qualified":
            need(len(head["coef"]) == dimension
                 and all(type(x) in (float, int) and math.isfinite(x) for x in head["coef"])
                 and type(head["intercept"]) in (float, int) and math.isfinite(head["intercept"]),
                 "Finite float64 coefficient/intercept required")
            if head["target"] in FIXED10:
                need(not head.get("warnings"), "Fixed10 final warning forbidden")
            qualified.append(head["target"])
        else:
            need(head["target"] not in FIXED10 and head["coef"] is None and head["intercept"] is None
                 and head["status"] in ("insufficient_training_classes", "nonconverged_or_nonfinite"),
                 "Fixed10 final head unqualified or auxiliary null contract differs")
    return {"train_success": n, "dimension": dimension, "qualified_targets": qualified,
            "mean_max_abs_difference": max_mean, "variance_max_abs_difference": max_variance,
            "scale_max_abs_difference": max_scale, "moment_tolerance": {"relative": 1e-12, "absolute": 1e-12},
            "algorithm": "stdlib math.fsum population moments on successful outertrain only"}


def independent_predictions(model, rows, features):
    """No H.predict_numeric, NumPy dot, estimator import or fitting."""
    mean, scale = model["scaler"]["mean"], model["scaler"]["scale"]
    output = []
    for row in rows:
        values = features[row[ID]]
        need(len(values) == model["feature_dim"], "Independent feature dimension differs")
        standardized = [(float(x) - m) / s for x, m, s in zip(values, mean, scale)]
        need(len(standardized) == model["feature_dim"] and all(math.isfinite(x) for x in standardized),
             "Independent input dimension/finite contract differs")
        result = []
        for head in model["heads"]:
            if head["status"] != "qualified":
                result.append(None)
                continue
            z = math.fsum(x * c for x, c in zip(standardized, head["coef"])) + head["intercept"]
            probability = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
            need(math.isfinite(probability), "Independent probability nonfinite")
            result.append(probability)
        output.append(result)
    return output


def compare_csvs(f64_path, f32_path, rows, independent):
    values = []
    for path in (f64_path, f32_path):
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            need(next(reader) == [ID, *TARGETS], "Prediction CSV header differs")
            records = list(reader)
        need(len(records) == len(rows) and [r[0] for r in records] == [r[ID] for r in rows]
             and all(len(r) == 13 for r in records), "Exact successful UID order/CSV shape differs")
        values.append([[None if x == "" else float(x) for x in r[1:]] for r in records])
    finite_cells, unsupported, max_abs = 0, 0, 0.
    for expected, a, b in zip(independent, values[0], values[1]):
        for ref, p64, p32 in zip(expected, a, b):
            if ref is None:
                need(p64 is p32 is None, "Unsupported target must stay absent, never zero-filled")
                unsupported += 1
                continue
            need(p64 is not None and p32 is not None and math.isfinite(p64) and math.isfinite(p32)
                 and 0 <= p64 <= 1 and 0 <= p32 <= 1, "Qualified probability missing/nonfinite")
            need(f32bits(ref) == f32bits(p64) == f32bits(p32), "Independent FP32 probability bit differs")
            need(abs(ref - p64) <= 1e-12, "Independent float64 probability differs beyond reporting tolerance")
            max_abs = max(max_abs, abs(ref - p64))
            finite_cells += 1
    return {"total_cells": len(rows) * 12, "finite_FP32_cells_bitwise_equal": finite_cells,
            "unsupported_cells_mask_equal": unsupported, "float64_max_abs_difference": max_abs,
            "FP32_mismatch_count": 0, "H_predict_numeric_or_sklearn_predict_used": False,
            "algorithm": "stdlib standardization, math.fsum dot and stable sigmoid; both CSVs independently parsed"}


def audit(run_dir, expected_run_sha):
    directory = Path(run_dir)
    run_path = directory / "run.json"
    need(type(expected_run_sha) is str and len(expected_run_sha) == 64
         and pin(run_path)["sha256"] == expected_run_sha, "Explicit completed run SHA required")
    run = read(run_path)
    validate_header(run)
    inputs = run["inputs"]
    for key, sha in (("policy", POLICY_SHA), ("qualified_regional_bundle", BUNDLE_SHA),
                     ("root_design_review", REVIEW_SHA), ("cohort", COHORT_SHA)):
        receipt = inputs[key]
        need(receipt["sha256"] == sha, key + " required pin differs")
        bound(receipt["path"], receipt)
    policy = read(inputs["policy"]["path"])
    for relative, expected in policy["immutable_before_pins"].items():
        bound(ROOT / relative, expected)
    need(inputs["environment"]["versions"] == ENV and inputs["environment"]["Torch_imported"] is False
         and inputs["environment"]["new_packages_installed"] is False, "Recorded CPU environment differs")
    source_list = (("orthodiff_weak_head.py", H_SHA), ("orthodiff_regional_features.py", LOADER_SHA),
                      ("orthodiff_regional_inner_c.py", PRODUCER_SHA),
                      ("fit_orthodiffusion_regional_inner_c.py", CLI_SHA), ("contracts.py", CONTRACTS_SHA))
    need(set(inputs["source_pins"]) == {x[0] for x in source_list}, "Exact source inventory differs")
    for name, sha in source_list:
        receipt = inputs["source_pins"][name]
        need(receipt["sha256"] == sha, "Frozen producer/loader source differs: " + name)
        bound(receipt["path"], receipt)
    seal_path = directory / "outer2-prediction-seal.json"
    bound(seal_path, run["prediction_seal"])
    seal = read(seal_path)
    need(seal["schema_version"] == "orthodiff_regional_innerC_outer2_prediction_seal_v1"
         and seal["inputs"] == inputs and seal["outer2_target_release"] is False
         and seal["source_guard_and_model_engineering_passed"] is True
         and seal["outer2_AUC_or_adoption_qualified"] is False, "Prediction seal scope/binding differs")
    names = {"started.json", "training-only-label-view.json", "inner-C-selection.json", "numeric-model.json",
             "outer2-success-probabilities-f64.csv", "outer2-success-probabilities-f32.csv", "outer2-metadata-only.json"}
    need(set(seal["outputs"]) == names, "Exact target-free output inventory differs")
    for name, receipt in seal["outputs"].items():
        need(Path(name).name == name, "Unsafe output path")
        bound(directory / name, receipt)
    started = read(directory / "started.json")
    need(started["schema_version"] == RUN_SCHEMA and started["outer2_target_release_or_evaluation_enabled"] is False
         and all(started["inputs"][k] == inputs[k] for k in started["inputs"]), "Started/sealed input chain differs")
    cohort = read(inputs["cohort"]["path"])
    need(cohort["Gold_UID_overlap"] == cohort["Gold_supplied_group_overlap"] == 0, "Pinned no-Gold metadata differs")
    need({k: inputs["original_weak_source"][k] for k in ("bytes", "sha256")} == cohort["manifest_pins"]["weak.csv"],
         "Original weak source receipt differs; raw weak CSV is intentionally not opened here")
    view = read(directory / "training-only-label-view.json")
    bound(directory / "training-only-label-view.json", inputs["training_only_label_view"])
    need(view["cohort"] == inputs["cohort"] and view["weak_source"] == inputs["original_weak_source"], "View source pins differ")
    training, outer2 = validate_view(view, cohort["requested"])
    need(len(training) == 3319 and len(outer2) == 888, "Original outertrain/holdout populations differ")
    # Import only the pinned numeric feature loader after source checks. Never
    # call the producer, H.predict_numeric, metric/evaluator or any fit function.
    from rsna_knee import orthodiff_regional_features as F
    _, features, failures, strata, uids = F.load_qualified_regional_features(inputs["qualified_regional_bundle"]["path"], BUNDLE_SHA)
    need(uids == [r[ID] for r in cohort["requested"]]
         and len(features) == 3063 and len(failures) == 1144, "Qualified frozen feature partition differs")
    train_success = [r for r in training if r[ID] in features]
    outer_success = [r for r in outer2 if r[ID] in features]
    need(len(train_success) == 2409 and len(outer_success) == 654, "Expected final train/outer2 feature counts differ")
    metadata = read(directory / "outer2-metadata-only.json")
    expected_rows = [{**r, "feature_success": r[ID] in features, "stratum": strata.get(r[ID]),
                      "feature_failure": failures.get(r[ID])} for r in outer2]
    need(metadata["all_rows"] == expected_rows and metadata["original_count"] == 888
         and metadata["success_count"] == 654 and metadata["failure_count"] == 234
         and metadata["target_values_or_support_counts_retained"] is False, "All888 UID/failure metadata differs")
    selection = read(directory / "inner-C-selection.json")
    selection_check = validate_selection(selection, training, features)
    chosen = selection_check["selected_C"]
    need(chosen == run["selected_common_C"] == seal["selected_common_C"], "Selected C chain differs")
    model = read(directory / "numeric-model.json")
    context = model["fit_context"]
    need(model["selected_common_C"] == chosen == context["selected_common_C"]
         and context["recipe_schema"] == RUN_SCHEMA and context["outer_heldout_fold"] == 2
         and context["inner_original_folds"] == list(FOLDS) and context["fixed_C_grid"] == list(GRID)
         and context["inputs"] == inputs and context["outer2_targets_read_or_used"] is False
         and context["inner_validation_labels_used_for_C_selection"] is True
         and context["successful_train"] == 2409 and context["successful_outer2"] == 654
         and context["train_UID_order_sha256"] == uid_hash(train_success)
         and context["outer2_success_UID_order_sha256"] == uid_hash(outer_success)
         and context["selection"] == seal["outputs"]["inner-C-selection.json"], "Final model/selection provenance differs")
    scaler = verify_scaler_and_heads(model, training, features, 1024)
    prediction = independent_predictions(model, outer_success, features)
    prediction_check = compare_csvs(directory / "outer2-success-probabilities-f64.csv",
                                    directory / "outer2-success-probabilities-f32.csv", outer_success, prediction)
    need(prediction_check["total_cells"] == 7848 and "torch" not in sys.modules, "Prediction cell count or Torch import differs")
    return {"schema_version": "orthodiff_innerC_target_free_actual_engineering_audit_v1",
            "passed": True, "target_free_prediction_seal_engineering_qualified": True,
            "run": {"path": str(run_path.resolve()), **pin(run_path)},
            "prediction_seal": {"path": str(seal_path.resolve()), **pin(seal_path)},
            "inputs": inputs, "output_pins": seal["outputs"],
            "selection_table": selection_check, "training_only_scaler_and_heads": scaler,
            "independent_predictions": prediction_check,
            "outer2_partition": {"all_UIDs": 888, "success": 654, "failure": 234, "targets_accessed": False},
            "C_refit_inner_refit_or_parameter_change": 0,
            "outer2_metric_or_target_access_Gold_Report_MRI_Torch_GPU_network": 0,
            "Public_candidate_adoption_qualified": False,
            "limitations": ["Inner model/prediction exports do not exist: recorded inner AUCs and convergence/eligibility cannot be independently replayed; table/mask arithmetic is checked only.",
                            "Weak training labels are checked from the sealed training-only view; raw shared weak CSV is not reopened, preserving the heldout-target boundary.",
                            "This engineering audit does not release outer2 targets, qualify its AUC, validate Public improvement or authorize a Submit."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-run-sha256", required=True)
    parser.add_argument("--expected-auditor-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    need(not args.output.exists(), "Use a new audit output path")
    try:
        need(pin(__file__)["sha256"] == args.expected_auditor_sha256, "Explicit auditor source SHA differs")
        result = audit(args.run_dir, args.expected_run_sha256)
        exit_code = 0
    except Exception as exc:
        result = {"schema_version": "orthodiff_innerC_target_free_actual_engineering_audit_v1",
                  "passed": False, "target_free_prediction_seal_engineering_qualified": False,
                  "error": {"type": type(exc).__name__, "message": str(exc)},
                  "Public_candidate_adoption_qualified": False}
        exit_code = 2
    result["auditor"] = {"path": str(Path(__file__).resolve()), **pin(__file__)}
    try:
        numpy_version = importlib.metadata.version("numpy")
    except importlib.metadata.PackageNotFoundError:
        numpy_version = None
    result["auditor_runtime"] = {"executable": sys.executable,
                                 "python": sys.version.split()[0],
                                 "installed_NumPy_version": numpy_version,
                                 "producer_environment_is_a_separate_input_receipt": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"passed": result["passed"], "output": str(args.output), **pin(args.output)}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
