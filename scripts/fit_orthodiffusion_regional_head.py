"""Manual CPU head comparison for hash-qualified fixed2x2 regional1024 pooling."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from rsna_knee import orthodiff_weak_head as H  # noqa: E402
from rsna_knee.contracts import ID, TARGETS  # noqa: E402

POLICY = ROOT / "artifacts/research/20261017-orthodiff-regional-head-preparation-v1/prospective-policy.json"
POLICY_SHA = "c6796180a208c1b56f7aeee28154d537081e34d7927874978cf4d255c4590ff7"
SCHEMA = "orthodiff_regional2x2_fixed20_head_policy_v1"
RUN_SCHEMA = "orthodiff_regional2x2_head_run_v1"
MANIFEST = ROOT / "data/manifests/v1-vmohitrao-research"
LOADER = ROOT / "src/rsna_knee/orthodiff_regional_features.py"


def import_pinned_script(path, expected, name):
    H.check_pin(path, expected)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_pins():
    paths = (Path(__file__), Path(H.__file__), ROOT / "src/rsna_knee/contracts.py", LOADER,
             ROOT / "scripts/fit_orthodiffusion_weak_head.py",
             ROOT / "scripts/fit_orthodiffusion_weak_head_fold1.py")
    return {path.name: H.pin(path) for path in paths}


def prepare_sources(policy_path, expected_policy_sha, expected_loader_sha):
    H.require(expected_policy_sha == POLICY_SHA == H.pin(policy_path)["sha256"], "Exact regional pooling policy SHA required")
    policy = H.read_json(policy_path)
    H.require(policy["schema_version"] == SCHEMA, "Regional pooling policy schema differs")
    for relative, expected in policy["immutable_old_source_pins"].items():
        H.check_pin(ROOT / relative, expected)
    H.check_pin(ROOT / policy["base_scientific_policy"]["path"], policy["base_scientific_policy"])
    H.require(len(expected_loader_sha) == 64 and all(x in "0123456789abcdef" for x in expected_loader_sha)
              and H.pin(LOADER)["sha256"] == expected_loader_sha, "Explicit qualified regional loader SHA required")
    from rsna_knee import orthodiff_regional_features as F
    base = import_pinned_script(ROOT / "scripts/fit_orthodiffusion_weak_head.py",
        policy["immutable_old_source_pins"]["scripts/fit_orthodiffusion_weak_head.py"], "fixed_pd_head_helpers")
    fold1 = import_pinned_script(ROOT / "scripts/fit_orthodiffusion_weak_head_fold1.py",
        policy["immutable_old_source_pins"]["scripts/fit_orthodiffusion_weak_head_fold1.py"], "fixed_original_fold1_loader")
    return policy, F, base, fold1


def joint_gate(fold0_delta, fold1_delta, engineering):
    return bool(engineering and fold0_delta is not None and fold1_delta is not None
                and np.isfinite(fold0_delta) and np.isfinite(fold1_delta)
                and fold0_delta > 0 and fold1_delta > 0)


def verify_new_fold0(args, pins):
    """Only a root review of THIS new recipe may authorize its original fold1."""
    H.require(args.fold0_run is not None and args.fold0_review is not None
              and args.expected_fold0_review_sha256, "New regional fold0 root review required before fold1")
    H.require(H.pin(args.fold0_review)["sha256"] == args.expected_fold0_review_sha256, "New root review SHA differs")
    review = H.read_json(args.fold0_review)
    H.require(review.get("root_review_passed") is True, "Explicit root continuation approval missing")
    run_path, evaluation_path = args.fold0_run / "run.json", args.fold0_run / "evaluation.json"
    for name in ("run.json", "evaluation.json", "numeric-model.json", "cv-rankmix20.csv", "head-valid-success.csv"):
        H.check_pin(args.fold0_run / name, review["inputs"][name])
    run, evaluation = H.read_json(run_path), H.read_json(evaluation_path)
    H.require(run.get("schema_version") == RUN_SCHEMA and run.get("status") == "complete"
              and run.get("fold") == 0 and run.get("seed") == 20261002
              and run.get("engineering_qualified") is True
              and run.get("continuation_screen_passed") is True
              and evaluation.get("continuation_screen_passed") is True
              and evaluation.get("fixed10_delta") is not None and evaluation["fixed10_delta"] > 0,
              "This regional pooling recipe's fold0 did not pass")
    H.require(run["inputs"]["policy"]["sha256"] == POLICY_SHA
              and run["inputs"]["qualified_regional_bundle"]["sha256"] == args.expected_bundle_sha256
              and run["inputs"]["source_pins"] == pins,
              "New fold0 policy/regional/source differs; old strict-PD approval cannot be reused")
    return {"approved": True, "fixed10_delta": evaluation["fixed10_delta"],
            "run": {"path": str(run_path.resolve()), **H.pin(run_path)},
            "evaluation": {"path": str(evaluation_path.resolve()), **H.pin(evaluation_path)},
            "root_review": {"path": str(args.fold0_review.resolve()), **H.pin(args.fold0_review)}}


def read_old_scores(reference, valid_uids, allow_subset=False):
    path = ROOT / reference["path"]
    H.check_pin(path, reference)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        H.require(reader.fieldnames == [ID, *TARGETS], "Strict-PD score columns differ")
        rows = H.index_rows(list(reader))
    H.require(set(rows).issubset(valid_uids) if allow_subset else set(rows) == set(valid_uids), "Strict-PD heldout UIDs differ")
    scores = np.full((len(valid_uids), len(TARGETS)), np.nan, dtype=np.float64)
    present = np.asarray([uid in rows for uid in valid_uids])
    for i, uid in enumerate(valid_uids):
        if uid in rows:
            for j, target in enumerate(TARGETS):
                cell = rows[uid][target]
                H.require(allow_subset or cell != "", "Full strict-PD mixture has missing cells")
                if cell != "":
                    value = float(cell)
                    H.require(np.isfinite(value) and 0 <= value <= 1, "Strict-PD score invalid")
                    scores[i, j] = value
    return scores, present


def secondary_comparison(labels, baseline, mixture, head, success_mask, old_mix, old_head, old_success):
    """Same full holdout and same per-target common masks; never an adoption gate."""
    result, deltas = {}, {}
    for j, target in enumerate(TARGETS):
        old_full, new_full = (H.metric(labels[:, j], values[:, j]) for values in (old_mix, mixture))
        left, right = old_full["auc"], new_full["auc"]
        deltas[target] = right - left if left is not None and right is not None else None
        common = np.asarray(success_mask) & np.asarray(old_success) & np.isfinite(head[:, j]) & np.isfinite(old_head[:, j])
        metrics = {name: H.metric(labels[common, j], values[common, j]) for name, values in
                   (("F006_rank_baseline", baseline), ("strict_PD_head", old_head), ("regional_head", head),
                    ("strict_PD_rankmix20", old_mix), ("regional_rankmix20", mixture))}
        result[target] = {"full_original_holdout_strict_PD_rankmix20": old_full,
                          "full_original_holdout_regional_rankmix20": new_full,
                          "full_holdout_delta_auc_vs_strict_PD": deltas[target],
                          "common_success_UID_count": int(common.sum()), "common_success_same_mask_metrics": metrics}
    return {"report_only_not_a_gate": True, "fixed10_full_holdout_delta_auc_vs_strict_PD": H.fixed_mean(deltas, H.FIXED10),
            "per_target": result, "pooling_only_same_success_UIDs_and_preprocessing": True}


def fit(args):
    output = args.output
    H.require(not output.exists() and not output.is_symlink(), "Use a new run directory")
    policy, F, base, fold1_loader = prepare_sources(args.policy, args.expected_policy_sha256, args.expected_loader_sha256)
    env, pins = H.environment(require_fixed=True), source_pins()
    H.require(env["Torch_imported"] is False, "CPU-only head must not import Torch")
    fold0_gate = verify_new_fold0(args, pins) if args.fold == 1 else None
    # Qualification precedes reading real weak targets or F006 probabilities.
    bundle, features, failures, strata, all_uids = F.load_qualified_regional_features(args.bundle, args.expected_bundle_sha256)
    fold = str(args.fold)
    spec = policy["folds"][fold]
    control = args.control or ROOT / spec["F006_E12_control"]["path"]
    H.check_pin(control, spec["F006_E12_control"])
    cohort_path = ROOT / policy["original_metadata_cohort"]["path"]
    descriptor = {"original_metadata_cohort": policy["original_metadata_cohort"],
                  "control": {"prediction_sha256_required": spec["F006_E12_control"]["sha256"]}}
    load = H.load_weak_inputs if args.fold == 0 else fold1_loader.load_fold1_weak_inputs
    rows, valid, raw_control = load(args.manifest, cohort_path, control, descriptor)
    H.require([row[ID] for row in rows] == all_uids and len(rows) == 4207
              and len(valid) == spec["original_valid"] and len(rows) - len(valid) == spec["original_train"],
              "Original4207 order/fold population differs")
    train = [row for row in rows if row["fold"] != fold and row[ID] in features]
    valid_success = [row for row in valid if row[ID] in features]
    H.require(train, "No successful training features")
    H.require(len(train) == spec["successful_train"] and len(valid_success) == spec["successful_valid"],
              "Regional success population differs from immutable parent")
    H.require(not {row["group_id"] for row in rows if row["fold"] != fold} & {row["group_id"] for row in valid}, "Group leakage")
    train_x = np.stack([features[row[ID]] for row in train])
    train_y = np.stack([H.labels_from_cells([row[name] for name in TARGETS]) for row in train])
    valid_x = np.stack([features[row[ID]] for row in valid_success]) if valid_success else np.empty((0, 1024), dtype=np.float64)
    H.require(train_x.shape[1] == valid_x.shape[1] == 1024, "Frozen1024 regional dimension required")
    valid_uids = [row[ID] for row in valid]
    strict_refs = policy["prior_global256_fixed20_recipe"]["observed_results"][fold]
    old_mix, _ = read_old_scores(strict_refs["full_rankmix20"], valid_uids)
    old_head, old_success = read_old_scores(strict_refs["success_head_predictions"], valid_uids, allow_subset=True)
    H.require(np.array_equal(np.asarray([row[ID] in features for row in valid]), old_success),
              "Regional and global256 heldout success UID sets differ")
    provenance = {"policy": {"path": str(args.policy.resolve()), **H.pin(args.policy)},
                  "qualified_regional_bundle": {"path": str(args.bundle.resolve()), **H.pin(args.bundle)},
                  "qualified_regional_receipt": bundle,
                  "F006_E12_control": {"path": str(control.resolve()), **H.pin(control)},
                  "cohort": {"path": str(cohort_path.resolve()), **H.pin(cohort_path)},
                  "strict_PD_comparison_pins": {key: strict_refs[key] for key in ("full_rankmix20", "success_head_predictions")},
                  "source_pins": pins, "environment": env}
    if fold0_gate is not None:
        provenance["approved_new_regional_fold0_gate"] = fold0_gate
    output.mkdir(parents=True)
    started = time.monotonic()
    H.save_json(output / "started.json", {"at_UTC": datetime.now(timezone.utc).isoformat(), "status": "running",
        "fold": args.fold, "seed": 20261002, "inputs": provenance, "actual_successful_train": len(train),
        "actual_successful_valid": len(valid_success), "parameter_search_or_retry": False})
    try:
        model, predictions, roundtrip = H.fit_numeric(train_x, train_y, valid_x)
        model["fit_context"] = {"fold": args.fold, "seed": 20261002, "inputs": provenance,
            "train_UID_order_sha256": hashlib.sha256("".join(row[ID] + "\n" for row in train).encode()).hexdigest(),
            "successful_training_groups": len({row["group_id"] for row in train})}
        H.save_json(output / "numeric-model.json", model)
        numeric = H.predict_numeric(H.read_json(output / "numeric-model.json"), valid_x)
        H.require(np.array_equal(numeric, predictions, equal_nan=True), "On-disk numeric roundtrip differs")
        roundtrip["on_disk_JSON_reload_exact_numeric_predictions"] = True
        success_mask = np.asarray([row[ID] in features for row in valid])
        head_full = np.full((len(valid), len(TARGETS)), np.nan, dtype=np.float64)
        head_full[success_mask] = numeric
        supported = {entry["target"]: entry["status"] == "qualified" for entry in model["heads"]}
        baseline, blend, details = H.rank_blend(raw_control, head_full, success_mask, supported)
        labels = np.stack([H.labels_from_cells([row[name] for name in TARGETS]) for row in valid])
        result = H.evaluate(labels, raw_control, baseline, blend, head_full, success_mask,
                            [row["group_id"] for row in valid], model, details)
        secondary = secondary_comparison(labels, baseline, blend, head_full, success_mask, old_mix, old_head, old_success)
        base.write_scores(output / "cv-rank-baseline.csv", valid_uids, baseline)
        base.write_scores(output / "cv-rankmix20.csv", valid_uids, blend)
        base.write_scores(output / "head-valid-success.csv", [row[ID] for row in valid_success], numeric, allow_unsupported=True)
        coverage = base.coverage_report(rows, features, strata)
        coverage["domains_remain_separate"] = True
        coverage["success_target_support_by_domain"] = {}
        for domain in sorted(set(strata.values())):
            selected = [row for row in rows if row[ID] in features and strata[row[ID]] == domain]
            supports = {}
            for name in TARGETS:
                values = H.labels_from_cells([row[name] for row in selected])
                supports[name] = {"positive": int(np.count_nonzero(values == 1)), "negative": int(np.count_nonzero(values == 0)),
                                  "missing": int(np.count_nonzero(np.isnan(values)))}
            coverage["success_target_support_by_domain"][domain] = {"successful_UIDs": len(selected),
                "original_fold_counts": dict(Counter(row["fold"] for row in selected)), "targets": supports}
        coverage["new_success_UIDs_vs_strict_PD_in_this_holdout"] = int(np.count_nonzero(success_mask & ~old_success))
        coverage["parent_success_lost_in_this_holdout"] = int(np.count_nonzero(old_success & ~success_mask))
        H.require(coverage["parent_success_lost_in_this_holdout"] == 0, "Parent success unexpectedly lost")
        H.save_json(output / "coverage.json", coverage)
        H.save_json(output / "feature-failures.json", {"count": len(failures), "failures": failures})
        H.save_json(output / "evaluation.json", result)
        H.save_json(output / "secondary-strict-PD-comparison.json", secondary)
        engineering = bool(result["fixed10_heads_qualified"] and roundtrip["on_disk_JSON_reload_exact_numeric_predictions"]
                           and roundtrip["sklearn_NumPy_FP32_bitwise_equal"])
        passed = bool(engineering and result["continuation_screen_passed"])
        joint = joint_gate(fold0_gate["fixed10_delta"], result["fixed10_delta"], engineering) if fold0_gate else None
        report = {"schema_version": RUN_SCHEMA, "status": "complete", "fold": args.fold, "seed": 20261002,
            "completed_at_UTC": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - started,
            "inputs": provenance, "original_train_count": spec["original_train"], "original_valid_count": len(valid),
            "successful_train_count": len(train), "successful_valid_count": len(valid_success), "failed_feature_count": len(failures),
            "head_status_counts": dict(Counter(entry["status"] for entry in model["heads"])), "numeric_model_roundtrip": roundtrip,
            "engineering_qualified": engineering, "fixed10_delta": result["fixed10_delta"], "continuation_screen_passed": passed,
            "joint_new_twofold_screen_passed": joint,
            "decision": "ROOT_REVIEW_FOR_NEW_FOLD1" if args.fold == 0 and passed else "ROOT_MAY_CONSIDER_PUBLIC_CANDIDATE" if joint else "HOLD_FIXED2X2_REGIONAL1024_FIXED20_RECIPE",
            "no_parameter_target_or_coefficient_search": True, "BCE_and_CI_report_only": True,
            "Gold_used": False, "Report_read": False, "actual_CPU_weak_head_fit_performed": True,
            "MRI_model_load_Torch_GPU_network_Save_Submit": 0, "automatic_followup": False,
            "old_global256_recipe_remains_HOLD": True, "existing_old256_adapter_qualifies_this_model": False,
            "outputs": {path.name: H.pin(path) for path in output.iterdir() if path.is_file()}}
        H.save_json(output / "run.json", report)
        return {key: report[key] for key in ("status", "fold", "fixed10_delta", "continuation_screen_passed", "joint_new_twofold_screen_passed", "decision", "automatic_followup")}
    except Exception as error:
        H.save_json(output / "failed.json", {"status": "incomplete", "error": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(), "elapsed_seconds": time.monotonic() - started,
            "partial_outputs_retained": True, "retries_or_parameter_changes": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("environment", help="Inspect existing CPU packages; never installs or fits")
    train = sub.add_parser("fit", help="Root invokes manually after actual regional and parent-parity qualification")
    train.add_argument("--fold", type=int, choices=(0, 1), required=True)
    train.add_argument("--bundle", type=Path, required=True)
    train.add_argument("--expected-bundle-sha256", required=True)
    train.add_argument("--expected-loader-sha256", required=True)
    train.add_argument("--policy", type=Path, default=POLICY)
    train.add_argument("--expected-policy-sha256", required=True)
    train.add_argument("--manifest", type=Path, default=MANIFEST)
    train.add_argument("--control", type=Path)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--fold0-run", type=Path)
    train.add_argument("--fold0-review", type=Path)
    train.add_argument("--expected-fold0-review-sha256")
    args = parser.parse_args()
    result = H.environment(require_fixed=True) if args.command == "environment" else fit(args)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
