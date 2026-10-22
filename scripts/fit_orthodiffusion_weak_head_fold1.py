"""Manual original-fold1 confirmation; reuse fixed fold0 mathematics without changing it."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# This process alone uses bounded CPU BLAS. Existing GPU training is untouched.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from rsna_knee import orthodiff_weak_head as H  # noqa: E402
from rsna_knee.contracts import ID, TARGETS  # noqa: E402

POLICY = ROOT / "artifacts/research/20261017-orthodiff-head-fold1-preparation-v1/weak-head-fold1-prospective-policy-v3.json"
CONTROL = ROOT / "artifacts/runs/f006-late-cosine-fold1-20261012-v1/epoch_012_weak.csv"
MANIFEST = ROOT / "data/manifests/v1-vmohitrao-research"

EXPECTED_POLICY_SHA = "663c99c6476fa81233399196574d715e6d3bd7a38290783059fe2d2aa4ca1787"
CONTROL_SHA = "7c052b7f16a86c9f8ee3bf0ef5f2807057acc7677f050e1337eac74b91274e7c"
BUNDLE_SHA = "525094b583c7cd821769fac56b9e64a524d5ef185a4a9eb689f91480b31ce83c"
MATH_SHA = "e1149f8e88eeba0f32418e8ee4986cd41ef5aae208a31142cd907689958d78e7"
ROOT_REVIEW = ROOT / "artifacts/research/20261017-orthodiff-head0-root-continuation-review-v1/review.json"
ROOT_REVIEW_SHA = "37d6fa77fce3a60be0421010af8ca194ee525d8094eb7a0532edf8a38e62a4d4"
require, pin, check_pin, read_json = H.require, H.pin, H.check_pin, H.read_json
labels_from_cells, index_rows, align_controls = H.labels_from_cells, H.index_rows, H.align_controls



def source_pins():
    return {path.name: H.pin(path) for path in (Path(__file__), ROOT / "src/rsna_knee/orthodiff_weak_head.py", ROOT / "src/rsna_knee/contracts.py")}


def write_scores(path, uids, scores, allow_unsupported=False):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([ID, *TARGETS])
        for uid, values in zip(uids, scores):
            H.require(len(values) == len(TARGETS), "Prediction width differs")
            cells = [float(np.float32(value)) if np.isfinite(value) else "" for value in values]
            H.require(allow_unsupported or all(value != "" for value in cells), "Full CV scores must be finite")
            writer.writerow([uid, *cells])


def coverage_report(rows, features, strata):
    folds, groups = {}, {}
    for row in rows:
        uid, fold, group = row[ID], row["fold"], row["group_id"]
        for store, key in ((folds, fold), (groups, group)):
            entry = store.setdefault(key, {"requested": 0, "success": 0, "failure": 0, "success_strata": {}})
            entry["requested"] += 1
            entry["success"] += uid in features
            entry["failure"] += uid not in features
            if uid in features:
                stratum = strata[uid]
                entry["success_strata"][stratum] = entry["success_strata"].get(stratum, 0) + 1
        groups[group]["fold"] = fold
    supports = {}
    for fold in ("0", "1", "2", "3", "4"):
        selected = [row for row in rows if row["fold"] == fold and row[ID] in features]
        supports[fold] = {}
        for name in TARGETS:
            values = H.labels_from_cells([row[name] for row in selected])
            supports[fold][name] = {"positive": int(np.count_nonzero(values == 1)), "negative": int(np.count_nonzero(values == 0)),
                                    "missing": int(np.count_nonzero(np.isnan(values)))}
    return {"by_original_fold": folds, "by_supplied_report_group": groups, "success_target_support_by_fold": supports,
            "patient_independence_verified": False}


def load_fold1_weak_inputs(manifest_dir, cohort_path, control_path, policy):
    """Called only after qualified-feature guards. Gold targets are never read."""
    require(pin(control_path)["sha256"] == CONTROL_SHA == policy["control"]["prediction_sha256_required"], "Fixed F006 originalfold1 E12 prediction SHA differs")
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
    valid = [row for row in rows if row["fold"] == "1"]
    require(len(valid) == len(controls) == 891 and set(controls) == {row[ID] for row in valid}, "F006/control exact891 UID set differs")
    raw = align_controls([row[ID] for row in valid], controls)
    return rows, valid, raw

def verify_fold0_gate(policy):
    require(pin(Path(H.__file__))["sha256"] == MATH_SHA, "Original numerical module source changed")
    reference = policy["based_on_fold0_v2"]
    require(reference["fold0_gate_approved_by_root"] is True, "Explicit fold0 gate approval required")
    run_path = ROOT / reference["fold0_run"]["path"]
    evaluation_path = ROOT / reference["fold0_evaluation"]["path"]
    check_pin(run_path, reference["fold0_run"])
    check_pin(evaluation_path, reference["fold0_evaluation"])
    require(pin(ROOT_REVIEW)["sha256"] == ROOT_REVIEW_SHA, "Exact root continuation review SHA required")
    require(policy["root_continuation_review"]["sha256"] == ROOT_REVIEW_SHA, "Policy root approval binding differs")
    root_review = read_json(ROOT_REVIEW)
    require(root_review["root_review_passed"] is True
            and root_review["inputs"]["run.json"]["sha256"] == reference["fold0_run"]["sha256"]
            and root_review["inputs"]["evaluation.json"]["sha256"] == reference["fold0_evaluation"]["sha256"],
            "Root review does not bind the same approved fold0 evidence")
    peer_reference = policy["fold0_actual_independent_review"]
    peer_path = ROOT / peer_reference["path"]
    check_pin(peer_path, peer_reference)
    peer = read_json(peer_path)
    require(peer["passed"] is True and peer["run_pin"]["sha256"] == reference["fold0_run"]["sha256"],
            "Independent fold0 audit does not bind the same approved run")
    run, evaluation = read_json(run_path), read_json(evaluation_path)
    require(run["status"] == "complete" and run["fold"] == 0 and run["seed"] == 20261002
            and run["continuation_screen_passed"] is True and evaluation["continuation_screen_passed"] is True
            and evaluation["fixed10_delta"] > 0, "The fixed original fold0 gate did not pass")
    require(run["inputs"]["qualified_bundle"]["sha256"] == BUNDLE_SHA
            and run["inputs"]["policy"]["sha256"] == H.POLICY_SHA,
            "Fold0 evidence uses a different feature bundle or policy")
    return {"approved": True, "fixed10_delta": evaluation["fixed10_delta"],
            "run": reference["fold0_run"], "evaluation": reference["fold0_evaluation"],
            "root_review": {"path": str(ROOT_REVIEW), **pin(ROOT_REVIEW)}}


def joint_twofold_gate(fold0_delta, fold1_delta, engineering_verified):
    return bool(engineering_verified and fold0_delta is not None and fold1_delta is not None
                and np.isfinite(fold0_delta) and np.isfinite(fold1_delta) and fold0_delta > 0 and fold1_delta > 0)


def fit(args):
    output = args.output
    H.require(not output.exists() and not output.is_symlink(), "Use a new run directory; prior results are immutable")
    H.require(args.expected_policy_sha256 == EXPECTED_POLICY_SHA == H.pin(args.policy)["sha256"], "Exact prospective policy-v2 SHA required")
    env = H.environment(require_fixed=True)
    H.require(env["Torch_imported"] is False, "CPU head process must not import Torch")
    policy = H.read_json(args.policy)
    H.require(policy["schema_version"] == "orthodiff_fixed_PD_weak_head_fold1_confirmation_policy_v2", "Policy v2 required")
    fold0_gate = verify_fold0_gate(policy)
    H.require(args.expected_bundle_sha256 == BUNDLE_SHA, "The SAME qualified full4207 feature bundle is mandatory")
    # Qualify every feature hash before reading any real weak target or control probability.
    bundle, features, failures, strata, all_uids = H.load_qualified_features(args.bundle, args.expected_bundle_sha256)
    cohort_path = ROOT / policy["original_metadata_cohort"]["path"]
    rows, valid, raw_control = load_fold1_weak_inputs(args.manifest, cohort_path, args.control, policy)
    H.require([row[ID] for row in rows] == all_uids, "Qualified feature cohort differs from exact original weak4207 order")
    H.require(len(rows) == 4207 and len(valid) == 891, "Original train3316/valid891 split required")
    train = [row for row in rows if row["fold"] != "1" and row[ID] in features]
    valid_success = [row for row in valid if row[ID] in features]
    H.require(len(train) == 2426 and len(valid_success) == 637, "Qualified fold1 success counts2426/637 differ")
    H.require(train, "No successful training features; no scaler/head fit permitted")
    train_groups = {row["group_id"] for row in rows if row["fold"] != "1"}
    H.require(not train_groups & {row["group_id"] for row in valid}, "Train/validation group overlap")
    train_x = np.stack([features[row[ID]] for row in train])
    train_y = np.stack([H.labels_from_cells([row[name] for name in TARGETS]) for row in train])
    valid_x = np.stack([features[row[ID]] for row in valid_success]) if valid_success else np.empty((0, 256), dtype=np.float64)
    H.require(train_x.shape[1] == valid_x.shape[1] == 256, "Frozen256 feature dimension required")
    output.mkdir(parents=True)
    started = time.monotonic()
    provenance = {"policy": {"path": str(args.policy.resolve()), **H.pin(args.policy)},
                  "qualified_bundle": {"path": str(args.bundle.resolve()), **H.pin(args.bundle)},
                  "qualification": bundle["qualification"], "qualified_feature_files": bundle["files"],
                  "F006_E12_control": {"path": str(args.control.resolve()), **H.pin(args.control)},
                  "cohort": {"path": str(cohort_path), **H.pin(cohort_path)}, "source_pins": source_pins(), "environment": env}
    provenance["approved_fold0_gate"] = fold0_gate
    H.save_json(output / "started.json", {"at_UTC": datetime.now(timezone.utc).isoformat(), "status": "running",
        "fold": 1, "seed": 20261002, "inputs": provenance, "actual_successful_train": len(train),
        "actual_successful_valid": len(valid_success), "no_parameter_search_or_retry": True,
        "pretrained_backbone_load_Torch_MRI_GPU_network_Save_Submit": 0})
    try:
        model, head_success, roundtrip = H.fit_numeric(train_x, train_y, valid_x)
        model["fit_context"] = {"fold": 1, "seed": 20261002, "inputs": provenance,
                                "train_UID_order_sha256": hashlib.sha256("".join(row[ID] + "\n" for row in train).encode()).hexdigest(),
                                "successful_training_groups": len({row["group_id"] for row in train})}
        H.save_json(output / "numeric-model.json", model)
        numeric = H.predict_numeric(H.read_json(output / "numeric-model.json"), valid_x)
        H.require(np.array_equal(numeric, head_success, equal_nan=True), "On-disk numeric model roundtrip differs")
        roundtrip["on_disk_JSON_reload_exact_numeric_predictions"] = True
        success_mask = np.asarray([row[ID] in features for row in valid])
        head_full = np.full((891, len(TARGETS)), np.nan, dtype=np.float64)
        head_full[success_mask] = numeric
        supported = {entry["target"]: entry["status"] == "qualified" for entry in model["heads"]}
        baseline, blend, details = H.rank_blend(raw_control, head_full, success_mask, supported)
        # Validation labels first enter the metric after the fixed head/model has been saved.
        labels = np.stack([H.labels_from_cells([row[name] for name in TARGETS]) for row in valid])
        result = H.evaluate(labels, raw_control, baseline, blend, head_full, success_mask,
                            [row["group_id"] for row in valid], model, details)
        result["joint_twofold_candidate_screen"] = {
            "passed": joint_twofold_gate(fold0_gate["fixed10_delta"], result["fixed10_delta"],
                                        result["fixed10_heads_qualified"] and roundtrip["on_disk_JSON_reload_exact_numeric_predictions"]),
            "fold0_delta": fold0_gate["fixed10_delta"], "fold1_delta": result["fixed10_delta"],
            "decision_fixed_before_fold1_metric": policy["joint_twofold_candidate_screen"]["fixed_before_fold1_metric"],
            "BCE_CIs_report_only": True, "no_twohead_CV_pooling": True,
            "if_false": "Hold this fixed20percent recipe; no coefficient or target search on the same results",
            "if_true": "Root may consider separate Public-candidate qualification; not automatic adoption or submission"}

        valid_uids = [row[ID] for row in valid]
        write_scores(output / "cv-rank-baseline.csv", valid_uids, baseline)
        write_scores(output / "cv-rankmix20.csv", valid_uids, blend)
        write_scores(output / "head-valid-success.csv", [row[ID] for row in valid_success], numeric, allow_unsupported=True)
        H.save_json(output / "coverage.json", coverage_report(rows, features, strata))
        H.save_json(output / "feature-failures.json", {"count": len(failures), "failures": failures})
        H.save_json(output / "evaluation.json", result)
        report = {"schema_version": "orthodiff_fixed_weak_head_fold1_run_v1", "status": "complete", "fold": 1, "seed": 20261002,
                  "started_at_UTC": H.read_json(output / "started.json")["at_UTC"], "completed_at_UTC": datetime.now(timezone.utc).isoformat(),
                  "elapsed_seconds": time.monotonic() - started, "inputs": provenance,
                  "original_train_count": 3316, "original_valid_count": 891, "successful_train_count": len(train),
                  "successful_valid_count": len(valid_success), "failed_feature_count": len(failures),
                  "head_status_counts": dict(Counter(entry["status"] for entry in model["heads"])), "numeric_model_roundtrip": roundtrip,
                  "fixed10_delta": result["fixed10_delta"], "continuation_screen_passed": result["continuation_screen_passed"],
                  "joint_candidate_screen_passed": result["joint_twofold_candidate_screen"]["passed"],
                  "followup_or_Submit_started": False, "root_review_required": True, "Gold_used": False,
                  "Report_read": False, "actual_CPU_weak_head_fit_performed": True,
                  "pretrained_backbone_load_Torch_MRI_GPU_network_Save_Submit": 0,
                  "outputs": {path.name: H.pin(path) for path in output.iterdir() if path.is_file()}}
        H.save_json(output / "run.json", report)
        return {"status": "complete", "output": str(output), "fixed10_delta": result["fixed10_delta"],
                "continuation_screen_passed": result["continuation_screen_passed"],
                "joint_candidate_screen_passed": result["joint_twofold_candidate_screen"]["passed"], "automatic_followup": False}
    except Exception as error:
        H.save_json(output / "failed.json", {"status": "incomplete", "error": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(), "elapsed_seconds": time.monotonic() - started,
            "partial_outputs_retained": True, "retries_or_parameter_changes": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("environment", help="Inspect the existing CPU environment only; no fitting")
    seal = sub.add_parser("seal-bundle", help="Bind independently qualified full4207 output hashes; no labels/fit")
    seal.add_argument("--audit", type=Path, required=True)
    seal.add_argument("--expected-audit-sha256", required=True)
    seal.add_argument("--records", type=Path, required=True)
    seal.add_argument("--summary", type=Path, required=True)
    seal.add_argument("--status", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    train = sub.add_parser("fit", help="Root manually invokes only after qualification; fixed head plus evaluation")
    train.add_argument("--bundle", type=Path, required=True)
    train.add_argument("--expected-bundle-sha256", required=True)
    train.add_argument("--policy", type=Path, default=POLICY)
    train.add_argument("--expected-policy-sha256", required=True)
    train.add_argument("--manifest", type=Path, default=MANIFEST)
    train.add_argument("--control", type=Path, default=CONTROL)
    train.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "environment":
        result = H.environment(require_fixed=True)
    elif args.command == "seal-bundle":
        result = H.seal_bundle(args.audit, args.expected_audit_sha256, args.records, args.summary, args.status, args.output)
    else:
        result = fit(args)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
