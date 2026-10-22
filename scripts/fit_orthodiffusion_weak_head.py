"""Manual CPU-only fixed weak-head fit/evaluation after independent full qualification."""
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

POLICY = ROOT / "artifacts/research/20261017-orthodiff-fullweak-preparation-v1/weak-head-prospective-policy-v2.json"
CONTROL = ROOT / "artifacts/runs/f006-late-cosine-fold0-20261012-v1/epoch_012_weak.csv"
MANIFEST = ROOT / "data/manifests/v1-vmohitrao-research"


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


def fit(args):
    output = args.output
    H.require(not output.exists() and not output.is_symlink(), "Use a new run directory; prior results are immutable")
    H.require(args.expected_policy_sha256 == H.POLICY_SHA == H.pin(args.policy)["sha256"], "Exact prospective policy-v2 SHA required")
    env = H.environment(require_fixed=True)
    H.require(env["Torch_imported"] is False, "CPU head process must not import Torch")
    policy = H.read_json(args.policy)
    H.require(policy["schema_version"] == "orthodiff_full4207_fixed_PD_weak_head_prospective_policy_v2", "Policy v2 required")
    # Qualify every feature hash before reading any real weak target or control probability.
    bundle, features, failures, strata, all_uids = H.load_qualified_features(args.bundle, args.expected_bundle_sha256)
    cohort_path = ROOT / policy["original_metadata_cohort"]["path"]
    rows, valid, raw_control = H.load_weak_inputs(args.manifest, cohort_path, args.control, policy)
    H.require([row[ID] for row in rows] == all_uids, "Qualified feature cohort differs from exact original weak4207 order")
    H.require(len(rows) == 4207 and len(valid) == 821, "Original train3386/valid821 split required")
    train = [row for row in rows if row["fold"] != "0" and row[ID] in features]
    valid_success = [row for row in valid if row[ID] in features]
    H.require(train, "No successful training features; no scaler/head fit permitted")
    train_groups = {row["group_id"] for row in rows if row["fold"] != "0"}
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
    H.save_json(output / "started.json", {"at_UTC": datetime.now(timezone.utc).isoformat(), "status": "running",
        "fold": 0, "seed": 20261002, "inputs": provenance, "actual_successful_train": len(train),
        "actual_successful_valid": len(valid_success), "no_parameter_search_or_retry": True,
        "pretrained_backbone_load_Torch_MRI_GPU_network_Save_Submit": 0})
    try:
        model, head_success, roundtrip = H.fit_numeric(train_x, train_y, valid_x)
        model["fit_context"] = {"fold": 0, "seed": 20261002, "inputs": provenance,
                                "train_UID_order_sha256": hashlib.sha256("".join(row[ID] + "\n" for row in train).encode()).hexdigest(),
                                "successful_training_groups": len({row["group_id"] for row in train})}
        H.save_json(output / "numeric-model.json", model)
        numeric = H.predict_numeric(H.read_json(output / "numeric-model.json"), valid_x)
        H.require(np.array_equal(numeric, head_success, equal_nan=True), "On-disk numeric model roundtrip differs")
        roundtrip["on_disk_JSON_reload_exact_numeric_predictions"] = True
        success_mask = np.asarray([row[ID] in features for row in valid])
        head_full = np.full((821, len(TARGETS)), np.nan, dtype=np.float64)
        head_full[success_mask] = numeric
        supported = {entry["target"]: entry["status"] == "qualified" for entry in model["heads"]}
        baseline, blend, details = H.rank_blend(raw_control, head_full, success_mask, supported)
        # Validation labels first enter the metric after the fixed head/model has been saved.
        labels = np.stack([H.labels_from_cells([row[name] for name in TARGETS]) for row in valid])
        result = H.evaluate(labels, raw_control, baseline, blend, head_full, success_mask,
                            [row["group_id"] for row in valid], model, details)
        valid_uids = [row[ID] for row in valid]
        write_scores(output / "cv-rank-baseline.csv", valid_uids, baseline)
        write_scores(output / "cv-rankmix20.csv", valid_uids, blend)
        write_scores(output / "head-valid-success.csv", [row[ID] for row in valid_success], numeric, allow_unsupported=True)
        H.save_json(output / "coverage.json", coverage_report(rows, features, strata))
        H.save_json(output / "feature-failures.json", {"count": len(failures), "failures": failures})
        H.save_json(output / "evaluation.json", result)
        report = {"schema_version": "orthodiff_fixed_weak_head_run_v1", "status": "complete", "fold": 0, "seed": 20261002,
                  "started_at_UTC": H.read_json(output / "started.json")["at_UTC"], "completed_at_UTC": datetime.now(timezone.utc).isoformat(),
                  "elapsed_seconds": time.monotonic() - started, "inputs": provenance,
                  "original_train_count": 3386, "original_valid_count": 821, "successful_train_count": len(train),
                  "successful_valid_count": len(valid_success), "failed_feature_count": len(failures),
                  "head_status_counts": dict(Counter(entry["status"] for entry in model["heads"])), "numeric_model_roundtrip": roundtrip,
                  "fixed10_delta": result["fixed10_delta"], "continuation_screen_passed": result["continuation_screen_passed"],
                  "followup_or_Submit_started": False, "root_review_required": True, "Gold_used": False,
                  "Report_read": False, "actual_CPU_weak_head_fit_performed": True,
                  "pretrained_backbone_load_Torch_MRI_GPU_network_Save_Submit": 0,
                  "outputs": {path.name: H.pin(path) for path in output.iterdir() if path.is_file()}}
        H.save_json(output / "run.json", report)
        return {"status": "complete", "output": str(output), "fixed10_delta": result["fixed10_delta"],
                "continuation_screen_passed": result["continuation_screen_passed"], "automatic_followup": False}
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
