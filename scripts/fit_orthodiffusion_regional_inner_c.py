"""Manual CPU-only inner-C fit/seal; no outer2 target release or evaluation."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from rsna_knee import orthodiff_regional_inner_c as I  # noqa: E402
from rsna_knee import orthodiff_weak_head as H  # noqa: E402
from rsna_knee.contracts import ID  # noqa: E402

MANIFEST = ROOT / "data/manifests/v1-vmohitrao-research"
COHORT = ROOT / "artifacts/research/20261017-orthodiff-fullweak-preparation-v1/cohort.json"
BUNDLE = ROOT / "artifacts/research/20261017-orthodiff-regional-actual-v1/qualified-feature-bundle.json"


def progress(value):
    print(json.dumps(value, ensure_ascii=True, allow_nan=False), flush=True)


def prepare_qualified_inputs(args):
    """Feature guards finish before any weak-target view is constructed."""
    H.require(args.expected_bundle_sha256 == I.BUNDLE_SHA
              and H.pin(args.feature_bundle)["sha256"] == I.BUNDLE_SHA,
              "Exact qualified frozen regional bundle SHA required")
    loader_path = ROOT / "src/rsna_knee/orthodiff_regional_features.py"
    H.require(H.pin(loader_path)["sha256"] == I.LOADER_SHA, "Regional feature loader source differs")
    from rsna_knee import orthodiff_regional_features as F
    bundle, features, failures, strata, uids = F.load_qualified_regional_features(
        args.feature_bundle, args.expected_bundle_sha256)
    H.require(bundle["feature_dimension"] == 1024 and len(uids) == 4207
              and len(features) == 3063 and len(failures) == 1144,
              "Frozen1024 feature population differs")
    H.require(all(np.asarray(value).shape == (1024,) and np.isfinite(value).all() for value in features.values()),
              "Finite1024 features required")
    view = I.load_training_view(args.manifest_dir, args.cohort)
    H.require([r[ID] for r in H.read_json(args.cohort)["requested"]] == uids,
              "Feature/cohort UID order differs")
    train, outer2 = view["training"], view["outer2_metadata"]
    H.require(sum(r[ID] in features for r in train) == 2409
              and sum(r[ID] in features for r in outer2) == 654,
              "Frozen outertrain/outer2 feature coverage differs")
    return bundle, features, failures, strata, view


def fit_and_seal(args):
    I.prepare_sources(args.policy, args.expected_policy_sha256,
                      args.expected_producer_sha256, Path(__file__),
                      args.expected_cli_sha256, args.expected_root_review_sha256)
    environment = H.environment(require_fixed=True)
    H.require("torch" not in sys.modules, "CPU head path must not import Torch")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_paths = (Path(__file__), Path(I.__file__), Path(H.__file__),
                    ROOT / "src/rsna_knee/contracts.py", ROOT / "src/rsna_knee/orthodiff_regional_features.py")
    inputs = {"policy": {"path": str(args.policy.resolve()), **H.pin(args.policy)},
              "root_design_review": {"path": str(I.REVIEW_PATH), **H.pin(I.REVIEW_PATH)},
              "qualified_regional_bundle": {"path": str(args.feature_bundle.resolve()), **H.pin(args.feature_bundle)},
              "environment": environment,
              "source_pins": {p.name: {"path": str(p.resolve()), **H.pin(p)} for p in source_paths}}
    H.save_json(output / "started.json", {"schema_version": I.RUN_SCHEMA,
                "at_UTC": datetime.now(timezone.utc).isoformat(), "inputs": inputs,
                "outer2_target_release_or_evaluation_enabled": False})
    try:
        _, features, failures, strata, view = prepare_qualified_inputs(args)
        H.save_json(output / "training-only-label-view.json", view)
        inputs["training_only_label_view"] = H.pin(output / "training-only-label-view.json")
        inputs["cohort"] = view["cohort"]
        inputs["original_weak_source"] = view["weak_source"]
        selection = I.select_common_C(view["training"], features, progress=progress)
        H.save_json(output / "inner-C-selection.json", selection)
        model, probabilities, numerical, train, valid = I.refit_selected(
            view["training"], view["outer2_metadata"], features, selection["selected_C"])
        H.require(model["feature_dim"] == 1024 and len(train) == 2409 and len(valid) == 654,
                  "Final refit population or dimensionality differs")
        model["fit_context"] = {"recipe_schema": I.RUN_SCHEMA, "outer_heldout_fold": 2,
            "inner_original_folds": list(I.INNER_FOLDS), "selected_common_C": selection["selected_C"],
            "fixed_C_grid": list(I.GRID), "successful_train": len(train),
            "train_UID_order_sha256": I._uid_hash(train), "successful_outer2": len(valid),
            "outer2_success_UID_order_sha256": I._uid_hash(valid),
            "inner_validation_labels_used_for_C_selection": True,
            "outer2_targets_read_or_used": False, "inputs": inputs,
            "selection": H.pin(output / "inner-C-selection.json")}
        H.save_json(output / "numeric-model.json", model)
        # Re-read exact JSON representation used by later transport.
        reloaded = H.read_json(output / "numeric-model.json")
        vx = np.asarray([features[r[ID]] for r in valid], dtype=np.float64)
        reproduced = H.predict_numeric(reloaded, vx)
        H.require(np.array_equal(np.isnan(probabilities), np.isnan(reproduced))
                  and np.array_equal(probabilities[np.isfinite(probabilities)].astype(np.float32).view(np.uint32),
                                     reproduced[np.isfinite(reproduced)].astype(np.float32).view(np.uint32)),
                  "Final numeric JSON model FP32 roundtrip differs")
        I.write_predictions(output / "outer2-success-probabilities-f64.csv", [r[ID] for r in valid], probabilities, "float64")
        I.write_predictions(output / "outer2-success-probabilities-f32.csv", [r[ID] for r in valid], probabilities, "float32")
        outer2_metadata = {"schema_version": "orthodiff_regional_innerC_outer2_target_free_metadata_v1",
                          "original_count": 888, "success_count": 654, "failure_count": 234,
                          "all_rows": [{**r, "feature_success": r[ID] in features,
                                        "stratum": strata.get(r[ID]),
                                        "feature_failure": failures.get(r[ID])} for r in view["outer2_metadata"]],
                          "target_values_or_support_counts_retained": False}
        H.save_json(output / "outer2-metadata-only.json", outer2_metadata)
        outputs = {p.name: H.pin(p) for p in sorted(output.iterdir()) if p.is_file()}
        seal = {"schema_version": "orthodiff_regional_innerC_outer2_prediction_seal_v1",
                "sealed_at_UTC": datetime.now(timezone.utc).isoformat(), "inputs": inputs,
                "selected_common_C": selection["selected_C"], "outputs": outputs,
                "source_guard_and_model_engineering_passed": True,
                "outer2_target_release": False, "outer2_AUC_or_adoption_qualified": False,
                "required_before_future_evaluation": "Qualified independent F006fold2 fixedE12 control888 UID seal, independent head/seal audit and separate root target-release review"}
        H.save_json(output / "outer2-prediction-seal.json", seal)
        run = {"schema_version": I.RUN_SCHEMA, "status": "SEALED_OUTER2_PREDICTIONS_NO_TARGET_EVALUATION",
               "completed_at_UTC": datetime.now(timezone.utc).isoformat(),
               "elapsed_seconds": time.monotonic() - started, "inputs": inputs,
               "selected_common_C": selection["selected_C"], "original_train": 3319,
               "successful_train": 2409, "original_outer2": 888, "successful_outer2": 654,
               "numeric_roundtrip": numerical, "prediction_seal": H.pin(output / "outer2-prediction-seal.json"),
               "inner_validation_labels_used_for_C_selection": True,
               "legacy_numeric_roundtrip_validation_label_flag_scope": "The cloned fit itself never reads validation labels; the separate inner-C algorithm uses inner-validation labels for C selection.",
               "outer2_targets_projected_converted_retained_or_scored": False,
               "actual_training_only_CPU_weak_fit_performed": True,
               "Gold_Report_MRI_Torch_GPU_network_Save_Submit": 0,
               "new_outer2_AUC": None, "Public_candidate_adoption_qualified": False,
               "old_C1_recipes_remain_HOLD": True, "automatic_followup": False}
        H.save_json(output / "run.json", run)
        progress({"status": run["status"], "selected_C": selection["selected_C"],
                  "output": str(output), "outer2_metrics_computed": False})
        return 0
    except Exception as exc:
        if hasattr(exc, "selection") and not (output / "inner-C-selection.json").exists():
            H.save_json(output / "inner-C-selection.json", exc.selection)
        H.save_json(output / "run.json", {"schema_version": I.RUN_SCHEMA,
            "status": "HOLD" if isinstance(exc, I.HoldRecipe) else "FAILED",
            "elapsed_seconds": time.monotonic() - started, "inputs": inputs,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "outer2_target_release_or_evaluation": False, "automatic_retry": False})
        traceback.print_exc()
        return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-and-seal", action="store_true", help="Root manual execution only after source/CPU QA review")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--policy", type=Path, default=I.POLICY_PATH)
    parser.add_argument("--expected-policy-sha256")
    parser.add_argument("--expected-root-review-sha256")
    parser.add_argument("--expected-producer-sha256")
    parser.add_argument("--expected-cli-sha256")
    parser.add_argument("--feature-bundle", type=Path, default=BUNDLE)
    parser.add_argument("--expected-bundle-sha256")
    parser.add_argument("--cohort", type=Path, default=COHORT)
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST)
    args = parser.parse_args()
    if not args.fit_and_seal:
        progress({"status": "DISABLED_MANUAL_FIT_FLAG_REQUIRED", "real_data_or_fit_performed": False,
                  "outer2_target_release_or_evaluation_available": False})
        return 0
    if args.output_dir is None:
        parser.error("--output-dir must name a new run directory")
    return fit_and_seal(args)


if __name__ == "__main__":
    raise SystemExit(main())
