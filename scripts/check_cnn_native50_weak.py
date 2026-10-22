"""Additional fixed family50 screen, only after a positive fixed-E12 primary.

Standard library, immutable existing auditors, no model/Torch/MRI/network work.
This screen never grants serving or formal-submission qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import re
import struct
import types
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY = "artifacts/research/20261021-CNN-native50-weak-prospective-v1/policy.json"
POLICY_SHA = "7337c3669df2f25771aebb306feb4b9f3746a263de0fd7c74b835a6f8aa45384"
SCHEMA = "CNN_native50_paired_weak_screen_v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def pin(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def frozen_primary(candidate):
    """Load exact stdlib auditor definitions, never invoke their audit entry."""
    require(type(candidate) is str and candidate in ("tiny", "small"), "One Tiny or Small candidate required")
    # Policy JSON is inert; its hash is checked before any endpoint path is used.
    import json

    raw = (ROOT / POLICY).read_bytes()
    require(pin(raw)["sha256"] == POLICY_SHA, "Exact prospective native50 policy")
    policy = json.loads(raw)
    ref = policy["primaries"][candidate]["source"]
    source = (ROOT / ref["path"]).read_bytes()
    require(pin(source) == {k: ref[k] for k in ("bytes", "sha256")}, "Exact immutable primary auditor")
    module = types.ModuleType("native50_frozen_" + candidate)
    module.__file__ = str(ROOT / ref["path"])
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    require(module.ROOT.resolve() == ROOT.resolve(), "Same workspace primary definitions")
    policy = module.strict_json(raw)
    require(policy["schema_version"] == "CNN_native50_paired_weak_prospective_v1"
            and policy["endpoint"]["coefficient"] == .5 and policy["endpoint"]["fixed_epoch"] == 12
            and policy["endpoint"]["excluded_targets"] == list(module.EXCLUDED)
            and policy["order"]["primary_result_required_first"] is True
            and policy["order"]["fold1_requires_positive_samecandidate_fold0screen"] is True,
            "Fixed endpoint and primary-first order")
    return module, policy


def artifact_ref(C, path, expected_sha):
    require(type(expected_sha) is str and re.fullmatch("[0-9a-f]{64}", expected_sha), "Explicit full artifact SHA required")
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    require(not path.is_symlink() and path.resolve().is_relative_to((ROOT / "artifacts/research").resolve())
            and path.suffix == ".json", "Research JSON evidence only")
    relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    C.workspace_path(ROOT, relative)
    return relative, expected_sha


def allowed_primary_artifacts(C, policy, fold):
    candidate = policy["candidate"]
    paths = {C.POLICY, policy["master_policy"]["path"], policy["pixel_index"]["path"], *policy["input_pins"]}
    paths.update(candidate[k]["path"] for k in ("config", "provenance", "resource_gate"))
    paths.update("src/rsna_knee/" + n for n in candidate["source_pins"])
    control = policy["controls"][str(fold)]
    paths.update(control["directory"] + "/" + n for n in control["file_pins"])
    directory = candidate["directories"][str(fold)]
    paths.update(directory + "/" + n for n in ("run.json", "config.json", "epoch_checkpoint_index.json"))
    paths.update(directory + f"/epoch_{e:03d}{suffix}" for e in range(1, 13) for suffix in (".pt", "_weak.csv"))
    return paths


def validate_primary_value(C, value, candidate, fold, spec, primary_policy):
    require(value["schema_version"] == spec["experiment"] + "_fixedE12_paired_weak_CSV_audit_v1"
            and value["complete"] is True and type(value["fold"]) is int and value["fold"] == fold
            and value["fixed_epoch"] == 12 and value["policy_SHA"] == spec["policy"]["sha256"]
            and value["master_policy_SHA"] == spec["master_policy"]["sha256"]
            and value["strict_metadata_and_all12_artifact_checks_passed_before_metrics"] is True
            and value["gold_numeric_labels_read"] is False
            and value["checkpoint_tensor_deserialize_GPU_MRI_network_Submit"] == 0
            and value["qualified_for_serving_or_Submit"] is False and value["Public_gain_established"] is False,
            "Complete same-candidate fixedE12 primary required before this screen")
    delta = C.primary_delta(value["evaluation"]["control"], value["evaluation"]["candidate"])
    require(value["primary"] == delta and delta["strictly_positive"] is True,
            "Nonpositive primary cannot be rescued by a family50 screen")
    require(value["source_scope"]["candidate"] == f"current_exact_source{spec['source_count']}"
            and value["paired_UIDs"] == primary_policy["candidate"]["counts"][str(fold)]["valid"],
            "Exact primary source/population scope")
    expected = allowed_primary_artifacts(C, primary_policy, fold)
    require(set(value["accessed_artifacts"]) == expected, "Exact primary artifact inventory, no aliases or extra paths")
    checkpoint = value["fixedE12_checkpoint_SHA"]
    control_dir = primary_policy["controls"][str(fold)]["directory"]
    candidate_dir = primary_policy["candidate"]["directories"][str(fold)]
    require(checkpoint["control"] == primary_policy["controls"][str(fold)]["file_pins"]["epoch_012.pt"]["sha256"]
            and checkpoint["control"] == value["accessed_artifacts"][control_dir + "/epoch_012.pt"]["sha256"]
            and checkpoint["candidate"] == value["accessed_artifacts"][candidate_dir + "/epoch_012.pt"]["sha256"],
            "Primary fixedE12 checkpoint binding")
    return delta


def bind_primary(C, snapshot, policy, candidate, fold, path, expected_sha):
    relative, expected_sha = artifact_ref(C, path, expected_sha)
    raw = snapshot.raw(relative)
    require(pin(raw)["sha256"] == expected_sha, "Primary result bytes differ from root binding")
    spec = policy["primaries"][candidate]
    primary_policy = C.load_policy(snapshot)
    require(pin((ROOT / C.POLICY).read_bytes())["sha256"] == spec["policy"]["sha256"]
            and C.MASTER_SHA == spec["master_policy"]["sha256"], "Exact candidate primary/master policy")
    value = C.strict_json(raw)
    validate_primary_value(C, value, candidate, fold, spec, primary_policy)
    # Hash-only revalidation is permitted here; no teacher or prediction floats.
    for name, expected in value["accessed_artifacts"].items():
        snapshot.pin(name, expected)
    return value, {"path": relative, **pin(raw)}, primary_policy


def validate_previous_value(C, value, candidate, policy, source_sha):
    require(value["schema_version"] == SCHEMA and value["candidate"] == candidate
            and type(value["fold"]) is int and value["fold"] == 0 and value["complete"] is True
            and value["policy_SHA"] == POLICY_SHA and value["source_SHA"] == source_sha
            and value["all_strict_checks_passed_before_new_numeric_reads"] is True
            and value["qualified_for_serving_or_Submit"] is False and value["Public_gain_established"] is False
            and value["gold_numeric_labels_read"] is False, "Same-candidate bound fold0 screen required before fold1")
    delta = C.primary_delta(value["native50_evaluation"]["control"], value["native50_evaluation"]["candidate"])
    require(value["screen"] == delta and delta["strictly_positive"] is True,
            "Fold0 screen must be strictly positive before fold1")
    require(value["primary_policy_SHA"] == policy["primaries"][candidate]["policy"]["sha256"], "Same primary namespace")


def bind_previous(C, snapshot, policy, candidate, path, expected_sha):
    relative, expected_sha = artifact_ref(C, path, expected_sha)
    raw = snapshot.raw(relative)
    require(pin(raw)["sha256"] == expected_sha, "Previous screen bytes differ from root binding")
    value = C.strict_json(raw)
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    validate_previous_value(C, value, candidate, policy, source_sha)
    primary = value["primary_evidence"]
    bound, ref, primary_policy = bind_primary(C, snapshot, policy, candidate, 0, primary["path"], primary["sha256"])
    require(ref == primary and value["fixedE12_checkpoint_SHA"] == bound["fixedE12_checkpoint_SHA"], "Previous primary binding")
    expected = allowed_primary_artifacts(C, primary_policy, 0) | {POLICY, ref["path"], policy["primaries"][candidate]["source"]["path"]}
    expected.update(r["path"] for r in policy["reference_pins"].values())
    expected.update(policy["REG"]["0"][role]["path"] for role in ("run", "prediction", "checkpoint"))
    require(set(value["accessed_artifacts"]) == expected, "Exact previous fold0 screen inventory")
    for name, expected_pin in value["accessed_artifacts"].items():
        snapshot.pin(name, expected_pin)
    return value, {"path": relative, **pin(raw)}


def verify_reg(C, snapshot, policy, fold, fold_audit, input_hashes):
    ref = policy["REG"][str(fold)]
    for role in ("run", "prediction", "checkpoint"):
        item = ref[role]
        snapshot.pin(item["path"], {k: item[k] for k in ("bytes", "sha256")})
    run = snapshot.json(ref["run"]["path"])
    require(run["status"] == "complete" and all(run.get(k) == v for k, v in ref["run_identity"].items())
            and run["fold"] == fold and run["seed"] == 20261002
            and run["gold_used_for_training_or_selection"] is False and run["gold_evaluated"] is False,
            "Exact historical REG run identity")
    expected = policy["primaries"]["tiny"]["candidate"]["counts"][str(fold)]
    require(run["train_studies"] == expected["train"] and run["valid_studies"] == expected["valid"]
            and run["audit"]["manifest_sha256"] == input_hashes and run["audit"]["fold_audit"] == fold_audit
            and run["audit"]["pixel_index_sha256"] == ref["run_identity"]["audit"]["pixel_index_sha256"],
            "Same REG heldout/group/cache/mask audit")
    binding = ref["original_pair_binding"]
    require(ref["selected_epoch"] == (8 if fold == 0 else 10) == binding["selected_epoch"]
            and ref["checkpoint"]["sha256"] == binding["checkpoint_sha256"]
            and ref["run"]["sha256"] == binding["run_json_sha256"]
            and run["config"] == binding["config"] and run["encoder_provenance"] == binding["encoder_provenance"],
            "Unchanged P039 REG selected E8/E10 and model binding")
    selected = run["selected"]
    history = run["history"]
    require([h["epoch"] for h in history] == list(range(1, 13))
            and selected["epoch"] == ref["selected_epoch"] and selected["checkpoint_sha256"] == ref["checkpoint"]["sha256"]
            and selected["prediction_sha256"] == ref["prediction"]["sha256"]
            and all(selected[k] == v for k, v in history[ref["selected_epoch"] - 1].items()), "REG selected writer correspondence")
    return run, selected, ref["prediction"]["path"]


def fp32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def mean_two(left, right):
    require(type(left) in (int, float) and type(right) in (int, float)
            and all(math.isfinite(v) and 0 <= v <= 1 for v in (left, right)), "Finite two probabilities")
    return fp32(fp32(fp32(left) + fp32(right)) * .5)


def family50(C, truth, predictions, regional):
    order = C.unique_ids(truth)
    require(order == C.unique_ids(predictions) == C.unique_ids(regional), "Same exact UID order for all family arms")
    result = {}
    for target in C.TARGETS:
        labels, scores, uids = [], [], []
        missing = soft = 0
        for row, left, right in zip(truth, predictions, regional):
            y = C.probability(row[target], missing=True)
            p = mean_two(C.probability(left[target]), C.probability(right[target]))
            if y is None:
                missing += 1
            elif y not in (0, 1):
                soft += 1
            else:
                labels.append(y)
                scores.append(p)
                uids.append(row[C.ID])
        result[target] = {"auc": C.binary_auc(labels, scores) if labels else None, "binary_observed": len(labels),
                          "positive": sum(labels), "missing_excluded": missing, "soft_excluded": soft,
                          "binary_UID_mask_SHA": C.object_sha(uids)}
    return {"per_target": result, "samefold_matchingheldout_only": True, "coefficient": .5}


def audit(candidate, fold, primary_result, primary_sha, output, *, previous_fold0=None, previous_sha=None):
    require(type(fold) is int and fold in (0, 1), "Fixed fold0 or fold1 required")
    require((fold == 0 and previous_fold0 is None and previous_sha is None)
            or (fold == 1 and previous_fold0 is not None and previous_sha is not None), "Fold1 requires positive fold0 evidence")
    output = Path(output).resolve()
    require(output.is_relative_to((ROOT / "artifacts/research").resolve()) and not output.exists(), "Fresh research output only")
    C, policy = frozen_primary(candidate)
    snapshot = C.Snapshot(ROOT)
    snapshot.pin(POLICY, pin((ROOT / POLICY).read_bytes()))
    for ref in policy["reference_pins"].values():
        snapshot.pin(ref["path"], {k: ref[k] for k in ("bytes", "sha256")})
    snapshot.pin(policy["primaries"][candidate]["source"]["path"],
                 {k: policy["primaries"][candidate]["source"][k] for k in ("bytes", "sha256")})
    primary, primary_ref, primary_policy = bind_primary(C, snapshot, policy, candidate, fold, primary_result, primary_sha)
    previous, previous_ref = (None, None) if fold == 0 else bind_previous(C, snapshot, policy, candidate, previous_fold0, previous_sha)
    # No new numerical teacher/prediction access is possible before both gates.
    metadata, fold_audit, input_hashes = C.verify_inputs(snapshot, primary_policy)
    arms = {role: C.verify_run(snapshot, primary_policy, fold, role, fold_audit, input_hashes) for role in ("control", "candidate")}
    reg_run, reg_selected, reg_prediction = verify_reg(C, snapshot, policy, fold, fold_audit, input_hashes)
    require({role: run["checkpoint"] for role, run in arms.items()} == primary["fixedE12_checkpoint_SHA"], "Current primary E12 models unchanged")
    header, cells = C.csv_cells(snapshot.raw(primary_policy["manifest_directory"] + "/weak.csv"))
    weak = [dict(zip(header, values)) for values in cells]
    require([{k: row[k] for k in (C.ID, "source", "group_id", "fold")} for row in weak] == metadata, "Teacher metadata unchanged")
    C.verify_teacher_coverage(weak, fold, fold_audit)
    truth = [row for row in weak if row["fold"] == str(fold)]
    order = C.unique_ids(truth)
    mask_sha = C.object_sha([[r[C.ID], r["group_id"], r["fold"], *[C.probability(r[t], missing=True) for t in C.TARGETS]] for r in truth])
    require(mask_sha == primary["teacher_group_fold_missingmask_SHA"], "Same exact primary group/fold/missing mask")
    raw_evaluation, predictions = {}, {}
    for role, run in arms.items():
        predictions[role] = C.prediction_rows(snapshot.raw(run["directory"] + "/epoch_012_weak.csv"), order)
        raw_evaluation[role] = C.evaluate_weak(truth, predictions[role])
        C.verify_writer(raw_evaluation[role], run["E12"], policy["tolerances"])
        require(raw_evaluation[role] == primary["evaluation"][role], "Current raw metrics equal bound primary")
    regional = C.prediction_rows(snapshot.raw(reg_prediction), order)
    reg_evaluation = C.evaluate_weak(truth, regional)
    C.verify_writer(reg_evaluation, reg_selected, policy["tolerances"])
    averaged = {role: family50(C, truth, values, regional) for role, values in predictions.items()}
    screen = C.primary_delta(averaged["control"], averaged["candidate"])
    snapshot.recheck()
    result = {"schema_version": SCHEMA, "at_UTC": datetime.now(timezone.utc).isoformat(), "complete": True,
              "candidate": candidate, "fold": fold, "policy_SHA": POLICY_SHA, "source_SHA": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "primary_policy_SHA": primary["policy_SHA"], "primary_evidence": primary_ref, "previous_fold0_evidence": previous_ref,
              "all_strict_checks_passed_before_new_numeric_reads": True, "fixed_epoch": 12, "coefficient": .5,
              "fixedE12_checkpoint_SHA": primary["fixedE12_checkpoint_SHA"], "REG_checkpoint_SHA": policy["REG"][str(fold)]["checkpoint"]["sha256"],
              "REG_epoch": reg_selected["epoch"], "paired_UIDs": len(order), "teacher_group_fold_missingmask_SHA": mask_sha,
              "primary_delta": primary["primary"], "screen": screen, "native50_evaluation": averaged, "REG_writer_evaluation": reg_evaluation,
              "source_scope": primary["source_scope"], "counter_summaries": primary["counter_summaries"], "accessed_artifacts": snapshot.accessed,
              "both_screen_folds_positive": bool(fold == 1 and previous is not None and screen["strictly_positive"]),
              "qualified_for_serving_or_Submit": False, "Public_gain_established": False, "gold_numeric_labels_read": False,
              "model_deserialize_GPU_MRI_network_external_action": 0, "original_F009_HOLD_unchanged": True,
              "limitations": policy["scope"], "required_next": policy["order"]["future_formal"],
              "next_stage": "root_may_consider_manual_fold1_only" if fold == 0 and screen["strictly_positive"] else
                            "bothscreens_positive_still_require_separate_current_model_serving_qualification" if screen["strictly_positive"] else
                            "reject_additional_screen_no_alpha_epoch_label_rescue"}
    output.mkdir(parents=True, exist_ok=False)
    import json

    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("tiny", "small"), required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--primary-result", type=Path, required=True)
    parser.add_argument("--primary-sha", required=True)
    parser.add_argument("--previous-fold0", type=Path)
    parser.add_argument("--previous-sha")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.candidate, args.fold, args.primary_result, args.primary_sha, args.output,
                   previous_fold0=args.previous_fold0, previous_sha=args.previous_sha)
    print(result["screen"]["fixed10_mean_delta"], "qualified_for_Submit=False")


if __name__ == "__main__":
    main()
