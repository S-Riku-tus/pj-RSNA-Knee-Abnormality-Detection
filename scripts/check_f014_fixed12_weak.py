"""Prospective, standard-library-only F014 fixed-E12 weak CSV audit.

No Torch import, tensor deserialization, MRI access, fitting, network or external
writes. BCE-best aliases are never opened. This is not a serving/Public gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import struct
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
POLICY = "artifacts/research/20261021-F014-fixed12-CSV-audit-prospective-v1/policy.json"
POLICY_SHA = "f9f250020d45858a1380814f6a7ffba577dbcb5052fb888812ac2f9cdb6bdb93"
MASTER_SHA = "c86f51554480c4cb3be08745f569d8c56111bd5e825d2ceecde80ce8a377a68b"
ID = "StudyInstanceUID"
TARGETS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
           "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")
COLUMNS = (ID, *TARGETS)
EXCLUDED = ("MCL", "Synovitis")
FIXED10 = tuple(t for t in TARGETS if t not in EXCLUDED)
EPOCHS = list(range(1, 13))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("Nonfinite JSON token: " + value)

    return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=pairs, parse_constant=nonfinite)


def workspace_path(root, relative):
    p = PurePosixPath(relative)
    require(not p.is_absolute() and ".." not in p.parts and ":" not in relative and "\\" not in relative,
            "Workspace-relative positive path required")
    path = Path(root) / Path(*p.parts)
    require(not path.is_symlink() and path.resolve().is_relative_to(Path(root).resolve()), "Workspace regular path required")
    return path


class Snapshot:
    """Hash-only checkpoint access and bounded JSON/CSV reads, verified again at end."""

    def __init__(self, root):
        self.root, self.accessed = Path(root), {}

    def pin(self, relative, expected=None):
        path = workspace_path(self.root, relative)
        require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= 512 * 1024**2,
                "Bounded regular artifact required: " + relative)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        value = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
        require(expected is None or value == expected, "Artifact bytes differ: " + relative)
        require(relative not in self.accessed or self.accessed[relative] == value, "Artifact changed during audit")
        self.accessed[relative] = value
        return value

    def raw(self, relative, expected=None):
        require(Path(relative).suffix in (".json", ".csv"), "Only JSON/CSV content may be read")
        value = self.pin(relative, expected)
        require(value["bytes"] <= 32 * 1024**2, "Bounded JSON/CSV required")
        raw = workspace_path(self.root, relative).read_bytes()
        require(len(raw) == value["bytes"] and hashlib.sha256(raw).hexdigest() == value["sha256"], "Read changed bytes")
        return raw

    def json(self, relative, expected=None):
        return strict_json(self.raw(relative, expected))

    def recheck(self):
        for name, value in list(self.accessed.items()):
            self.pin(name, value)


def load_policy(snapshot):
    raw = snapshot.raw(POLICY)
    require(hashlib.sha256(raw).hexdigest() == POLICY_SHA, "Exact prospective CSV policy required")
    value = strict_json(raw)
    require(value["schema_version"] == "F014_fixedE12_CSV_audit_prospective_v1"
            and value["master_policy"]["sha256"] == MASTER_SHA and value["fixed_epoch"] == 12
            and value["fixed_epochs"] == EPOCHS and value["excluded_targets"] == list(EXCLUDED)
            and value["strict_checks_before_metrics"] is True and value["checkpoint_tensor_deserialize"] is False,
            "Fixed E12/10-target/no-tensor policy required")
    return value


def probability(text, *, missing=False):
    if missing and text.strip().lower() in ("", "nan", "na", "null"):
        return None
    require(isinstance(text, str) and text.strip() != "", "Numeric cell required")
    value = float(text)
    require(math.isfinite(value) and 0 <= value <= 1, "Finite probability in [0,1] required")
    return value


def csv_cells(raw):
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    header = next(reader, None)
    require(header and len(header) == len(set(header)), "Unique nonempty CSV header required")
    rows = list(reader)
    require(rows and all(len(row) == len(header) for row in rows), "Nonempty rectangular CSV required")
    return header, rows


def unique_ids(rows):
    ids = [row[ID] for row in rows]
    require(all(ids) and len(ids) == len(set(ids)) and all(uid.strip() == uid and uid not in (".", "..")
            and not any(c in uid for c in "/\\:") and not any(ord(c) < 32 or ord(c) == 127 for c in uid) for uid in ids),
            "Unique nonempty clean study IDs required")
    return ids


def prediction_rows(raw, expected_ids):
    header, cells = csv_cells(raw)
    require(tuple(header) == COLUMNS, "Exactly ordered 13 prediction columns required")
    rows = [dict(zip(header, values)) for values in cells]
    require(unique_ids(rows) == list(expected_ids), "Prediction UID order/population differs")
    for row in rows:
        for target in TARGETS:
            probability(row[target])
    return rows


def metadata_rows(raw, fields):
    """Project metadata only: Gold target strings are never retained/converted."""
    header, cells = csv_cells(raw)
    require(set(fields) <= set(header), "Required CSV metadata missing")
    indices = {key: header.index(key) for key in fields}
    result = [{key: values[index] for key, index in indices.items()} for values in cells]
    unique_ids(result)
    return result


def validate_groups(weak, gold, *, folds=5):
    weak_ids, gold_ids = set(unique_ids(weak)), set(unique_ids(gold))
    require(not weak_ids.intersection(gold_ids), "Gold UID intersects weak data")
    groups = {}
    for row in weak:
        require(row["source"] == "report_weak" and row["group_id"] and row["fold"] in [str(f) for f in range(folds)],
                "Exact report-weak group/fold metadata required")
        require(row["group_id"] not in groups or groups[row["group_id"]] == row["fold"], "Supplied/report group crosses folds")
        groups[row["group_id"]] = row["fold"]
    require(all(row["group_id"] for row in gold) and not set(groups).intersection(row["group_id"] for row in gold),
            "Gold group intersects weak data")


def validate_history(run, index, expected_identity, counts, checkpoint_schema):
    """No CSV numeric read is permitted until this complete-run guard succeeds."""
    require(run["status"] == "complete" and len(run["history"]) == 12, "All12 epochs must be complete")
    require(all(run.get(key) == value for key, value in expected_identity.items()), "Frozen run identity differs")
    require(run["train_studies"] == counts["train"] and run["valid_studies"] == counts["valid"], "Fixed split counts differ")
    require(run["targets"] == list(TARGETS) and run["gold_used_for_training_or_selection"] is False
            and run["gold_evaluated"] is False and run["patient_independence_verified"] is False,
            "Weak-only exact target/provenance scope required")
    require(index["schema_version"] == "native392_epoch_suffix_head_retention_v1" and index["checkpoint_schema"] == checkpoint_schema
            and index["fold"] == run["fold"] and index["targets"] == list(TARGETS)
            and index["source_sha256"] == run["source_sha256"] and index["selection_data"] == "weak_validation"
            and index["selection_rule"] == "minimum observed weak BCE; earliest exact tie" and index["resume_supported"] is False,
            "Exact retention identity required; BCE alias is not the candidate endpoint")
    require(all(type(h["epoch"]) is int for h in (*run["history"], *index["checkpoints"]))
            and [h["epoch"] for h in run["history"]] == [h["epoch"] for h in index["checkpoints"]] == EPOCHS,
            "Exactly positive ordered epochs1..12 required")
    opportunities = (counts["train"] + 15) // 16
    prior_elapsed, total_steps, total_skips = 0., 0, 0
    for history, entry in zip(run["history"], index["checkpoints"]):
        epoch = history["epoch"]
        require(all(entry.get(k) == v for k, v in history.items()), "Writer history/retention correspondence differs")
        require(entry["checkpoint"] == f"epoch_{epoch:03d}.pt" and entry["predictions_csv"] == f"epoch_{epoch:03d}_weak.csv",
                "Fixed retained epoch names required; best/BCE aliases forbidden")
        steps, offered, skipped = [history[k] for k in ("optimizer_steps", "optimizer_step_opportunities", "amp_skipped_steps")]
        require(all(type(v) is int for v in (steps, offered, skipped)) and offered == opportunities and steps >= 1
                and skipped >= 0 and steps + skipped == offered, "Actual AMP updates/opportunities/skips differ")
        elapsed = history["elapsed_seconds"]
        require(isinstance(elapsed, (int, float)) and math.isfinite(elapsed) and prior_elapsed < elapsed <= 43200,
                "Finite ordered epoch runtime within12h required")
        prior_elapsed = elapsed
        require(all(isinstance(history[k], (int, float)) and not isinstance(history[k], bool) and math.isfinite(history[k])
                    and history[k] >= 0 for k in ("train_bce", "weak_bce", "gpu_peak_reserved_bytes")), "Finite writer diagnostics required")
        multiplier = 1. if epoch <= 6 else .1 + .9 * (1 + math.cos(math.pi * (epoch - 6) / 6)) / 2
        require(math.isclose(history["learning_rate_multiplier"], multiplier, rel_tol=0, abs_tol=1e-15)
                and set(history["learning_rates"]) == {"head", "last2"}
                and all(math.isclose(history["learning_rates"][name], base * multiplier, rel_tol=0, abs_tol=1e-18)
                        for name, base in (("head", .0003), ("last2", .00001))), "Fixed late-cosine learning rates differ")
        total_steps, total_skips = total_steps + steps, total_skips + skipped
    require(math.isfinite(run["elapsed_seconds"]) and prior_elapsed <= run["elapsed_seconds"] <= 43200, "Run runtime differs")
    return {"actual_updates": total_steps, "opportunities": opportunities * 12, "AMP_skips": total_skips}


def verify_inputs(snapshot, policy):
    master = policy["master_policy"]
    snapshot.pin(master["path"], {key: master[key] for key in ("bytes", "sha256")})
    for path, pin in policy["input_pins"].items():
        snapshot.pin(path, pin)
    index_ref, candidate = policy["pixel_index"], policy["candidate"]
    index = snapshot.json(index_ref["path"], {key: index_ref[key] for key in ("bytes", "sha256")})
    for key in ("config", "provenance", "resource_gate"):
        ref = candidate[key]
        value = snapshot.json(ref["path"], {k: ref[k] for k in ("bytes", "sha256")})
        require(object_sha(value) == ref["object_sha256"], "Pinned config/provenance/gate object differs")
    require(len(candidate["source_pins"]) == 16, "Exact current source16 inventory required")
    for name, pin in candidate["source_pins"].items():
        snapshot.pin("src/rsna_knee/" + name, pin)
        require(pin["sha256"] == candidate["run_identity"]["source_sha256"][name], "Current source16/run pins differ")
    manifest_dir = policy["manifest_directory"]
    manifest = snapshot.json(manifest_dir + "/manifest.json")
    audit = snapshot.json(manifest_dir + "/fold-audit-v2.json")
    require(object_sha(manifest) == candidate["shared_manifest_object_sha256"] and object_sha(audit) == candidate["shared_fold_audit_object_sha256"],
            "Exact frozen input audit objects required")
    require(manifest["seed"] == 20261002 and manifest["n_folds"] == 5 and manifest["weak_studies"] == 4207
            and manifest["label_provenance"]["gold_used_for_tuning"] is False
            and manifest["patient_independence_verified"] is False and audit["integrity_valid"] is audit["ready_for_training"] is True
            and audit["gold_used_for_tuning"] is False, "Reviewed weak eligibility metadata differs")
    weak_meta = metadata_rows(snapshot.raw(manifest_dir + "/weak.csv"), (ID, "source", "group_id", "fold"))
    gold_meta = metadata_rows(snapshot.raw(manifest_dir + "/gold.csv"), (ID, "group_id"))
    validate_groups(weak_meta, gold_meta)
    require(len(weak_meta) == manifest["weak_studies"] == audit["weak_studies"] and len(gold_meta) == manifest["gold_studies"] == audit["gold_studies"],
            "Manifest population differs")
    require(index["complete"] is True and index["split"] == "train" and index["limited"] is False
            and len(index["study_ids"]) == len(set(index["study_ids"])) == 4207
            and set(index["study_ids"]) == {r[ID] for r in weak_meta}
            and index["contract"]["preprocess"] == candidate["run_identity"]["config"]["preprocess"], "Processed cache population/preprocess differs")
    input_hashes = {Path(path).name: pin["sha256"] for path, pin in policy["input_pins"].items()}
    require(index["manifest_sha256"] == input_hashes, "Processed cache manifest hashes differ")
    for fold in range(5):
        require(sum(r["fold"] == str(fold) for r in weak_meta) == manifest["fold_counts"][str(fold)] == audit["fold_counts"][str(fold)],
                "Fixed group fold population differs")
    return weak_meta, audit, input_hashes


def verify_run(snapshot, policy, fold, role, fold_audit, input_hashes):
    candidate = policy["candidate"]
    if role == "control":
        control = policy["controls"][str(fold)]
        directory, identity = control["directory"], control["run_identity"]
        for name, pin in control["file_pins"].items():
            snapshot.pin(directory + "/" + name, pin)
        checkpoint_schema = "native392_generic_dino_suffix_head_fast16_v1"
    else:
        require(role == "candidate", "Known comparison role required")
        directory, identity = candidate["directories"][str(fold)], {**candidate["run_identity"], "fold": fold}
        checkpoint_schema = candidate["checkpoint_schema"]
    run = snapshot.json(directory + "/run.json")
    index = snapshot.json(directory + "/epoch_checkpoint_index.json")
    summary = validate_history(run, index, identity, candidate["counts"][str(fold)], checkpoint_schema)
    require(snapshot.json(directory + "/config.json") == identity["config"], "Run config file differs")
    audit = run["audit"]
    require(audit["manifest_sha256"] == input_hashes and audit["fold_audit"] == fold_audit
            and audit["pixel_index_sha256"] == policy["pixel_index"]["sha256"], "Run input/fold/cache audit differs")
    if role == "control":
        require(object_sha(audit) == control["audit_object_sha256"] and summary["AMP_skips"] == control["known_AMP_skips"],
                "Historical control audit/counters changed")
    else:
        require(audit["resource_gate_sha256"] == candidate["resource_gate"]["sha256"]
                and object_sha(audit["resource_gate"]) == candidate["resource_gate"]["object_sha256"]
                and object_sha(audit["manifest"]) == candidate["shared_manifest_object_sha256"]
                and object_sha(run["environment"]) == candidate["environment_object_sha256"], "Current candidate gate/environment differs")
    for entry in index["checkpoints"]:
        snapshot.pin(directory + "/" + entry["checkpoint"], {"bytes": entry["checkpoint_bytes"], "sha256": entry["checkpoint_sha256"]})
        require(snapshot.pin(directory + "/" + entry["predictions_csv"])["sha256"] == entry["prediction_sha256"],
                "Retained prediction bytes/writer hash differ")
    return {"directory": directory, "run": run, "E12": run["history"][11], "counters": summary,
            "checkpoint": index["checkpoints"][11]["checkpoint_sha256"], "source_scope":
            "current_exact_source16" if role == "candidate" else "historical_recorded_source_pins_not_current_source16"}


def binary_auc(labels, scores):
    """Independent Mann-Whitney pair comparisons: ties count one half."""
    require(len(labels) == len(scores) and labels and all(type(y) in (int, float) and y in (0, 1) for y in labels),
            "Binary paired labels required")
    require(all(math.isfinite(p) and 0 <= p <= 1 for p in scores), "Finite AUC scores required")
    positive = [p for y, p in zip(labels, scores) if y == 1]
    negative = [p for y, p in zip(labels, scores) if y == 0]
    if not positive or not negative:
        return None
    doubled_wins = sum(2 * (a > b) + (a == b) for a in positive for b in negative)
    return doubled_wins / (2 * len(positive) * len(negative))


def verify_teacher_coverage(weak, fold, fold_audit):
    for partition, valid in (("train", False), ("valid", True)):
        rows = [r for r in weak if (r["fold"] == str(fold)) == valid]
        for target in TARGETS:
            values = [probability(row[target], missing=True) for row in rows]
            observed = [v for v in values if v is not None]
            expected = fold_audit["per_fold"][str(fold)][partition][target]
            counts = {"observed": len(observed), "missing": len(values) - len(observed),
                      "exact_zero": observed.count(0.), "exact_one": observed.count(1.),
                      "soft": sum(v not in (0, 1) for v in observed)}
            require(all(expected[k] == v for k, v in counts.items()), "Original teacher/missing-mask coverage differs")


def evaluate_weak(truth, predictions):
    require(unique_ids(truth) == unique_ids(predictions), "Paired teacher/prediction UID order required")
    result, loss_terms, observed_count = {}, [], 0
    for target in TARGETS:
        labels, scores, soft, missing = [], [], 0, 0
        for row, prediction in zip(truth, predictions):
            y, p = probability(row[target], missing=True), probability(prediction[target])
            if y is None:
                missing += 1
                continue
            observed_count += 1
            # Writer targets and probabilities are FP32. Logits cannot be
            # recovered exactly from sigmoid CSV; this is a tolerance diagnostic.
            y32 = struct.unpack("<f", struct.pack("<f", y))[0]
            require(not (p == 0 and y32 > 0 or p == 1 and y32 < 1), "Saturated contradictory CSV cannot identify finite logits BCE")
            loss_terms.append(-(y32 * math.log(p) if y32 else 0.) - ((1 - y32) * math.log1p(-p) if y32 != 1 else 0.))
            if y in (0, 1):
                labels.append(y)
                scores.append(p)
            else:
                soft += 1
        result[target] = {"auc": binary_auc(labels, scores) if labels else None, "binary_observed": len(labels),
                          "positive": sum(labels), "soft_excluded": soft, "missing_excluded": missing}
    defined = [row["auc"] for row in result.values() if row["auc"] is not None]
    require(observed_count > 0, "Observed weak cells required")
    return {"per_target": result, "macro_auc_12": sum(defined) / 12 if len(defined) == 12 else None,
            "available_class_macro_auc": sum(defined) / len(defined) if defined else None,
            "defined_classes": len(defined), "CSV_reconstructed_BCE": math.fsum(loss_terms) / observed_count,
            "observed_cells": observed_count}


def verify_writer(calculated, item, tolerances):
    metrics = item["metrics"]
    require(set(metrics["per_target"]) == set(TARGETS), "Writer exact12 metric inventory required")
    for target in TARGETS:
        for key in ("auc", "binary_observed", "positive", "soft_excluded"):
            actual, expected = calculated["per_target"][target][key], metrics["per_target"][target][key]
            require(actual is None and expected is None or actual is not None and expected is not None
                    and math.isclose(actual, expected, rel_tol=0, abs_tol=tolerances["AUC_abs"]), "Current writer metric differs: " + target + "/" + key)
    for key in ("macro_auc_12", "available_class_macro_auc", "defined_classes"):
        actual, expected = calculated[key], metrics[key]
        require(actual is None and expected is None or actual is not None and expected is not None
                and math.isclose(actual, expected, rel_tol=0, abs_tol=tolerances["AUC_abs"]), "Writer aggregate differs: " + key)
    require(item["weak_macro_auc_12"] == metrics["macro_auc_12"], "Writer E12 macro correspondence differs")
    require(abs(calculated["CSV_reconstructed_BCE"] - item["weak_bce"]) <= tolerances["BCE_abs"],
            "CSV/logits BCE tolerance diagnostic differs")


def primary_delta(control, candidate):
    require(all(control["per_target"][t]["auc"] is not None and candidate["per_target"][t]["auc"] is not None for t in FIXED10),
            "All fixed10 AUCs must be defined; no denominator substitution")
    differences = {t: candidate["per_target"][t]["auc"] - control["per_target"][t]["auc"] for t in FIXED10}
    delta = math.fsum(differences.values()) / 10
    return {"targets": list(FIXED10), "per_target_delta_report_only": differences, "fixed10_mean_delta": delta,
            "strictly_positive": delta > 0, "control_fixed10": math.fsum(control["per_target"][t]["auc"] for t in FIXED10) / 10,
            "candidate_fixed10": math.fsum(candidate["per_target"][t]["auc"] for t in FIXED10) / 10}


def audit(fold, output):
    require(type(fold) is int and fold in (0, 1), "Only fixed folds0/1 permitted")
    output = Path(output).resolve()
    require(output.is_relative_to((ROOT / "artifacts/research").resolve()) and not output.exists() and not output.is_symlink(), "Fresh research output required")
    snapshot = Snapshot(ROOT)
    policy = load_policy(snapshot)
    metadata, fold_audit, input_hashes = verify_inputs(snapshot, policy)
    # Both whole-run gates, hashes, counters and exact retained epochs precede
    # any numerical teacher or prediction read. No partial-run endpoint exists.
    runs = {role: verify_run(snapshot, policy, fold, role, fold_audit, input_hashes) for role in ("control", "candidate")}
    require(all(len([r for r in metadata if (r["fold"] == str(fold)) == valid]) == policy["candidate"]["counts"][str(fold)]["valid" if valid else "train"] for valid in (True, False)),
            "Exact paired fold counts required")
    weak_path = policy["manifest_directory"] + "/weak.csv"
    header, cells = csv_cells(snapshot.raw(weak_path))
    require(set(COLUMNS) <= set(header), "Weak teacher columns missing")
    weak = [dict(zip(header, row)) for row in cells]
    require([{k: row[k] for k in (ID, "source", "group_id", "fold")} for row in weak] == metadata, "Teacher metadata pass changed")
    # Validate every weak cell, including training rows, without fitting or
    # converting any numeric Gold label. Every study must have observations.
    for row in weak:
        require(any(probability(row[t], missing=True) is not None for t in TARGETS), "Entirely missing weak study")
        for target in TARGETS:
            probability(row[target], missing=True)
    verify_teacher_coverage(weak, fold, fold_audit)
    truth = [row for row in weak if row["fold"] == str(fold)]
    order = unique_ids(truth)
    evaluated = {}
    for role, run in runs.items():
        predicted = prediction_rows(snapshot.raw(run["directory"] + "/epoch_012_weak.csv"), order)
        evaluated[role] = evaluate_weak(truth, predicted)
        verify_writer(evaluated[role], run["E12"], policy["tolerances"])
    delta = primary_delta(evaluated["control"], evaluated["candidate"])
    mask_teacher = [[r[ID], r["group_id"], r["fold"], *[probability(r[t], missing=True) for t in TARGETS]] for r in truth]
    snapshot.recheck()
    result = {"schema_version": "F014_fixedE12_paired_weak_CSV_audit_v1", "at_UTC": datetime.now(timezone.utc).isoformat(),
              "complete": True, "fold": fold, "fixed_epoch": 12, "policy_SHA": POLICY_SHA, "master_policy_SHA": MASTER_SHA,
              "strict_metadata_and_all12_artifact_checks_passed_before_metrics": True, "paired_UIDs": len(order),
              "teacher_group_fold_missingmask_SHA": object_sha(mask_teacher), "gold_numeric_labels_read": False,
              "checkpoint_tensor_deserialize_GPU_MRI_network_Submit": 0, "primary": delta,
              "counter_summaries": {role: run["counters"] for role, run in runs.items()},
              "fixedE12_checkpoint_SHA": {role: run["checkpoint"] for role, run in runs.items()},
              "source_scope": {role: run["source_scope"] for role, run in runs.items()},
              "evaluation": evaluated, "accessed_artifacts": snapshot.accessed,
              "next_stage": "eligible_for_root_to_consider_manual_fold1_only" if fold == 0 and delta["strictly_positive"] else
                            "fold1_positive_requires_separate_bothfold_binding_and_model_qualification" if delta["strictly_positive"] else "reject_fixedE12_encoder_choice_no_epoch_or_label_rescue",
              "qualified_for_serving_or_Submit": False, "Public_gain_established": False,
              "limits": policy["accuracy_limits"], "BCE_scope": "CSV sigmoid reconstruction with prespecified tolerance; not exact reconstruction of original FP32 logits loss"}
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.fold, args.output)
    print(json.dumps({"complete": True, "fold": args.fold, **result["primary"], "qualified_for_Submit": False}))


if __name__ == "__main__":
    main()
