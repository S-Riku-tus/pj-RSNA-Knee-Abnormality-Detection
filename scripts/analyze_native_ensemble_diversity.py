"""Audit completed native runs on the same weak fold; no models, MRI or Gold."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

from rsna_knee.contracts import ID, SUBMISSION_COLUMNS, TARGETS, index_studies, probability, read_csv, sha256
from rsna_knee.ensemble_diversity import compare_target

FIXED10 = tuple(t for t in TARGETS if t not in ("MCL", "Synovitis"))
WEAK_COLUMNS = (*SUBMISSION_COLUMNS, "source", "group_id", "fold")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_run(directory, checkpoint):
    run = json.loads((directory / "run.json").read_text(encoding="utf-8-sig"))
    require(run.get("status") == "complete", "Both runs must be complete before any prediction/teacher access")
    require(run.get("fold") in (0, 1), "Only previously evaluated folds0/1 allowed; outer2 remains sealed")
    history = run.get("history", [])
    require(len(history) == 12 and all(type(r.get("epoch")) is int for r in history)
            and [r["epoch"] for r in history] == list(range(1, 13)), "Exactly twelve completed epochs required")
    require(run.get("targets") == list(TARGETS) and run.get("selection_data") == "weak_validation"
            and run.get("selection_rule") in ("bce", "minimum global observed-cell weak BCE; earliest exact tie"),
            "Exact target order and weak-BCE selection required")
    require(run.get("gold_used_for_training_or_selection") is False and run.get("gold_evaluated") is False,
            "Gold must remain unused")
    require(not any(run.get(k, False) for k in ("artificial", "synthetic", "fixture", "diagnostic_mode")),
            "Only genuine full held-out runs allowed")
    require(not run.get("training_augmentation") or checkpoint == "fixed12",
            "Augmented F009 comparison is fixed E12 only")
    if checkpoint == "selected":
        require(all(type(r.get("weak_bce")) in (int, float) and math.isfinite(r["weak_bce"]) for r in history),
                "Finite completed weak BCE required")
        reference = run["selected"]
        chosen = min(history, key=lambda row: row["weak_bce"])
        require(all(reference.get(k) == v for k, v in chosen.items()), "Selected receipt differs from first minimum BCE")
        name = "weak_valid_predictions.csv"
    else:
        reference, name = history[-1], "epoch_012_weak.csv"
    return run, reference, directory / name


def prediction_csv(path, reference):
    require(sha256(path) == reference["prediction_sha256"], "Saved prediction SHA differs from completed receipt")
    fields, rows = read_csv(path)
    require(fields == list(SUBMISSION_COLUMNS), "Exact prediction columns required; Report forbidden")
    values = index_studies(rows)
    require(bool(values), "Empty predictions")
    return {uid: {t: probability(row[t]) for t in TARGETS} for uid, row in values.items()}


def weak_partition(path, fold):
    """Read group metadata for every fold; interpret target values only for requested fold0/1."""
    truth, all_ids, group_folds, counts = {}, set(), {}, {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == list(WEAK_COLUMNS), "Exact weak columns required; Gold/Report forbidden")
        for row in reader:
            require(None not in row and all(v is not None for v in row.values()), "Malformed weak CSV")
            key, group = row[ID], row["group_id"]
            require(key and key not in all_ids and group and row["source"] == "report_weak", "Invalid weak UID/group/source")
            all_ids.add(key)
            assigned = int(row["fold"])
            require(0 <= assigned < 5, "Invalid group fold")
            require(group_folds.setdefault(group, assigned) == assigned, "Supplied group spans folds")
            counts[assigned] = counts.get(assigned, 0) + 1
            if assigned == fold:
                truth[key] = {t: probability(row[t], missing=True) for t in TARGETS}
    require(bool(truth), "Empty requested weak fold")
    return truth, counts, len(group_folds)


def analyze(left_dir, right_dir, weak, checkpoint):
    # Complete checks on BOTH arms precede every prediction/teacher payload access.
    left, a, ap = read_run(left_dir, checkpoint)
    right, b, bp = read_run(right_dir, checkpoint)
    require(left["fold"] == right["fold"], "Same validation fold required; never pool cross-fold predictions")
    require(left["seed"] == right["seed"], "Same seed required for this native comparison")
    for key in ("manifest_sha256", "pixel_index_sha256"):
        require(left["audit"][key] == right["audit"][key], "Immutable manifest/group/pixel identity differs")
    require(left["audit"]["manifest"]["grouping"] == "duplicate_report + supplied_groups",
            "Existing duplicate-report and supplied-group partition required")
    require(sha256(weak) == left["audit"]["manifest_sha256"]["weak.csv"], "Weak manifest SHA differs")
    p, q = prediction_csv(ap, a), prediction_csv(bp, b)
    truth, counts, groups = weak_partition(weak, left["fold"])
    require(p.keys() == q.keys() == truth.keys(), "Both prediction UID sets must equal the exact held-out fold")
    require(len(truth) == left["valid_studies"] == right["valid_studies"]
            and sum(counts.values()) - len(truth) == left["train_studies"] == right["train_studies"],
            "Recorded full train/validation counts differ")
    targets = {}
    order = sorted(truth)
    for target in TARGETS:
        ids = [uid for uid in order if truth[uid][target] in (0, 1)]
        item = compare_target([truth[u][target] for u in ids], [p[u][target] for u in ids], [q[u][target] for u in ids])
        item["missing_excluded"] = sum(truth[u][target] is None for u in order)
        item["soft_excluded"] = len(order) - len(ids) - item["missing_excluded"]
        item["binary_support_SHA256"] = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        # Recompute AUC and the exact pair identity independently of saved scalar metrics.
        if ids:
            for side, reference in (("left", a), ("right", b)):
                recorded = reference["metrics"]["per_target"][target]
                require(recorded["binary_observed"] == len(ids) and recorded["positive"] == item["positive"],
                        "Saved target support differs")
                require(item["auc"][side] == recorded["auc"], "CSV AUC differs from saved completed metric")
            if item["pair_transitions"]["auc_delta"] is not None:
                delta = item["auc"]["right"] - item["auc"]["left"]
                require(abs(delta - item["pair_transitions"]["auc_delta"]) < 1e-12, "Exact pair/AUC identity differs")
        targets[target] = item
    summaries = {}
    for name, selected in (("fixed10", FIXED10), ("all12_auxiliary", TARGETS)):
        values = {side: [targets[t]["auc"][side] if targets[t]["auc"] else None for t in selected]
                  for side in ("left", "right", "fixed_FP32_probability_mean50")}
        aucs = {side: math.fsum(v) / len(v) if all(x is not None for x in v) else None for side, v in values.items()}
        summaries[name] = {"targets": list(selected), "macro_auc": aucs,
                           "mean50_delta_from_left": (aucs["fixed_FP32_probability_mean50"] - aucs["left"]
                                                      if aucs["left"] is not None else None),
                           "mean50_delta_from_right": (aucs["fixed_FP32_probability_mean50"] - aucs["right"]
                                                       if aucs["right"] is not None else None)}
    return {"schema_version": "completed_samefold_native_diversity_v1", "at_UTC": datetime.now(timezone.utc).isoformat(),
            "checkpoint_rule": checkpoint, "fold": left["fold"], "seed": left["seed"], "valid_studies": len(truth),
            "weak_manifest_SHA256": sha256(weak), "group_count_all_folds": groups,
            "UID_order_SHA256": hashlib.sha256("\n".join(order).encode()).hexdigest(),
            "arms": {side: {"directory": str(directory), "run_SHA256": sha256(directory / "run.json"),
                            "epoch": ref["epoch"], "predictions_SHA256": sha256(path)}
                     for side, directory, ref, path in (("left", left_dir, a, ap), ("right", right_dir, b, bp))},
            "per_target": targets, "summaries": summaries,
            "runtime": {"python": platform.python_version(), "platform": platform.platform()},
            "public958_parent_sameholdout_predictions": "not acquired; complementarity with public parent unmeasured",
            "limits": ["Weak report labels: these AUCs cannot be compared directly with Public0.958.",
                       "Exploratory reuse of prior weak holdouts; selected epochs have selection bias.",
                       "FP32 probability mean50 is descriptive, not the deployed rank overlay or a weight search.",
                       "Residual correlation includes shared labels; low correlation alone is not improvement or independence.",
                       "No cross-fold pooling or patient-independence claim. Sparse MCL/Synovitis are auxiliary.",
                       "Fold2 targets not interpreted or evaluated; Gold, Report, model, MRI, GPU and remote actions zero."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-run", type=Path, required=True)
    parser.add_argument("--right-run", type=Path, required=True)
    parser.add_argument("--weak-csv", type=Path, default=Path("data/manifests/v1-vmohitrao-research/weak.csv"))
    parser.add_argument("--checkpoint", choices=("selected", "fixed12"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Fresh output required; existing experiments must never be overwritten")
    report = analyze(args.left_run, args.right_run, args.weak_csv, args.checkpoint)
    report["source_SHA256"] = {str(p): sha256(p) for p in (Path(__file__), Path(__file__).resolve().parents[1] / "src/rsna_knee/ensemble_diversity.py")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "SHA256": sha256(args.output), "fold": report["fold"],
                      "summary": report["summaries"]["fixed10"]}))


if __name__ == "__main__":
    main()
