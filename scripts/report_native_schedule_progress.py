"""Read saved run.json metrics only; never read teachers, predictions, models, or GPU."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

EXCLUDED = ("MCL", "Synovitis")
FIXED_TEN = ("ACL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
             "Effusion", "Baker's", "Contusion", "Fracture")
REVIEWED_NATIVE = ("0f07e0e28867d17bfa269c5cb3053696837bfade3a045819371a5a8e4813ecb6",
                   "2ffa41e2fc650b61c5ccf2d5cf7c49974aff001025e604764238f6711efd44ef")


def require(value, message):
    if value is not True:
        raise ValueError(message)


def digest(body):
    return hashlib.sha256(body).hexdigest()


def semantic(value):
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())


def read_run(path):
    path = Path(path)
    body = path.read_bytes()
    value = json.loads(body)
    require(type(value) is dict and value.get("schema_version") == "native392_training_run_fast16_v1", "Native run required")
    return value, {"path": str(path.resolve()), "raw_sha256": digest(body), "status": value["status"],
                   "observed_completed_epochs": [row["epoch"] for row in value["history"]]}


def base_config(config):
    value = copy.deepcopy(config)
    value.pop("reason", None)
    value["train"].pop("lr_schedule", None)
    return value


def rates(run, row):
    config = run["config"]
    schedule = config["train"].get("lr_schedule", {"name": "constant"})
    require(schedule.get("name") in ("constant", "late_cosine"), "Unreviewed LR schedule")
    factor = 1.0
    if schedule["name"] == "late_cosine":
        require(schedule == {"name": "late_cosine", "constant_epochs": 6, "min_factor": 0.1}
                and config["train"]["epochs"] == 12, "Reviewed fixed12 late cosine required")
        if row["epoch"] > 6:
            progress = (row["epoch"] - 6) / 6
            factor = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    expected = {"head": config["train"]["learning_rate"] * factor,
                "last2": config["online"]["encoder_learning_rate"] * factor}
    recorded = row.get("learning_rates")
    if recorded is not None:
        require(set(recorded) == set(expected) and all(math.isclose(recorded[k], expected[k], rel_tol=1e-12) for k in expected),
                "Recorded epoch LR differs from schedule")
        require(math.isclose(row["learning_rate_multiplier"], factor, rel_tol=1e-12), "Recorded LR factor differs")
    return {"multiplier": factor, "groups": recorded or expected,
            "origin": "run.history" if recorded is not None else "inferred from original constant config"}


def metrics(run, row, previous_elapsed=0.0):
    per_target = row["metrics"]["per_target"]
    require(set(FIXED_TEN).issubset(per_target), "All ten fixed targets required")
    auc = {name: per_target[name]["auc"] for name in FIXED_TEN}
    require(all(type(value) in (int, float) and math.isfinite(value) for value in auc.values()), "Undefined fixed10 AUC; no replacement/drop")
    return {"epoch": row["epoch"], "train_bce": row["train_bce"], "weak_bce": row["weak_bce"],
            "weak_macro_auc_12": row["weak_macro_auc_12"], "weak_fixed10_macro_auc": sum(auc.values()) / 10,
            "per_target_auc_fixed10": auc, "optimizer_steps": row["optimizer_steps"],
            "optimizer_step_opportunities": row["optimizer_step_opportunities"], "amp_skipped_steps": row["amp_skipped_steps"],
            "cumulative_elapsed_seconds": row["elapsed_seconds"], "epoch_elapsed_seconds": row["elapsed_seconds"] - previous_elapsed,
            "gpu_peak_reserved_bytes": row["gpu_peak_reserved_bytes"], "rates": rates(run, row),
            "prediction_sha256_metadata_only": row["prediction_sha256"]}


def compare(old, new, through_epoch=None, require_complete=False):
    oa, na = old["audit"], new["audit"]
    same_fields = ("seed", "fold", "train_studies", "valid_studies", "targets", "parameters", "encoder_provenance",
                   "environment", "gpu", "microbatch_studies", "effective_batch_studies", "denominator", "engineering_lineage")
    checks = {name + "_same": old[name] == new[name] for name in same_fields}
    checks.update({"manifest_same": oa["manifest"] == na["manifest"], "manifest_hashes_same": oa["manifest_sha256"] == na["manifest_sha256"],
                   "fold_group_audit_same": oa["fold_audit"] == na["fold_audit"], "pixel_cache_index_same": oa["pixel_index_sha256"] == na["pixel_index_sha256"],
                   "config_except_reason_and_lr_schedule_same": base_config(old["config"]) == base_config(new["config"]),
                   "weak_BCE_selection_Gold0": all(run["selection_data"] == "weak_validation" and run["selection_rule"] == "bce"
                        and run["gold_used_for_training_or_selection"] is False and run["gold_evaluated"] is False for run in (old, new))})
    source_changes = {name: {"old": old["source_sha256"].get(name), "new": new["source_sha256"].get(name)}
                      for name in set(old["source_sha256"]) | set(new["source_sha256"])
                      if old["source_sha256"].get(name) != new["source_sha256"].get(name)}
    checks["reviewed_native_source_pair_only"] = (set(source_changes) == {"native_training_fast16.py"}
        and (old["source_sha256"]["native_training_fast16.py"], new["source_sha256"]["native_training_fast16.py"]) == REVIEWED_NATIVE)
    require(all(checks.values()), "Comparison inputs/config/source differ: " + ",".join(name for name, passed in checks.items() if not passed))
    if require_complete:
        require(old["status"] == new["status"] == "complete", "Final comparison requires both runs complete")
    histories = []
    for run in (old, new):
        require([row["epoch"] for row in run["history"]] == list(range(1, len(run["history"]) + 1)), "Sequential epoch history required")
        histories.append({row["epoch"]: row for row in run["history"]})
    available = min(len(histories[0]), len(histories[1]))
    last = available if through_epoch is None else through_epoch
    require(type(last) is int and 1 <= last <= available, "Requested completed epoch unavailable")
    rows = []
    numerical = ("train_bce", "weak_bce", "weak_macro_auc_12", "weak_fixed10_macro_auc", "optimizer_steps",
                 "optimizer_step_opportunities", "amp_skipped_steps", "cumulative_elapsed_seconds", "epoch_elapsed_seconds")

    def support(row):
        return {name: {key: value for key, value in item.items() if key != "auc"}
                for name, item in row["metrics"]["per_target"].items()}

    for epoch in range(1, last + 1):
        values = [metrics(run, history[epoch], history.get(epoch - 1, {}).get("elapsed_seconds", 0.0))
                  for run, history in zip((old, new), histories)]
        require(support(histories[0][epoch]) == support(histories[1][epoch]), "Per-target validation support changed")
        rows.append({"epoch": epoch, "old": values[0], "new": values[1],
                     "delta_new_minus_old": {name: values[1][name] - values[0][name] for name in numerical},
                     "per_target_auc_delta_fixed10": {name: values[1]["per_target_auc_fixed10"][name] - values[0]["per_target_auc_fixed10"][name] for name in FIXED_TEN},
                     "LR_factor_active": epoch > 6})
    selected = []
    for run, history in zip((old, new), histories):
        best = min(run["history"], key=lambda row: (row["weak_bce"], row["epoch"]))
        require(run["selected"]["epoch"] == best["epoch"] and run["selected"]["weak_bce"] == best["weak_bce"], "Selected is not earliest minimum weak BCE")
        selected.append({"status": run["status"], "selection_scope": "final" if run["status"] == "complete" else "provisional so far",
                         **metrics(run, history[best["epoch"]], history.get(best["epoch"] - 1, {}).get("elapsed_seconds", 0.0)),
                         "checkpoint_sha256_metadata_only": run["selected"].get("checkpoint_sha256")})
    return {"checks": checks, "inputs": {"seed": new["seed"], "fold": new["fold"], "train_studies": new["train_studies"], "valid_studies": new["valid_studies"],
            "manifest_sha256": na["manifest_sha256"], "manifest_inputs_sha256": na["manifest"]["inputs_sha256"],
            "grouping": na["manifest"]["grouping"], "pixel_index_sha256": na["pixel_index_sha256"], "patient_independence_verified": new["patient_independence_verified"]},
            "config": {"old_semantic_sha256": semantic(old["config"]), "new_semantic_sha256": semantic(new["config"]),
                       "common_without_reason_schedule_sha256": semantic(base_config(new["config"])),
                       "only_experimental_change": "train.lr_schedule", "metadata_change": "reason",
                       "old_schedule": old["config"]["train"].get("lr_schedule", {"name": "constant"}), "new_schedule": new["config"]["train"].get("lr_schedule")},
            "source": {"changed": source_changes, "unchanged": {name: value for name, value in new["source_sha256"].items() if name not in source_changes},
                       "reviewed_scope": "epoch checkpoint retention plus optional LR schedule/rate records; original model/cache/loss and earliest-BCE selection retained",
                       "CLI_sha256": {"old": oa["resource_gate"]["CLI_sha256"], "new": na["resource_gate"]["CLI_sha256"]},
                       "CLI_change_scope": "QA target arm consumes requested config; CUDA-first gate maintained",
                       "resource_gate_sha256": {"old": oa["resource_gate_sha256"], "new": na["resource_gate_sha256"]}},
            "metric_scope": {"teacher": "weak only", "fixed10_targets": list(FIXED_TEN), "excluded_targets": list(EXCLUDED), "Gold_or_Public_evaluation": False},
            "compared_through_epoch": last, "epoch_comparisons": rows,
            "first6_numeric_differences": [row for row in rows if row["epoch"] <= 6],
            "BCE_selected_comparison": {"old": selected[0], "new": selected[1], "both_final": old["status"] == new["status"] == "complete",
                "delta_new_minus_old": {name: selected[1][name] - selected[0][name] for name in ("weak_bce", "weak_fixed10_macro_auc", "weak_macro_auc_12")}},
            "interpretation": "Epochs1-6 use equal LR. Tiny initial metric differences are numerical observations, not an LR improvement claim. LR-factor assessment starts epoch7; final earliest-BCE comparison is required. Wall time is observed, not causally attributed. Weak gains do not prove Public gains."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run", type=Path, required=True)
    parser.add_argument("--new-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--through-epoch", type=int)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    require(not args.output.exists(), "Fresh output required; never overwrite prior observations")
    old, old_snapshot = read_run(args.old_run)
    new, new_snapshot = read_run(args.new_run)
    report = compare(old, new, args.through_epoch, args.require_complete)
    report.update(schema_version="native_schedule_progress_metrics_only_v1", at_utc=datetime.now(timezone.utc).isoformat(),
                  snapshots={"old": old_snapshot, "new": new_snapshot}, reporter_sha256=digest(Path(__file__).read_bytes()),
                  teacher_prediction_model_GPU_API_reads_or_calls=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output.read_bytes()), "through_epoch": report["compared_through_epoch"],
                      "epoch_deltas": [{"epoch": row["epoch"], **row["delta_new_minus_old"]} for row in report["epoch_comparisons"]]}, indent=2))


if __name__ == "__main__":
    main()
