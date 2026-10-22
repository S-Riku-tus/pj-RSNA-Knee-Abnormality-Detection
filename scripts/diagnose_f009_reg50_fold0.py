"""CSV-only REG50 report after completed guarded F009 primary; fold0 only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsna_knee.contracts import ID, TARGETS, probability  # noqa: E402

HERE = ROOT / "artifacts/research/20261017-f009-reg50-secondary-preparation-v1"
POLICY_SHA = "7f5cbc2d4769e6f39d4ad0c09634d53c3dd5197a2a9875a5fa8f466387932455"
BASE = "artifacts/research/20261017-f009-gamma-preparation-v1"
META = (ID, "group_id", "fold", "source")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_bytes())


def pin(path):
    raw = Path(path).read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def bound(path, expected):
    require(pin(path) == {k: expected[k] for k in ("bytes", "sha256")}, "Input SHA/size differs: " + str(path))


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def validate_primary(guard, result, guard_pin, result_pin, policy):
    """Metadata only; primary gate may be negative and is never recomputed."""
    require(guard["schema_version"] == "F009_fixed_E12_guarded_evaluation_receipt_v1"
            and guard["complete"] is True and guard["original_evaluate_calls"] == 1
            and guard["original_policy_thresholds_CI_BCE_output_modified"] is False
            and guard["original_result_pin"] == result_pin
            and guard["result_status"] == result["status"]
            and guard["fold1_automatic_start"] is False and guard["root_review_still_required"] is True
            and guard["wrapper_pin"] == {k: policy["guard_wrapper"][k] for k in ("bytes", "sha256")},
            "Completed hash-bound original guarded primary required first")
    g = guard["guards"]
    require(g["schema_version"] == "F009_fixed_E12_input_guard_v1" and g["complete"] is True
            and g["Gold_body_opened_or_hashed"] is False and g["model_deserialized"] is False
            and g["target_projection_or_statistics_before_legacy_evaluate"] is False,
            "Existing primary guard qualification differs")
    require(result["status"] == "F009_E12_WEAK_COMPARISON_ONLY_POLICY_V2"
            and result["candidate_epoch"] == result["control_epoch"] == 12
            and result["policy_sha256"] == policy["parent_policy"]["sha256"]
            and result["ordered_UID_sha256"] == policy["original_v2_sorted_UID_json_sha256"]
            and result["fold1_automatic_start"] is False and result["root_review_still_required"] is True
            and guard["fold1_gate_conditions_met"] == result["fold1_gate_conditions_met"]
            and type(result["fold1_gate_conditions_met"]) is bool, "Original primary result identity differs")
    for label in ("control", "candidate"):
        original, completed = result["inputs"][label], g["completed_E12_control_candidate"][label]
        require(original["run_sha256"] == completed["run"]["sha256"]
                and original["epoch_index_sha256"] == completed["index"]["sha256"]
                and original["prediction_sha256"] == completed["E12_prediction"]["sha256"]
                and original["checkpoint_sha256_reference_only_not_loaded"] == completed["E12_checkpoint_body_stream_hash"]["sha256"],
                "Guard/result fixedE12 linkage differs: " + label)
    require(g["completed_E12_control_candidate"]["control"]["E12_prediction"] ==
            {k: policy["fixed_control_E12"][k] for k in ("bytes", "sha256")}, "Original CLS E12 control changed")
    require(len(guard_pin["sha256"]) == 64, "Explicit primary guard SHA required")
    return g["completed_E12_control_candidate"]


def validate_REG0(regional, fixed):
    selected = regional["selected"]
    require(regional["status"] == "complete" and regional["fold"] == 0
            and regional["seed"] == 20261002 and regional["targets"] == list(TARGETS)
            and regional["gold_used_for_training_or_selection"] is False and regional["gold_evaluated"] is False
            and selected["epoch"] == fixed["selected_epoch"] == 8
            and selected["prediction_sha256"] == fixed["prediction"]["sha256"]
            and selected["checkpoint_sha256"] == fixed["checkpoint_SHA_reference_only_not_loaded"],
            "Only the frozen already-selected REG0 epoch8 is allowed")


def f32(value):
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def mean2_fp32(left, right):
    a, b = probability(left), probability(right)
    return f32(f32(f32(a) + f32(b)) / f32(2.))


def predictions(path, ordered):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        require(next(reader) == [ID, *TARGETS], "Exact13 prediction columns required")
        records = list(reader)
    require(len(records) == len(ordered) and all(len(r) == 13 for r in records)
            and len({r[0] for r in records}) == len(records)
            and {r[0] for r in records} == {r[ID] for r in ordered}, "All prediction UID sets must match exactly")
    return {r[0]: {ID: r[0], **{t: probability(v) for t, v in zip(TARGETS, r[1:])}} for r in records}


def project_fold0(header, token_rows, metadata_rows):
    require(len(header) == len(set(header)) and set((*META, *TARGETS)).issubset(header), "Required unique weak columns missing")
    index = {name: header.index(name) for name in (*META, *TARGETS)}
    truth, groups, count = {}, {}, 0
    for count, tokens in enumerate(token_rows, 1):
        require(count <= len(metadata_rows) and len(tokens) == len(header), "Weak row count/width differs")
        metadata = {name: tokens[index[name]] for name in META}
        require(metadata == metadata_rows[count - 1] and metadata["source"] == "report_weak", "Original weak metadata/order differs")
        require(groups.setdefault(metadata["group_id"], metadata["fold"]) == metadata["fold"], "Supplied group crosses folds")
        if metadata["fold"] != "0":
            continue
        cells = {name: tokens[index[name]] for name in TARGETS}
        for value in cells.values():
            probability(value, missing=True, binary=True)
        truth[metadata[ID]] = {**metadata, **cells}
    require(count == len(metadata_rows) and len(truth) == sum(r["fold"] == "0" for r in metadata_rows), "Exact fold0 target projection required")
    return truth


def load_original_paired(policy):
    path = ROOT / policy["original_evaluator"]["path"]
    spec = importlib.util.spec_from_file_location("f009_REG50_pinned_original_v2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.ROOT.resolve() == ROOT and tuple(module.TARGETS) == tuple(TARGETS)
            and Path(module.B.__file__).resolve() == (ROOT / "scripts/bootstrap_weak_comparison.py").resolve()
            and Path(module.B.HELPERS.__file__).resolve() == (ROOT / "scripts/summarize_controlled_runs.py").resolve(),
            "Pinned original paired/helper import origin differs")
    return module.paired


def compare_secondary(truth, cls_control, cls_candidate, REG0, paired_function):
    require(truth.keys() == cls_control.keys() == cls_candidate.keys() == REG0.keys(), "Identical full UID sets required")
    left, right = {}, {}
    for uid in truth:
        left[uid] = {ID: uid, **{t: mean2_fp32(cls_control[uid][t], REG0[uid][t]) for t in TARGETS}}
        right[uid] = {ID: uid, **{t: mean2_fp32(cls_candidate[uid][t], REG0[uid][t]) for t in TARGETS}}
    values = paired_function(truth, left, right, seed=20261012, repeats=1000)
    return left, right, values


def write_scores(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([ID, *TARGETS])
        for uid, row in rows.items():
            writer.writerow([uid, *[format(row[t], ".9g") for t in TARGETS]])


def run(args):
    require(pin(__file__)["sha256"] == args.expected_cli_sha256, "Explicit secondary CLI SHA differs")
    policy_path = HERE / "policy.json"
    require(pin(policy_path)["sha256"] == POLICY_SHA, "Fixed REG50 secondary policy differs")
    policy = read(policy_path)
    for name in ("parent_policy", "tracked_parent_policy", "original_evaluator", "guard_wrapper", "cohort"):
        receipt = policy[name]
        bound(ROOT / receipt["path"], receipt)
    for receipt in policy["helpers"].values():
        bound(ROOT / receipt["path"], receipt)
    guard_pin, result_pin = pin(args.guard_receipt), pin(args.primary_result)
    require(guard_pin["sha256"] == args.expected_guard_receipt_sha256
            and result_pin["sha256"] == args.expected_primary_result_sha256, "Explicit completed primary/guard SHA differs")
    guard, primary = read(args.guard_receipt), read(args.primary_result)
    completed = validate_primary(guard, primary, guard_pin, result_pin, policy)
    control_path = ROOT / policy["fixed_control_E12"]["path"]
    candidate_path = ROOT / policy["fixed_candidate_E12_path"]
    require(Path(primary["control"]).resolve() == control_path.parent.resolve()
            and Path(primary["candidate"]).resolve() == candidate_path.parent.resolve()
            and Path(guard["guards"]["candidate_run"]).resolve() == candidate_path.parent.resolve(),
            "Only the original completed fold0 runs are allowed")
    for path, label in ((control_path, "control"), (candidate_path, "candidate")):
        bound(path, completed[label]["E12_prediction"])
    fixed = policy["fixed_REG0"]
    bound(ROOT / fixed["run"]["path"], fixed["run"])
    validate_REG0(read(ROOT / fixed["run"]["path"]), fixed)
    reg_path = ROOT / fixed["prediction"]["path"]
    bound(reg_path, fixed["prediction"])
    cohort = read(ROOT / policy["cohort"]["path"])
    require(cohort["Gold_UID_overlap"] == cohort["Gold_supplied_group_overlap"] == 0, "No-Gold metadata differs")
    ordered = [r for r in cohort["requested"] if r["fold"] == "0"]
    require(len(ordered) == 821 and hashlib.sha256("".join(r[ID] + "\n" for r in ordered).encode()).hexdigest() == policy["original_fold0_UID_order_sha256"], "Exact821 metadata cohort required")
    control, candidate, REG0 = (predictions(p, ordered) for p in (control_path, candidate_path, reg_path))
    weak_path = ROOT / policy["manifest_dir"] / "weak.csv"
    bound(weak_path, policy["weak_csv_pin"])
    paired = load_original_paired(policy)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    save(args.output_dir / "started.json", {"status": "PRIMARY_GUARDS_COMPLETE_BEFORE_SECONDARY_TARGET_PROJECTION",
         "primary_guard": guard_pin, "primary_result": result_pin, "primary_gate_will_not_change": True})
    with weak_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        truth = project_fold0(next(reader), reader, cohort["requested"])
    require(list(truth) == [r[ID] for r in ordered], "Exact ordered821 truth required")
    left, right, values = compare_secondary(truth, control, candidate, REG0, paired)
    write_scores(args.output_dir / "CLScontrolE12-REG0-mean2-f32.csv", left)
    write_scores(args.output_dir / "F009E12-REG0-mean2-f32.csv", right)
    result = {"schema_version": "F009_REG50_fold0_secondary_diagnostic_v1", "status": "COMPLETE_REPORT_ONLY",
        "fold": 0, "seed": policy["seed"], "UID_count": 821, "policy": {"path": str(policy_path), **pin(policy_path)},
        "secondary_CLI": {"path": str(Path(__file__).resolve()), **pin(__file__)},
        "runtime": {"executable": sys.executable, "python": sys.version.split()[0], "FP32": "stdlib IEEE754 binary32 struct"},
        "primary_guard": {"path": str(args.guard_receipt), **guard_pin},
        "original_primary_result": {"path": str(args.primary_result), **result_pin},
        "original_primary_fold1_gate_conditions_met_unmodified": primary["fold1_gate_conditions_met"],
        "secondary_can_override_primary_or_fold1_gate": False,
        "REG0": fixed, "FP32_rule": policy["FP32_rule"], "secondary_paired": values,
        "outputs": {p.name: pin(p) for p in args.output_dir.iterdir() if p.is_file()},
        "newfit_otherfoldmodel_Gold_Report_MRI_Torch_GPU_API_Save_Submit": 0,
        "C_gamma_alpha_targets_epoch_or_fold1_gate_change": False,
        "limits": [policy["interpretation"], "Repeated weak holdout/supplied groups; no independent-patient or Public0.960 guarantee"]}
    save(args.output_dir / "secondary-result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-result", type=Path, required=True)
    parser.add_argument("--expected-primary-result-sha256", required=True)
    parser.add_argument("--guard-receipt", type=Path, required=True)
    parser.add_argument("--expected-guard-receipt-sha256", required=True)
    parser.add_argument("--expected-cli-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps({"status": result["status"], "secondary_report_only": True, "primary_or_fold1_gate_changed": False}))


if __name__ == "__main__":
    main()
