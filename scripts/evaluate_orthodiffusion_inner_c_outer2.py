"""One root-released outer2 comparison. Default template cannot release targets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from rsna_knee import orthodiff_weak_head as H  # noqa: E402
from rsna_knee.contracts import ID, TARGETS  # noqa: E402

PLAN = ROOT / "artifacts/research/20261017-orthodiff-innerC-outer2-evaluator-design-v1"
TEMPLATE = PLAN / "root-release-template.json"
DESIGN_SHA = "1969da87673ad3f693d7f2d7d9d5431db8ac8b5d9287b209c906318f270494ce"
IMPLEMENTATION_REVIEW_SHA = "7595fea6c5f07504a3916a872a475635b09b0b74dfea9b67ced38e56912efac2"
POLICY_SHA = "dd26d15c5b27c624c08eeee84e65d828b4c0df572854dc301af149954a1c2c6f"
H_SHA = "e1149f8e88eeba0f32418e8ee4986cd41ef5aae208a31142cd907689958d78e7"
SCHEMA = "orthodiff_innerC_outer2_one_evaluation_run_v1"
META = (ID, "group_id", "fold", "source")
GATE_FLAGS = ("native392_pixel16_complete", "native392_byte_QC_passed", "compression_measured",
              "direct_archive_read_speed_measured", "artificial_CUDA_gradient_and_roundtrip_passed")


def path_in_root(value):
    path = Path(value)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    H.require(path.is_relative_to(ROOT), "Input/output must stay inside this workspace")
    return path


def receipt_file(receipt):
    path = path_in_root(receipt["path"])
    H.check_pin(path, receipt)
    return path


def json_from_receipt(receipt):
    return H.read_json(receipt_file(receipt))


def uid_hash(rows):
    return hashlib.sha256("".join(r[ID] + "\n" for r in rows).encode()).hexdigest()


def validate_control_records(run, index, gate, preflight, config, expected_sources,
                             expected_gate_sha, csv_pin, checkpoint_pin):
    """Metadata only. Never choose a checkpoint using any recorded AUC/BCE."""
    H.require(run["schema_version"] == "native392_training_run_fast16_fold2_v1"
              and run["status"] == "complete" and type(run["fold"]) is int and run["fold"] == 2
              and run["seed"] == 20261002 and run["train_studies"] == 3319 and run["valid_studies"] == 888
              and run["targets"] == list(TARGETS) and run["config"] == config
              and run["source_sha256"] == expected_sources
              and run["selection_data"] == "weak_validation" and run["selection_rule"] == "bce"
              and run["gold_used_for_training_or_selection"] is False and run["gold_evaluated"] is False
              and run["resume_supported"] is False, "Completed independent fixed-recipe fold2 run required")
    H.require(config["seed"] == 20261002 and config["online"]["folds"] == [2]
              and config["train"]["epochs"] == 12 and "augmentation" not in config,
              "Unaugmented original fold2 twelve-epoch config required")
    semantic = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    H.require(gate["schema_version"] == "native392_training_resource_gate_fast16_fold2_v1"
              and all(gate.get(k) is True for k in GATE_FLAGS)
              and gate["source_sha256"] == expected_sources
              and gate["config_semantic_sha256"]["2"] == semantic
              and run["audit"]["resource_gate_sha256"] == expected_gate_sha
              and run["audit"]["resource_gate"] == gate, "Existing native resource/source/config binding differs")
    H.require(preflight["ready"] is True and preflight["train_studies"] == 3319
              and preflight["valid_studies"] == 888 and preflight["cache_studies"] == 4207
              and preflight["fitting_performed"] is False, "Existing manual CUDA preflight population differs")
    history = run["history"]
    H.require(len(history) == 12 and [r["epoch"] for r in history] == list(range(1, 13)), "Complete ordered twelve-epoch history required")
    for row in history:
        steps, skips, opportunities = (row[k] for k in ("optimizer_steps", "amp_skipped_steps", "optimizer_step_opportunities"))
        H.require(all(type(x) is int for x in (steps, skips, opportunities))
                  and steps > 0 and skips >= 0 and steps + skips == opportunities == 208
                  and all(math.isfinite(row[k]) for k in ("train_bce", "weak_bce")),
                  "Optimizer/AMP accounting or completed finite loss differs")
    H.require(index["schema_version"] == "native392_epoch_suffix_head_retention_v1"
              and index["checkpoint_schema"] == "native392_generic_dino_suffix_head_fast16_fold2_v1"
              and type(index["fold"]) is int and index["fold"] == 2
              and index["source_sha256"] == expected_sources and index["targets"] == list(TARGETS)
              and index["selection_data"] == "weak_validation" and index["resume_supported"] is False
              and index["selection_rule"] == "minimum observed weak BCE; earliest exact tie"
              and len(index["checkpoints"]) == 12
              and [r["epoch"] for r in index["checkpoints"]] == list(range(1, 13)),
              "New fold2 retention schema/lineage differs")
    final = index["checkpoints"][11]
    H.require(final["predictions_csv"] == "epoch_012_weak.csv" and final["checkpoint"] == "epoch_012.pt"
              and final["prediction_sha256"] == history[11]["prediction_sha256"] == csv_pin["sha256"]
              and final["checkpoint_sha256"] == checkpoint_pin["sha256"]
              and final["checkpoint_bytes"] == checkpoint_pin["bytes"]
              and final["weak_bce"] == history[11]["weak_bce"], "Unconditional E12 checkpoint/CSV linkage differs")
    return {"fixed_epoch": 12, "optimizer_opportunities_per_epoch": 208,
            "AMP_skips_total": sum(r["amp_skipped_steps"] for r in history),
            "checkpoint_binary_deserialized_or_rehashed_here": False,
            "checkpoint_hash_binding": "Root engineering attestation and hash-bound retention index; no alias selected"}


def read_scores(path, expected_rows):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        H.require(next(reader) == [ID, *TARGETS], "Exact13 prediction columns required")
        records = list(reader)
    H.require(all(len(r) == 13 for r in records) and len(records) == len(expected_rows)
              and len({r[0] for r in records}) == len(records)
              and {r[0] for r in records} == {r[ID] for r in expected_rows},
              "Prediction UID set must match exactly; no join/drop")
    keyed = {r[0]: r[1:] for r in records}
    values = np.asarray([[float(x) for x in keyed[r[ID]]] for r in expected_rows], dtype=np.float64)
    H.require(np.isfinite(values).all() and np.all((values >= 0) & (values <= 1)), "Finite0..1 control/head scores required")
    return values


def prepare_before_targets(release, expected_evaluator_sha):
    """Every real engineering/source/UID guard precedes target projection."""
    H.require(release["schema_version"] == "orthodiff_innerC_outer2_root_target_release_v1"
              and release.get("authorized") is True
              and release.get("status") == "AUTHORIZED_ONE_FIXED_OUTER2_COMPARISON"
              and release.get("decision") == "RELEASE_FIXED_OUTER2_TARGETS_FOR_ONE_PREDECLARED_COMPARISON"
              and release.get("root_release_created_at_UTC")
              and release["permitted_target_fold"] == 2 and release["permitted_target_rows"] == 888
              and release["fixed_alpha"] == .2 and release["one_primary_comparison_only"] is True,
              "Explicit populated root release required")
    H.require(release["design"]["sha256"] == DESIGN_SHA, "Fixed evaluator design differs")
    design = json_from_receipt(release["design"])
    H.require(H.pin(PLAN / "implementation-review.json")["sha256"] == IMPLEMENTATION_REVIEW_SHA,
              "Root implementation review differs")
    H.require(release["policy"] == design["policy"] and release["policy"]["sha256"] == POLICY_SHA,
              "Frozen policy differs")
    policy = json_from_receipt(release["policy"])
    for relative, expected in policy["immutable_before_pins"].items():
        H.check_pin(ROOT / relative, expected)
    H.require(H.pin(H.__file__)["sha256"] == H_SHA and release["H_math"]["sha256"] == H_SHA,
              "Immutable H numerical source differs")
    receipt_file(release["H_math"])
    evaluator = release["evaluator"]
    H.require(evaluator["sha256"] == expected_evaluator_sha == H.pin(__file__)["sha256"], "Explicit evaluator source SHA differs")
    receipt_file(evaluator)
    qa = json_from_receipt(evaluator["synthetic_CPU_QA_receipt"])
    H.require(qa["schema_version"] == "orthodiff_innerC_outer2_evaluator_artificial_qualification_v1"
              and qa["passed"] is True and qa["actual_target_projection_or_comparison_performed"] is False
              and qa["sources"]["scripts/evaluate_orthodiffusion_inner_c_outer2.py"]["sha256"] == expected_evaluator_sha,
              "Hash-bound synthetic evaluator QA differs")
    environment = H.environment(require_fixed=True)
    H.require(environment["versions"] == release["expected_CPU_versions"] and "torch" not in sys.modules,
              "Fixed CPU environment/no Torch required")
    H.require(release["head"] == design["head_authoritative_pins"], "Canonical sealed head differs")
    for receipt in release["head"].values():
        receipt_file(receipt)
    head_run = json_from_receipt(release["head"]["run"])
    audit = json_from_receipt(release["head"]["target_free_actual_audit"])
    H.require(head_run["status"] == "SEALED_OUTER2_PREDICTIONS_NO_TARGET_EVALUATION"
              and head_run["selected_common_C"] == .01 and head_run["new_outer2_AUC"] is None
              and audit["passed"] is audit["target_free_prediction_seal_engineering_qualified"] is True
              and audit["run"]["sha256"] == release["head"]["run"]["sha256"], "Qualified target-free head required")
    cohort = json_from_receipt(release["cohort"])
    native_cohort = json_from_receipt(release["native_cohort"])
    H.require(release["cohort"] == design["target_release"]["metadata_cohort"]
              and release["native_cohort"] == design["target_release"]["native_metadata_cohort"]
              and cohort["requested"] == native_cohort["requested"]
              and cohort["Gold_UID_overlap"] == cohort["Gold_supplied_group_overlap"] == 0
              and native_cohort["train_validation_group_overlap"] == 0, "Original no-Gold group metadata differs")
    outer = [r for r in cohort["requested"] if r["fold"] == "2"]
    metadata = json_from_receipt(release["head"]["outer2-metadata-only.json"])
    H.require(len(outer) == 888 and uid_hash(outer) == release["outer2_UID_order_sha256"] == native_cohort["valid_UID_order_sha256"]
              and [{k: r[k] for k in META} for r in metadata["all_rows"]] == outer,
              "Exact888 outer2 metadata order differs")
    success_rows = [r for r in metadata["all_rows"] if r["feature_success"]]
    failure_rows = [r for r in metadata["all_rows"] if not r["feature_success"]]
    H.require(len(success_rows) == 654 and len(failure_rows) == 234
              and uid_hash(success_rows) == release["success_UID_order_sha256"]
              and uid_hash(failure_rows) == release["failure_UID_order_sha256"], "654/234 frozen head partition differs")
    control = release["native_control"]
    control_design = design["native_control"]
    H.require(control["primary_control_epoch"] == 12 and control["trained_fold"] == 2
              and control["config"] == control_design["config"] and control["CLI"] == control_design["CLI"]
              and control["sources"] == control_design["sources"], "Only original fixed-recipe F006fold2 E12 control allowed")
    native_directory = path_in_root(control_design["planned_run_directory"])
    for field, filename in (("run", "run.json"), ("retention_index", "epoch_checkpoint_index.json"),
                            ("epoch12_checkpoint", "epoch_012.pt"), ("epoch12_CSV", "epoch_012_weak.csv")):
        H.require(path_in_root(control[field]["path"]) == native_directory / filename,
                  "Control files must belong to the one planned native fold2 run")
    receipt_file(control["CLI"])
    expected_sources = {}
    for name, receipt in control["sources"].items():
        source_path = receipt_file(receipt)
        normalized = hashlib.sha256(source_path.read_text(encoding="utf-8").replace("\r\n", "\n").encode()).hexdigest()
        H.require(normalized == receipt["normalized_source_sha256"], "Normalized native source differs")
        expected_sources[name] = normalized
    config = json_from_receipt(control["config"])
    native_run, index, gate, preflight = (json_from_receipt(control[k]) for k in
                                         ("run", "retention_index", "resource_gate", "manual_preflight_receipt"))
    checkpoint_pin, csv_pin = control["epoch12_checkpoint"], control["epoch12_CSV"]
    H.require(Path(checkpoint_pin["path"]).name == "epoch_012.pt"
              and Path(csv_pin["path"]).name == "epoch_012_weak.csv", "FixedE12 filenames required")
    engineering = validate_control_records(native_run, index, gate, preflight, config, expected_sources,
                                          control["resource_gate"]["sha256"], csv_pin, checkpoint_pin)
    attest = control["completed_control_engineering_attestation"]
    H.require(attest["root_checked"] is True
              and attest["source_config_group_noGold_epoch12_checkpoint_and_CSV_linkage_passed"] is True
              and attest["completed_before_target_release"] is True, "Root completed-control engineering attestation required")
    H.require(native_run["audit"]["manifest_sha256"] == {k: v["sha256"] for k, v in cohort["manifest_pins"].items()}
              and native_run["audit"]["pixel_index_sha256"] == native_cohort["cache_index"]["sha256"],
              "Native control uses different manifest/cache identity")
    raw = read_scores(receipt_file(csv_pin), outer)
    successful_head = read_scores(receipt_file(release["head"]["outer2-success-probabilities-f32.csv"]), success_rows)
    model = json_from_receipt(release["head"]["numeric-model.json"])
    H.require(model["targets"] == list(TARGETS) and model["feature_dim"] == 1024
              and model["selected_common_C"] == .01
              and [r["target"] for r in model["heads"]] == list(TARGETS)
              and all(r["status"] == "qualified" for r in model["heads"]), "Canonical twelve qualified head metadata differs")
    success_set = {r[ID] for r in success_rows}
    success_mask = np.asarray([r[ID] in success_set for r in outer], dtype=bool)
    head = np.full((888, 12), np.nan, dtype=np.float64)
    head[success_mask] = successful_head
    H.require(release["weak_csv"] == design["target_release"]["weak_csv_pin"], "Original weak CSV pin differs")
    weak_path = receipt_file(release["weak_csv"])  # Hash bytes; no target projection here.
    return {"cohort": cohort, "outer": outer, "weak_path": weak_path, "raw": raw,
            "head": head, "success_mask": success_mask, "model": model,
            "environment": environment, "control_engineering": engineering}


def project_fold2(header, token_rows, cohort_rows):
    H.require(len(header) == len(set(header)) and set((*META, *TARGETS)).issubset(header), "Unique required label CSV columns missing")
    columns = {name: header.index(name) for name in (*META, *TARGETS)}
    projected, groups = [], {}
    count = 0
    for count, tokens in enumerate(token_rows, 1):
        H.require(count <= len(cohort_rows) and len(tokens) == len(header), "Label CSV row count/width differs")
        metadata = {name: tokens[columns[name]] for name in META}
        H.require(metadata == cohort_rows[count - 1] and metadata["source"] == "report_weak", "Label source/cohort order differs")
        H.require(groups.setdefault(metadata["group_id"], metadata["fold"]) == metadata["fold"], "Supplied/report group crosses folds")
        if metadata["fold"] != "2":
            continue
        labels = H.labels_from_cells([tokens[columns[name]] for name in TARGETS])
        projected.append({**metadata, "targets": [None if np.isnan(x) else float(x) for x in labels]})
    H.require(count == len(cohort_rows), "Full metadata cohort must be inspected")
    return projected


def decision(evaluation, engineering=True):
    delta = evaluation["fixed10_delta"]
    passed = bool(engineering and evaluation["fixed10_heads_qualified"] and delta is not None
                  and math.isfinite(delta) and delta > 0)
    return {"weak_screen_passed": passed,
            "decision": "ROOT_MAY_CONSIDER_ONE_SEALED_MODEL" if passed else "HOLD",
            "BCE_CI_report_only": True, "automatic_followup": False,
            "Public_adoption_or_Submit_authorized": False,
            "C_alpha_target_search_or_all4207_refit_permitted": False}


def compare_once(labels, raw, head, success_mask, groups, model):
    supported = {r["target"]: r["status"] == "qualified" for r in model["heads"]}
    baseline, blend, details = H.rank_blend(raw, head, success_mask, supported)
    evaluation = H.evaluate(labels, raw, baseline, blend, head, success_mask, groups, model, details)
    return baseline, blend, evaluation, decision(evaluation)


def write_scores(path, rows, values):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([ID, *TARGETS])
        for row, scores in zip(rows, values):
            writer.writerow([row[ID], *[format(float(x), ".9g") for x in scores]])


def execute_release(release, release_path, expected_release_sha, expected_evaluator_sha, output):
    H.require(H.pin(release_path)["sha256"] == expected_release_sha, "Explicit actual root-release SHA differs")
    prepared = prepare_before_targets(release, expected_evaluator_sha)
    output = path_in_root(output)
    H.require(output == path_in_root(release["output_dir"])
              and output.is_relative_to((ROOT / "artifacts/runs").resolve()), "Release-bound fresh output directory required")
    output.mkdir(parents=True, exist_ok=False)
    H.save_json(output / "comparison-started.json", {"schema_version": SCHEMA,
                "release": {"path": str(Path(release_path).resolve()), **H.pin(release_path)},
                "all_engineering_guards_passed_before_target_projection": True,
                "one_comparison_only": True, "target_projection_started": True})
    try:
        with prepared["weak_path"].open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            projected = project_fold2(next(reader), reader, prepared["cohort"]["requested"])
        H.require(len(projected) == 888 and [{k: r[k] for k in META} for r in projected] == prepared["outer"], "Exactly888 fold2 targets required")
        labels = np.asarray([[np.nan if x is None else x for x in r["targets"]] for r in projected], dtype=np.float64)
        baseline, blend, evaluation, result = compare_once(labels, prepared["raw"], prepared["head"],
            prepared["success_mask"], [r["group_id"] for r in projected], prepared["model"])
        write_scores(output / "cv-rank-baseline.csv", projected, baseline)
        write_scores(output / "cv-rankmix20.csv", projected, blend)
        H.save_json(output / "evaluation.json", evaluation)
        H.save_json(output / "decision.json", result)
        target_view_hash = hashlib.sha256(json.dumps(projected, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        H.save_json(output / "run.json", {"schema_version": SCHEMA, "status": "complete",
            "release": {"path": str(Path(release_path).resolve()), **H.pin(release_path)},
            "evaluator": {"path": str(Path(__file__).resolve()), **H.pin(__file__)},
            "environment": prepared["environment"], "control_engineering": prepared["control_engineering"],
            "logical_target_projection": {"fold": 2, "rows": 888, "otherfold_target_values_accessed": False,
                "view_SHA256": target_view_hash, "CSV_syntax_hash_bytes_can_include_otherfold_targets": True},
            "feature_partition": {"all": 888, "success": 654, "failure": 234},
            "fixed10_delta": evaluation["fixed10_delta"], "decision": result,
            "outputs": {p.name: H.pin(p) for p in output.iterdir() if p.is_file()},
            "newfit_features_Torch_GPU_network_Save_Submit": 0,
            "target_release_scope": "NewC evaluator view only; native trainer ordinarily validates onfold2 each epoch with auxiliary bestBCE; primary here alwaysE12"})
        return 0
    except Exception as exc:
        H.save_json(output / "run.json", {"schema_version": SCHEMA, "status": "failed_no_automatic_retry",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "root_release_consumed_for_this_output": True,
                    "C_alpha_target_search_permitted": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-release", type=Path, default=TEMPLATE)
    parser.add_argument("--expected-root-release-sha256")
    parser.add_argument("--expected-evaluator-sha256")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    release = H.read_json(args.root_release)
    if release.get("authorized") is not True:
        print(json.dumps({"status": "DISABLED_NO_ACTUAL_ROOT_RELEASE", "target_projection_or_comparison_performed": False}))
        return 0
    H.require(args.output_dir is not None and args.expected_root_release_sha256 and args.expected_evaluator_sha256,
              "Explicit output/root-release/evaluator SHA arguments required")
    return execute_release(release, args.root_release, args.expected_root_release_sha256,
                           args.expected_evaluator_sha256, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
