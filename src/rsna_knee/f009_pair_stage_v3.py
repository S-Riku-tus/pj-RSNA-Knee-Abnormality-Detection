"""Local F009 pair sealer v3 for the separately amended joint evaluation.

Only exact six-document qualification and actual strict CPU receipts may
open checkpoint copy paths. Artificial tests remain explicitly artificial.
No old gate/result is rewritten, no model is deserialized, and no API exists.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from . import f009_pair_stage_v2 as previous
from . import f009_source_namespace as sources

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS_SHA = "b1687e813ecdecfadff8b5adc21e02577a5c57998cc0891899fa4034f1043813"
CPU_PRODUCER_SHA = "fc2fbf7ab581bc6258817ac9fb189de966ce8b2d074b531e7bf3b642d1fe0705"
CPU_PRODUCER_CLI_SHA = "8f57bd7688d5507af8fce7151793acb1c2e0a185401ef474234a4b4605d9667d"
APPROVED_ACTUAL_CPU_PRODUCERS = frozenset({CPU_PRODUCER_SHA})
CPU_SCHEMA = "f009_strict_CPU_amended_eval_v2"
PAIR_SCHEMA = "f009_fixed_E12_two_CLS_amended_source14_local_pair_v3"
BUNDLE_SCHEMA = "f009_fixed_E12_single_CLS_amended_source14_local_bundle_v3"
SPEC_SCHEMA = "f009_pair_stage_amended_root_pins_v3"
QUALIFICATION_SCHEMA = "F009_amended_joint_metadata_v2"
QUALIFICATION_KEYS = {"result0", "guard0", "result1", "guard1", "legacy_guard1", "legacy_comparison1"}
BINDING_FILENAMES = {"policy0": "policy0.json", "policy1": "original-policy1.json",
    "policy2": "amended-policy2.json", "declaration": "amendment-declaration.json",
    "snapshot": "epoch0-snapshot.json", "legacy_evaluator": "original-fold1-evaluator.py",
    "amended_wrapper": "amended-fold1-wrapper.py"}
TARGETS = previous.TARGETS
HELPER_SHA = previous.HELPER_SHA
require, digest, pin = previous.require, previous.digest, previous.pin
expected_tensor_shapes = previous.expected_tensor_shapes
copy_exact, atomic_json = previous.copy_exact, previous.atomic_json


def _validate_cpu_base(cpu, *, fold, run_pin, index_pin, checkpoint, prefix, fixture_only):
    require(cpu.get("complete") is True and cpu.get("producer_CLI_sha256") == CPU_PRODUCER_CLI_SHA
            and cpu.get("eval_gamma_RNG_unchanged") is True
            and cpu.get("strict_apply_identity_and_tensor_bytes_exact") is True
            and cpu.get("checkpoint_loader") == {"source_sha256": sources.CORE_SHA["native_training_fast16.py"],
                "map_location": "cpu", "weights_only": True, "allow_artificial": False},
            "New explicit CPU completion/producer CLI/strict loader/eval evidence required")
    require(cpu["schema_version"] == CPU_SCHEMA and type(cpu["fold"]) is int and cpu["fold"] == fold and cpu["epoch"] == 12
            and cpu["source_sha256"] == sources.CORE_SHA and cpu["support_sha256"] == sources.SUPPORT_SHA
            and cpu["namespace_helper_sha256"] == HELPER_SHA and cpu["CLI_sha256"] == sources.CLI_SHA
            and cpu["config_sha256"] == sources.CONFIG_SHA and cpu["targets"] == TARGETS
            and cpu["run_pin"] == run_pin and cpu["retained_index_pin"] == index_pin
            and cpu["checkpoint_pin"] == checkpoint and cpu["frozen_prefix_sha256"] == prefix,
            "New CPU proof/E12/source/config/sidecar bindings differ")
    require(digest(cpu["producer_sha256"]), "Pinned independent CPU producer SHA required")
    require(re.fullmatch(r"_rsna_f009_[a-z0-9_]+", cpu["namespace"]) is not None
            and cpu["model_class"] == cpu["namespace"] + ".finetune_model.NativeTripletMIL"
            and cpu["head_class"] == cpu["namespace"] + ".feature_model.FrozenFeatureHead",
            "Actual model/head must belong to the new pinned F009 namespace")
    if fixture_only:
        require(cpu["evidence_scope"] == "artificial_fixture_only" and cpu["actual_CPU_strict_reconstruction"] is False,
                "Artificial bytes cannot claim actual strict CPU reconstruction")
    else:
        require(cpu["producer_sha256"] in APPROVED_ACTUAL_CPU_PRODUCERS
                and cpu["evidence_scope"] == "actual_CPU_tensor_reconstruction"
                and cpu["actual_CPU_strict_reconstruction"] is True,
                "Actual F009 strict CPU producer is not implemented/reviewed/pinned yet")
    require(cpu["generic_factory_calls"] == 1 and cpu["model_forward_calls"] == cpu["optimizer_steps"] == 0
            and cpu["model_training"] is cpu["encoder_training"] is False and cpu["gamma_draws"] == 0
            and cpu["teacher_MRI_Gold_Report_read"] is False and cpu["GPU_used"] is False,
            "CPU reconstruction/eval gamma and no-data scope differs")
    for family, expected in expected_tensor_shapes().items():
        tensors = cpu["tensor_metadata"][family]
        require(len(tensors) == len(expected) and {row["name"] for row in tensors} == set(expected)
                and all(row["shape"] == expected[row["name"]] and row["dtype"] == "torch.float32"
                        and row["finite"] is True and digest(row["tensor_sha256"]) for row in tensors)
                and digest(cpu[family + "_tensor_sha256"]), "Exact CPU FP32 tensor key/shape/dtype/hash contract differs")


def read(path):
    path = Path(path)
    require(path.suffix == ".json" and pin(path)["bytes"] <= 16 * 1024**2, "Bounded JSON metadata only")

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON field")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def bound(root, receipt):
    require(set(receipt) == {"path", "bytes", "sha256"} and isinstance(receipt["path"], str), "Exact path/bytes/SHA metadata pin required")
    path = root / receipt["path"]
    require(path.suffix in {".json", ".py"} and path.resolve().is_relative_to(root)
            and pin(path) == {key: receipt[key] for key in ("bytes", "sha256")},
            "Explicit local metadata pin differs")
    return path


def validate_cpu(cpu, *, qualification, qualification_binding_pins, **kwargs):
    _validate_cpu_base(cpu, **kwargs)
    require(cpu["qualification_schema"] == QUALIFICATION_SCHEMA
            and cpu["qualification_pins"] == qualification
            and cpu["qualification_binding_pins"] == qualification_binding_pins,
            "CPU proof must bind all six exact amended qualification documents and sources")
    require(cpu["runtime_or_adoption_qualified"] is False, "A CPU proof is not a deployment qualification")


def consumed_qualification(validated, qualification):
    """Check the consumed contract as well as the shared validator's pass flag."""
    r0, g0, r1, g1, lg, lr = (validated[key] for key in
        ("fold0_result", "fold0_guard", "fold1_result", "fold1_guard", "legacy_guard1", "legacy_comparison1"))
    refs = {name: {key: descriptor[key] for key in ("bytes", "sha256")} for name, descriptor in qualification.items()}
    require(g0["complete"] is True and g0["original_evaluate_calls"] == 1 and g0["original_result_pin"] == refs["result0"]
            and r0["fold1_gate_conditions_met"] is g0["fold1_gate_conditions_met"] is True
            and previous.positive(r0["fixed10_delta"]), "Consumed fold0 qualification differs")
    require(r1["schema_version"] == "F009_fold1_known_control_AMP_amended_joint_result_v2"
            and g1["schema_version"] == "F009_fold1_amended_evaluation_receipt_v2"
            and g1["complete"] is True and r1["actual_comparison"] is g1["actual_comparison"] is True
            and g1["legacy_comparison_calls"] == 1 and g1["result_pin"] == refs["result1"]
            and g1["legacy_guard_pin"] == r1["legacy_guard_pin"] == refs["legacy_guard1"]
            and g1["legacy_comparison_pin"] == r1["legacy_comparison_pin"] == refs["legacy_comparison1"]
            and r1["v2_joint_candidate_screen_passed"] is g1["v2_joint_candidate_screen_passed"] is True
            and previous.positive(r1["fixed10_delta"]), "Consumed actual amended comparison and six cross-pins differ")
    require(r1["legacy_v1_joint_screen"] == "HOLD" and r1["legacy_v1_machine_passed"] is False
            and lg["machine_engineering_passed"] is False and lg["AMP_skipped_steps"] == {"control": 1, "candidate": 0}
            and lr["fold1_machine_engineering_passed"] is lr["joint_candidate_screen_passed"] is False
            and lr["fold1_independent_positive_screen"] is False and g1["original_v1_preserved_as_HOLD"] is True,
            "Consumed original v1 must remain HOLD")
    require(all(r1[key] == lr[key] for key in ("fixed10_delta", "all12_auxiliary_delta", "fixed10_interval",
            "all12_auxiliary_interval", "per_target", "bootstrap", "weak_BCE", "ordered_UID_sha256"))
            and r1["inputs"] == lr["inputs"] == lg["completed_E12_control_candidate"]
            and r1["fold0"] == lr["fold0"] == lg["fold0_positive_sha_bound"],
            "Consumed statistical copy and E12 lineage differ")


def registry_inventory(root):
    require(pin(root / "src/rsna_knee/f009_pair_stage_v2.py")["sha256"] == PREVIOUS_SHA, "Pinned old pure-copy/tensor-shape helpers changed")
    require(digest(CPU_PRODUCER_SHA) and digest(CPU_PRODUCER_CLI_SHA), "New CPU producer registry is not frozen")
    entries = {
        "CPU_producer_v2.py": ("src/rsna_knee/f009_cpu_proof_v2.py", CPU_PRODUCER_SHA),
        "CPU_producer_CLI_v2.py": ("scripts/check_f009_E12_cpu_v2.py", CPU_PRODUCER_CLI_SHA),
        "CPU_producer_v1_helper.py": ("src/rsna_knee/f009_cpu_proof.py", "1d2cbdc18212454d97aca2a768ed2dc6f89492a757644c7e91de03e0eeb066f1"),
        "CPU_frozen_v1_tensor_contract.py": ("artifacts/research/20261017-f009-strict-CPU-producer-preparation-v1/f009_pair_stage-v1.py", "d3a951a2913798d77c7e93977997bff68ad334173cc7b8c0382e96d9f5772781"),
        "sealer_v2_pure_helpers.py": ("src/rsna_knee/f009_pair_stage_v2.py", PREVIOUS_SHA),
        "source_namespace_helper.py": ("src/rsna_knee/f009_source_namespace.py", HELPER_SHA),
    }
    result = {}
    for name, (relative, expected) in entries.items():
        path = root / relative
        observed = pin(path)
        require(observed["sha256"] == expected, "Reviewed CPU/sealer helper source differs: " + name)
        result[name] = {"path": relative, **observed}
    return result


def preflight(root, spec, *, fixture_only=False):
    """All metadata/CPU/source checks finish before checkpoint copy access."""
    root = Path(root).resolve()
    require(spec["schema_version"] == SPEC_SCHEMA and set(spec) == {"schema_version", "qualification", "folds"}
            and set(spec["folds"]) == {"0", "1"} and set(spec["qualification"]) == QUALIFICATION_KEYS,
            "Explicit amended v3 six-document pair spec required; legacy schemas refused")
    registry = registry_inventory(root)
    from . import f009_cpu_proof_v2 as producer

    require(Path(producer.__file__).resolve() == (root / "src/rsna_knee/f009_cpu_proof_v2.py").resolve()
            or fixture_only, "Actual amended producer module path differs")
    validated = producer.validate_amended_joint_metadata(root, spec["qualification"])
    require(validated["schema_version"] == QUALIFICATION_SCHEMA and validated["passed"] is True
            and validated["qualification_pins"] == spec["qualification"]
            and set(validated["binding_pins"]) == set(BINDING_FILENAMES),
            "Shared validation must preserve all six explicit pins and the seven fixed binding documents")
    consumed_qualification(validated, spec["qualification"])
    base, config = validated["base"], validated["config"]
    policy0, policy1 = validated["policy0"], validated["policy1"]
    guards = validated["completed_guards"]
    require(base["source_sha256"] == previous.BASE_SOURCE and base["blend_policy"] == "success-subset"
            and base["native_executor"] == "isolated-uid", "Original four-model REG references changed")
    require(pin(root / previous.BASE)["sha256"] == previous.BASE_SHA, "Pinned P035 reference manifest changed")
    require(pin(root / "configs/experiments/f009-dinov2-native-last2-late-cosine-study-gamma.json")["sha256"] == sources.CONFIG_SHA,
            "Fixed gamma config changed")
    bodies = sources._sources(root / "src/rsna_knee", {**sources.CORE_SHA, **sources.SUPPORT_SHA})
    require(len(bodies) == 14 and pin(root / "scripts/train_native_dinov2_fast16.py")["sha256"] == sources.CLI_SHA,
            "Fixed source14 and training CLI required")
    plan = []
    for fold in (0, 1):
        item = spec["folds"][str(fold)]
        require(set(item) == {"run", "index", "CPU", "expected_CPU_producer_sha256"}, "Exact per-fold run/index/CPU/producer pins required")
        run_path, index_path, cpu_path = (bound(root, item[key]) for key in ("run", "index", "CPU"))
        expected_directory = root / (policy0["planned_run_directory"] if fold == 0 else policy1["new_run"])
        require(run_path.resolve() == expected_directory.resolve() / "run.json"
                and index_path.resolve() == expected_directory.resolve() / "epoch_checkpoint_index.json",
                "Only the two existing F009 run directories permitted")
        run, index, cpu = (read(path) for path in (run_path, index_path, cpu_path))
        run_pin, index_pin = pin(run_path), pin(index_path)
        completed = guards[fold]["completed_E12_control_candidate"]["candidate"]
        require(run_pin == completed["run"] and index_pin == completed["index"]
                and run["status"] == "complete" and type(run["fold"]) is int and run["fold"] == fold
                and run["seed"] == 20261002 and run["config"] == config and run["source_sha256"] == sources.CORE_SHA
                and run["targets"] == TARGETS and run["gold_used_for_training_or_selection"] is False
                and run["gold_evaluated"] is False and [row["epoch"] for row in run["history"]] == list(range(1, 13))
                and all(type(row["amp_skipped_steps"]) is int and row["amp_skipped_steps"] == 0 for row in run["history"]),
                "Actual candidate fold/run/gate metadata differs")
        entries = [row for row in index["checkpoints"] if row["epoch"] == 12]
        require(len(entries) == 1, "Exactly one retained E12 checkpoint entry required")
        row, checkpoint = entries[0], completed["E12_checkpoint_body_stream_hash"]
        require(index["fold"] == fold and index["source_sha256"] == sources.CORE_SHA and row["checkpoint"] == "epoch_012.pt"
                and row["predictions_csv"] == "epoch_012_weak.csv"
                and checkpoint == {"bytes": row["checkpoint_bytes"], "sha256": row["checkpoint_sha256"]}
                and type(row["weak_bce"]) in (float, int) and math.isfinite(row["weak_bce"])
                and row["weak_bce"] == run["history"][-1]["weak_bce"]
                and completed["E12_prediction"]["sha256"] == row["prediction_sha256"] == run["history"][-1]["prediction_sha256"],
                "Exact E12 index/checkpoint/CSV bindings differ")
        require(cpu["producer_sha256"] == item["expected_CPU_producer_sha256"] == CPU_PRODUCER_SHA,
                "Expected new CPU producer registry identity differs")
        prefix = base["native"]["model_fingerprints"]["REG:0"]["frozen_prefix_sha256"]
        validate_cpu(cpu, fold=fold, run_pin=run_pin, index_pin=index_pin, checkpoint=checkpoint, prefix=prefix,
                     fixture_only=fixture_only, qualification=spec["qualification"],
                     qualification_binding_pins=validated["binding_pins"])
        require(cpu["checkpoint_schema"] == "native392_generic_dino_suffix_head_fast16_v1"
                and cpu["weak_bce"] == row["weak_bce"] and cpu["encoder_provenance"] == run["encoder_provenance"],
                "CPU checkpoint schema/loss/generic provenance differs")
        plan.append({"fold": fold, "checkpoint_path": run_path.parent / "epoch_012.pt", "checkpoint_pin": checkpoint,
                     "run_path": run_path, "index_path": index_path, "CPU_path": cpu_path, "CPU": cpu,
                     "metadata_pins": {"run": run_pin, "index": index_pin, "CPU": pin(cpu_path)}})
    require(plan[0]["checkpoint_pin"]["sha256"] != plan[1]["checkpoint_pin"]["sha256"], "Distinct fold checkpoints required")
    return base, plan, validated, registry


def _seal(root, spec, output, *, fixture_only=False, root_spec=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(output.is_relative_to(root) and not output.exists(), "Fresh workspace output directory required")
    require(fixture_only or root_spec is not None, "Actual sealing requires a root-SHA-bound input spec")
    base, plan, validated, registry = preflight(root, spec, fixture_only=fixture_only)
    audit = {key + ".json": value for key, value in spec["qualification"].items()}
    for name, descriptor in validated["binding_pins"].items():
        require(Path(descriptor["path"]).suffix == Path(BINDING_FILENAMES[name]).suffix, "Binding file type differs")
        audit[BINDING_FILENAMES[name]] = descriptor
    audit.update(registry)
    audit["sealer_v3.py"] = {"path": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"), **pin(__file__)}
    audit["sealer_CLI_v3.py"] = {"path": "scripts/seal_f009_E12_pair_v3.py", **pin(root / "scripts/seal_f009_E12_pair_v3.py")}
    if root_spec is not None:
        audit["root-spec.json"] = root_spec
    # Validate every prospective inventory item before creating any output.
    for descriptor in audit.values():
        bound(root, descriptor)
    output.mkdir(parents=True, exist_ok=False)
    try:
        folders, input_inventory = {}, dict(audit)
        for item in plan:
            fold, folder = item["fold"], output / ("fold" + str(item["fold"]))
            sources.stage_f009_sources(root / "src/rsna_knee", root / "scripts/train_native_dinov2_fast16.py",
                root / "configs/experiments/f009-dinov2-native-last2-late-cosine-study-gamma.json", folder / "source14")
            copy_exact(item["checkpoint_path"], folder / "trained/epoch_012.pt", item["checkpoint_pin"])
            for key in ("run", "index", "CPU"):
                path, observed = item[key + "_path"], item["metadata_pins"][key]
                copy_exact(path, folder / "audit" / (key + ".json"), observed)
                input_inventory["fold" + str(fold) + "/" + key] = {"path": path.relative_to(root).as_posix(), **observed}
            for name, descriptor in audit.items():
                copy_exact(root / descriptor["path"], folder / "audit" / name,
                           {key: descriptor[key] for key in ("bytes", "sha256")})
            manifest = {"schema_version": BUNDLE_SCHEMA, "fold": fold, "epoch": 12,
                "checkpoint_pin": item["checkpoint_pin"], "CPU_receipt": item["metadata_pins"]["CPU"],
                "CPU_producer_sha256": CPU_PRODUCER_SHA, "CPU_producer_CLI_sha256": CPU_PRODUCER_CLI_SHA,
                "qualification_schema": QUALIFICATION_SCHEMA, "qualification_pins": spec["qualification"],
                "qualification_binding_pins": validated["binding_pins"],
                "source_manifest_pin": pin(folder / "source14/source-manifest.json"),
                "source_sha256": sources.CORE_SHA, "support_sha256": sources.SUPPORT_SHA,
                "assets": {path.relative_to(folder).as_posix(): pin(path) for path in sorted(folder.rglob("*")) if path.is_file()},
                "artificial_fixture_only": fixture_only, "actual_strict_CPU_accepted": not fixture_only,
                "runtime_transport_or_Submit_qualified": False}
            atomic_json(folder / "bundle.json", manifest)
            folders[str(fold)] = pin(folder / "bundle.json")
        # Rehash source metadata, raw checkpoints and every copied output before
        # publishing the pair or its completion marker. No model deserialize.
        for descriptor in input_inventory.values():
            bound(root, descriptor)
        registry_inventory(root)
        sources._sources(root / "src/rsna_knee", {**sources.CORE_SHA, **sources.SUPPORT_SHA})
        for item in plan:
            require(pin(item["checkpoint_path"]) == item["checkpoint_pin"], "Source checkpoint changed during copy")
            folder = output / ("fold" + str(item["fold"]))
            manifest = read(folder / "bundle.json")
            require(pin(folder / "bundle.json") == folders[str(item["fold"])], "New bundle changed before publication")
            for relative, observed in manifest["assets"].items():
                require(pin(folder / relative) == observed, "Copied bundle asset changed before publication")
        retained = {key: {"transport": base["native"]["transport_jobs"][key],
                          "fingerprint": base["native"]["model_fingerprints"][key]} for key in ("REG:0", "REG:1")}
        pair = {"schema_version": PAIR_SCHEMA, "local_pair_complete": True, "CLS_E12_bundles": folders,
            "qualification_schema": QUALIFICATION_SCHEMA, "qualification_pins": spec["qualification"],
            "qualification_binding_pins": validated["binding_pins"], "old_v1_joint_screen": "HOLD",
            "fixed_REG_references_only": retained, "REG_checkpoint_body_read_or_copied": False,
            "logical_models": 4, "old_CLS_included": False, "artificial_fixture_only": fixture_only,
            "actual_CLS_stage_constructed": not fixture_only, "four_model_transport_runtime_qualified": False,
            "source_input_spec": spec, "root_spec_pin": root_spec, "adoption_Save_Submit_authorized": False}
        atomic_json(output / "pair-sidecar.json", pair)
        atomic_json(output / "BUILD_COMPLETE.json", {"complete": True, "pair_sidecar": pin(output / "pair-sidecar.json"),
                    "artificial_fixture_only": fixture_only, "four_model_transport_runtime_qualified": False})
        return pair
    except BaseException as error:
        atomic_json(output / "BUILD_FAILED.json", {"complete": False, "error_type": type(error).__name__,
                    "artificial_fixture_only": fixture_only, "partial_output_preserved": True})
        raise


def seal_f009_pair(spec_path, expected_spec_sha256, output):
    """Manual local-only entry; actual amended two-fold/CPU evidence required."""
    spec_path = Path(spec_path).resolve()
    require(spec_path.suffix == ".json" and spec_path.is_relative_to(ROOT) and pin(spec_path)["sha256"] == expected_spec_sha256,
            "Root expected input-spec SHA required")
    descriptor = {"path": spec_path.relative_to(ROOT).as_posix(), **pin(spec_path)}
    return _seal(ROOT, read(spec_path), output, fixture_only=False, root_spec=descriptor)

