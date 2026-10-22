"""Amended-evaluation strict CPU producer; actual models remain manual future work.

The six qualification documents are independently SHA-bound. Original legacy
HOLD fields stay false; only the pinned historical E10 accounting is excepted.
No model/checkpoint body is opened before all metadata guards. No forward,
optimizer, MRI, teacher, Gold, GPU, API, upload or automatic continuation route.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

from . import f009_source_namespace as sources

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = "artifacts/research/20261017-f009-strict-CPU-producer-preparation-v1/f009_pair_stage-v1.py"
SNAPSHOT_SHA = "d3a951a2913798d77c7e93977997bff68ad334173cc7b8c0382e96d9f5772781"
LEGACY_PRODUCER = "src/rsna_knee/f009_cpu_proof.py"
LEGACY_PRODUCER_SHA = "1d2cbdc18212454d97aca2a768ed2dc6f89492a757644c7e91de03e0eeb066f1"
GENERIC_SHA = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
AMENDED_WRAPPER = "artifacts/research/20261017-f009-fold1-amended-evaluator-v2/evaluate_amended.py"
AMENDED_WRAPPER_SHA = "bcf9fbe5dad5c7f1faae6761548b1d8786d5a24d1c25525f5e9f5d2ea289bcce"
AMENDED_BINDINGS = {
    "policy_sha256": "bf53e1828784df4d0da793cea6014f055ca9e5948dc5013cf368b646dd328a90",
    "declaration_sha256": "939394e0a10c01ab109a441ba1ab898a8e2038d60bf18ec85145444475f7ee6e",
    "legacy_sha256": "c8e244461dbf45be68fff13328f2a774df1fbf5ed69e8c109ee84b0f421e352c",
}
QUALIFICATION_SCHEMA = "F009_amended_joint_metadata_v2"
CPU_SCHEMA = "f009_strict_CPU_amended_eval_v2"
CLI_REL = "scripts/check_f009_E12_cpu_v2.py"
QUALIFICATION_KEYS = frozenset({"result0", "guard0", "result1", "guard1", "legacy_guard1", "legacy_comparison1"})
STATISTICS_KEYS = ("fixed10_delta", "all12_auxiliary_delta", "fixed10_interval", "all12_auxiliary_interval",
                   "per_target", "bootstrap", "weak_BCE", "ordered_UID_sha256")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _contracts():
    """Frozen v1 guards/shapes, independent of a future consumer registry v2."""
    require(sha(ROOT / LEGACY_PRODUCER) == LEGACY_PRODUCER_SHA, "Frozen numerical producer source changed")
    path = ROOT / SNAPSHOT
    require(sha(path) == SNAPSHOT_SHA, "Frozen consumer v1 source changed")
    spec = importlib.util.spec_from_file_location("rsna_knee._f009_cpu_frozen_stage_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



def pin(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= 128 * 1024**2,
            "Bounded regular source/metadata/model member required")
    size = path.stat().st_size
    checksum = sha(path)
    require(path.stat().st_size == size, "Member size changed")
    return {"bytes": size, "sha256": checksum}


def read(path):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "Duplicate JSON field")
            value[key] = item
        return value
    require(Path(path).suffix == ".json" and 0 < Path(path).stat().st_size <= 16 * 1024**2,
            "Bounded JSON metadata required")
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def bound(root, ref):
    root = Path(root).resolve()
    require(isinstance(ref, dict) and {"path", "bytes", "sha256"} <= set(ref), "Explicit file pin required")
    path = root / ref["path"]
    require(path.suffix in {".json", ".py"} and path.resolve().is_relative_to(root)
            and not path.is_symlink(), "Pinned metadata/source member leaves workspace or type differs")
    require(pin(path) == {key: ref[key] for key in ("bytes", "sha256")}, "Explicit metadata pin differs")
    return path


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _amended_wrapper():
    path = ROOT / AMENDED_WRAPPER
    require(pin(path)["sha256"] == AMENDED_WRAPPER_SHA, "Frozen amended wrapper source changed")
    spec = importlib.util.spec_from_file_location("_f009_amended_metadata_contract", path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def _retained_metadata(root, folder, completed, fold):
    """Inspect bound JSON only, never checkpoint, prediction or teacher bodies."""
    folder = Path(folder).resolve()
    require(folder.is_relative_to(root), "Run leaves workspace")
    run_path, index_path = folder / "run.json", folder / "epoch_checkpoint_index.json"
    require(pin(run_path) == completed["run"] and pin(index_path) == completed["index"],
            "Completed run/index differs from evaluation guard")
    run, index = read(run_path), read(index_path)
    require(run["status"] == "complete" and type(run["fold"]) is int and run["fold"] == fold
            and run["seed"] == 20261002 and run["gold_used_for_training_or_selection"] is False
            and run["gold_evaluated"] is False, "Complete fixed run/fold/seed/noGold metadata required")
    history = run["history"]
    require([row["epoch"] for row in history] == list(range(1, 13)), "Exactly completed epochs1..12 required")
    require(all(all(type(row[key]) is int and row[key] >= 0 for key in
                    ("optimizer_steps", "optimizer_step_opportunities", "amp_skipped_steps"))
                and row["optimizer_step_opportunities"] > 0
                and row["optimizer_steps"] + row["amp_skipped_steps"] == row["optimizer_step_opportunities"]
                and all(type(row[key]) in (int, float) and math.isfinite(row[key]) and row[key] >= 0
                        for key in ("train_bce", "weak_bce")) for row in history),
            "Finite losses and explained exact AMP accounting required")
    entries = [row for row in index["checkpoints"] if row["epoch"] == 12]
    require(index["fold"] == fold and index["source_sha256"] == run["source_sha256"] and len(entries) == 1,
            "Exact retained E12 source/fold/index required")
    row = entries[0]
    require(row["checkpoint"] == "epoch_012.pt" and row["predictions_csv"] == "epoch_012_weak.csv"
            and completed["E12_checkpoint_body_stream_hash"] == {"bytes": row["checkpoint_bytes"], "sha256": row["checkpoint_sha256"]}
            and row["weak_bce"] == history[-1]["weak_bce"]
            and completed["E12_prediction"]["sha256"] == row["prediction_sha256"] == history[-1]["prediction_sha256"],
            "E12 index/history/checkpoint-reference/prediction bindings differ")
    return {"run": run, "row": row, "run_path": run_path, "index_path": index_path,
            "run_pin": completed["run"], "retained_index_pin": completed["index"],
            "checkpoint_path": folder / "epoch_012.pt", "checkpoint_pin": completed["E12_checkpoint_body_stream_hash"]}


def validate_amended_joint_metadata(root, qualification):
    """Shared stdlib gate; exact six pins, actual amended comparison, legacy HOLD."""
    root = Path(root).resolve()
    require(isinstance(qualification, dict) and set(qualification) == QUALIFICATION_KEYS
            and all(isinstance(ref, dict) and set(ref) == {"path", "bytes", "sha256"}
                    for ref in qualification.values()), "Exact six qualification document pins required")
    documents = {key: read(bound(root, ref)) for key, ref in qualification.items()}
    r0, g0, r1, g1, lg, lr = (documents[key] for key in
        ("result0", "guard0", "result1", "guard1", "legacy_guard1", "legacy_comparison1"))
    wrapper = _amended_wrapper()
    policy2, declaration, snapshot = wrapper.premetric_guard(root, AMENDED_BINDINGS)
    c = _contracts()
    policy0 = read(bound(root, policy2["fold0_policy_immutable"]))
    policy1 = read(bound(root, policy2["original_policy_immutable"]))
    qpin = {key: {name: ref[name] for name in ("bytes", "sha256")} for key, ref in qualification.items()}
    require(qualification["result0"] == policy2["fold0_primary_result"]
            and qualification["guard0"] == policy2["fold0_guard"], "Same sealed fold0 documents required")
    require(g0["schema_version"] == "F009_fixed_E12_guarded_evaluation_receipt_v1" and g0["complete"] is True
            and g0["original_evaluate_calls"] == 1 and g0["wrapper_pin"]["sha256"] == c.F0_WRAPPER_SHA
            and g0["original_result_pin"] == qpin["result0"] and g0["fold1_gate_conditions_met"] is True
            and g0["original_policy_thresholds_CI_BCE_output_modified"] is False
            and r0["status"] == "F009_E12_WEAK_COMPARISON_ONLY_POLICY_V2"
            and r0["policy_sha256"] == policy2["fold0_policy_immutable"]["sha256"]
            and r0["candidate_epoch"] == r0["control_epoch"] == 12 and r0["candidate_AMP_skipped_steps"] == 0
            and r0["fold1_gate_conditions_met"] is True and positive(r0["fixed10_delta"]),
            "Completed independently positive unchanged fold0 required")
    require(r1["schema_version"] == "F009_fold1_known_control_AMP_amended_joint_result_v2"
            and r1["status"] == "FIXED12_PREMETRIC_ENGINEERING_AMENDMENT_V2"
            and g1["schema_version"] == "F009_fold1_amended_evaluation_receipt_v2"
            and g1["complete"] is True and r1["actual_comparison"] is g1["actual_comparison"] is True
            and r1["policy2_sha256"] == AMENDED_BINDINGS["policy_sha256"]
            and r1["declaration_sha256"] == AMENDED_BINDINGS["declaration_sha256"]
            and g1["wrapper_pin"] == pin(ROOT / AMENDED_WRAPPER)
            and g1["policy_pin"] == pin(root / wrapper.POLICY_REL)
            and g1["declaration_pin"] == pin(root / wrapper.DECLARATION_REL)
            and g1["epoch0_snapshot_pin"] == {key: policy2["candidate_epoch0_snapshot"][key] for key in ("bytes", "sha256")}
            and g1["legacy_evaluator_pin"] == policy2["original_evaluator_immutable"]
            and g1["legacy_comparison_calls"] == 1 and g1["result_pin"] == qpin["result1"]
            and g1["legacy_guard_pin"] == r1["legacy_guard_pin"] == qpin["legacy_guard1"]
            and g1["legacy_comparison_pin"] == r1["legacy_comparison_pin"] == qpin["legacy_comparison1"],
            "Actual amended result/receipt and all six cross-pins required")
    require(r1["legacy_v1_joint_screen"] == "HOLD" and r1["legacy_v1_machine_passed"] is False
            and g1["original_v1_preserved_as_HOLD"] is True and g1["legacy_consumer_compatible"] is False
            and r1["legacy_CPU_producer_sealer_compatible"] is False and r1["root_review_still_required"] is True
            and r1["BCE_CI_REG50_report_only"] is True
            and r1["automatic_fit_stage_Save_Submit"] is g1["automatic_fit_stage_Save_Submit"] is False
            and lg["schema_version"] == "F009_fold1_fixed12_input_guard_v1" and lg["complete"] is True
            and lg["machine_engineering_passed"] is False and lg["AMP_skipped_steps"] == {"control": 1, "candidate": 0}
            and lr["status"] == "F009_FOLD1_FIXED12_PAIRED_AND_INDEPENDENT_TWO_FOLD_SCREEN_V1"
            and lr["policy0_sha256"] == policy2["fold0_policy_immutable"]["sha256"]
            and lr["policy1_sha256"] == policy2["original_policy_immutable"]["sha256"]
            and lr["fold1_machine_engineering_passed"] is False and lr["joint_candidate_screen_passed"] is False
            and lr["fold1_independent_positive_screen"] is False
            and lr["AMP_skipped_steps"] == {"control": 1, "candidate": 0},
            "Original HOLD and consumer incompatibility must remain explicit")
    prior = lg["fold0_positive_sha_bound"]
    require(prior["result"] == qpin["result0"] and prior["guard_receipt"] == qpin["guard0"]
            and prior["fixed10_delta"] == r0["fixed10_delta"] and prior["engineering_gate_passed"] is True
            and r1["fold0"] == lr["fold0"] == prior
            and r1["inputs"] == lr["inputs"] == lg["completed_E12_control_candidate"]
            and r1["fold"] == lr["fold"] == 1
            and r1["candidate_epoch"] == r1["control_epoch"] == lr["candidate_epoch"] == lr["control_epoch"] == 12
            and all(r1[key] == lr[key] for key in STATISTICS_KEYS), "Unchanged statistics and both-fold lineage required")
    expected_inputs = {"src/rsna_knee/" + key: value for key, value in sources.CORE_SHA.items()}
    expected_inputs.update({policy0[key]["path"]: policy0[key]["sha256"] for key in ("config", "resource_gate", "input_cache")})
    for guard in (g0["guards"], lg):
        require(guard["complete"] is True and guard["Gold_body_opened_or_hashed"] is False
                and guard["Gold_hash_metadata_equal"] is True and guard["model_deserialized"] is False
                and all(guard["input_source_pins"][key]["sha256"] == value for key, value in expected_inputs.items()),
                "Guard source/config/cache/Gold metadata differs")
    require(g0["guards"]["target_projection_or_statistics_before_legacy_evaluate"] is False
            and lg["fold1_target_projection_or_statistics_before_guards"] is False,
            "Original target guard scope differs")
    for arm, original in r0["inputs"].items():
        completed = g0["guards"]["completed_E12_control_candidate"][arm]
        require(original["run_sha256"] == completed["run"]["sha256"]
                and original["epoch_index_sha256"] == completed["index"]["sha256"]
                and original["prediction_sha256"] == completed["E12_prediction"]["sha256"]
                and original["checkpoint_sha256_reference_only_not_loaded"] == completed["E12_checkpoint_body_stream_hash"]["sha256"],
                "Fold0 result/guard E12 join differs")
    require(policy0["source_sha256"] == sources.CORE_SHA
            and pin(bound(root, policy0["config"]))["sha256"] == sources.CONFIG_SHA,
            "Exact gamma config/source8 required")
    config = read(root / policy0["config"]["path"])
    cache = read(bound(root, policy0["input_cache"]))
    resource = read(bound(root, policy0["resource_gate"]))
    require(cache["manifest_sha256"] == policy0["input_cache"]["manifest_sha256"]
            and resource["source_sha256"] == sources.CORE_SHA, "Current cache/resource metadata differs")
    require(len(sources._sources(root / "src/rsna_knee", {**sources.CORE_SHA, **sources.SUPPORT_SHA})) == 14
            and pin(root / "scripts/train_native_dinov2_fast16.py")["sha256"] == sources.CLI_SHA
            and pin(root / "src/rsna_knee/f009_source_namespace.py")["sha256"] == c.HELPER_SHA,
            "Exact isolated source14/CLI/helper required")
    require(pin(root / c.BASE)["sha256"] == c.BASE_SHA, "Frozen P035 base manifest differs")
    base = read(root / c.BASE)
    require(base["source_sha256"] == c.BASE_SOURCE and base["blend_policy"] == "success-subset"
            and base["native_executor"] == "isolated-uid", "Original P035 base contract differs")
    contexts = {}
    for fold, folder, completed in ((0, root / policy0["planned_run_directory"], g0["guards"]["completed_E12_control_candidate"]["candidate"]),
                                    (1, root / policy1["new_run"], lg["completed_E12_control_candidate"]["candidate"])):
        context = _retained_metadata(root, folder, completed, fold)
        run = context["run"]
        require(run["config"] == config and run["source_sha256"] == sources.CORE_SHA and run["targets"] == c.TARGETS
                and all(row["amp_skipped_steps"] == 0 for row in run["history"])
                and run["audit"]["pixel_index_sha256"] == policy0["input_cache"]["sha256"]
                and run["audit"]["manifest_sha256"] == policy0["input_cache"]["manifest_sha256"]
                and run["audit"]["resource_gate_sha256"] == policy0["resource_gate"]["sha256"],
                "Candidate source/config/cache/audit or zero-skip metadata differs")
        contexts[fold] = context
    control_spec = policy2["immutable_historical_control"]
    control_folder = bound(root, control_spec["run"]).parent
    require(pin(bound(root, control_spec["retention_index"])) == lg["completed_E12_control_candidate"]["control"]["index"],
            "Exact historical control index required")
    control = _retained_metadata(root, control_folder, lg["completed_E12_control_candidate"]["control"], 1)
    require(lg["completed_E12_control_candidate"]["control"]["E12_prediction"] ==
            {key: control_spec["epoch12_predictions"][key] for key in ("bytes", "sha256")},
            "Exact historical control E12 prediction reference required")
    engineering = wrapper.engineering_guard(policy2, snapshot, lg,
        {"control": control["run"], "candidate": contexts[1]["run"]})
    require(engineering == r1["v2_engineering"] == g1["engineering_v2"]
            and r1["v2_joint_candidate_screen_passed"] is g1["v2_joint_candidate_screen_passed"] is True
            and wrapper.amended_joint(r0["fixed10_delta"], r1["fixed10_delta"], True, engineering["passed"]),
            "Exact engineering and independent strictly positive AND required")
    require(contexts[0]["checkpoint_pin"]["sha256"] != contexts[1]["checkpoint_pin"]["sha256"],
            "Separate fold checkpoints required")
    binding_pins = {"policy0": policy2["fold0_policy_immutable"], "policy1": policy2["original_policy_immutable"],
        "policy2": {"path": wrapper.POLICY_REL, **pin(root / wrapper.POLICY_REL)},
        "declaration": {"path": wrapper.DECLARATION_REL, **pin(root / wrapper.DECLARATION_REL)},
        "snapshot": policy2["candidate_epoch0_snapshot"], "legacy_evaluator": policy2["original_evaluator_immutable"],
        "amended_wrapper": {"path": AMENDED_WRAPPER, **pin(ROOT / AMENDED_WRAPPER)}}
    return {"schema_version": QUALIFICATION_SCHEMA, "passed": True, "qualification_pins": copy.deepcopy(qualification),
        "fold0_result": r0, "fold0_guard": g0, "fold1_result": r1, "fold1_guard": g1,
        "legacy_guard1": lg, "legacy_comparison1": lr, "completed_guards": {0: g0["guards"], 1: lg},
        "policy0": policy0, "policy1": policy1, "policy2": policy2, "base": base, "config": config,
        "binding_pins": binding_pins, "candidate_contexts": contexts}


def prepare_inputs(root, spec, fold):
    require(type(fold) is int and fold in (0, 1), "Explicit fold0 or fold1 required")
    require(spec["schema_version"] == "f009_CPU_proof_amended_root_inputs_v2"
            and set(spec) == {"schema_version", "qualification", "folds", "provenance"}
            and set(spec["folds"]) == {"0", "1"}
            and all(set(item) == {"run", "index"} for item in spec["folds"].values()),
            "Exact amended CPU input specification required")
    gate = validate_amended_joint_metadata(root, spec["qualification"])
    for n in (0, 1):
        context = gate["candidate_contexts"][n]
        for field, expected in (("run", context["run_pin"]), ("index", context["retained_index_pin"])):
            path = bound(root, spec["folds"][str(n)][field])
            require(path.resolve() == context[field + "_path"].resolve() and pin(path) == expected,
                    "Explicit producer run/index differs from qualified context")
    provenance = read(bound(root, spec["provenance"]))
    require(provenance["weights_sha256"] == GENERIC_SHA
            and all(gate["candidate_contexts"][n]["run"]["encoder_provenance"] == provenance for n in (0, 1)),
            "Same audited generic provenance required")
    context = dict(gate["candidate_contexts"][fold])
    context.update(fold=fold, config=gate["config"], provenance=provenance,
        prefix=gate["base"]["native"]["model_fingerprints"]["REG:0"]["frozen_prefix_sha256"],
        producer_sha256=sha(__file__), gate_input_spec=copy.deepcopy(spec),
        qualification_pins=gate["qualification_pins"], qualification_binding_pins=gate["binding_pins"])
    return context


def tensor_digest(state):
    checksum = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        checksum.update(name.encode())
        checksum.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        checksum.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return checksum.hexdigest()


def _metadata(state, expected, torch):
    require(set(state) == set(expected), "Exact suffix/head tensor keys required")
    rows = []
    for name, shape in sorted(expected.items()):
        tensor = state[name]
        require(
            isinstance(tensor, torch.Tensor)
            and list(tensor.shape) == shape
            and tensor.device.type == "cpu"
            and tensor.dtype == torch.float32
            and bool(torch.isfinite(tensor).all()),
            "Finite exact CPU FP32 tensor shape/dtype required: " + name,
        )
        rows.append(
            {
                "name": name,
                "shape": shape,
                "dtype": str(tensor.dtype),
                "finite": True,
                "tensor_sha256": hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest(),
            }
        )
    return rows


def _reconstruct(context, loaded, repo, weights, *, fixture_only=False):
    """Only this private kernel permits toy factories, always under fixture scope."""
    import torch

    if not fixture_only:
        require(
            isinstance(loaded, sources.LoadedF009Sources) and loaded.receipt["production_F009_pins"] is True,
            "Actual source14 loader and producer identities required",
        )
    trainer = loaded.trainer
    require(
        trainer.native_source_hashes() == sources.CORE_SHA
        and trainer.NativeTripletMIL is loaded.model.NativeTripletMIL,
        "Pinned F009 trainer/model identity differs",
    )
    checkpoint = trainer.load_native_checkpoint(context["checkpoint_path"], allow_artificial=False)
    require(
        checkpoint["schema_version"] == "native392_generic_dino_suffix_head_fast16_v1"
        and type(checkpoint["fold"]) is int
        and checkpoint["fold"] == context["fold"]
        and checkpoint["epoch"] == 12
        and checkpoint["weak_bce"] == context["row"]["weak_bce"]
        and checkpoint["config"] == context["config"]
        and checkpoint["targets"] == _contracts().TARGETS
        and checkpoint["source_sha256"] == sources.CORE_SHA
        and checkpoint["encoder_provenance"] == context["provenance"]
        and checkpoint["frozen_prefix_sha256"] == context["prefix"]
        and checkpoint["artificial"] is False,
        "CPU checkpoint exact E12/run/config/source/provenance/prefix differs",
    )
    shapes = _contracts().expected_tensor_shapes()
    original = {part: _metadata(checkpoint[part], expected, torch) for part, expected in shapes.items()}
    # Exactly one audited generic factory; never use the legacy COMMON factory/class.
    model = trainer.new_native_model(repo, weights, context["provenance"], context["config"], device="cpu")
    require(
        type(model) is loaded.model.NativeTripletMIL
        and type(model).__module__ == loaded.namespace + ".finetune_model"
        and type(model.head) is loaded.modules["feature_model"].FrozenFeatureHead
        and type(model.head).__module__ == loaded.namespace + ".feature_model",
        "Exact isolated model/head class identity required; module-name spoofing is insufficient",
    )
    require(trainer.prefix_digest(model) == context["prefix"], "Generic CPU prefix/norm differs before apply")
    require(
        trainer.apply_native_checkpoint(model, checkpoint, context["provenance"]) is model,
        "Strict apply must preserve model identity",
    )
    model.eval().requires_grad_(False)
    suffix = {name: model.encoder.state_dict()[name] for name in trainer.suffix_keys(model)}
    head = model.head.state_dict()
    reconstructed = {"encoder_suffix": suffix, "head": head}
    metadata = {part: _metadata(state, shapes[part], torch) for part, state in reconstructed.items()}
    require(
        metadata == original
        and all(
            torch.equal(value, checkpoint[part][name])
            for part, state in reconstructed.items()
            for name, value in state.items()
        )
        and trainer.prefix_digest(model) == context["prefix"],
        "CPU exact suffix/head bytes or frozen prefix changed",
    )
    require(
        not model.training
        and not model.encoder.training
        and all(
            p.device.type == "cpu" and p.dtype == torch.float32 and not p.requires_grad for p in model.parameters()
        ),
        "CPU FP32 eval/no-grad required",
    )
    rng = torch.get_rng_state().clone()
    gamma_state = model._gamma_generator.get_state().clone()
    require(
        model._gamma_draws == 0
        and model.sample_training_gamma(1, "cpu") is None
        and model._gamma_draws == 0
        and torch.equal(rng, torch.get_rng_state())
        and torch.equal(gamma_state, model._gamma_generator.get_state()),
        "Eval gamma must perform no draw, power operation or RNG change",
    )
    return {
        "schema_version": CPU_SCHEMA,
        "complete": True,
        "evidence_scope": "artificial_fixture_only" if fixture_only else "actual_CPU_tensor_reconstruction",
        "actual_CPU_strict_reconstruction": not fixture_only,
        "fold": context["fold"],
        "epoch": 12,
        "producer_sha256": context["producer_sha256"],
        "source_sha256": sources.CORE_SHA,
        "support_sha256": sources.SUPPORT_SHA,
        "producer_CLI_sha256": sha(ROOT / CLI_REL),
        "qualification_schema": QUALIFICATION_SCHEMA,
        "qualification_pins": copy.deepcopy(context["qualification_pins"]),
        "qualification_binding_pins": copy.deepcopy(context["qualification_binding_pins"]),
        "legacy_v1_joint_screen": "HOLD",
        "namespace_helper_sha256": _contracts().HELPER_SHA,
        "CLI_sha256": sources.CLI_SHA,
        "config_sha256": sources.CONFIG_SHA,
        "namespace": loaded.namespace,
        "model_class": type(model).__module__ + "." + type(model).__name__,
        "head_class": type(model.head).__module__ + "." + type(model.head).__name__,
        "targets": _contracts().TARGETS,
        "run_pin": context["run_pin"],
        "retained_index_pin": context["retained_index_pin"],
        "checkpoint_pin": context["checkpoint_pin"],
        "checkpoint_schema": checkpoint["schema_version"],
        "weak_bce": checkpoint["weak_bce"],
        "encoder_provenance": context["provenance"],
        "frozen_prefix_sha256": context["prefix"],
        "tensor_metadata": metadata,
        **{part + "_tensor_sha256": tensor_digest(state) for part, state in reconstructed.items()},
        "generic_factory_calls": 1,
        "model_forward_calls": 0,
        "optimizer_steps": 0,
        "model_training": False,
        "encoder_training": False,
        "gamma_draws": 0,
        "eval_gamma_RNG_unchanged": True,
        "teacher_MRI_Gold_Report_read": False,
        "GPU_used": False,
        "checkpoint_loader": {
            "source_sha256": sources.CORE_SHA["native_training_fast16.py"],
            "map_location": "cpu",
            "weights_only": True,
            "allow_artificial": False,
        },
        "strict_apply_identity_and_tensor_bytes_exact": True,
        "runtime_or_adoption_qualified": False,
    }


def produce(spec_path, expected_spec_sha256, fold, output, repo, weights, expected_producer_sha256):
    c = _contracts()
    require(
        sha(__file__) == expected_producer_sha256 and sha(spec_path) == expected_spec_sha256,
        "Explicit reviewed producer/spec SHAs required",
    )
    spec = c.read(spec_path)
    context = prepare_inputs(ROOT, spec, fold)
    output = Path(output).resolve()
    require(
        output.is_relative_to(ROOT / "artifacts/qa") and not output.exists(),
        "Fresh artifacts/qa proof directory required",
    )
    # First model-body access occurs only after both complete positive evaluation guards.
    require(c.pin(context["checkpoint_path"]) == context["checkpoint_pin"], "Actual E12 checkpoint bytes changed")
    output.mkdir(parents=True, exist_ok=False)
    loaded = None
    try:
        sources.stage_f009_sources(
            ROOT / "src/rsna_knee",
            ROOT / "scripts/train_native_dinov2_fast16.py",
            ROOT / "configs/experiments/f009-dinov2-native-last2-late-cosine-study-gamma.json",
            output / "source14",
        )
        loaded = sources.load_f009_sources(output / "source14", "_rsna_f009_strict_cpu_fold" + str(fold))
        result = _reconstruct(context, loaded, repo, weights)
        require(
            c.pin(context["checkpoint_path"]) == context["checkpoint_pin"]
            and c.pin(context["run_path"]) == context["run_pin"]
            and c.pin(context["index_path"]) == context["retained_index_pin"]
            and sha(spec_path) == expected_spec_sha256
            and sha(__file__) == expected_producer_sha256
            and loaded.trainer.native_source_hashes() == sources.CORE_SHA,
            "Input/source changed during reconstruction",
        )
        after = validate_amended_joint_metadata(ROOT, spec["qualification"])
        require(after["qualification_pins"] == context["qualification_pins"]
                and after["binding_pins"] == context["qualification_binding_pins"]
                and sha(ROOT / CLI_REL) == result["producer_CLI_sha256"], "Qualification/source changed during reconstruction")
        result.update(
            input_spec={"path": str(Path(spec_path).resolve()), **c.pin(spec_path)},
            local_source14_import=dict(loaded.receipt),
        )
        # A namespace cleanup failure must not leave a complete CPU receipt.
        loaded.close()
        loaded = None
        final = validate_amended_joint_metadata(ROOT, spec["qualification"])
        require(sha(spec_path) == expected_spec_sha256 and sha(__file__) == expected_producer_sha256
                and sha(ROOT / CLI_REL) == result["producer_CLI_sha256"]
                and final["qualification_pins"] == context["qualification_pins"]
                and final["binding_pins"] == context["qualification_binding_pins"],
                "Final spec/source/qualification changed after namespace cleanup")
        c.atomic_json(output / "CPU.json", result)
        return result
    except BaseException as error:
        c.atomic_json(
            output / "FAILED.json",
            {"complete": False, "actual_CPU_strict_reconstruction": False, "error_type": type(error).__name__},
        )
        raise
    finally:
        if loaded is not None:
            loaded.close()

