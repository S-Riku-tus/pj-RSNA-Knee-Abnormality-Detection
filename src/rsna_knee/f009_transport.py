"""Offline, hash-bound F009 CLS transport. No tensor import, network or uploader.

Root paths copied into audit JSON are provenance strings, never runtime paths.
This transports two source14/CLS bundles; the generic encoder and fixed REG
weights are external unchanged Inputs. Successful transport is not inference.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import zipfile
from pathlib import Path, PurePosixPath

from . import f009_source_namespace as sources

PLAN_REL = "experiments/f009-P039-parent-fixed20-CLS-pair-conditional-20261017.json"
PLAN_SHA = "37bb0c60826da6deddda0e3929fd8aeb4a2f56ad91371af9dab365835a69bba2"
PLAN_OBJECT_SHA = "d7a80608b6adca6d3329b358cfe22fed75dc5d86c9edc502b709b4885cf3172a"
SCHEMA = "f009_amended_CLS_transport_v1"
PAIR_SCHEMA = "f009_fixed_E12_two_CLS_amended_source14_local_pair_v3"
BUNDLE_SCHEMA = "f009_fixed_E12_single_CLS_amended_source14_local_bundle_v3"
CPU_SCHEMA = "f009_strict_CPU_amended_eval_v2"
QUAL_SCHEMA = "F009_amended_joint_metadata_v2"
PRODUCER_SHA = "fc2fbf7ab581bc6258817ac9fb189de966ce8b2d074b531e7bf3b642d1fe0705"
PRODUCER_CLI_SHA = "8f57bd7688d5507af8fce7151793acb1c2e0a185401ef474234a4b4605d9667d"
GENERIC_SHA = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
           "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
QUAL_KEYS = {"result0", "guard0", "result1", "guard1", "legacy_guard1", "legacy_comparison1"}
BINDINGS = {"policy0": "policy0.json", "policy1": "original-policy1.json", "policy2": "amended-policy2.json",
            "declaration": "amendment-declaration.json", "snapshot": "epoch0-snapshot.json",
            "legacy_evaluator": "original-fold1-evaluator.py", "amended_wrapper": "amended-fold1-wrapper.py"}
FIXED_BINDINGS = {"policy0": "de5a6aa743d2d663f98244322958bf4b8154908238384112a7badf2df1e2d42c",
                  "policy1": "9f42cd771db3f6bac92d275769636a451d3ef4dacd35e245a53d888c17ec2420",
                  "policy2": "bf53e1828784df4d0da793cea6014f055ca9e5948dc5013cf368b646dd328a90",
                  "declaration": "939394e0a10c01ab109a441ba1ab898a8e2038d60bf18ec85145444475f7ee6e",
                  "snapshot": "9ebb83ba5df1b708f73f0097781a50c0c0d814bdbc36cd273ddb927bfe974e70",
                  "legacy_evaluator": "c8e244461dbf45be68fff13328f2a774df1fbf5ed69e8c109ee84b0f421e352c",
                  "amended_wrapper": "bcf9fbe5dad5c7f1faae6761548b1d8786d5a24d1c25525f5e9f5d2ea289bcce"}
HELPERS = {"CPU_producer_v2.py": PRODUCER_SHA, "CPU_producer_CLI_v2.py": PRODUCER_CLI_SHA,
           "CPU_producer_v1_helper.py": "1d2cbdc18212454d97aca2a768ed2dc6f89492a757644c7e91de03e0eeb066f1",
           "CPU_frozen_v1_tensor_contract.py": "d3a951a2913798d77c7e93977997bff68ad334173cc7b8c0382e96d9f5772781",
           "sealer_v2_pure_helpers.py": "b1687e813ecdecfadff8b5adc21e02577a5c57998cc0891899fa4034f1043813",
           "source_namespace_helper.py": "cd2b5fd71355801b67f4f4596487d42728e4974f84b10eac15155ef35c6e5526",
           "sealer_v3.py": "cc7c77f012c320452f0a3232196d77f38da077a6a91e1576903d6967bda157e7",
           "sealer_CLI_v3.py": "9e8559205736624df2b84537c71345f375f0f15ddac43cf7c6a3e3c0bafce84e"}
FILES = {"manifest.json", "bundle.zip", "pair-sidecar.json", "BUILD_COMPLETE.json", "conditional-plan.json", "TRANSPORT_COMPLETE.json"}
MAX_META = 16 * 1024**2
MAX_MEMBER = 128 * 1024**2
MAX_TOTAL = 192 * 1024**2
MAX_ARCHIVE = MAX_TOTAL + 1024**2
STATISTICS = ("fixed10_delta", "all12_auxiliary_delta", "fixed10_interval", "all12_auxiliary_interval",
              "per_target", "bootstrap", "weak_BCE", "ordered_UID_sha256")


def require(value, message):
    if not value:
        raise ValueError(message)


def member(name):
    require(isinstance(name, str) and "\\" not in name and ":" not in name, "Unsafe member name")
    path = PurePosixPath(name)
    require(path.parts and not path.is_absolute() and path.as_posix() == name
            and all(p not in {".", "..", ""} and not p.endswith((".", " ")) for p in path.parts), "Noncanonical member")
    require(all(not re.fullmatch(r"(?i:con|prn|aux|nul|com[1-9]|lpt[1-9])", p.split(".", 1)[0]) for p in path.parts), "Reserved member")
    return name


def pin(path, limit=MAX_MEMBER):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= limit, "Bounded regular file required")
    size, digest = path.stat().st_size, hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    require(path.stat().st_size == size, "File changed during hashing")
    return {"bytes": size, "sha256": digest.hexdigest()}


def read(path):
    pin(path, MAX_META)

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def json_new(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as stream:
        stream.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())
    os.replace(temporary, path)


def copy_stream(source, destination, expected):
    require(pin(source) == expected, "Source copy pin differs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest, size = hashlib.sha256(), 0
    with Path(source).open("rb") as incoming, destination.open("xb") as outgoing:
        for chunk in iter(lambda: incoming.read(1024**2), b""):
            size += len(chunk)
            require(size <= expected["bytes"], "Copy grew")
            digest.update(chunk)
            outgoing.write(chunk)
    require({"bytes": size, "sha256": digest.hexdigest()} == expected and pin(source) == expected, "Copy changed")


def asset_names(root_spec):
    names = {"trained/epoch_012.pt", "source14/config.json", "source14/producer-CLI.py", "source14/source-manifest.json",
             "source14/package/__init__.py", "audit/run.json", "audit/index.json", "audit/CPU.json"}
    names.update("source14/package/" + name for name in {**sources.CORE_SHA, **sources.SUPPORT_SHA})
    names.update("audit/" + key + ".json" for key in QUAL_KEYS)
    names.update("audit/" + name for name in BINDINGS.values())
    names.update("audit/" + name for name in HELPERS)
    if root_spec is not None:
        names.add("audit/root-spec.json")
    return names


def inventory(directory):
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(), "Regular directory required")
    result = set()
    for path in directory.rglob("*"):
        require(not path.is_symlink(), "Symlink in bundle")
        if path.is_file():
            result.add(member(path.relative_to(directory).as_posix()))
        else:
            require(path.is_dir(), "Nonregular bundle member")
    require(len(result) == len({x.casefold() for x in result}), "Case-fold collision")
    return result


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def expected_shapes():
    suffix = {}
    for block in (10, 11):
        for name, shape in {"norm1.weight": [384], "norm1.bias": [384], "attn.qkv.weight": [1152, 384],
                            "attn.qkv.bias": [1152], "attn.proj.weight": [384, 384], "attn.proj.bias": [384],
                            "ls1.gamma": [384], "norm2.weight": [384], "norm2.bias": [384],
                            "mlp.fc1.weight": [1536, 384], "mlp.fc1.bias": [1536], "mlp.fc2.weight": [384, 1536],
                            "mlp.fc2.bias": [384], "ls2.gamma": [384]}.items():
            suffix["blocks." + str(block) + "." + name] = shape
    return {"encoder_suffix": suffix, "head": {"project.0.weight": [384], "project.0.bias": [384],
            "project.1.weight": [192, 384], "project.1.bias": [192], "plane.weight": [4, 192],
            "geometry.weight": [192, 4], "geometry.bias": [192], "slice_score.weight": [12, 192],
            "slice_score.bias": [12], "series_query": [12, 192], "output_weight": [12, 192], "output_bias": [12]}}


def _qualification(folder, bundle, cpu, *, fixture_only):
    q, bindings = bundle["qualification_pins"], bundle["qualification_binding_pins"]
    require(set(q) == QUAL_KEYS and set(bindings) == set(BINDINGS), "Exact six qualification/seven binding pins required")
    require(cpu["qualification_pins"] == q and cpu["qualification_binding_pins"] == bindings
            and cpu["qualification_schema"] == bundle["qualification_schema"] == QUAL_SCHEMA, "CPU qualification differs")
    docs = {}
    for key, ref in q.items():
        require(set(ref) == {"path", "bytes", "sha256"} and isinstance(ref["path"], str), "Declared provenance pin required")
        path = folder / "audit" / (key + ".json")
        require(pin(path) == {k: ref[k] for k in ("bytes", "sha256")}, "Copied qualification SHA differs")
        docs[key] = read(path)
    for key, ref in bindings.items():
        path = folder / "audit" / BINDINGS[key]
        require(pin(path) == {k: ref[k] for k in ("bytes", "sha256")}, "Copied source/policy binding differs")
        if not fixture_only or key in {"legacy_evaluator", "amended_wrapper"}:
            require(pin(path)["sha256"] == FIXED_BINDINGS[key], "Frozen policy/source binding differs")
    r0, g0, r1, g1, lg, lr = (docs[k] for k in ("result0", "guard0", "result1", "guard1", "legacy_guard1", "legacy_comparison1"))
    qp = {key: {k: ref[k] for k in ("bytes", "sha256")} for key, ref in q.items()}
    require(g0["complete"] is True and g0["original_evaluate_calls"] == 1 and g0["original_result_pin"] == qp["result0"]
            and g0["schema_version"] == "F009_fixed_E12_guarded_evaluation_receipt_v1"
            and g0["wrapper_pin"]["sha256"] == "21c4e3089953d59f6da30fcb82fff3e27f5f3ecef1e20f4a96e22d561c5ba76b"
            and r0["policy_sha256"] == bindings["policy0"]["sha256"]
            and r0["fold1_gate_conditions_met"] is g0["fold1_gate_conditions_met"] is True
            and r0["candidate_epoch"] == r0["control_epoch"] == 12 and r0["candidate_AMP_skipped_steps"] == 0
            and positive(r0["fixed10_delta"]), "Positive completed fold0 required")
    require(r1["schema_version"] == "F009_fold1_known_control_AMP_amended_joint_result_v2"
            and g1["schema_version"] == "F009_fold1_amended_evaluation_receipt_v2"
            and g1["complete"] is True and r1["actual_comparison"] is g1["actual_comparison"] is True
            and g1["legacy_comparison_calls"] == 1 and g1["result_pin"] == qp["result1"]
            and g1["legacy_guard_pin"] == r1["legacy_guard_pin"] == qp["legacy_guard1"]
            and g1["legacy_comparison_pin"] == r1["legacy_comparison_pin"] == qp["legacy_comparison1"]
            and r1["v2_joint_candidate_screen_passed"] is g1["v2_joint_candidate_screen_passed"] is True
            and positive(r1["fixed10_delta"]), "Actual positive amended result/crosspins required")
    require(r1["legacy_v1_joint_screen"] == "HOLD" and r1["legacy_v1_machine_passed"] is False
            and g1["original_v1_preserved_as_HOLD"] is True and g1["legacy_consumer_compatible"] is False
            and lg["complete"] is True and lg["machine_engineering_passed"] is False
            and lg["AMP_skipped_steps"] == lr["AMP_skipped_steps"] == {"control": 1, "candidate": 0}
            and lr["fold1_machine_engineering_passed"] is lr["joint_candidate_screen_passed"] is lr["fold1_independent_positive_screen"] is False,
            "Legacy HOLD must remain unchanged")
    require(all(r1[k] == lr[k] for k in STATISTICS)
            and lr["policy0_sha256"] == bindings["policy0"]["sha256"] and lr["policy1_sha256"] == bindings["policy1"]["sha256"]
            and r1["inputs"] == lr["inputs"] == lg["completed_E12_control_candidate"]
            and r1["fold0"] == lr["fold0"] == lg["fold0_positive_sha_bound"], "Copied statistics/lineage differs")
    prior = lg["fold0_positive_sha_bound"]
    require(prior["result"] == qp["result0"] and prior["guard_receipt"] == qp["guard0"]
            and prior["fixed10_delta"] == r0["fixed10_delta"] and prior["engineering_gate_passed"] is True, "Fold0 crossbinding differs")
    p2 = read(folder / "audit/amended-policy2.json")
    declaration = read(folder / "audit/amendment-declaration.json")
    snapshot = read(folder / "audit/epoch0-snapshot.json")
    require(declaration["policy"] == bindings["policy2"] and declaration["candidate_epoch0_snapshot"] == bindings["snapshot"]
            and p2["original_policy_immutable"] == bindings["policy1"] and p2["original_evaluator_immutable"] == bindings["legacy_evaluator"]
            and p2["fold0_primary_result"] == q["result0"] and p2["fold0_guard"] == q["guard0"]
            and p2["candidate_epoch0_snapshot"] == bindings["snapshot"] and snapshot["history"] == []
            and snapshot["status"] == "running" and snapshot["fold"] == 1 and snapshot["seed"] == 20261002
            and declaration["observed_candidate_history_length"] == 0 and declaration["candidate_fold1_metric_read"] is False,
            "Premetric snapshot/declaration binding differs")
    require(r1["policy2_sha256"] == bindings["policy2"]["sha256"]
            and r1["declaration_sha256"] == bindings["declaration"]["sha256"]
            and g1["wrapper_pin"] == {k: bindings["amended_wrapper"][k] for k in ("bytes", "sha256")}
            and g1["policy_pin"] == {k: bindings["policy2"][k] for k in ("bytes", "sha256")}
            and g1["declaration_pin"] == {k: bindings["declaration"][k] for k in ("bytes", "sha256")}
            and g1["epoch0_snapshot_pin"] == {k: bindings["snapshot"][k] for k in ("bytes", "sha256")}
            and g1["legacy_evaluator_pin"] == bindings["legacy_evaluator"], "Amended policy/source crosspins differ")
    e = r1["v2_engineering"]
    require(e == g1["engineering_v2"] and e["passed"] is True and e["legacy_machine_engineering_passed"] is False
            and e["legacy_v1_joint_screen"] == "HOLD" and e["actual_update_count_parity_claimed"] is False
            and e["candidate_zero_skip_exception"] is False
            and e["exact_accounting"]["control"] == p2["required_control_totals"] == {"optimizer_steps": 2495, "optimizer_step_opportunities": 2496, "amp_skipped_steps": 1}
            and e["exact_accounting"]["candidate"] == p2["required_candidate_totals"] == {"optimizer_steps": 2496, "optimizer_step_opportunities": 2496, "amp_skipped_steps": 0},
            "Exact historical exception/candidate zero accounting required")
    return docs


def _consume(folder, *, expected_bundle_pin, pair, conditional_plan, fold, fixture_only=False, metadata_only=False):
    folder = Path(folder)
    require(type(fold) is int and fold in (0, 1), "Exact CLS fold required")
    require(pin(folder / "bundle.json") == expected_bundle_pin, "Explicit bundle manifest pin differs")
    b = read(folder / "bundle.json")
    require(hashlib.sha256(json.dumps(conditional_plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest() == PLAN_OBJECT_SHA,
            "Entire fixed conditional plan differs")
    require(b["schema_version"] == BUNDLE_SCHEMA and b["fold"] == fold and b["epoch"] == 12
            and b["CPU_producer_sha256"] == PRODUCER_SHA and b["CPU_producer_CLI_sha256"] == PRODUCER_CLI_SHA,
            "Fixed amended bundle producer/fold differs")
    require(pair["schema_version"] == PAIR_SCHEMA and pair["local_pair_complete"] is True
            and set(pair["CLS_E12_bundles"]) == {"0", "1"} and pair["CLS_E12_bundles"][str(fold)] == expected_bundle_pin
            and pair["logical_models"] == 4 and pair["old_CLS_included"] is False and pair["old_v1_joint_screen"] == "HOLD",
            "Complete two-fold pair required")
    require(pair["artificial_fixture_only"] is b["artificial_fixture_only"] is fixture_only
            and pair["actual_CLS_stage_constructed"] is b["actual_strict_CPU_accepted"] is (not fixture_only)
            and pair["four_model_transport_runtime_qualified"] is False and b["runtime_transport_or_Submit_qualified"] is False,
            "Artificial/actual scope differs")
    require(pair["qualification_pins"] == b["qualification_pins"]
            and pair["qualification_binding_pins"] == b["qualification_binding_pins"], "Pair qualification differs")
    require(fixture_only or isinstance(pair["root_spec_pin"], dict), "Actual sealed root spec required")
    require(conditional_plan["frozen_control"]["source"]["sha256"] == "6453b3d09787abab07205d832c0176f0e78d050127e98d83c9f835bc5f799452"
            and pair["fixed_REG_references_only"] == {key: {"transport": conditional_plan["intended_intervention"]["old_REG_transport_refs"][key],
               "fingerprint": conditional_plan["intended_intervention"]["old_REG_fingerprints"][key]} for key in ("REG:0", "REG:1")},
            "Fixed P039 parent or unchanged REG references differ")
    allowed = asset_names(pair["root_spec_pin"])
    present = {"bundle.json", *allowed} - ({"trained/epoch_012.pt"} if metadata_only else set())
    require(set(b["assets"]) == allowed and inventory(folder) == present, "Exact nondata inventory required")
    require(sum(ref["bytes"] for ref in b["assets"].values()) <= MAX_TOTAL, "Total bundle cap exceeded")
    for name, expected in b["assets"].items():
        require(member(name) == name, "Unsafe asset name")
        if not name.endswith(".pt"):
            require(pin(folder / name, MAX_META) == expected, "Bundle member bytes/SHA differs")
    for name, digest in HELPERS.items():
        require(pin(folder / "audit" / name)["sha256"] == digest, "Pinned producer/helper source differs")
    core = {**sources.CORE_SHA, **sources.SUPPORT_SHA}
    require(b["source_sha256"] == sources.CORE_SHA and b["support_sha256"] == sources.SUPPORT_SHA, "Source14 identity differs")
    for name, expected in core.items():
        require(pin(folder / "source14/package" / name)["sha256"] == expected, "Source14 body differs")
    require((folder / "source14/package/__init__.py").read_bytes() == sources.INIT
            and pin(folder / "source14/config.json")["sha256"] == sources.CONFIG_SHA
            and pin(folder / "source14/producer-CLI.py")["sha256"] == sources.CLI_SHA
            and pin(folder / "source14/source-manifest.json") == b["source_manifest_pin"], "Source14 auxiliary binding differs")
    sm = read(folder / "source14/source-manifest.json")
    require(sm["source_sha256"] == core and sm["production_F009_pins"] is True and sm["init_sha256"] == hashlib.sha256(sources.INIT).hexdigest(), "Source14 manifest differs")
    cpu, run, index = (read(folder / "audit" / (name + ".json")) for name in ("CPU", "run", "index"))
    require(pin(folder / "audit/CPU.json") == b["CPU_receipt"] and cpu["schema_version"] == CPU_SCHEMA
            and cpu["complete"] is True and cpu["producer_sha256"] == PRODUCER_SHA and cpu["producer_CLI_sha256"] == PRODUCER_CLI_SHA
            and cpu["actual_CPU_strict_reconstruction"] is (not fixture_only)
            and cpu["evidence_scope"] == ("artificial_fixture_only" if fixture_only else "actual_CPU_tensor_reconstruction")
            and cpu["runtime_or_adoption_qualified"] is False, "Actual strict CPU proof required")
    require(cpu["run_pin"] == pin(folder / "audit/run.json") and cpu["retained_index_pin"] == pin(folder / "audit/index.json")
            and cpu["checkpoint_pin"] == b["checkpoint_pin"] == b["assets"]["trained/epoch_012.pt"]
            and cpu["fold"] == run["fold"] == index["fold"] == fold and cpu["epoch"] == 12
            and run["status"] == "complete" and run["seed"] == 20261002 and run["source_sha256"] == cpu["source_sha256"] == sources.CORE_SHA
            and cpu["support_sha256"] == sources.SUPPORT_SHA and cpu["targets"] == run["targets"] == TARGETS
            and cpu["encoder_provenance"] == run["encoder_provenance"]
            and cpu["encoder_provenance"]["weights_sha256"] == GENERIC_SHA, "Run/index/checkpoint/provenance differs")
    require(cpu["namespace_helper_sha256"] == HELPERS["source_namespace_helper.py"] and cpu["CLI_sha256"] == sources.CLI_SHA
            and cpu["config_sha256"] == sources.CONFIG_SHA and run["config"] == read(folder / "source14/config.json")
            and cpu["checkpoint_schema"] == "native392_generic_dino_suffix_head_fast16_v1"
            and cpu["frozen_prefix_sha256"] == conditional_plan["intended_intervention"]["old_REG_fingerprints"]["REG:0"]["frozen_prefix_sha256"]
            and re.fullmatch(r"_rsna_f009_[a-z0-9_]+", cpu["namespace"])
            and cpu["model_class"] == cpu["namespace"] + ".finetune_model.NativeTripletMIL"
            and cpu["head_class"] == cpu["namespace"] + ".feature_model.FrozenFeatureHead"
            and cpu["model_training"] is cpu["encoder_training"] is False, "Pinned CPU class/source/config/prefix differs")
    for family, shapes in expected_shapes().items():
        rows = cpu["tensor_metadata"][family]
        require(len(rows) == len(shapes) and {r["name"] for r in rows} == set(shapes)
                and all(r["shape"] == shapes[r["name"]] and r["dtype"] == "torch.float32" and r["finite"] is True
                        and re.fullmatch("[0-9a-f]{64}", r["tensor_sha256"]) for r in rows), "Exact40 tensor metadata required")
    require([r["epoch"] for r in run["history"]] == list(range(1, 13)) and all(r["amp_skipped_steps"] == 0 for r in run["history"])
            and cpu["generic_factory_calls"] == 1 and cpu["model_forward_calls"] == cpu["optimizer_steps"] == cpu["gamma_draws"] == 0
            and cpu["eval_gamma_RNG_unchanged"] is cpu["strict_apply_identity_and_tensor_bytes_exact"] is True
            and cpu["GPU_used"] is cpu["teacher_MRI_Gold_Report_read"] is False
            and cpu["checkpoint_loader"] == {"source_sha256": sources.CORE_SHA["native_training_fast16.py"], "map_location": "cpu", "weights_only": True, "allow_artificial": False}, "Strict CPU/no-data contract differs")
    docs = _qualification(folder, b, cpu, fixture_only=fixture_only)
    completed = docs["guard0"]["guards"]["completed_E12_control_candidate"]["candidate"] if fold == 0 else docs["legacy_guard1"]["completed_E12_control_candidate"]["candidate"]
    require(cpu["run_pin"] == completed["run"] and cpu["retained_index_pin"] == completed["index"]
            and cpu["checkpoint_pin"] == completed["E12_checkpoint_body_stream_hash"], "CPU and actual evaluation E12 join differs")
    selected = [r for r in index["checkpoints"] if r["epoch"] == 12]
    require(len(selected) == 1 and selected[0]["checkpoint_sha256"] == cpu["checkpoint_pin"]["sha256"]
            and selected[0]["prediction_sha256"] == completed["E12_prediction"]["sha256"] == run["history"][-1]["prediction_sha256"], "Retained E12 reference differs")
    if not metadata_only:
        require(pin(folder / "trained/epoch_012.pt") == b["checkpoint_pin"], "Qualified checkpoint body differs")
    fingerprint = {"model_class": cpu["model_class"], "checkpoint_sha256": cpu["checkpoint_pin"]["sha256"],
                   "frozen_prefix_sha256": cpu["frozen_prefix_sha256"], "encoder_suffix_tensor_sha256": cpu["encoder_suffix_tensor_sha256"],
                   "head_tensor_sha256": cpu["head_tensor_sha256"], "CPU_receipt_sha256": b["CPU_receipt"]["sha256"],
                   "run_json_sha256": cpu["run_pin"]["sha256"], "selected_epoch": 12, "weak_bce": cpu["weak_bce"]}
    return {"schema_version": "f009_consumed_CLS_bundle_v1", "fold": fold, "bundle_root": folder,
            "source14": folder / "source14", "checkpoint_path": folder / "trained/epoch_012.pt",
            "config": read(folder / "source14/config.json"), "cpu": cpu, "run": run, "index": index,
            "fingerprint": fingerprint, "qualification_pins": b["qualification_pins"],
            "qualification_binding_pins": b["qualification_binding_pins"], "manifest": b, "bundle": b, "pair": pair,
            "bundle_pin": expected_bundle_pin, "conditional_plan": conditional_plan,
            "cpu_pin": b["CPU_receipt"], "run_pin": cpu["run_pin"], "index_pin": cpu["retained_index_pin"],
            "checkpoint_pin": b["checkpoint_pin"],
            "checkpoint_body_verified": not metadata_only,
            "actual_transport": not fixture_only and not metadata_only, "runtime_transport_qualified": False}


def consume_f009_bundle(bundle_root, *, expected_bundle_pin, pair, conditional_plan, fold):
    return _consume(bundle_root, expected_bundle_pin=expected_bundle_pin, pair=pair,
                    conditional_plan=conditional_plan, fold=fold, fixture_only=False)


def _pack(pair_directory, output, *, expected_pair_pin, expected_complete_pin, conditional_plan_path, fixture_only=False):
    source, output = Path(pair_directory).resolve(), Path(output).resolve()
    implementation_pin = pin(__file__)
    require(not output.exists() and not output.is_symlink() and not output.is_relative_to(source), "Fresh separate pack output required")
    published = output
    intended_parent = published.parent.resolve()
    staging = published.with_name(published.name + ".partial")
    require(staging.resolve() == intended_parent / staging.name and not staging.exists() and not staging.is_symlink()
            and not staging.resolve().is_relative_to(source), "Fresh owned sibling staging directory required")
    require(pin(source / "pair-sidecar.json") == expected_pair_pin and pin(source / "BUILD_COMPLETE.json") == expected_complete_pin, "Explicit completed pair pins differ")
    complete, pair = read(source / "BUILD_COMPLETE.json"), read(source / "pair-sidecar.json")
    require(complete["complete"] is True and complete["pair_sidecar"] == expected_pair_pin
            and complete["artificial_fixture_only"] is fixture_only, "Sealer completion required")
    require(pin(conditional_plan_path)["sha256"] == PLAN_SHA, "Frozen conditional plan differs")
    plan = read(conditional_plan_path)
    require(inventory(source) == {"pair-sidecar.json", "BUILD_COMPLETE.json", *(
        "fold" + str(f) + "/" + name for f in (0, 1) for name in {"bundle.json", *asset_names(pair["root_spec_pin"])})}, "Complete twofold sealer output only")
    for fold in (0, 1):
        _consume(source / ("fold" + str(fold)), expected_bundle_pin=pair["CLS_E12_bundles"][str(fold)],
                 pair=pair, conditional_plan=plan, fold=fold, fixture_only=fixture_only)
    # Build every file and both completion copies privately. The requested
    # destination does not exist until the entire owned directory is published.
    output = staging
    output.mkdir(parents=True, exist_ok=False)
    try:
        summaries = {}
        for fold in (0, 1):
            target, folder = output / ("fold" + str(fold)), source / ("fold" + str(fold))
            target.mkdir()
            for filename, incoming in (("pair-sidecar.json", source / "pair-sidecar.json"), ("BUILD_COMPLETE.json", source / "BUILD_COMPLETE.json"), ("conditional-plan.json", Path(conditional_plan_path))):
                copy_stream(incoming, target / filename, pin(incoming))
            names = sorted(inventory(folder))
            members = {**read(folder / "bundle.json")["assets"], "bundle.json": pair["CLS_E12_bundles"][str(fold)]}
            archive = target / "bundle.zip.partial"
            with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED, allowZip64=False) as book:
                for name in names:
                    require(pin(folder / name) == members[name], "Pre-archive member changed")
                    info = zipfile.ZipInfo(name, (2026, 10, 17, 0, 0, 0))
                    info.external_attr = (stat.S_IFREG | 0o600) << 16
                    with (folder / name).open("rb") as incoming, book.open(info, "w") as outgoing:
                        while chunk := incoming.read(1024**2):
                            outgoing.write(chunk)
                    require(pin(folder / name) == members[name], "Member changed during packing")
            os.replace(archive, target / "bundle.zip")
            transport = {"schema_version": SCHEMA, "fold": fold, "archive": pin(target / "bundle.zip", MAX_ARCHIVE),
                "bundle": pair["CLS_E12_bundles"][str(fold)], "pair": expected_pair_pin, "complete": expected_complete_pin,
                "plan": pin(conditional_plan_path), "members": members, "artificial_fixture_only": fixture_only,
                "actual_transport": not fixture_only, "runtime_or_Public_qualified": False,
                "transport_source_sha256": implementation_pin["sha256"]}
            json_new(target / "manifest.json", transport)
            summaries[str(fold)] = pin(target / "manifest.json")
        require(pin(source / "pair-sidecar.json") == expected_pair_pin and pin(source / "BUILD_COMPLETE.json") == expected_complete_pin, "Pair changed during packing")
        for fold in (0, 1):
            _consume(source / ("fold" + str(fold)), expected_bundle_pin=pair["CLS_E12_bundles"][str(fold)],
                     pair=pair, conditional_plan=plan, fold=fold, fixture_only=fixture_only)
            target = output / ("fold" + str(fold))
            require(pin(target / "manifest.json") == summaries[str(fold)], "Transport manifest changed during packing")
            transport = read(target / "manifest.json")
            require(pin(target / "bundle.zip", MAX_ARCHIVE) == transport["archive"], "Packed archive changed before completion")
        require(pin(__file__) == implementation_pin, "Transport implementation changed during packing")
        completion = {"complete": True, "fold_manifests": summaries, "pair": expected_pair_pin,
            "artificial_fixture_only": fixture_only, "actual_transport": not fixture_only, "runtime_or_Public_qualified": False}
        json_new(output / "TRANSPORT_COMPLETE.json", completion)
        for fold in (0, 1):
            copy_stream(output / "TRANSPORT_COMPLETE.json", output / ("fold" + str(fold)) / "TRANSPORT_COMPLETE.json", pin(output / "TRANSPORT_COMPLETE.json"))
        require(output.resolve() == intended_parent / staging.name
                and published.resolve() == intended_parent / published.name
                and not output.is_symlink() and not published.exists() and not published.is_symlink(),
                "Checked same-parent fresh directory publication required")
        os.rename(output, published)
        return summaries
    except BaseException as error:
        # These exact paths were created by this call under the fresh owned
        # staging root. Withdraw all completion markers even from retained
        # partial evidence, so no single fold can be accepted after failure.
        for relative in ("TRANSPORT_COMPLETE.json", "fold0/TRANSPORT_COMPLETE.json", "fold1/TRANSPORT_COMPLETE.json"):
            marker = output / relative
            require(marker.resolve().is_relative_to(staging.resolve()), "Completion rollback leaves owned staging")
            if marker.exists():
                marker.unlink()
        json_new(output / "TRANSPORT_FAILED.json", {"complete": False, "error_type": type(error).__name__, "artificial_fixture_only": fixture_only})
        raise


def pack_f009_pair(pair_directory, output, *, expected_pair_pin, expected_complete_pin, conditional_plan_path):
    return _pack(pair_directory, output, expected_pair_pin=expected_pair_pin, expected_complete_pin=expected_complete_pin,
                 conditional_plan_path=conditional_plan_path, fixture_only=False)


def _unpack(input_root, output, *, expected_manifest_sha256, fold, fixture_only=False):
    incoming, output = Path(input_root), Path(output)
    require(inventory(incoming) == FILES and not output.exists() and not output.is_symlink()
            and not output.resolve().is_relative_to(incoming.resolve()), "Exact transport files and fresh separate output required")
    require(pin(incoming / "manifest.json")["sha256"] == expected_manifest_sha256, "Explicit transport manifest SHA differs")
    t = read(incoming / "manifest.json")
    require(t["schema_version"] == SCHEMA and type(fold) is int and t["fold"] == fold and fold in (0, 1)
            and t["artificial_fixture_only"] is fixture_only and t["actual_transport"] is (not fixture_only)
            and t["runtime_or_Public_qualified"] is False, "Transport fold/scope differs")
    require(t["transport_source_sha256"] == pin(__file__)["sha256"], "Pinned transport implementation differs")
    packed_complete = read(incoming / "TRANSPORT_COMPLETE.json")
    require(packed_complete["complete"] is True and set(packed_complete["fold_manifests"]) == {"0", "1"}
            and packed_complete["fold_manifests"][str(fold)] == pin(incoming / "manifest.json") and packed_complete["pair"] == t["pair"]
            and packed_complete["artificial_fixture_only"] is fixture_only and packed_complete["actual_transport"] is (not fixture_only),
            "Twofold transport completion required")
    archive_path = incoming / "bundle.zip"
    require(not archive_path.is_symlink() and archive_path.is_file() and 0 < archive_path.stat().st_size <= MAX_ARCHIVE
            and archive_path.stat().st_size == t["archive"]["bytes"], "Bounded archive file required")
    require(pin(incoming / "pair-sidecar.json") == t["pair"]
            and pin(incoming / "BUILD_COMPLETE.json") == t["complete"] and pin(incoming / "conditional-plan.json")["sha256"] == PLAN_SHA
            and pin(incoming / "conditional-plan.json") == t["plan"], "Transport sidecar/archive pins differ")
    pair, complete, plan = (read(incoming / name) for name in ("pair-sidecar.json", "BUILD_COMPLETE.json", "conditional-plan.json"))
    require(complete["complete"] is True and complete["pair_sidecar"] == t["pair"] and complete["artificial_fixture_only"] is fixture_only, "Sealer incomplete")
    names = {"bundle.json", *asset_names(pair["root_spec_pin"])}
    require(set(t["members"]) == names and sum(r["bytes"] for r in t["members"].values()) <= MAX_TOTAL, "Exact bounded archive inventory required")
    with zipfile.ZipFile(incoming / "bundle.zip") as archive:
        entries = archive.infolist()
        require(len(entries) == len(names) and {e.filename for e in entries} == names, "Extra/missing/duplicate archive member")
        for entry in entries:
            mode = entry.external_attr >> 16
            require(member(entry.filename) == entry.filename and not entry.is_dir() and stat.S_IFMT(mode) in (0, stat.S_IFREG)
                    and entry.compress_type == zipfile.ZIP_STORED and entry.flag_bits & 1 == 0
                    and entry.file_size == t["members"][entry.filename]["bytes"]
                    and 0 < entry.file_size <= (MAX_MEMBER if entry.filename.endswith(".pt") else MAX_META), "Unsafe/oversized archive member")
        output.mkdir(parents=True, exist_ok=False)
        try:
            folder = output / "bundle"
            def extract(entry):
                path = folder / entry.filename
                path.parent.mkdir(parents=True, exist_ok=True)
                checksum, size = hashlib.sha256(), 0
                with archive.open(entry) as stream, path.open("xb") as target:
                    while chunk := stream.read(1024**2):
                        size += len(chunk)
                        require(size <= entry.file_size, "Archive member grew")
                        checksum.update(chunk)
                        target.write(chunk)
                require({"bytes": size, "sha256": checksum.hexdigest()} == t["members"][entry.filename], "Extracted member digest differs")
            for entry in entries:
                if entry.filename != "trained/epoch_012.pt":
                    extract(entry)
            _consume(folder, expected_bundle_pin=t["bundle"], pair=pair, conditional_plan=plan, fold=fold,
                     fixture_only=fixture_only, metadata_only=True)
            require(pin(incoming / "bundle.zip", MAX_ARCHIVE) == t["archive"], "Qualified archive bytes differ")
            extract(next(entry for entry in entries if entry.filename == "trained/epoch_012.pt"))
            descriptor = _consume(folder, expected_bundle_pin=t["bundle"], pair=pair, conditional_plan=plan, fold=fold, fixture_only=fixture_only)
            require(pin(incoming / "manifest.json")["sha256"] == expected_manifest_sha256 and pin(incoming / "bundle.zip", MAX_ARCHIVE) == t["archive"], "Transport changed during extraction")
            receipt = {"complete": True, "fold": fold, "transport_manifest_sha256": expected_manifest_sha256,
                       "bundle": t["bundle"], "archive": t["archive"], "artificial_fixture_only": fixture_only,
                       "actual_transport": not fixture_only, "runtime_or_Public_qualified": False}
            json_new(output / "TRANSPORT_UNPACKED.json", receipt)
            descriptor["transport_receipt"] = receipt
            return descriptor
        except BaseException as error:
            json_new(output / "TRANSPORT_FAILED.json", {"complete": False, "error_type": type(error).__name__, "artificial_fixture_only": fixture_only})
            raise


def unpack_f009_transport(input_root, output, *, expected_manifest_sha256, fold):
    return _unpack(input_root, output, expected_manifest_sha256=expected_manifest_sha256, fold=fold, fixture_only=False)
