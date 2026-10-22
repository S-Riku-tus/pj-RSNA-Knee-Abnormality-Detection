"""Local-only F009 E12 pair sealer preparation; actual CPU producer pending.

Production sealing is closed until an actual strict-CPU producer is reviewed
and pinned. The private fixture engine copies artificial bytes only in tests.
No upload, model import/deserialization, GPU, inference, fit or queue code.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

from . import f009_source_namespace as sources

ROOT = Path(__file__).resolve().parents[2]
BASE = "artifacts/kaggle/p035-p029m-f006-isolated20-20261017-v2/manifest.json"
BASE_SOURCE = "0b67f4d7b8131b5855421e456b108c89b6a407e8a680441fd6f1b11d555ab9ea"
BASE_SHA = "2ff6c2fe1bd32e44f383997d89338a5dc5efde34593e611869382e2e3cc1dbee"
POLICY0 = "experiments/f009-evaluation-policy-v2-prescore-20261017.json"
POLICY0_SHA = "de5a6aa743d2d663f98244322958bf4b8154908238384112a7badf2df1e2d42c"
POLICY1 = "experiments/f009-fold1-fixed12-joint-screen-prospective-20261017.json"
POLICY1_SHA = "9f42cd771db3f6bac92d275769636a451d3ef4dacd35e245a53d888c17ec2420"
HELPER_SHA = "cd2b5fd71355801b67f4f4596487d42728e4974f84b10eac15155ef35c6e5526"
F0_WRAPPER_SHA = "21c4e3089953d59f6da30fcb82fff3e27f5f3ecef1e20f4a96e22d561c5ba76b"
F1_EVALUATOR_SHA = "c8e244461dbf45be68fff13328f2a774df1fbf5ed69e8c109ee84b0f421e352c"
CPU_SCHEMA = "f009_fixed_E12_CPU_strict_reconstruction_v1"
PAIR_SCHEMA = "f009_fixed_E12_two_CLS_source14_local_pair_v1"
APPROVED_ACTUAL_CPU_PRODUCERS = frozenset()  # Consumer schema only; no real producer exists yet.
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def require(value, reason):
    if not value:
        raise ValueError(reason)


def positive(value):
    return type(value) in (float, int) and math.isfinite(value) and value > 0


def digest(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef") and value != "0" * 64


def pin(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= 128 * 1024**2, "Bounded regular member required")
    size, checksum = path.stat().st_size, hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            checksum.update(block)
    require(path.stat().st_size == size, "Member size changed")
    return {"bytes": size, "sha256": checksum.hexdigest()}


def read(path):
    require(Path(path).suffix == ".json" and Path(path).stat().st_size <= 16 * 1024**2, "Bounded JSON metadata only")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def bound(root, receipt):
    path = root / receipt["path"]
    require(path.resolve().is_relative_to(root) and pin(path) == {k: receipt[k] for k in ("bytes", "sha256")}, "Explicit metadata pin differs")
    return path


def expected_tensor_shapes():
    suffix = {}
    for block in (10, 11):
        for stem, shape in {"norm1.weight": [384], "norm1.bias": [384], "attn.qkv.weight": [1152, 384],
                            "attn.qkv.bias": [1152], "attn.proj.weight": [384, 384], "attn.proj.bias": [384],
                            "ls1.gamma": [384], "norm2.weight": [384], "norm2.bias": [384],
                            "mlp.fc1.weight": [1536, 384], "mlp.fc1.bias": [1536],
                            "mlp.fc2.weight": [384, 1536], "mlp.fc2.bias": [384], "ls2.gamma": [384]}.items():
            suffix["blocks." + str(block) + "." + stem] = shape
    head = {"project.0.weight": [384], "project.0.bias": [384], "project.1.weight": [192, 384],
            "project.1.bias": [192], "plane.weight": [4, 192], "geometry.weight": [192, 4],
            "geometry.bias": [192], "slice_score.weight": [12, 192], "slice_score.bias": [12],
            "series_query": [12, 192], "output_weight": [12, 192], "output_bias": [12]}
    return {"encoder_suffix": suffix, "head": head}


def validate_cpu(cpu, *, fold, run_pin, index_pin, checkpoint, prefix, fixture_only):
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


def preflight(root, spec, *, fixture_only=False):
    """Complete all metadata/CPU checks before touching checkpoint member paths."""
    root = Path(root).resolve()
    require(spec["schema_version"] == "f009_pair_stage_root_pins_v1"
            and set(spec) == {"schema_version", "result0", "guard0", "result1", "guard1", "folds"}
            and set(spec["folds"]) == {"0", "1"}, "Explicit new pair spec required")
    require(pin(root / POLICY0)["sha256"] == POLICY0_SHA and pin(root / POLICY1)["sha256"] == POLICY1_SHA,
            "Frozen twofold policies required")
    policy0, policy1 = read(root / POLICY0), read(root / POLICY1)
    documents = {key: read(bound(root, spec[key])) for key in ("result0", "guard0", "result1", "guard1")}
    r0, g0, r1, g1 = (documents[k] for k in ("result0", "guard0", "result1", "guard1"))
    p0, p1 = ({k: spec["result" + str(n)][k] for k in ("bytes", "sha256")} for n in (0, 1))
    require(g0["complete"] is True and g0["original_evaluate_calls"] == 1
            and g0["wrapper_pin"]["sha256"] == F0_WRAPPER_SHA and g0["original_result_pin"] == p0
            and r0["status"] == "F009_E12_WEAK_COMPARISON_ONLY_POLICY_V2"
            and r0["candidate_epoch"] == r0["control_epoch"] == 12
            and r0["policy_sha256"] == POLICY0_SHA
            and positive(r0["fixed10_delta"]) and r0["candidate_AMP_skipped_steps"] == 0
            and g0["fold1_gate_conditions_met"] is r0["fold1_gate_conditions_met"] is True,
            "Independent completed positive fold0 required")
    prior = g1["guards"]["fold0_positive_sha_bound"]
    require(g1["complete"] is True and g1["comparison_calls"] == 1
            and g1["evaluator_pin"]["sha256"] == F1_EVALUATOR_SHA and g1["result_pin"] == p1
            and prior["result"] == p0 and prior["guard_receipt"] == {k: spec["guard0"][k] for k in ("bytes", "sha256")}
            and prior["fixed10_delta"] == r0["fixed10_delta"] and prior["engineering_gate_passed"] is True
            and r1["status"] == "F009_FOLD1_FIXED12_PAIRED_AND_INDEPENDENT_TWO_FOLD_SCREEN_V1"
            and r1["fold"] == 1 and r1["candidate_epoch"] == r1["control_epoch"] == 12
            and r1["policy0_sha256"] == POLICY0_SHA and r1["policy1_sha256"] == POLICY1_SHA
            and positive(r1["fixed10_delta"]) and r1["joint_candidate_screen_passed"] is True
            and g1["joint_candidate_screen_passed"] is True and r1["fold1_machine_engineering_passed"] is True
            and g1["guards"]["machine_engineering_passed"] is True
            and r1["AMP_skipped_steps"] == {"control": 0, "candidate": 0},
            "Separate positive fold1 and strict positive AND engineering required")
    expected_inputs = {"src/rsna_knee/" + key: value for key, value in sources.CORE_SHA.items()}
    expected_inputs.update({policy0[key]["path"]: policy0[key]["sha256"] for key in ("config", "resource_gate", "input_cache")})
    for guard in (g0, g1):
        g = guard["guards"]
        require(g["complete"] is True and g["Gold_body_opened_or_hashed"] is False and g["model_deserialized"] is False
                and all(g["input_source_pins"][key]["sha256"] == value for key, value in expected_inputs.items()),
                "Original guarded source/config/cache lineage differs")
    require(r1["inputs"] == g1["guards"]["completed_E12_control_candidate"] and r1["fold0"] == prior,
            "Fold1 result/guard lineage differs")
    for arm, original in r0["inputs"].items():
        completed = g0["guards"]["completed_E12_control_candidate"][arm]
        require(original["run_sha256"] == completed["run"]["sha256"]
                and original["epoch_index_sha256"] == completed["index"]["sha256"]
                and original["prediction_sha256"] == completed["E12_prediction"]["sha256"]
                and original["checkpoint_sha256_reference_only_not_loaded"] == completed["E12_checkpoint_body_stream_hash"]["sha256"],
                "Fold0 result/guard E12 lineage differs")
    require(pin(root / BASE)["sha256"] == BASE_SHA, "Original P035 base manifest bytes changed")
    base = read(root / BASE)
    require(base["source_sha256"] == BASE_SOURCE and base["blend_policy"] == "success-subset"
            and base["native_executor"] == "isolated-uid", "Exact original P035 four-model reference required")
    require(sources._sha((root / "src/rsna_knee/f009_source_namespace.py").read_bytes()) == HELPER_SHA,
            "Fixed source14 namespace helper required")
    config = read(root / "configs/experiments/f009-dinov2-native-last2-late-cosine-study-gamma.json")
    require(pin(root / "configs/experiments/f009-dinov2-native-last2-late-cosine-study-gamma.json")["sha256"] == sources.CONFIG_SHA,
            "Fixed gamma config required")
    source_bodies = sources._sources(root / "src/rsna_knee", {**sources.CORE_SHA, **sources.SUPPORT_SHA})
    require(len(source_bodies) == 14 and pin(root / "scripts/train_native_dinov2_fast16.py")["sha256"] == sources.CLI_SHA,
            "Fixed source14/CLI required")
    plan = []
    for fold, guard in ((0, g0), (1, g1)):
        item = spec["folds"][str(fold)]
        require(set(item) == {"run", "index", "CPU", "expected_CPU_producer_sha256"}, "Explicit per-fold source/CPU receipts only")
        run_path, index_path, cpu_path = (bound(root, item[key]) for key in ("run", "index", "CPU"))
        expected_directory = root / (policy0["planned_run_directory"] if fold == 0 else policy1["new_run"])
        require(run_path.resolve() == expected_directory.resolve() / "run.json"
                and index_path.resolve() == expected_directory.resolve() / "epoch_checkpoint_index.json",
                "Only the two prospective F009 run directories permitted")
        run, index, cpu = (read(p) for p in (run_path, index_path, cpu_path))
        run_pin, index_pin = pin(run_path), pin(index_path)
        completed = guard["guards"]["completed_E12_control_candidate"]["candidate"]
        require(run_pin == completed["run"] and index_pin == completed["index"]
                and run["status"] == "complete" and run["fold"] == fold and run["seed"] == 20261002
                and run["config"] == config and run["source_sha256"] == sources.CORE_SHA
                and run["targets"] == TARGETS and run["gold_used_for_training_or_selection"] is False
                and run["gold_evaluated"] is False and [r["epoch"] for r in run["history"]] == list(range(1, 13))
                and all(r["amp_skipped_steps"] == 0 for r in run["history"]), "Actual per-fold run/gate metadata differs")
        row = index["checkpoints"][-1]
        cp = completed["E12_checkpoint_body_stream_hash"]
        require(index["fold"] == fold and index["source_sha256"] == sources.CORE_SHA and row["epoch"] == 12
                and row["checkpoint"] == "epoch_012.pt" and row["predictions_csv"] == "epoch_012_weak.csv"
                and cp == {"bytes": row["checkpoint_bytes"], "sha256": row["checkpoint_sha256"]}
                and type(row["weak_bce"]) in (float, int) and math.isfinite(row["weak_bce"])
                and row["weak_bce"] == run["history"][-1]["weak_bce"]
                and completed["E12_prediction"]["sha256"] == row["prediction_sha256"] == run["history"][-1]["prediction_sha256"],
                "Exact E12 index/checkpoint/CSV guard binding differs")
        require(cpu["producer_sha256"] == item["expected_CPU_producer_sha256"], "Explicit CPU producer identity differs")
        prefix = base["native"]["model_fingerprints"]["REG:0"]["frozen_prefix_sha256"]
        validate_cpu(cpu, fold=fold, run_pin=run_pin, index_pin=index_pin, checkpoint=cp, prefix=prefix, fixture_only=fixture_only)
        require(cpu["checkpoint_schema"] == "native392_generic_dino_suffix_head_fast16_v1"
                and cpu["weak_bce"] == row["weak_bce"] and cpu["encoder_provenance"] == run["encoder_provenance"],
                "CPU checkpoint schema/BCE/generic provenance differs")
        plan.append({"fold": fold, "checkpoint_path": run_path.parent / "epoch_012.pt", "checkpoint_pin": cp,
                     "run_path": run_path, "index_path": index_path, "CPU_path": cpu_path, "CPU": cpu,
                     "metadata_pins": {"run": run_pin, "index": index_pin, "CPU": pin(cpu_path)},
                     "gate_paths": {key: root / spec[key]["path"] for key in ("result0", "guard0", "result1", "guard1")},
                     "gate_pins": {key: {k: spec[key][k] for k in ("bytes", "sha256")} for key in ("result0", "guard0", "result1", "guard1")}})
    require(plan[0]["checkpoint_pin"]["sha256"] != plan[1]["checkpoint_pin"]["sha256"], "Two separate fold checkpoints required")
    return base, plan


def copy_exact(source, destination, expected):
    require(pin(source) == expected, "Approved checkpoint/copy bytes changed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open("rb") as src, destination.open("xb") as dst:
        for block in iter(lambda: src.read(1024**2), b""):
            dst.write(block)
    require(pin(destination) == expected, "Copied checkpoint/copy SHA differs")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    require(not path.exists(), "Never replace an existing sidecar")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _seal(root, spec, output, *, fixture_only=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(output.is_relative_to(root) and not output.exists(), "New workspace output directory required")
    base, plan = preflight(root, spec, fixture_only=fixture_only)
    output.mkdir(parents=True, exist_ok=False)
    try:
        folders = {}
        for item in plan:
            fold = item["fold"]
            folder = output / ("fold" + str(fold))
            sources.stage_f009_sources(root / "src/rsna_knee", root / "scripts/train_native_dinov2_fast16.py",
                root / "configs/experiments/f009-dinov2-native-last2-late-cosine-study-gamma.json", folder / "source14")
            copy_exact(item["checkpoint_path"], folder / "trained/epoch_012.pt", item["checkpoint_pin"])
            for field in ("run", "index", "CPU"):
                path = item[field + "_path"]
                copy_exact(path, folder / "audit" / (field + ".json"), item["metadata_pins"][field])
            for field, path in item["gate_paths"].items():
                copy_exact(path, folder / "audit" / (field + ".json"), item["gate_pins"][field])
            for field, relative, expected_sha in (("policy0.json", POLICY0, POLICY0_SHA), ("policy1.json", POLICY1, POLICY1_SHA),
                ("source_namespace_helper.py", "src/rsna_knee/f009_source_namespace.py", HELPER_SHA)):
                path = root / relative
                observed = pin(path)
                require(observed["sha256"] == expected_sha, "Source/policy changed after preflight")
                copy_exact(path, folder / "audit" / field, observed)
            manifest = {"schema_version": "f009_fixed_E12_single_CLS_source14_local_bundle_v1", "fold": fold, "epoch": 12,
                "checkpoint_pin": item["checkpoint_pin"], "CPU_receipt": pin(item["CPU_path"]),
                "source_manifest_pin": pin(folder / "source14/source-manifest.json"),
                "source_sha256": sources.CORE_SHA, "support_sha256": sources.SUPPORT_SHA,
                "assets": {str(path.relative_to(folder)).replace("\\", "/"): pin(path)
                           for path in sorted(folder.rglob("*")) if path.is_file()},
                "artificial_fixture_only": fixture_only, "actual_strict_CPU_accepted": not fixture_only,
                "runtime_transport_or_Submit_qualified": False}
            atomic_json(folder / "bundle.json", manifest)
            folders[str(fold)] = pin(folder / "bundle.json")
        retained = {key: {"transport": base["native"]["transport_jobs"][key],
                          "fingerprint": base["native"]["model_fingerprints"][key]} for key in ("REG:0", "REG:1")}
        pair = {"schema_version": PAIR_SCHEMA, "local_pair_complete": True, "CLS_E12_bundles": folders,
                "fixed_REG_references_only": retained, "REG_checkpoint_body_read_or_copied": False,
                "logical_models": 4, "old_CLS_included": False, "artificial_fixture_only": fixture_only,
                "actual_CLS_stage_constructed": not fixture_only, "four_model_transport_runtime_qualified": False,
                "source_input_spec": spec, "adoption_Save_Submit_authorized": False}
        atomic_json(output / "pair-sidecar.json", pair)
        atomic_json(output / "BUILD_COMPLETE.json", {"complete": True, "pair_sidecar": pin(output / "pair-sidecar.json"),
                    "artificial_fixture_only": fixture_only, "four_model_transport_runtime_qualified": False})
        return pair
    except BaseException as exc:
        atomic_json(output / "BUILD_FAILED.json", {"complete": False, "error_type": type(exc).__name__,
                    "artificial_fixture_only": fixture_only, "partial_output_preserved": True})
        raise


def seal_f009_pair(spec_path, expected_spec_sha256, output):
    """Future production entry. Actual producer registry is intentionally empty."""
    require(pin(spec_path)["sha256"] == expected_spec_sha256, "Root expected spec SHA required")
    return _seal(ROOT, read(spec_path), output, fixture_only=False)
