"""Assemble a local F009 root release from qualified metadata, never model bodies.

An explicit root call and this assembler's SHA are required. The release file
is the publication boundary: metadata copies use stable final paths, while a
private candidate is validated by the frozen context/real CPU metadata consumer.
No archive, checkpoint, MRI, GPU, network or upload operation is implemented.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path

from . import f009_kaggle_context as context
from . import f009_transport as transport

ROOT = Path(__file__).resolve().parents[2]
SPEC_SCHEMA = "f009_root_release_assembly_inputs_v1"
RECEIPT_SCHEMA = "f009_metadata_root_release_assembly_v1"
CONTEXT_SHA = "37a058c38f3002a9857a26c88601fb73fc9df73710af6d0c1b4248eef06fea97"
TRANSPORT_SHA = "8c613c2c2d24fc6b1d1c6222482c581b03ac5027c15b10f5e7116e7f7056af47"
BUILDER_SHA = "c9ac8bf2ad667c0f733ae19a96686c66c916bb74051352d647211b7a8afa651e"
CHECKPOINT_MEMBER = "trained/epoch_012.pt"
MAX_FILE = 16 * 1024**2
MAX_TOTAL = 64 * 1024**2


def require(value, message):
    if not value:
        raise ValueError(message)


def raw_pin(body):
    return {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def _path(path, suffixes):
    path = Path(path).absolute()
    require(path.suffix in suffixes, "Only declared metadata/source suffixes may be opened")
    current = path
    while current != current.parent:
        require(not current.is_symlink(), "Symlink metadata/source ancestor prohibited")
        current = current.parent
    require(path.is_file() and 0 < path.stat().st_size <= MAX_FILE, "Bounded regular metadata/source required")
    return path.resolve(strict=True)


def _body(path, *, suffixes=frozenset({".json"}), expected=None):
    path = _path(path, suffixes)
    body = path.read_bytes()
    require(0 < len(body) <= MAX_FILE and (expected is None or raw_pin(body) == expected), "Metadata/source bytes or SHA differs")
    return body


def _decode(body):
    return context._decode(body)


def _reference(ref):
    require(isinstance(ref, dict) and set(ref) == {"path", "bytes", "sha256"}, "Exact path/bytes/SHA reference required")
    context._pin_spec({k: ref[k] for k in ("bytes", "sha256")})
    return _body(ref["path"], expected={key: ref[key] for key in ("bytes", "sha256")})


def _implementation_pins(expected_assembler_sha256):
    require(isinstance(expected_assembler_sha256, str) and re.fullmatch("[0-9a-f]{64}", expected_assembler_sha256),
            "Explicit assembler source SHA required")
    paths = {"assembler": Path(__file__), "context": Path(context.__file__), "transport": Path(transport.__file__),
             "frozen_builder": ROOT / "scripts/prepare_f009_p033_native20.py"}
    expected = {"assembler": expected_assembler_sha256, "context": CONTEXT_SHA,
                "transport": TRANSPORT_SHA, "frozen_builder": BUILDER_SHA}
    result = {}
    for key, path in paths.items():
        body = _body(path, suffixes={".py"})
        require(raw_pin(body)["sha256"] == expected[key], "Exact frozen assembler/dependency source required")
        result[key] = {"path": str(path.resolve()), **raw_pin(body)}
    return result


def _frozen_control():
    """Import only SHA-checked stdlib tooling; inspect old carriers as literals."""
    path = ROOT / "scripts/prepare_f009_p033_native20.py"
    require(raw_pin(_body(path, suffixes={".py"}))["sha256"] == BUILDER_SHA, "Frozen builder source differs")
    spec = importlib.util.spec_from_file_location("_f009_release_frozen_literal_reader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.inspect_control(), module.runtime_paths(), module.PARENT_DATASETS


def _pin_pair(ref):
    return {key: ref[key] for key in ("bytes", "sha256")}


def _collect(spec, control, required_paths, parent_datasets):
    require(set(spec) == {"schema_version", "pair_directory", "pair", "complete", "transports", "CLS_datasets", "source_inventory", "runtime_paths"}
            and spec["schema_version"] == SPEC_SCHEMA, "Exact release assembly input schema required")
    source_inventory = _decode(_reference(spec["source_inventory"]))
    runtime_paths = _decode(_reference(spec["runtime_paths"]))
    context._inventory_spec(source_inventory)
    context._runtime_spec(runtime_paths, source_inventory)
    require(runtime_paths == required_paths, "Frozen serving runtime paths differ")
    require(set(spec["CLS_datasets"]) == set(spec["transports"]) == {"0", "1"}, "Both explicit CLS folds required")
    refs = spec["CLS_datasets"]
    require(all(isinstance(ref, str) and re.fullmatch(r"rsraki/[a-z0-9][a-z0-9-]*/1", ref) for ref in refs.values())
            and refs["0"] != refs["1"], "Two distinct own private CLS dataset refs V1 required")
    pair_root = Path(spec["pair_directory"]).resolve(strict=True)
    require(pair_root.is_dir() and not Path(spec["pair_directory"]).is_symlink(), "Regular explicit pair directory required")
    pair_raw, complete_raw = _reference(spec["pair"]), _reference(spec["complete"])
    require(_path(spec["pair"]["path"], {".json"}) == pair_root / "pair-sidecar.json"
            and _path(spec["complete"]["path"], {".json"}) == pair_root / "BUILD_COMPLETE.json", "Pair authority paths differ")
    pair, complete = _decode(pair_raw), _decode(complete_raw)
    require(pair["schema_version"] == transport.PAIR_SCHEMA and pair["local_pair_complete"] is True
            and pair["artificial_fixture_only"] is False and pair["actual_CLS_stage_constructed"] is True
            and pair["four_model_transport_runtime_qualified"] is False and pair["old_v1_joint_screen"] == "HOLD"
            and isinstance(pair["root_spec_pin"], dict) and set(pair["CLS_E12_bundles"]) == {"0", "1"}, "Actual amended sealer3 pair required")
    require(complete["complete"] is True and complete["artificial_fixture_only"] is False
            and complete["pair_sidecar"] == _pin_pair(spec["pair"]), "Actual completed pair pin required")
    files = {**control["reg_metadata"], "metadata/conditional-plan.json": control["plan_raw"],
             "metadata/CLS/pair-sidecar.json": pair_raw, "metadata/CLS/BUILD_COMPLETE.json": complete_raw}
    source_refs = [spec["pair"], spec["complete"], spec["source_inventory"], spec["runtime_paths"]]
    rows, jobs, completions = {}, {}, []
    for fold in (0, 1):
        key = str(fold)
        folder = pair_root / ("fold" + key)
        bundle_raw = _body(folder / "bundle.json", expected=pair["CLS_E12_bundles"][key])
        bundle = _decode(bundle_raw)
        require(bundle["schema_version"] == transport.BUNDLE_SCHEMA and type(bundle["fold"]) is int and bundle["fold"] == fold
                and bundle["artificial_fixture_only"] is False and bundle["actual_strict_CPU_accepted"] is True,
                "Actual strict CPU bundle required before metadata copies")
        assets = bundle["assets"]
        require(set(assets) == transport.asset_names(pair["root_spec_pin"]), "Exact source14/audit/checkpoint inventory required")
        prefix = "metadata/CLS/fold" + key
        files[prefix + "/bundle/bundle.json"] = bundle_raw
        source_refs.append({"path": str(folder / "bundle.json"), **raw_pin(bundle_raw)})
        for name, expected in assets.items():
            transport.member(name)
            if name == CHECKPOINT_MEMBER:
                require(isinstance(expected, dict) and set(expected) == {"bytes", "sha256"}
                        and type(expected["bytes"]) is int and 0 < expected["bytes"] <= transport.MAX_MEMBER
                        and isinstance(expected["sha256"], str) and re.fullmatch("[0-9a-f]{64}", expected["sha256"]),
                        "Bounded checkpoint expected pin metadata required")
                continue  # Expected pin only: never stat/open/hash the checkpoint.
            context._pin_spec(expected)
            require(sum(map(len, files.values())) + expected["bytes"] <= MAX_TOTAL, "Metadata copy total exceeds cap")
            require(Path(name).suffix in {".py", ".json"}, "Nonmetadata bundle asset prohibited")
            body = _body(folder / name, suffixes={".py", ".json"}, expected=expected)
            files[prefix + "/bundle/" + name] = body
            source_refs.append({"path": str(folder / name), **expected})
        incoming = spec["transports"][key]
        require(set(incoming) == {"manifest", "complete"}, "Explicit transport manifest/completion pins required")
        manifest_raw, completion_raw = _reference(incoming["manifest"]), _reference(incoming["complete"])
        manifest, completion = _decode(manifest_raw), _decode(completion_raw)
        require(manifest["schema_version"] == transport.SCHEMA and type(manifest["fold"]) is int and manifest["fold"] == fold
                and manifest["actual_transport"] is True and manifest["artificial_fixture_only"] is False
                and manifest["transport_source_sha256"] == TRANSPORT_SHA and manifest["runtime_or_Public_qualified"] is False
                and manifest["pair"] == _pin_pair(spec["pair"]) and manifest["complete"] == _pin_pair(spec["complete"])
                and manifest["plan"] == raw_pin(control["plan_raw"]) and manifest["bundle"] == raw_pin(bundle_raw)
                and manifest["members"] == {**assets, "bundle.json": raw_pin(bundle_raw)}, "Actual transport crosspins differ")
        require(completion["complete"] is True and completion["actual_transport"] is True
                and completion["artificial_fixture_only"] is False and completion["pair"] == _pin_pair(spec["pair"])
                and set(completion["fold_manifests"]) == {"0", "1"}
                and completion["fold_manifests"][key] == raw_pin(manifest_raw), "Whole-pair transport completion differs")
        files[prefix + "/transport-manifest.json"] = manifest_raw
        files[prefix + "/TRANSPORT_COMPLETE.json"] = completion_raw
        rows[key] = {"bundle_root": prefix + "/bundle",
                     "transport_manifest": {"path": prefix + "/transport-manifest.json", **raw_pin(manifest_raw)},
                     "transport_complete": {"path": prefix + "/TRANSPORT_COMPLETE.json", **raw_pin(completion_raw)}}
        jobs["CLS:" + key] = {"input_root": "/kaggle/input/datasets/" + refs[key].rsplit("/", 1)[0],
            "dataset_ref": refs[key], "manifest_sha": raw_pin(manifest_raw)["sha256"],
            "archive_sha": manifest["archive"]["sha256"], "bundle_sha": manifest["bundle"]["sha256"]}
        completions.append(completion)
        source_refs.extend(incoming.values())
    require(completions[0] == completions[1], "Both transport completion copies must agree")
    plan = control["plan"]
    for fold in (0, 1):
        key = "REG:" + str(fold)
        jobs[key] = dict(plan["intended_intervention"]["old_REG_transport_refs"][key])
        jobs[key].update(root_permission=runtime_paths["REG_metadata"][key]["permission"],
                         peer_review=runtime_paths["REG_metadata"][key]["peer"])
    datasets = [*parent_datasets, refs["0"], refs["1"], jobs["REG:0"]["dataset_ref"], jobs["REG:1"]["dataset_ref"], "rsraki/rsna-dicom-decoders-py313-v1/1"]
    require(len(datasets) == len(set(datasets)) == len({s.rsplit("/", 1)[0] for s in datasets}) == 7, "Seven distinct fixed dataset mounts required")
    claims = {"conditional_plan": {"path": "metadata/conditional-plan.json", **raw_pin(control["plan_raw"])},
        "CLS_pair": {"path": "metadata/CLS/pair-sidecar.json", **raw_pin(pair_raw)},
        "CLS_complete": {"path": "metadata/CLS/BUILD_COMPLETE.json", **raw_pin(complete_raw)},
        "CLS_metadata": rows, "parent": plan["frozen_control"], "integration": plan["intended_intervention"]["integration_contract"],
        "inputs": {"datasets": datasets, "kernels": ["rsraki/p029-r6v1-stage-20261015-b2/1"],
                   "competitions": ["rsna-knee-abnormality-detection"], "total": 9},
        "decoder_root": context.DECODER_ROOT, "transport_jobs": jobs}
    require(set(claims) == context.CLAIMS and 1 <= len(files) <= 256 and sum(map(len, files.values())) <= MAX_TOTAL,
            "Exact bounded release context/metadata required")
    return source_inventory, runtime_paths, files, claims, source_refs


def _atomic(path, body):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    require(not path.exists() and not temporary.exists() and not path.is_symlink() and not temporary.is_symlink(), "Fresh metadata publication required")
    temporary.write_bytes(body)
    os.replace(temporary, path)


def assemble_f009_release(spec_path=None, expected_spec_sha256=None, output=None, *, expected_assembler_sha256=None, execute=False):
    """Actual metadata only. Default returns without opening any input body."""
    if not execute:
        return {"status": "SOURCE_PREPARATION_ONLY", "actual_release_assembled": False, "input_body_reads": 0}
    implementation = _implementation_pins(expected_assembler_sha256)
    require(spec_path is not None and output is not None and isinstance(expected_spec_sha256, str)
            and re.fullmatch("[0-9a-f]{64}", expected_spec_sha256), "Explicit spec path/SHA and fresh output required")
    spec_raw = _body(spec_path)
    require(raw_pin(spec_raw)["sha256"] == expected_spec_sha256, "Root assembly spec SHA differs")
    spec = _decode(spec_raw)
    control, required_paths, parent_datasets = _frozen_control()
    inventory, paths, files, claims, sources = _collect(spec, control, required_paths, parent_datasets)
    requested_output = Path(output).absolute()
    current = requested_output
    while current != current.parent:
        require(not current.is_symlink(), "Output symlink ancestor prohibited")
        current = current.parent
    parent = requested_output.parent.resolve(strict=True)
    output = parent / requested_output.name
    require(requested_output == output and not output.exists() and not output.is_symlink(), "Fresh resolved same-parent local output required")
    # Stable final provenance paths are required by the frozen context assembler.
    # Metadata may remain after failure; no valid release/context marker remains.
    output.mkdir(exist_ok=False)
    candidate = output / ".release-candidate.json"
    try:
        for name, body in files.items():
            require(transport.member(name).startswith("metadata/") and Path(name).suffix in {".py", ".json"}, "Metadata-only copy target required")
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
        release = {"schema_version": context.RELEASE_SCHEMA, "actual_qualified_release": True,
            "artificial_fixture_only": False, "root_authorized_CPU_permission_derivation": True,
            "context": claims, "metadata_files": {name: {"path": str(output / name), **raw_pin(body)} for name, body in files.items()},
            "authority_scope": "Actual CPU/sealer/transport evidence metadata only; private Input existence and runtime qualification are not verified.",
            "assembler_sha256": implementation["assembler"]["sha256"], "assembly_input_spec_pin": raw_pin(spec_raw),
            "private_Inputs_exist_or_mounted_verified": False, "checkpoint_or_archive_body_verified_by_this_assembler": False,
            "GPU_Runtime_Public_Submit_qualified": False}
        release_raw = json_bytes(release)
        candidate.write_bytes(release_raw)
        prepared, metadata = context.prepare_context(candidate, raw_pin(release_raw)["sha256"], source_inventory=inventory, runtime_paths=paths)
        require(prepared["actual_entry_enabled"] is True and metadata["metadata/root-release.json"] == release_raw,
                "Frozen actual metadata validation required")
        for name, body in metadata.items():
            target = output / transport.member(name)
            require(Path(name).suffix in {".py", ".json"} and len(body) <= MAX_FILE, "Frozen context returned nonmetadata")
            if target.exists():
                require(_body(target, suffixes={".py", ".json"}) == body, "Context cannot replace copied metadata")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(body)
        context_raw = json_bytes(prepared)
        _atomic(output / "f009-context.json", context_raw)
        for ref in sources:
            require(raw_pin(_body(ref["path"], suffixes={".py", ".json"})) == _pin_pair(ref), "Source metadata changed before release publication")
        for name, body in files.items():
            require(_body(output / name, suffixes={".py", ".json"}) == body, "Copied metadata changed before publication")
        require(_body(spec_path) == spec_raw and _implementation_pins(expected_assembler_sha256) == implementation,
                "Spec or assembler/dependency source changed before publication")
        receipt = {"schema_version": RECEIPT_SCHEMA, "complete": True, "actual_release_assembled": True,
            "metadata_only": True, "release_pin": raw_pin(release_raw), "context_pin": raw_pin(context_raw),
            "spec_pin": {"path": str(Path(spec_path).resolve()), **raw_pin(spec_raw)}, "implementation_pins": implementation,
            "source_metadata_pins": sources, "metadata_copies": {name: raw_pin(body) for name, body in files.items()},
            "expected_private_CLS_refs": spec["CLS_datasets"], "private_Input_existence_verified": False,
            "checkpoint_archive_MRI_GPU_API_reads": 0, "runtime_Public_Submit_qualified": False,
            "publication": "Stable final metadata paths; release.json file gate, completion last; not atomic directory publication."}
        _atomic(output / "assembly-receipt.json", json_bytes(receipt))
        os.replace(candidate, output / "release.json")
        _atomic(output / "ASSEMBLY_COMPLETE.json", json_bytes({"complete": True, "release": raw_pin(release_raw),
            "receipt": raw_pin(json_bytes(receipt)), "metadata_only": True, "runtime_Public_Submit_qualified": False}))
        return receipt
    except BaseException as error:
        gates = (".release-candidate.json", "release.json", "ASSEMBLY_COMPLETE.json", "f009-context.json", "metadata/root-release.json", "assembly-receipt.json")
        for name in (*gates, *(name + ".partial" for name in gates)):
            target = output / name
            if target.exists():
                require(not target.is_symlink() and target.resolve().is_relative_to(output), "Failure marker withdrawal leaves owned output")
                target.unlink()
        _atomic(output / "ASSEMBLY_FAILED.json", json_bytes({"complete": False, "actual_qualified_release": False,
            "error_type": type(error).__name__, "metadata_partial_output_preserved": True}))
        raise
