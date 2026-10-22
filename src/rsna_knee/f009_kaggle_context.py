"""Metadata-only gate for the new F009 Kaggle execution path.

This module never opens a checkpoint, archive, MRI or network connection.
The existing frozen transport consumer independently validates the actual
CPU proofs copied as metadata; source preparation cannot issue a release.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from . import f009_transport as transport

SCHEMA = "f009_P033_native20_Kaggle_context_v1"
RELEASE_SCHEMA = "f009_P039_qualified_root_release_v1"
MAX_FILE = 16 * 1024**2
MAX_METADATA = 64 * 1024**2
FIXED_SUPPORT = {
    "f009_transport.py": "8c613c2c2d24fc6b1d1c6222482c581b03ac5027c15b10f5e7116e7f7056af47",
    "f009_mixed_runtime.py": "73695814776300ed7c2edd8f2ba535f99b2af9afe4c335346ff06ac1530f6a90",
    "f009_source_namespace.py": "cd2b5fd71355801b67f4f4596487d42728e4974f84b10eac15155ef35c6e5526",
    "isolated_uid_executor.py": "685ec00ccc595ccb274817f960c28bdd3bc5926657f52e222c86863e5806d8f8",
    "operational_uid_recovery.py": "40337f6bc2eaa5d75ab1157bcc2a9dec76bd0022445df2c012b6d89c1d45a814",
    "contracts.py": "aa246fd1a49ecbe88f1795f49904ef9675f119c935e99b9ffd5eee8220cbc71b",
}
SOURCE_ROLES = {"common", "regional", "mixed", "serving", "REG_transport", "support"}
CLAIMS = {"conditional_plan", "CLS_pair", "CLS_complete", "CLS_metadata", "parent", "integration",
          "inputs", "decoder_root", "transport_jobs"}
FIXED_INPUTS = {
    "heliosli/rsna-orthofoundation-fold0-epoch7-20261004/1",
    "nartaa/rsna-knee-publication-swa-weights-20261007/1",
    "rsraki/rsna-dicom-decoders-py313-v1/1",
}
DECODER_ROOT = "/kaggle/input/datasets/rsraki/rsna-dicom-decoders-py313-v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def raw_pin(body):
    return {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def _decode(body):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(body, object_pairs_hook=pairs,
                       parse_constant=lambda _: require(False, "Nonfinite JSON"))
    require(isinstance(value, dict), "JSON object required")
    return value


def _pin_spec(ref, *, path=False):
    require(isinstance(ref, dict) and set(ref) == ({"bytes", "sha256", "path"} if path else {"bytes", "sha256"}),
            "Exact bytes/SHA pin required")
    require(type(ref["bytes"]) is int and 0 <= ref["bytes"] <= MAX_FILE
            and isinstance(ref["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]), "Invalid bounded pin")
    if path:
        transport.member(ref["path"])
    return {key: ref[key] for key in ("bytes", "sha256")}


def _path(root, relative):
    transport.member(relative)
    root = Path(root).resolve(strict=True)
    candidate = root / relative
    current = candidate
    while current != root:
        require(not current.is_symlink() and current != current.parent, "Symlink ancestor prohibited")
        current = current.parent
    require(candidate.resolve().is_relative_to(root), "Runtime path escaped root")
    return candidate


def _body(path, expected=None):
    path = Path(path)
    require(path.suffix in {".py", ".json", ".ipynb"}, "Checkpoint/archive/data bodies prohibited")
    require(not path.is_symlink() and path.is_file() and path.stat().st_size <= MAX_FILE, "Bounded metadata/source file required")
    body = path.read_bytes()
    require(len(body) <= MAX_FILE, "File grew beyond cap")
    if expected is not None:
        require(raw_pin(body) == expected, "Metadata/source bytes or SHA differs")
    return body


def _reference(root, ref):
    expected = _pin_spec(ref, path=True)
    require(Path(ref["path"]).suffix == ".json", "Structured metadata must be JSON")
    return _decode(_body(_path(root, ref["path"]), expected))


def _inventory_spec(value):
    require(isinstance(value, dict) and 1 <= len(value) <= 512, "Bounded positive source inventory required")
    require(len(value) == len({key.casefold() for key in value}), "Source case collision")
    for name, expected in value.items():
        transport.member(name)
        _pin_spec(expected)
        require(Path(name).suffix in {".py", ".json", ".ipynb"}, "Only source files belong in source inventory")
        require(not name.startswith("metadata/") and name not in {"source-inventory.json", "f009-context.json", "f009-context.sha256"},
                "Context/metadata cannot be included in its own source inventory")


def _runtime_spec(paths, inventory):
    require(isinstance(paths, dict) and set(paths) == {"source_roots", "router", "parent", "child", "extension", "REG_metadata"},
            "Exact runtime path roles required")
    require(set(paths["source_roots"]) == SOURCE_ROLES, "Exact six source directory roles required")
    for name in (*paths["source_roots"].values(), *(paths[key] for key in ("router", "parent", "child", "extension"))):
        transport.member(name)
    for key in ("router", "parent", "child", "extension"):
        require(paths[key] in inventory, "Runtime entry point missing from source inventory")
    for role in SOURCE_ROLES:
        require(any(name.startswith(paths["source_roots"][role] + "/") for name in inventory), "Empty source directory role")
    support = paths["source_roots"]["support"]
    for name in (*FIXED_SUPPORT, "f009_kaggle_context.py", "f009_kaggle_driver.py", "__init__.py"):
        require(support + "/" + name in inventory, "Incomplete dedicated support package")
    for name, digest in FIXED_SUPPORT.items():
        require(inventory[support + "/" + name]["sha256"] == digest, "Frozen support source differs")
    require(set(paths["REG_metadata"]) == {"REG:0", "REG:1"}, "Both REG metadata roles required")
    for row in paths["REG_metadata"].values():
        require(set(row) == {"permission", "peer"}, "REG permission/peer roles required")
        for name in row.values():
            require(transport.member(name).startswith("metadata/REG/"), "REG metadata must be copied locally")


def context_template(*, source_inventory, runtime_paths):
    """An explicitly unready template; this does not read qualification inputs."""
    _inventory_spec(source_inventory)
    _runtime_spec(runtime_paths, source_inventory)
    body = json_bytes(source_inventory)
    context = {"schema_version": SCHEMA, "actual_entry_enabled": False, "artificial_fixture_only": False,
               "status": "SOURCE_PREPARATION_ONLY_NOT_MODEL_RELEASE", "runtime_paths": runtime_paths,
               "source_inventory": {"path": "source-inventory.json", **raw_pin(body)}}
    return context, {"source-inventory.json": body}


def _release(root, context):
    release = _reference(root, context["root_release"])
    require(release["schema_version"] == RELEASE_SCHEMA and release["actual_qualified_release"] is True
            and release["artificial_fixture_only"] is False
            and release["root_authorized_CPU_permission_derivation"] is True, "Actual new root release required")
    claims = release["context"]
    require(set(claims) == CLAIMS and all(context.get(key) == value for key, value in claims.items()), "Root release/context claims differ")
    files = release["metadata_files"]
    require(isinstance(files, dict) and 1 <= len(files) <= 256
            and len(files) == len({name.casefold() for name in files}), "Bounded positive metadata inventory required")
    require(sum(ref["bytes"] for ref in files.values()) <= MAX_METADATA, "Metadata total exceeds cap")
    for name, ref in files.items():
        require(transport.member(name).startswith("metadata/") and name != context["root_release"]["path"]
                and Path(name).suffix in {".json", ".py"}, "Only qualification metadata/source copies allowed")
        require(isinstance(ref, dict) and set(ref) == {"path", "bytes", "sha256"}, "Root metadata provenance pin required")
        # The original machine's path is provenance here; only its copied target is opened.
        _body(_path(root, name), _pin_spec({key: ref[key] for key in ("bytes", "sha256")}))
    return release


def _qualified_metadata(root, context):
    """No checkpoint/archive bodies: reuse the frozen real CPU metadata consumer."""
    plan = _reference(root, context["conditional_plan"])
    require(context["conditional_plan"]["sha256"] == transport.PLAN_SHA, "Conditional plan SHA differs")
    require(context["parent"] == plan["frozen_control"]
            and context["integration"] == plan["intended_intervention"]["integration_contract"], "P039 parent or numeric contract changed")
    pair = _reference(root, context["CLS_pair"])
    complete = _reference(root, context["CLS_complete"])
    pair_pin = _pin_spec(context["CLS_pair"], path=True)
    require(complete["complete"] is True and complete["pair_sidecar"] == pair_pin
            and complete["artificial_fixture_only"] is False, "Actual complete pair required")
    require(set(context["CLS_metadata"]) == {"0", "1"}, "Two CLS metadata folders required")
    manifests = {}
    for fold in (0, 1):
        row = context["CLS_metadata"][str(fold)]
        require(set(row) == {"bundle_root", "transport_manifest", "transport_complete"}, "Exact CLS metadata roles required")
        folder = _path(root, row["bundle_root"])
        descriptor = transport._consume(folder, expected_bundle_pin=pair["CLS_E12_bundles"][str(fold)],
            pair=pair, conditional_plan=plan, fold=fold, fixture_only=False, metadata_only=True)
        require(descriptor["checkpoint_body_verified"] is False and descriptor["actual_transport"] is False,
                "Metadata validation cannot claim checkpoint transport")
        t = _reference(root, row["transport_manifest"])
        tc = _reference(root, row["transport_complete"])
        require(t["schema_version"] == transport.SCHEMA and type(t["fold"]) is int and t["fold"] == fold
                and t["actual_transport"] is True and t["artificial_fixture_only"] is False
                and t["transport_source_sha256"] == FIXED_SUPPORT["f009_transport.py"]
                and t["runtime_or_Public_qualified"] is False, "Actual pinned new transport metadata required")
        require(t["pair"] == pair_pin and t["complete"] == _pin_spec(context["CLS_complete"], path=True)
                and t["plan"] == _pin_spec(context["conditional_plan"], path=True)
                and t["bundle"] == pair["CLS_E12_bundles"][str(fold)], "Transport sidecar crosspins differ")
        bundle = _decode(_body(folder / "bundle.json", t["bundle"]))
        require(t["members"] == {**bundle["assets"], "bundle.json": t["bundle"]}, "Exact transport members differ")
        require(tc["complete"] is True and tc["actual_transport"] is True and tc["artificial_fixture_only"] is False
                and tc["pair"] == pair_pin and set(tc["fold_manifests"]) == {"0", "1"}
                and tc["fold_manifests"][str(fold)] == _pin_spec(row["transport_manifest"], path=True), "Twofold transport completion differs")
        manifests[str(fold)] = (t, tc)
    require(manifests["0"][1] == manifests["1"][1], "Fold transport completions must agree")
    jobs = context["transport_jobs"]
    require(set(jobs) == {"CLS:0", "CLS:1", "REG:0", "REG:1"}, "Exactly four model roles required")
    old = plan["intended_intervention"]["old_REG_transport_refs"]
    for fold in (0, 1):
        job, t = jobs["CLS:" + str(fold)], manifests[str(fold)][0]
        require(set(job) == {"input_root", "dataset_ref", "manifest_sha", "archive_sha", "bundle_sha"}
                and job["manifest_sha"] == context["CLS_metadata"][str(fold)]["transport_manifest"]["sha256"]
                and job["archive_sha"] == t["archive"]["sha256"] and job["bundle_sha"] == t["bundle"]["sha256"], "CLS Input/transport pins differ")
        require(re.fullmatch(r"rsraki/[a-z0-9][a-z0-9-]*/[1-9][0-9]*", job["dataset_ref"])
                and job["input_root"] == "/kaggle/input/datasets/" + job["dataset_ref"].rsplit("/", 1)[0], "Versioned private CLS Input binding required")
        key = "REG:" + str(fold)
        expected = dict(old[key])
        paths = context["runtime_paths"]["REG_metadata"][key]
        expected.update(root_permission=paths["permission"], peer_review=paths["peer"])
        require(jobs[key] == expected, "Unchanged REG transport metadata required")
        for pathkey, shakey in (("root_permission", "root_permission_sha"), ("peer_review", "peer_sha")):
            require(raw_pin(_body(_path(root, expected[pathkey])))["sha256"] == expected[shakey], "REG authority copy SHA differs")
    inputs = context["inputs"]
    expected_datasets = FIXED_INPUTS | {jobs[key]["dataset_ref"] for key in jobs}
    require(set(inputs) == {"datasets", "kernels", "competitions", "total"} and len(expected_datasets) == 7
            and len({ref.rsplit("/", 1)[0] for ref in expected_datasets}) == 7
            and len({job["input_root"] for job in jobs.values()}) == 4
            and len(inputs["datasets"]) == 7 and set(inputs["datasets"]) == expected_datasets
            and inputs["kernels"] == ["rsraki/p029-r6v1-stage-20261015-b2/1"]
            and inputs["competitions"] == ["rsna-knee-abnormality-detection"] and type(inputs["total"]) is int and inputs["total"] == 9,
            "Exact seven datasets/one kernel/one competition required")
    require(context["decoder_root"] == DECODER_ROOT, "Frozen decoder Input mount required")
    return plan, pair


def validate_context(context_path, expected_sha256, *, runtime_root):
    """Validate all new metadata and source bodies before any model transport."""
    root = Path(runtime_root).resolve(strict=True)
    path = Path(context_path)
    require(path.resolve().is_relative_to(root) and not path.is_symlink(), "Context must be inside restored runtime")
    body = _body(path)
    require(raw_pin(body)["sha256"] == expected_sha256, "Explicit context SHA differs")
    context = _decode(body)
    require(context["schema_version"] == SCHEMA and context["actual_entry_enabled"] is True
            and context["artificial_fixture_only"] is False, "Source preparation cannot enable execution")
    inventory = _reference(root, context["source_inventory"])
    _inventory_spec(inventory)
    _runtime_spec(context["runtime_paths"], inventory)
    for name, expected in inventory.items():
        _body(_path(root, name), expected)
    release = _release(root, context)
    plan, pair = _qualified_metadata(root, context)
    require(inventory[context["runtime_paths"]["parent"]]["sha256"] == plan["frozen_control"]["parent_source_sha256"], "Original parent Notebook differs")
    jobs = {key: dict(value) for key, value in context["transport_jobs"].items()}
    for key in ("REG:0", "REG:1"):
        for name in ("root_permission", "peer_review"):
            jobs[key][name] = str(_path(root, jobs[key][name]))
    return {"context": context, "source_root": root,
            "source_roots": {key: _path(root, value) for key, value in context["runtime_paths"]["source_roots"].items()},
            "source_inventory": inventory, "conditional_plan": plan, "CLS_pair": pair,
            "root_release": release, "transport_jobs": jobs}


def prepare_context(release_path, expected_release_sha256, *, source_inventory, runtime_paths):
    """Assemble metadata copies from a future actual release; never copy model bytes."""
    _inventory_spec(source_inventory)
    _runtime_spec(runtime_paths, source_inventory)
    raw = _body(release_path)
    require(raw_pin(raw)["sha256"] == expected_release_sha256, "Explicit root release SHA differs")
    release = _decode(raw)
    require(release["schema_version"] == RELEASE_SCHEMA and release["actual_qualified_release"] is True
            and release["artificial_fixture_only"] is False and release["root_authorized_CPU_permission_derivation"] is True,
            "Actual new root release required")
    require(set(release["context"]) == CLAIMS, "Exact release context claims required")
    context = dict(release["context"])
    inventory_body = json_bytes(source_inventory)
    context.update(schema_version=SCHEMA, actual_entry_enabled=True, artificial_fixture_only=False,
                   runtime_paths=runtime_paths,
                   source_inventory={"path": "source-inventory.json", **raw_pin(inventory_body)},
                   root_release={"path": "metadata/root-release.json", **raw_pin(raw)})
    metadata = {"source-inventory.json": inventory_body, "metadata/root-release.json": raw}
    files = release["metadata_files"]
    require(isinstance(files, dict) and 1 <= len(files) <= 256
            and len(files) == len({name.casefold() for name in files}), "Bounded positive metadata inventory required")
    require(sum(ref["bytes"] for ref in files.values()) <= MAX_METADATA, "Metadata total exceeds cap")
    for name, ref in files.items():
        require(transport.member(name).startswith("metadata/") and name != "metadata/root-release.json"
                and Path(name).suffix in {".json", ".py"}, "Only qualification metadata/source copies allowed")
        require(isinstance(ref, dict) and set(ref) == {"path", "bytes", "sha256"}, "Original metadata path pin required")
        require(Path(ref["path"]).suffix == Path(name).suffix, "Original metadata suffix differs")
        expected = _pin_spec({key: ref[key] for key in ("bytes", "sha256")})
        metadata[name] = _body(ref["path"], expected)
    # Validate metadata with the same real consumer as execution. A temporary
    # metadata-only tree cannot contain a checkpoint or archive by construction.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="f009-qualified-metadata-") as temporary:
        root = Path(temporary)
        for name, body in metadata.items():
            target = _path(root, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
        _release(root, context)
        plan, _ = _qualified_metadata(root, context)
        require(source_inventory[runtime_paths["parent"]]["sha256"] == plan["frozen_control"]["parent_source_sha256"],
                "Original parent Notebook differs")
    return context, metadata
