"""Conditional F009/P039 native entry; source-only by default, no API or jobs.

The old bootstrap, model preflight, CLS transports and isolated binder are never
called. Only independently hash-pinned component source is reused. The builder
vendors this module under _f009_kaggle_support, not canonical rsna_knee.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
import importlib.machinery
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

SUPPORT_PACKAGE = "_f009_kaggle_support"
CONTEXT_SCHEMA = "f009_P033_native20_Kaggle_context_v1"
RESULT_SCHEMA = "f009_amended_CLS_P039_REG_owned_UID_native_result_v1"
ADAPTER_SCHEMA = "f009_P033_isolated_uid_native_adapter_v1"
RAW_SCHEMA = "f009_amended_CLS_P039_REG_raw_native_component_v1"
ORIGINAL_ROUTER_SHA = "6dfb95abb56055c04da11a697b9491a72e1bca774d2fdc36285f296b34c4f02c"
PARENT_SHA = "57284b71d271c3d4bf09595a6acade490ea911c8f7438a595724f260f3f28ccb"
SUPPORT_FIXED_SHA = {
    "f009_mixed_runtime.py": "73695814776300ed7c2edd8f2ba535f99b2af9afe4c335346ff06ac1530f6a90",
    "f009_transport.py": "8c613c2c2d24fc6b1d1c6222482c581b03ac5027c15b10f5e7116e7f7056af47",
    "f009_source_namespace.py": "cd2b5fd71355801b67f4f4596487d42728e4974f84b10eac15155ef35c6e5526",
    "operational_uid_recovery.py": "40337f6bc2eaa5d75ab1157bcc2a9dec76bd0022445df2c012b6d89c1d45a814",
    "isolated_uid_executor.py": "685ec00ccc595ccb274817f960c28bdd3bc5926657f52e222c86863e5806d8f8",
    "contracts.py": "aa246fd1a49ecbe88f1795f49904ef9675f119c935e99b9ffd5eee8220cbc71b",
}
UID_DEADLINE_SECONDS = 180.0


def require(condition, message):
    if condition is not True:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Regular pinned file required")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def pin(path):
    return {"bytes": Path(path).stat().st_size, "sha256": sha(path)}


def read_json(path, expected=None):
    path = Path(path)
    require(path.suffix == ".json" and not path.is_symlink() and 0 < path.stat().st_size <= 16*1024**2,
            "Bounded metadata JSON required")
    require(expected is None or pin(path) == expected, "Explicit metadata bytes/SHA differs")
    return json.loads(path.read_bytes(), parse_constant=lambda _: require(False, "Nonfinite JSON"))


def dump(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _statement(text):
    return ast.parse(text).body[0]


def derive_router_source(original_source):
    """Produce a new named spawn module; only declared integration nodes differ."""
    require(type(original_source) is bytes and hashlib.sha256(original_source).hexdigest() == ORIGINAL_ROUTER_SHA,
            "Exact original router source required; old bootstrap is not executable input")
    tree = ast.parse(original_source.decode("utf-8"))
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.FunctionDef) and node.name in ("preflight", "main"))
        and not (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                 and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__")]
    counts = {name: 0 for name in ("contracts_import", "mixed_import", "result", "constructor", "expected", "CPU_close", "GPU_close", "finally")}
    for index, node in enumerate(tree.body):
        if isinstance(node, ast.Import) and any(alias.name == "mixed_serving_contracts" for alias in node.names):
            tree.body[index] = _statement("from _f009_kaggle_support import f009_kaggle_driver as c")
            counts["contracts_import"] += 1
    execute = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "execute")

    final_assignment = next(statement for node in ast.walk(execute) if isinstance(node, ast.Try)
                            for statement in node.finalbody if isinstance(statement, ast.Assign)
                            and [n.id for n in statement.targets if isinstance(n, ast.Name)] == ["cpu_models", "models"])

    class Adapt(ast.NodeTransformer):
        def visit_Import(self, node):
            if [(n.name, n.asname) for n in node.names] == [("isolated_mixed_runtime", "mixed")]:
                counts["mixed_import"] += 1
                return _statement('mixed = plan["mixed"]')
            return node

        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "result" and isinstance(node.value, ast.Dict):
                counts["result"] += 1
                return _statement("result = c.initial_component(args, plan, determinism)")
            if isinstance(node.value, ast.Constant) and node.value.value is None:
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    target = node.targets[0].id
                    if target in ("cpu_models", "models"):
                        counts["CPU_close" if target == "cpu_models" else "GPU_close"] += 1
                        return [_statement(target+".close()"), node]
                if node is final_assignment:
                    counts["finally"] += 1
                    return [_statement("c.close_models(cpu_models, models)"), node]
            return self.generic_visit(node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "mixed" and node.func.attr == "reconstruct_cpu_models":
                counts["constructor"] += 1
                replacement = ast.parse('''mixed.reconstruct_cpu_models(
                    plan["models_roots"], pair, plan["views"], plan["documents"],
                    plan["permission"]["reconstruction_permission"], core=core,
                    regional=plan["regional"], boundary=plan["boundary"], regional_original=original,
                    cls_pair=plan["CLS_pair"], conditional_plan=plan["conditional_plan"],
                    cls_transport=plan["CLS_transport"], namespace_helper=plan["namespace_helper"],
                    dependency_pins=plan["dependency_pins"], permission_path=plan["CPU_permission_path"],
                    expected_permission_sha256=plan["CPU_permission_sha256"], execute=True)''').body[0].value
                require(ast.dump(ast.Tuple(elts=node.args, ctx=ast.Load())) == ast.dump(ast.Tuple(elts=replacement.args, ctx=ast.Load())),
                        "Original constructor positional binding changed")
                return replacement
            return self.generic_visit(node)

        def visit_Subscript(self, node):
            if ast.dump(node) == ast.dump(ast.parse('plan["values"]["CPU_smoke"]["model_fingerprints"]', mode="eval").body):
                counts["expected"] += 1
                return ast.parse("mixed.model_fingerprints_from_pair(pair)", mode="eval").body
            return self.generic_visit(node)

    Adapt().visit(execute)
    require(counts == {name: 1 for name in counts}, "Exact limited router integration edits required: " + str(counts))
    ast.fix_missing_locations(tree)
    raw = (ast.unparse(tree)+"\n").encode("utf-8")
    compile(raw, "F009/serving_router.py", "exec")
    return raw


def initial_component(args, plan, determinism):
    """New descriptors; no old F006/P018 model authority is inherited."""
    import torch

    pair = plan["pair"]
    return {"schema_version": RAW_SCHEMA, "complete": False, "phase_contract_passed": False,
        "startup_complete": False, "selected_parent_notebook_sha256": PARENT_SHA,
        "production_permission_sha256": args.permission_sha, "native_context_sha256": plan["context_sha256"],
        "pair_fingerprint_sha256": pair["pair_fingerprint_sha256"],
        "family_fold_manifest_sha256": pair["family_fold_manifest_sha256"], "model_classes": pair["model_classes"],
        "new_driver_source_sha256": sha(__file__), "new_source_inventory_pin": plan["source_inventory_pin"],
        "new_router_source_pin": plan["router_pin"], "determinism": determinism,
        "input_receipt": plan["input_receipt"], "baseline_csv_sha256": plan["baseline_csv_sha256"],
        "teacher_Report_read": False, "own_Gold_fit_or_selection": False,
        "trusted_native_cache_reference_SHA_available": False, "canonical_training_cache_match_claimed": False,
        "final_CSV_written": False, "baseline_cleanup_performed": False, "additional_cublas_clears": 0,
        "serving_candidate_gate_passed": False, "accuracy_evaluated": False,
        "new_diagnostic_qualification_receipt_written": False, "P019_reader_recovery_enabled": True,
        "old_runtime_or_CPU_qualification_inherited": False, "remaining_child_budget_seconds": args.budget_seconds,
        "actual_library_environment": {"torch": torch.__version__, "CUDA": torch.version.cuda,
            "cuDNN": torch.backends.cudnn.version(), "devices": [torch.cuda.get_device_name(i) for i in (0, 1)]}}


def close_models(cpu_models, gpu_models):
    """Attempt both leases even when one close raises; propagate cleanup failure."""
    errors = []
    for models in (gpu_models, cpu_models):
        if models is not None and not models.closed:
            try:
                models.close()
            except BaseException as error:
                errors.append(error)
    if errors:
        raise errors[0]


def pin_spawn_router_source(directory, expected_sha):
    directory = Path(directory).resolve(strict=True)
    path = directory / "serving_router.py"
    require(sha(path) == expected_sha, "New router source SHA differs")
    sys.path[:] = [str(directory), *(entry for entry in sys.path if entry != str(directory))]
    spec = importlib.machinery.PathFinder.find_spec("serving_router", sys.path)
    require(spec is not None and spec.origin is not None and Path(spec.origin).resolve() == path,
            "Spawn would resolve another serving_router")
    cached = sys.modules.get("serving_router")
    require(cached is not None and Path(cached.__file__).resolve() == path,
            "Cached router is not new spawn task module")
    return {"module": "serving_router", "source": str(path), "sha256": expected_sha,
            "fresh_PathFinder_origin": spec.origin, "parent_cached_origin": cached.__file__,
            "spawn_sys_path_binding_verified": True}


def _support(name):
    require(__package__ == SUPPORT_PACKAGE, "Actual driver must use the dedicated support package, never canonical rsna_knee")
    module = importlib.import_module("."+name, SUPPORT_PACKAGE)
    path = Path(module.__file__).resolve()
    require(path.parent == Path(__file__).resolve().parent
            and sys.modules.get(module.__name__) is module
            and Path(module.__spec__.origin).resolve() == path, "Actual same-package support identity required")
    if name+".py" in SUPPORT_FIXED_SHA:
        require(sha(path) == SUPPORT_FIXED_SHA[name+".py"], "Fixed support source differs: " + name)
    return module


def _named_module(name, path):
    """Import through a fresh origin check; do not accept a cached shadow."""
    path = Path(path).resolve(strict=True)
    spec = importlib.machinery.PathFinder.find_spec(name, sys.path)
    require(spec is not None and spec.origin is not None and Path(spec.origin).resolve() == path,
            "Fresh named component origin differs: " + name)
    module = importlib.import_module(name)
    require(Path(module.__file__).resolve() == path and sys.modules.get(name) is module,
            "Cached component origin differs: " + name)
    return module


def _derive_CPU_permission(plan, work):
    """Only a separately pinned new root release may authorize this derivation."""
    release, mixed = plan["root_release"], plan["mixed"]
    require(release["root_authorized_CPU_permission_derivation"] is True,
            "Explicit new root authority for CPU permission derivation required")
    value = {"schema_version": mixed.PERMISSION_SCHEMA, "root_authorized_CPU_reconstruction": True,
        "conditional_plan_sha256": mixed.PLAN_SHA, "pair_fingerprint_sha256": plan["pair"]["pair_fingerprint_sha256"],
        "runtime_source_sha256": sha(mixed.__file__), "dependency_pins": plan["dependency_pins"],
        "derived_under_new_root_release_pin": plan["root_release_pin"],
        "old_runtime_permission_reused": False}
    path = Path(work) / "f009-CPU-permission.json"
    dump(path, value)
    plan.update(permission={"reconstruction_permission": value}, CPU_permission_path=path,
                CPU_permission_sha256=sha(path))


def _transport_models(validated, work, transport, reg_transport):
    roots, receipts = {"CLS": {}, "REG": {}}, {}
    for fold in (0, 1):
        job = validated["transport_jobs"]["CLS:"+str(fold)]
        value = transport.unpack_f009_transport(job["input_root"], work / ("CLS-"+str(fold)),
                                              expected_manifest_sha256=job["manifest_sha"], fold=fold)
        require(value["actual_transport"] is True and value["pair"] == validated["CLS_pair"],
                "Fresh actual CLS transport must bind the same complete amended pair")
        roots["CLS"][fold] = Path(value["bundle_root"])
        receipts["CLS:"+str(fold)] = {"bundle_pin": value["bundle_pin"], "checkpoint_pin": value["checkpoint_pin"],
            "qualification_pins": value["qualification_pins"], "qualification_binding_pins": value["qualification_binding_pins"],
            "actual_transport": True, "model_root": str(value["bundle_root"])}
    for fold in (0, 1):
        job = dict(validated["transport_jobs"]["REG:"+str(fold)])
        for field in ("root_permission", "peer_review"):
            value = Path(job[field])
            job[field] = str(value if value.is_absolute() else Path(validated["source_root"])/value)
        destination = work / ("REG-"+str(fold))
        receipt = reg_transport.execute(SimpleNamespace(**job, output=str(destination), work_root="/kaggle/working"))
        require(receipt.get("complete") is True and receipt.get("actual_remote_transport_bytes_verified") is True
                and receipt.get("bundle_manifest_sha256") == job["bundle_sha"], "Exact unchanged REG transport required")
        roots["REG"][fold], receipts["REG:"+str(fold)] = destination / "bundle", receipt
    return roots, receipts


def _dependency_pins(core, regional, boundary, transport, helper, reg_root):
    paths = {"core": Path(core.__file__), "regional": Path(regional.__file__), "boundary": Path(boundary.__file__),
        "cls_transport": Path(transport.__file__), "namespace_helper": Path(helper.__file__),
        "regional_original": reg_root / "code/src/rsna_knee/regional_training_fast16.py",
        "reg_model": reg_root / "code/src/rsna_knee/regional_triplet_model.py",
        "old_native_model": reg_root / "code/src/rsna_knee/finetune_model.py",
        "reg_head": reg_root / "code/src/rsna_knee/feature_model.py"}
    return {name: {"path": str(path.resolve()), "sha256": sha(path)} for name, path in paths.items()}


def prepare_plan(args):
    """Actual-only root/source/metadata guards precede all model transports."""
    require(__package__ == SUPPORT_PACKAGE, "Use the new named dedicated-package entry")
    require(not args.output.exists() and not args.output.is_symlink() and args.output.suffix == ".json" and args.output.parent.is_dir(),
            "Fresh output in an existing owned workspace required")
    require(args.context.suffix == ".json", "Context is metadata JSON, never a model/archive path")
    context = read_json(args.context, {"bytes": args.context.stat().st_size, "sha256": args.context_sha})
    require(context.get("schema_version") == CONTEXT_SCHEMA and context.get("actual_entry_enabled") is True,
            "UNREADY/source-only context cannot execute")
    validator = _support("f009_kaggle_context")
    validated = validator.validate_context(args.context, args.context_sha, runtime_root=args.context.parent)
    require(set(validated) == {"context", "source_root", "source_roots", "source_inventory", "conditional_plan",
                               "CLS_pair", "root_release", "transport_jobs"}, "New context return contract differs")
    source_root, roots = Path(validated["source_root"]), validated["source_roots"]
    inventory = validated["source_inventory"]
    require(Path(roots["support"]).resolve() == Path(__file__).resolve().parent,
            "Executing support package differs from the validated source tree")
    package = sys.modules[SUPPORT_PACKAGE]
    init = Path(__file__).resolve().parent / "__init__.py"
    require(Path(package.__file__).resolve() == Path(package.__spec__.origin).resolve() == init
            and inventory[init.relative_to(source_root).as_posix()] == pin(init),
            "Actual support package init identity differs")
    for path in Path(__file__).resolve().parent.glob("*.py"):
        relative = path.relative_to(source_root).as_posix()
        require(inventory.get(relative) == pin(path), "Executing support source is not positively inventoried: " + relative)
        if path.name != "__init__.py":
            actual_module = _support(path.stem)
            require(Path(actual_module.__file__).resolve() == path,
                    "Executing module origin differs from the restored positive inventory")
    mixed, transport, helper, operational = (_support(name) for name in
        ("f009_mixed_runtime", "f009_transport", "f009_source_namespace", "operational_uid_recovery"))
    # Only component source paths are exposed. No historical bootstrap/preflight runs.
    for role in ("serving", "mixed", "regional", "REG_transport"):
        directory = str(roots[role])
        sys.path[:] = [directory, *(item for item in sys.path if item != directory)]
    reg_transport = _named_module("regional_transport", Path(roots["REG_transport"])/"regional_transport.py")
    work = args.output.parent
    model_roots, transport_receipts = _transport_models(validated, work, transport, reg_transport)
    dump(work / "transport-receipts.json", transport_receipts)
    regional = _named_module("regional_serving_contracts", Path(roots["regional"])/"regional_serving_contracts.py")
    boundary = _named_module("regional_runtime", Path(roots["regional"])/"regional_runtime.py")
    core, io = boundary.trusted_imports(roots["common"], model_roots["REG"][0]/"code")
    require("torch" not in sys.modules, "Before-CUDA old REG bootstrap must remain first Torch import")
    router_path = source_root / context["runtime_paths"]["router"]
    old_router = Path(roots["serving"])/"serving_router.py"
    # Inventory binds the exact generated bytes; AST equality avoids depending
    # on ast.unparse formatting being identical in local3.12 and Kaggle3.13.
    require(ast.dump(ast.parse(router_path.read_bytes()))
            == ast.dump(ast.parse(derive_router_source(old_router.read_bytes()))),
            "New router must be the exact limited AST derivation, never original model preflight")
    directory = str(router_path.parent.resolve())
    sys.path[:] = [directory, *(item for item in sys.path if item != directory)]
    router = _named_module("serving_router", router_path)
    spawn_binding = pin_spawn_router_source(router_path.parent, sha(router_path))
    recovery = _named_module("uid_recovery_adapter", Path(roots["serving"])/"uid_recovery_adapter.py")
    input_manifests = {"CLS": {fold: transport_receipts["CLS:"+str(fold)]["bundle_pin"] for fold in (0, 1)},
        "REG": {fold: validated["transport_jobs"]["REG:"+str(fold)]["bundle_sha"] for fold in (0, 1)}}
    dependency_pins = _dependency_pins(core, regional, boundary, transport, helper, model_roots["REG"][0])
    pair, views, documents = mixed.bind_pairs(core, regional, model_roots, input_manifests,
        cls_pair=validated["CLS_pair"], conditional_plan=validated["conditional_plan"], cls_transport=transport,
        namespace_helper=helper, dependency_pins=dependency_pins, execute=True)
    root, test, sample, series, input_receipt = io.resolve_test_input(Path("/kaggle/input"))
    require(args.baseline.suffix == ".csv" and not args.baseline.is_symlink()
            and 0 < args.baseline.stat().st_size <= 64*1024**2, "Bounded completed parent CSV required")
    baseline_body = args.baseline.read_bytes()
    baseline_strings = recovery.baseline_csv_strings(baseline_body, test, io.TARGETS)
    policy = recovery.StructuralReaderPolicy(roots["common"], model_roots["REG"][0]/"code")
    release_pin = {key: context["root_release"][key] for key in ("bytes", "sha256")}
    plan = {**validated, "context_sha256": args.context_sha, "roots": roots, "models_roots": model_roots,
        "regional": regional, "boundary": boundary, "core": core, "io": io, "mixed": mixed,
        "CLS_transport": transport, "namespace_helper": helper, "dependency_pins": dependency_pins,
        "pair": pair, "views": views, "documents": documents, "input_root": root, "test": test, "sample": sample,
        "series": series, "input_receipt": input_receipt, "baseline_strings": baseline_strings,
        "baseline_csv_sha256": hashlib.sha256(baseline_body).hexdigest(), "recovery_policy": policy,
        "p": SimpleNamespace(FAMILIES=("CLS", "REG")), "root_release_pin": release_pin,
        "source_inventory_pin": {key: context["source_inventory"][key] for key in ("bytes", "sha256")},
        "router_pin": pin(router_path), "spawn_router_source": spawn_binding}
    _derive_CPU_permission(plan, work)
    active = SimpleNamespace(output=args.output, decoder_root=Path(context["decoder_root"]),
                             budget_seconds=args.budget_seconds, permission_sha=release_pin["sha256"])
    return active, plan, router, operational


def _finish_native(args, plan, router, operational, *, started, temporary_parent=None):
    """Reclassify a new-model execution; retain the old operational contract only as a component."""
    require(not args.output.exists() and not args.output.is_symlink(), "Fresh new native output required")
    with tempfile.TemporaryDirectory(prefix="f009-native-component-", dir=temporary_parent) as temporary:
        component_args = copy.copy(args)
        component_args.output = Path(temporary) / "operational-component.json"
        raw = operational.execute_router_with_operational_recovery(router, component_args, plan, started,
            task_source_sha256=plan["router_pin"]["sha256"], uid_deadline_seconds=UID_DEADLINE_SECONDS)
        require(raw["schema_version"] == operational.RESULT_SCHEMA and raw["complete"] is True
                and raw["operational_recovery_contract_passed"] is True
                and json.loads(component_args.output.read_bytes()) == raw, "Completed new-model operational component required")
        projection = operational.write_native_projection(raw, args.output.parent / "isolated-uid-projection")
        result = dict(raw)
        result.update(schema_version=RESULT_SCHEMA, operational_component_schema=operational.RESULT_SCHEMA,
            operational_component_sha256=sha(component_args.output), old_runtime_qualification_inherited=False,
            model_source_change={"CLS0": "F009 fixed E12 study gamma", "CLS1": "F009 fixed E12 study gamma",
                                 "REG0": "unchanged selected E8", "REG1": "unchanged selected E10"},
            runtime_changes={"reader_source_bytes_changed": False, "CLS_model_source_bytes_changed": True,
                "REG_model_source_bytes_changed": False, "success_four_fold_FP32_math_changed": False,
                "executor": "isolated_owned_UID_subprocess", "operational_failure_projection": "explicit structural/operational union"},
            f009_execution_contract={"schema_version": "f009_new_four_role_execution_contract_v1",
                "new_context_and_source_inventory_bound": True, "new_pair_and_actual_CPU_bound": True,
                "old_F006_preflight_or_bootstrap_executed": False, "own_UID_deadline_seconds": UID_DEADLINE_SECONDS,
                "new_router_pin": plan["router_pin"], "new_driver_pin": pin(__file__),
                "root_release_pin": plan["root_release_pin"], "accuracy_or_Submit_qualified": False})
        if "unchanged_model_startup_component" in result:
            result["new_model_startup_component"] = result.pop("unchanged_model_startup_component")
            result.pop("unchanged_model_startup_component_scope", None)
    result["operational_component_quarantine_removed"] = True
    dump(args.output, result)
    failed = result["failure_UID_union"]
    adapter = {"schema_version": ADAPTER_SCHEMA, "complete": True, "mode": "serving", "native_executor": "isolated-uid",
        "studies": result["studies"], "successful_studies": result["coverage"]["S"],
        "structural_failure_count": len(result["structural_failures"]), "structural_failure_UIDs": list(result["structural_failures"]),
        "operational_failure_count": len(result["operational_failures"]), "operational_failure_UIDs": list(result["operational_failures"]),
        "total_failure_count": len(failed), "failure_UID_union": failed, "failure_mask_is_structural_operational_union": True,
        "structural_failures_forwarded_to_cell_failure_mask": True, "old_structural_failure_claimed_for_operational": False,
        "native_csv": projection["native_csv"], "native_csv_sha256": projection["native_csv_sha256"],
        "failure_mask_csv": projection["failure_mask_csv"], "failure_mask_sha256": projection["failure_mask_sha256"],
        "native_result_sha256": sha(args.output), "native_result_schema": RESULT_SCHEMA,
        "model_fingerprints": result["reconstruction_component"]["model_fingerprints"],
        "pair_fingerprint_sha256": plan["pair"]["pair_fingerprint_sha256"], "coverage": result["coverage"],
        "executor_receipts": result["executor_receipts"], "legacy_quarantine_removed": result["legacy_quarantine_removed"],
        "operational_recovery_contract_passed": True, "old_F006_qualification_inherited": False,
        "legacy_phase_and_fallback_assertions_not_new_qualification": True,
        "spawn_router_source": plan["spawn_router_source"], "elapsed_seconds": time.monotonic()-started,
        "inference_math_unchanged": True, "rank_or_final_CSV_written_here": False,
        "own_Gold_fit_select_or_evaluate": False, "Public_accuracy_measured": False}
    dump(args.output.parent / "adapter-receipt.json", adapter)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    for key in ("context", "output", "baseline"):
        parser.add_argument("--"+key, type=Path)
    parser.add_argument("--context-sha")
    parser.add_argument("--budget-seconds", type=float)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"status": "SOURCE_PREPARATION_ONLY", "actual_native_execution": False,
                          "context_model_MRI_GPU_read": False}))
        return
    require(all(getattr(args, key) is not None for key in ("context", "context_sha", "output", "baseline", "budget_seconds")),
            "Explicit actual context/output/parent/budget required")
    require(math.isfinite(args.budget_seconds) and 120 <= args.budget_seconds <= 28800,
            "Unchanged finite native budget required")
    import faulthandler

    started = time.monotonic()
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)
    try:
        active, plan, router, operational = prepare_plan(args)
        result = _finish_native(active, plan, router, operational, started=started)
        print("F009_NATIVE_COMPLETE", json.dumps({"complete": result["complete"], "studies": result["studies"],
              "native_result_schema": RESULT_SCHEMA, "final_submission_CSV_written": False}), flush=True)
        return result
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
