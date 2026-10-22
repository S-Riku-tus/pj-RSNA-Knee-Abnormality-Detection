"""Fixed192 engineering diagnosis for F009. Source-only unless --execute.

The frozen serving startup is derived at one input-resolution boundary. Its
globals are copied, never mutated. Old REG transport requires model temporaries
under /kaggle/working; they are removed before evidence publication. MRI links
stay outside working. No labels or uploader.
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
import os
import secrets
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

PACKAGE = "_f009_diagnostic_support"
CORE_PACKAGE = "_f009_kaggle_support"
DRIVER_SHA = "af9fbbce18c254019f4fb728f04d328ac1e78bf5e70cc0af4ccab5b9e88ba6e9"
CONTEXT_SHA = "37a058c38f3002a9857a26c88601fb73fc9df73710af6d0c1b4248eef06fea97"
SELECTION_SHA = "2f004a7cf359f62bc1fa56926f47bd63e5bc909388a304dcbc5080043bea54b0"
SELECTION_BYTES = 14406
ORDER_SHA = "2815c9fc459f5a02ade7d305e9acb17ca3b2e4028125b923d9e931b02abfd7c4"
SERIES_SHA = "573c1d80772bf41211c91b149c95677385a1c22d63f485c347f1b46c0177aef3"
RESULT_SCHEMA = "f009_fixed192_native_diagnostic_result_v1"
SELECTION_REL = "diagnostics/fixed192-selection.json"
MAX_JSON = 16 * 1024**2


def require(value, message):
    if not value:
        raise ValueError(message)


def raw_pin(body):
    return {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def pin(path):
    path = Path(path)
    require(path.suffix in {".py", ".json"} and path.is_file() and not path.is_symlink()
            and 0 < path.stat().st_size <= MAX_JSON, "Bounded source/metadata only; no model bodies")
    body = path.read_bytes()
    require(len(body) <= MAX_JSON, "Metadata grew beyond cap")
    return raw_pin(body)


def derive_startup_source(driver_source):
    """Only the input-root argument and new function signature may differ."""
    require(type(driver_source) is bytes and raw_pin(driver_source)["sha256"] == DRIVER_SHA,
            "Exact frozen serving startup source required")
    tree = ast.parse(driver_source)
    original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "prepare_plan")
    derived = copy.deepcopy(original)
    derived.name = "prepare_diagnostic_plan"
    derived.args.args.append(ast.arg(arg="fixture_input_root"))
    target = ast.parse('io.resolve_test_input(Path("/kaggle/input"))', mode="eval").body
    count = 0

    class Adapt(ast.NodeTransformer):
        def visit_Call(self, node):
            nonlocal count
            if ast.dump(node) == ast.dump(target):
                count += 1
                node.args = [ast.Name(id="fixture_input_root", ctx=ast.Load())]
                return node
            return self.generic_visit(node)

    Adapt().visit(derived)
    require(count == 1, "Exactly one serving input-resolution boundary required")
    ast.fix_missing_locations(derived)
    # Reverse the declared change, then compare the entire original function.
    restored = copy.deepcopy(derived)
    restored.name = original.name
    restored.args.args.pop()
    changed = ast.parse("io.resolve_test_input(fixture_input_root)", mode="eval").body

    class Restore(ast.NodeTransformer):
        def visit_Call(self, node):
            if ast.dump(node) == ast.dump(changed):
                return copy.deepcopy(target)
            return self.generic_visit(node)

    Restore().visit(restored)
    require(ast.dump(restored) == ast.dump(original), "Undeclared diagnostic startup changes")
    return (ast.unparse(derived) + "\n").encode()


def _startup(driver, args, fixture_root):
    source = Path(driver.__file__).read_bytes()
    derived = derive_startup_source(source)
    # Preserve the frozen core package/file used by its source-origin guards.
    # Helper function objects still have their original immutable globals.
    namespace = dict(driver.__dict__)
    exec(compile(derived, str(Path(__file__)) + "::derived-startup", "exec"), namespace)
    return namespace["prepare_diagnostic_plan"](args, fixture_root)


def _module_identity(module, expected_path, expected_pin):
    expected_path = Path(expected_path).resolve(strict=True)
    require(Path(module.__file__).resolve() == expected_path
            and module.__spec__ is not None and Path(module.__spec__.origin).resolve() == expected_path
            and sys.modules.get(module.__name__) is module and pin(expected_path) == expected_pin,
            "Executing module origin/spec/source differs")
    fresh = importlib.machinery.PathFinder.find_spec(module.__name__, [str(expected_path.parent)])
    require(fresh is not None and fresh.origin is not None and Path(fresh.origin).resolve() == expected_path,
            "Fresh module resolution differs")


def _preflight(args):
    require(__package__ == PACKAGE, "Use the distinct diagnostic support package")
    require(args.context.suffix == ".json" and args.selection.suffix == ".json", "Metadata JSON paths only")
    require(args.selection_sha == SELECTION_SHA, "Exact old fixed192 selection SHA required")
    driver = importlib.import_module(CORE_PACKAGE + ".f009_kaggle_driver")
    validator = importlib.import_module(CORE_PACKAGE + ".f009_kaggle_context")
    require(pin(driver.__file__)["sha256"] == DRIVER_SHA and pin(validator.__file__)["sha256"] == CONTEXT_SHA,
            "Frozen core source versions differ")
    validated = validator.validate_context(args.context, args.context_sha, runtime_root=args.context.parent)
    root, inventory = Path(validated["source_root"]), validated["source_inventory"]
    for module in (driver, validator):
        expected = root / CORE_PACKAGE / (module.__name__.split(".")[-1] + ".py")
        _module_identity(module, expected, inventory[expected.relative_to(root).as_posix()])
    package_root = root / PACKAGE
    package = sys.modules[PACKAGE]
    require(Path(package.__file__).resolve() == Path(package.__spec__.origin).resolve() == package_root / "__init__.py",
            "Diagnostic package init origin differs")
    names = {"__init__.py", "f009_kaggle_diagnostic.py", "f009_diagnostic_fixture.py"}
    require({path.name for path in package_root.glob("*.py")} == names, "Exact independent diagnostic source package required")
    for name in names:
        expected = package_root / name
        require(pin(expected) == inventory[expected.relative_to(root).as_posix()], "Diagnostic source inventory pin differs")
        if name != "__init__.py":
            module = importlib.import_module(PACKAGE + "." + Path(name).stem)
            _module_identity(module, expected, inventory[expected.relative_to(root).as_posix()])
    require(args.selection.resolve() == (root / SELECTION_REL).resolve()
            and inventory[SELECTION_REL] == {"bytes": SELECTION_BYTES, "sha256": SELECTION_SHA}
            and pin(args.selection) == inventory[SELECTION_REL], "Fixed192 selection is not the inventoried metadata")
    fixture = sys.modules[PACKAGE + ".f009_diagnostic_fixture"]
    return validated, driver, fixture


def _competition_root(explicit):
    candidates = [Path(explicit)] if explicit is not None else [
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection")]
    existing = sorted({p.resolve(strict=True) for p in candidates if (p / "train_series.csv").is_file()})
    require(len(existing) == 1, "One explicit competition metadata root required")
    root = existing[0].resolve(strict=True)
    require(root.is_relative_to(Path("/kaggle/input").resolve(strict=True)), "Actual metadata must come from mounted Kaggle Input")
    return root


@contextmanager
def _owned_work(parent):
    parent = Path(parent).resolve(strict=True)
    require(parent.is_dir() and not parent.is_symlink(), "Existing owned temporary parent required")
    root = Path(tempfile.mkdtemp(prefix="f009-diagnostic-work-", dir=parent)).resolve(strict=True)
    require(root.parent == parent and root.name.startswith("f009-diagnostic-work-") and not root.is_symlink(), "Owned work root differs")
    identity = (root.stat().st_dev, root.stat().st_ino)
    owner = secrets.token_hex(24)
    marker = root / ".f009-owner"
    marker.write_text(owner, encoding="ascii")
    try:
        yield root
    finally:
        require(root.resolve() == root and root.parent == parent and root.name.startswith("f009-diagnostic-work-")
                and root.is_dir() and not root.is_symlink() and (root.stat().st_dev, root.stat().st_ino) == identity
                and marker.is_file() and not marker.is_symlink() and marker.read_text(encoding="ascii") == owner,
                "Refusing unexpected recursive cleanup target")
        shutil.rmtree(root)


def _finish_diagnostic(active, plan, router, operational, *, started, fixture_receipt, fixture_only=False):
    """Run the exact ownedUID engine, publish an explicitly diagnostic projection."""
    root = active.output.parent
    component = copy.copy(active)
    component.output = root / "operational-component.json"
    raw = operational.execute_router_with_operational_recovery(router, component, plan, started,
        task_source_sha256=plan["router_pin"]["sha256"], uid_deadline_seconds=180.0, temporary_parent=root)
    require(raw["schema_version"] == operational.RESULT_SCHEMA and raw["complete"] is True
            and raw["operational_recovery_contract_passed"] is True
            and json.loads(component.output.read_bytes()) == raw, "Completed diagnostic component required")
    order = fixture_receipt["UIDs"]
    require(len(order) == len(set(order)) == 192 and list(plan["test"]) == order
            and [r["StudyInstanceUID"] for r in raw["probabilities_test_order"]] == order, "Exact fixed192 order required")
    coverage = raw["coverage"]
    require(coverage["N"] == raw["studies"] == 192 and coverage["N"] == coverage["S"] + coverage["F_structural"] + coverage["F_operational"]
            and raw["full_fold_model_forward_calls"] == 4 * coverage["S"], "Diagnostic forward/coverage contract differs")
    projection = operational.write_native_projection(raw, root / "native-projection")
    exports = {"native-mean.csv": Path(projection["native_csv"]).read_bytes(),
               "native-failures.csv": Path(projection["failure_mask_csv"]).read_bytes()}
    keys = ("targets", "probabilities_test_order", "sample_UID_order", "perUID_receipts", "coverage", "structural_failures",
            "operational_failures", "failure_UID_union", "full_fold_model_forward_calls", "executor_receipts",
            "reconstruction_component", "resources", "resources_passed", "transport", "torch_peak_reserved_bytes",
            "spawn_workers_remaining", "native_temp_files_removed", "elapsed_seconds", "new_model_startup_component",
            "unchanged_model_startup_component", "actual_library_environment", "determinism")
    result = {key: raw[key] for key in keys if key in raw}
    resources_ok = raw.get("resources_passed") is True and raw.get("spawn_workers_remaining") == 0 and raw.get("native_temp_files_removed") is True
    result.update(schema_version=RESULT_SCHEMA, complete=True, studies=192,
        diagnostic_engineering_passed=(not fixture_only and coverage["S"] == 192 and resources_ok),
        artificial_fixture_only=fixture_only, actual_native_inference_executed=not fixture_only,
        operational_recovery_contract_passed=True, fixture_receipt=fixture_receipt,
        fixed_selection_sha256=SELECTION_SHA, fixed_UID_order_sha256=ORDER_SHA, fixed_train_series_sha256=SERIES_SHA,
        baseline_kind="synthetic_constant_0.5_for_engineering_only", P033_parent_cells_executed=0,
        accuracy_or_weak_Gold_metric_evaluated=False, ranks_or_final_submission_written=False,
        previous_192_or_serving_qualification_inherited=False, new_CLS_comparison_requires_separate_independent_audit=True,
        model_role_changes={"CLS:0": "F009 fixedE12", "CLS:1": "F009 fixedE12", "REG:0": "unchanged E8", "REG:1": "unchanged E10"},
        pair_fingerprint_sha256=plan["pair"]["pair_fingerprint_sha256"], root_release_pin=plan["root_release_pin"],
        context_sha256=plan["context_sha256"], diagnostic_entry_source=pin(__file__), core_driver_sha256=DRIVER_SHA,
        router_source_pin=plan["router_pin"], spawn_router_source=plan["spawn_router_source"],
        operational_component_pin=raw_pin(component.output.read_bytes()),
        failure_policy="Exact structural/operational union; all192success required for new diagnostic engineering pass",
        actual_Runtime_or_Public_or_Submit_qualified=False)
    return result, exports


def _publish(output, result, exports):
    """Publish only small evidence after both temporary owners are cleaned."""
    output = Path(output).absolute()
    destination = output.parent
    parent = destination.parent.resolve(strict=True)
    stage = destination.with_name(destination.name + ".partial")
    require(output.suffix == ".json" and destination.resolve() == parent / destination.name
            and stage.resolve() == parent / stage.name and not destination.exists() and not destination.is_symlink()
            and not stage.exists() and not stage.is_symlink(), "Fresh checked same-parent diagnostic publication required")
    require(result["fixture_cleanup_complete"] is True and result["work_cleanup_complete"] is True, "Cleanup must complete before publication")
    require(set(exports) == {"native-mean.csv", "native-failures.csv"}
            and output.name not in {*exports, "fixture-receipt.json", "DIAGNOSTIC_FAILED.json"}, "Only native evidence files allowed")
    require(all(type(body) is bytes and 0 < len(body) <= MAX_JSON for body in exports.values()), "Bounded evidence only")
    result = copy.deepcopy(result)
    result["native_projection"] = {"native_csv": str(destination / "native-mean.csv"),
        "native_csv_pin": raw_pin(exports["native-mean.csv"]), "failure_mask_csv": str(destination / "native-failures.csv"),
        "failure_mask_pin": raw_pin(exports["native-failures.csv"])}
    payloads = {**exports, "fixture-receipt.json": json_bytes(result["fixture_receipt"]), output.name: json_bytes(result)}
    require(all(len(body) <= MAX_JSON for body in payloads.values()), "Diagnostic metadata exceeds cap")
    stage.mkdir(exist_ok=False)
    try:
        for name, body in payloads.items():
            (stage / name).write_bytes(body)
        require(stage.resolve() == parent / stage.name and not stage.is_symlink()
                and destination.resolve() == parent / destination.name and not destination.exists() and not destination.is_symlink(),
                "Checked fresh diagnostic publication changed")
        os.rename(stage, destination)
    except BaseException as error:
        require(stage.resolve() == parent / stage.name and not stage.is_symlink(), "Publication failure stage ownership differs")
        marker = stage / output.name
        if marker.exists():
            require(not marker.is_symlink(), "Unexpected diagnostic marker symlink")
            marker.unlink()
        (stage / "DIAGNOSTIC_FAILED.json").write_bytes(json_bytes({"complete": False, "error_type": type(error).__name__}))
        raise
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    for key in ("context", "output", "selection", "temporary-parent", "competition-root"):
        parser.add_argument("--" + key, type=Path)
    for key in ("context-sha", "selection-sha"):
        parser.add_argument("--" + key)
    parser.add_argument("--budget-seconds", type=float)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"status": "SOURCE_PREPARATION_ONLY", "actual_diagnostic_execution": False, "context_data_model_GPU_API_reads": 0}))
        return None
    require(all(getattr(args, key) is not None for key in ("context", "context_sha", "output", "selection", "selection_sha", "temporary_parent", "budget_seconds")),
            "Explicit fixed192 context/selection/output/temp/budget required")
    require(math.isfinite(args.budget_seconds) and 120 <= args.budget_seconds <= 3600, "Fixed diagnostic budget120..3600 required")
    require(args.output.suffix == ".json" and not args.output.parent.exists() and not args.output.parent.is_symlink(), "Fresh diagnostic output directory required")
    started = time.monotonic()
    validated, driver, fixture_module = _preflight(args)
    competition = _competition_root(args.competition_root)
    fixture_root, baseline, fixture_receipt, cleanup = fixture_module.prepare_fixed192_fixture(
        competition, args.selection, args.selection_sha, args.temporary_parent, execute=True)
    try:
        order = fixture_receipt["UIDs"]
        order_sha = hashlib.sha256(json.dumps(order, separators=(",", ":")).encode()).hexdigest()
        require(fixture_receipt["actual_fixture_created"] is True and fixture_receipt["artificial_fixture_only"] is False
                and fixture_receipt["selection_pin"] == {"bytes": SELECTION_BYTES, "sha256": SELECTION_SHA}
                and fixture_receipt["train_series_csv_sha256"] == SERIES_SHA
                and fixture_receipt["ordered_UID_sha256"] == order_sha == ORDER_SHA
                and fixture_receipt["studies"] == len(order) == len(set(order)) == 192,
                "Actual fixed192 metadata handoff differs before model transport")
        # The unchanged REG transport explicitly requires this working root.
        # Only the MRI virtual fixture may contain links, and it stays in /tmp.
        with _owned_work(Path("/kaggle/working")) as work:
            active_args = SimpleNamespace(context=args.context, context_sha=args.context_sha,
                output=work / "diagnostic-native.json", baseline=Path(baseline), budget_seconds=args.budget_seconds)
            active, plan, router, operational = _startup(driver, active_args, fixture_root)
            require(plan["context_sha256"] == args.context_sha and plan["CLS_pair"] == validated["CLS_pair"], "Fresh diagnostic model/context handoff differs")
            result, exports = _finish_diagnostic(active, plan, router, operational, started=started, fixture_receipt=fixture_receipt)
        result["work_cleanup_complete"] = True
    finally:
        cleanup()
    require(fixture_receipt["cleanup_complete"] is True, "Fixture cleanup was not completed")
    result.update(fixture_cleanup_complete=True, fixture_receipt=fixture_receipt,
                  model_temporaries_under_working_removed_before_publication=True,
                  MRI_links_under_working=False,
                  total_elapsed_seconds=time.monotonic() - started)
    require(result["total_elapsed_seconds"] < args.budget_seconds, "Diagnostic deadline expired before evidence publication")
    return _publish(args.output, result, exports)


if __name__ == "__main__":
    main()
