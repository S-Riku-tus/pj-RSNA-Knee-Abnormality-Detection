"""Kaggle-only F006 child adapter; the builder supplies the pinned bootstrap.

No network, training, upload or competition submission is implemented here.
Both diagnostic and serving modes call the unchanged four-model execute path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

# BOOTSTRAP is injected by the builder. Its execution supplies the _p18_*
# globals; targeted F821 annotations avoid assigning over those restored values.


def require(condition, message):
    if condition is not True:
        raise ValueError(message)


def dump(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def progress(stage, state, **details):
    import time

    now = time.monotonic()
    print("P032_NATIVE_PROGRESS", json.dumps({
        "stage": stage, "state": state, "monotonic_seconds": now,
        "elapsed_seconds": now - globals().get("STARTED", now), **details,
    }, allow_nan=False), flush=True)


def pin_spawn_router_source(serving_directory, expected_sha):
    """Make a fresh spawn import resolve the same sealed router as the parent."""
    import importlib.machinery
    import sys

    directory = Path(serving_directory).resolve(strict=True)
    source = directory / "serving_router.py"
    require(not source.is_symlink() and source.is_file(), "Regular sealed serving router required")
    require(hashlib.sha256(source.read_bytes()).hexdigest() == expected_sha,
            "Sealed serving router SHA differs before spawn binding")
    # trusted_imports prepends common/, which also contains a serving_router.py.
    # Spawn inherits sys.path, but does not inherit the parent's cached module.
    # Keep the intended source first before any named task is pickled.
    sys.path[:] = [str(directory), *(entry for entry in sys.path if entry != str(directory))]
    spec = importlib.machinery.PathFinder.find_spec("serving_router", sys.path)
    require(spec is not None and spec.origin is not None
            and Path(spec.origin).resolve(strict=True) == source,
            "Fresh spawn would import another serving router")
    cached = sys.modules.get("serving_router")
    require(cached is not None and Path(cached.__file__).resolve(strict=True) == source,
            "Parent cached serving router differs from fresh spawn resolution")
    return {"module": "serving_router", "source": str(source), "sha256": expected_sha,
            "fresh_PathFinder_origin": spec.origin, "parent_cached_origin": cached.__file__,
            "spawn_sys_path_binding_verified": True}


def diagnostic_fixture(plan, work, count, *, temporary_root=None):
    """Select new studies using series metadata only, never train.csv/Report/labels."""
    import random

    io = plan["io"]
    original, _, _, _, _ = io.resolve_test_input(Path("/kaggle/input"))
    cohort, _ = _p18_contracts.read_pin(  # noqa: F821
        {"path": str(_p18_source / "serving/cohorts/cohort64.json"),  # noqa: F821
         "sha256": plan["p"].COHORTS["native64"]}
    )
    excluded = set(cohort["source_cohort"]["uids"])
    grouped = {}
    with (original / "train_series.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(set(io.SERIES_COLUMNS) <= set(reader.fieldnames or []), "Series metadata header differs")
        for source in reader:
            row = {key: source[key] for key in io.SERIES_COLUMNS}
            if row[io.ID] not in excluded:
                grouped.setdefault(row[io.ID], []).append(row)
    buckets = {}
    for key, rows in grouped.items():
        signature = (min(len(rows), 10),
                     tuple(sorted({r["Anatomical_Plane"] for r in rows})),
                     tuple(sorted({(r["Fluid_Sensitive"], r["Fat_Suppression"]) for r in rows})))
        buckets.setdefault(signature, []).append(key)
    rng = random.Random(20261016)
    keys = sorted(buckets, key=repr)
    for key in keys:
        buckets[key].sort()
        rng.shuffle(buckets[key])
    selected = []
    while len(selected) < count:
        progress = False
        for key in keys:
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop())
                progress = True
        require(progress, "Insufficient new diagnostic UIDs")
    require(not set(selected) & excluded, "Old native64 overlap refused")
    # MRI directory links must never be under /kaggle/working: Output collection
    # may follow them. Only small JSON/CSV receipts are kept in working.
    temporary_root = Path(temporary_root) if temporary_root is not None else (
        Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
    )
    temporary_root = temporary_root.resolve(strict=True)
    temporary = Path(tempfile.mkdtemp(prefix="p032-diagnostic-", dir=temporary_root)).resolve(strict=True)
    require(temporary.parent == temporary_root and temporary.name.startswith("p032-diagnostic-")
            and not temporary.is_symlink(), "Exclusive diagnostic temporary root required")

    def remove_fixture():
        import shutil

        if temporary.exists():
            require(temporary.resolve() == temporary and temporary.parent == temporary_root
                    and not temporary.is_symlink(), "Refusing unexpected diagnostic cleanup target")
            shutil.rmtree(temporary)
        atexit.unregister(remove_fixture)

    import atexit

    atexit.register(remove_fixture)
    root = temporary / "rsna-knee-abnormality-detection"
    root.mkdir(exist_ok=False)
    (root / "test_series").symlink_to(original / "train_series", target_is_directory=True)
    with (root / "test.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([io.ID])
        writer.writerows([[key] for key in selected])
    with (root / "test_series.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, io.SERIES_COLUMNS)
        writer.writeheader()
        for key in selected:
            writer.writerows(grouped[key])
    baseline = work / "diagnostic-baseline.csv"
    for target in (root / "sample_submission.csv", baseline):
        with target.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([io.ID, *io.TARGETS])
            writer.writerows([[key, *(["0.5"] * 12)] for key in selected])
    receipt = {
        "selection": "round_robin_series_count_plane_fluid_fat_metadata_strata",
        "seed": 20261016, "studies": count, "UIDs": selected,
        "old_native64_UID_overlap": 0,
        "train_series_csv_sha256": io.sha(original / "train_series.csv"),
        "train_csv_Report_labels_read": False,
        "Gold_image_UID_presence_audited": False,
        "Gold_labels_fit_select_or_evaluate": False,
        "MRI_directory_link_under_Output_working": False,
        "temporary_fixture_parent": str(temporary),
        "scope": "engineering only; no labeled accuracy or public-parent OOF",
    }
    dump(work / "diagnostic-selection.json", receipt)
    return root.parent, baseline, remove_fixture


def execute_native_router(args, active, plan, router, *, task_source_sha256, started):
    """Opt-in operational wrapper; default calls the original router directly."""
    if getattr(args, "native_executor", "legacy-pool") == "legacy-pool":
        return router.execute(active, plan, started)
    require(args.native_executor == "isolated-uid", "Unknown native executor")
    from p032blend.operational_uid_recovery import execute_router_with_operational_recovery

    return execute_router_with_operational_recovery(
        router, active, plan, started, task_source_sha256=task_source_sha256, uid_deadline_seconds=180.0)


def emit_native_adapter(args, result, work, io, pair, spawn_router, *, started):
    """Separate typed operational failures, preserving the legacy output branch."""
    import time

    isolated = getattr(args, "native_executor", "legacy-pool") == "isolated-uid"
    if isolated:
        from p032blend.operational_uid_recovery import RESULT_SCHEMA, write_native_projection

        require(result.get("schema_version") == RESULT_SCHEMA and result.get("complete") is True
                and result.get("operational_recovery_contract_passed") is True,
                "New operational native execution did not complete its own contract")
        projection = write_native_projection(result, work / "isolated-uid-projection")
        failure = result["structural_failures"]
        operational = result["operational_failures"]
        native, mask = Path(projection["native_csv"]), Path(projection["failure_mask_csv"])
        failed_uids = result["failure_UID_union"]
    else:
        require(result.get("complete") is True and result.get("phase_contract_passed") is True,
                "New native execution did not complete")
        failure, operational = result["structural_failures"], {}
        failed_uids = list(failure)
        native = work / "native-mean.csv"
        with native.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([io.ID, *io.TARGETS])
            for row in result["probabilities_test_order"]:
                writer.writerow([row[io.ID], *row["probabilities"]])
        mask = work / "native-failures.csv"
        with mask.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([io.ID])
            writer.writerows([[row[io.ID]] for row in result["probabilities_test_order"] if row[io.ID] in failure])
    adapter = {
        "schema_version": "p032_f006_native_adapter_v1", "complete": True,
        "mode": args.mode, "studies": result["studies"], "successful_studies": result["studies"] - len(failed_uids),
        "structural_failure_count": len(failure), "structural_failure_UIDs": list(failure),
        "structural_failures_forwarded_to_column_disable_mask": True,
        "native_csv": str(native), "failure_mask_csv": str(mask),
        "native_csv_sha256": io.sha(native), "failure_mask_sha256": io.sha(mask),
        "native_result_sha256": io.sha(args.output),
        "model_fingerprints": result["reconstruction_component"]["model_fingerprints"],
        "pair_fingerprint_sha256": pair["pair_fingerprint_sha256"],
        "elapsed_seconds": time.monotonic() - started,
        "new_parent_integration_not_old_S6_qualification": True,
        "spawn_router_source": spawn_router,
        "historical_native_result_parent_fields_scope": "inherited metadata only; not new-parent certificates",
        "inference_math_unchanged": True, "rank_or_final_CSV_written_here": False,
        "own_Gold_fit_select_or_evaluate": False, "Public_accuracy_measured": False,
    }
    if isolated:
        adapter.pop("structural_failures_forwarded_to_column_disable_mask", None)
        adapter.pop("structural_failures_forwarded_to_cell_failure_mask", None)
        adapter.update(schema_version="p035_f006_isolated_uid_adapter_v1", native_executor="isolated-uid",
                       native_result_schema=result["schema_version"], operational_recovery_contract_passed=True,
                       structural_failures_forwarded_to_cell_failure_mask=True,
                       operational_failure_count=len(operational), operational_failure_UIDs=list(operational),
                       total_failure_count=len(failed_uids), failure_UID_union=failed_uids,
                       failure_mask_is_structural_operational_union=True,
                       old_structural_failure_claimed_for_operational=False,
                       coverage=result["coverage"], executor_receipts=result["executor_receipts"],
                       legacy_quarantine_removed=result["legacy_quarantine_removed"],
                       legacy_phase_and_fallback_assertions_not_new_qualification=True)
    dump(work / "adapter-receipt.json", adapter)
    return adapter


def run(args):
    import sys

    # BOOTSTRAP is assigned by the offline builder. It is the exact recovered
    # native64 source cell: positive source hashes and historical metadata only.
    # The enclosing __main__ guard prevents spawn children from staging it again.
    progress("bootstrap", "begin", mode=args.mode, output=str(args.output))
    exec(compile(BOOTSTRAP, "F006-native64-recovered-bootstrap", "exec"), globals())  # noqa: F821
    progress("bootstrap", "end", mode=args.mode)
    c = _p18_contracts  # noqa: F821
    progress("preflight", "begin", mode=args.mode)
    plan = c.preflight(_p18_context)  # noqa: F821
    progress("preflight", "end", mode=args.mode)
    work = args.output.parent
    for directory in (_p18_source / "serving", plan["roots"]["mixed"],  # noqa: F821
                      plan["roots"]["common"] / "reused",
                      plan["roots"]["common"] / "reused/code/scripts"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    import diagnostic_notebook
    import isolated_contracts
    import serving_router
    import uid_recovery_adapter as recovery

    progress("transport", "begin", mode=args.mode)
    model_roots, transport = diagnostic_notebook.transport_models(plan, work)
    dump(work / "transport-receipts.json", transport)
    progress("transport", "end", mode=args.mode)
    progress("bind", "begin", mode=args.mode)
    regional, boundary = plan["p"].dependencies(plan["roots"]["regional"])
    core, io = boundary.trusted_imports(plan["roots"]["common"], model_roots["REG"][0] / "code")
    spawn_router = pin_spawn_router_source(_p18_source / "serving", c.source_shas()["serving_router.py"])  # noqa: F821
    progress("spawn_router_source", "bound", **spawn_router)
    manifest_shas = {
        family: {int(fold): value for fold, value in values.items()}
        for family, values in plan["review"]["family_fold_manifest_sha256"].items()
    }
    pair, views, documents = isolated_contracts.bind_pairs(core, regional, model_roots, manifest_shas)
    c.require(c.canonical(pair) == c.canonical(plan["pair"]), "Exact historical F006 pair differs")
    progress("bind", "end", mode=args.mode)
    progress("fixture", "begin", mode=args.mode)
    input_root, baseline = Path("/kaggle/input"), args.baseline

    def remove_fixture():
        pass

    if args.mode == "diagnostic":
        input_root, baseline, remove_fixture = diagnostic_fixture(plan, work, args.diagnostic_count)
    root, test, sample, series, input_receipt = io.resolve_test_input(input_root)
    body = baseline.read_bytes()
    baseline_strings = recovery.baseline_csv_strings(body, test, io.TARGETS)
    policy = recovery.StructuralReaderPolicy(plan["roots"]["common"], model_roots["REG"][0] / "code")
    progress("fixture", "end", mode=args.mode, studies=len(test), data_root=str(root))
    # This is a new P032 execution plan, not a certificate for the old S6 parent.
    # Only old exact-model/source metadata and component execution are reused.
    plan.update(models_roots=model_roots, regional=regional, boundary=boundary,
                core=core, io=io, pair=pair, views=views, documents=documents,
                input_root=root, test=test, sample=sample, series=series,
                input_receipt=input_receipt, baseline_strings=baseline_strings,
                baseline_csv_sha256=hashlib.sha256(body).hexdigest(),
                recovery_policy=policy, permission={"reconstruction_permission": {}})
    decoder, resolution = __import__("mixed_parent").resolve_decoder_root()
    dump(work / "decoder-resolution.json", resolution)
    active = SimpleNamespace(output=args.output, decoder_root=decoder,
                             budget_seconds=args.budget_seconds,
                             permission_sha="P032_USER_AUTHORIZED_NEW_PARENT_ADAPTER")
    progress("execute", "begin", mode=args.mode, studies=len(test), budget_seconds=args.budget_seconds)
    try:
        result = execute_native_router(args, active, plan, serving_router,
                                       task_source_sha256=c.source_shas()["serving_router.py"], started=STARTED)
    except BaseException as error:
        progress("execute", "error", mode=args.mode, error_type=type(error).__name__)
        raise
    finally:
        remove_fixture()
    progress("execute", "end", mode=args.mode, complete=result.get("complete"),
             phase_contract_passed=result.get("phase_contract_passed"),
             operational_recovery_contract_passed=result.get("operational_recovery_contract_passed"))
    require(result["studies"] == len(test) and result["targets"] == list(io.TARGETS),
            "New native UID/target contract differs")
    require(set(result["structural_failures"]) <= set(test), "Unknown structural failure UID")
    if getattr(args, "native_executor", "legacy-pool") == "isolated-uid":
        require(set(result["operational_failures"]) <= set(test), "Unknown operational failure UID")
    emit_native_adapter(args, result, work, io, pair, spawn_router, started=STARTED)


def main():
    import faulthandler

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("serving", "diagnostic"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--budget-seconds", type=float, required=True)
    parser.add_argument("--diagnostic-count", type=int, default=192)
    parser.add_argument("--native-executor", choices=("legacy-pool", "isolated-uid"), default="legacy-pool")
    args = parser.parse_args()
    require(args.mode == "diagnostic" or args.baseline is not None, "Serving requires completed parent")
    require(120 <= args.budget_seconds <= 28800 and 1 <= args.diagnostic_count <= 256, "Finite child limits required")
    require(not args.output.exists(), "Fresh native result required")
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)
    try:
        run(args)
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    import time

    STARTED = time.monotonic()
    main()
