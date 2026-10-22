"""Offline fixed192 diagnostic source composition; no jobs, uploads or API.

The default CLI opens no input bodies. Source-only Notebook preparation stops
before fixture/model work and emits no Save payload. Actual metadata releases
are future prerequisites, not evidence that a diagnostic has run.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "scripts/prepare_f009_p033_native20.py"
BASE_SHA = "c9ac8bf2ad667c0f733ae19a96686c66c916bb74051352d647211b7a8afa651e"
BASE_INVENTORY = (ROOT / "artifacts/research/20261017-f009-notebook-preparation-v1/"
                  "source-only-pack-v1/source-inventory.json")
BASE_INVENTORY_SHA = "72f2a70e1bc302d6eba7150a50fc2dfd70a2aadbce366cc913d5de5da51aa813"
SELECTION_PATH = (ROOT / "artifacts/research/20261017-p035-execution-v1/observations/"
                  "p035d-isolated192-20261017-bfc2fa11b4/20261017T024620074489Z/files/P032/diagnostic-selection.json")
SELECTION_SHA = "2f004a7cf359f62bc1fa56926f47bd63e5bc909388a304dcbc5080043bea54b0"
SELECTION_BYTES = 14406
SELECTION_NAME = "diagnostics/fixed192-selection.json"
DIAG_PACKAGE = "_f009_diagnostic_support"


def require(value, message):
    if not value:
        raise ValueError(message)


def raw_pin(body):
    return {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def load_base():
    raw = BASE_PATH.read_bytes()
    require(raw_pin(raw)["sha256"] == BASE_SHA, "Frozen serving source builder differs")
    spec = importlib.util.spec_from_file_location("_f009_diagnostic_frozen_builder", BASE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixed_selection():
    body = SELECTION_PATH.read_bytes()
    require(raw_pin(body) == {"bytes": SELECTION_BYTES, "sha256": SELECTION_SHA}, "Exact observed P035 selection required")
    selection = json.loads(body)
    require(selection["studies"] == 192 and len(selection["UIDs"]) == len(set(selection["UIDs"])) == 192
            and selection["train_csv_Report_labels_read"] is False
            and selection["Gold_labels_fit_select_or_evaluate"] is False, "Fixed metadata-only ordered192 selection differs")
    return body


def assemble_sources(base, control, driver):
    files = base.assemble_sources(control, driver)
    inventory_raw = BASE_INVENTORY.read_bytes()
    require(raw_pin(inventory_raw)["sha256"] == BASE_INVENTORY_SHA, "Frozen serving inventory differs")
    original = json.loads(inventory_raw)
    require({name: raw_pin(body) for name, body in files.items()} == original,
            "All original108 serving sources must remain exact")
    files[DIAG_PACKAGE + "/__init__.py"] = b'"""Fixed192 diagnostic adapter; no parent submission workflow."""\n'
    for name in ("f009_kaggle_diagnostic.py", "f009_diagnostic_fixture.py"):
        body = (ROOT / "src/rsna_knee" / name).read_bytes()
        compile(body, DIAG_PACKAGE + "/" + name, "exec")
        files[DIAG_PACKAGE + "/" + name] = body
    files[SELECTION_NAME] = fixed_selection()
    require(len(files) == len(original) + 4, "Only two diagnostic modules/init/selection may extend old sources")
    return files, original


def make_notebook(base, files, control, context, *, entry_enabled):
    # Generate the exact same source-restoration cell, then discard every
    # parent/external-extension execution cell. The original science is inert.
    serving_raw, archive, inventory = base.make_notebook(files, control, context, entry_enabled=entry_enabled)
    serving = json.loads(serving_raw)
    restore_cells = [copy.deepcopy(c) for c in serving["cells"] if c["cell_type"] == "code"][:2]
    run = ("import sys as _f009_diag_sys\n"
           "from pathlib import Path as _f009_diag_Path\n"
           "_f009_diag_sys.path.insert(0,str(_f009_runtime))\n"
           "from _f009_diagnostic_support import f009_kaggle_diagnostic as _f009_diag\n"
           "_f009_diag_tmp = '/kaggle/temp' if _f009_diag_Path('/kaggle/temp').is_dir() else '/tmp'\n"
           "_f009_diagnostic_result = _f009_diag.main([\n"
           "    '--context',str(_f009_runtime/'f009-context.json'),\n"
           "    '--context-sha',(_f009_runtime/'f009-context.sha256').read_text(encoding='ascii').strip(),\n"
           "    '--output','/kaggle/working/F009_FIXED192_DIAGNOSTIC/diagnostic-result.json',\n"
           f"    '--selection',str(_f009_runtime/{SELECTION_NAME!r}),\n"
           f"    '--selection-sha',{SELECTION_SHA!r},\n"
           "    '--temporary-parent',_f009_diag_tmp,'--budget-seconds','3600','--execute'])\n"
           "if (not isinstance(_f009_diagnostic_result,dict) or _f009_diagnostic_result.get('complete') is not True\n"
           "        or _f009_diagnostic_result.get('schema_version') != 'f009_fixed192_native_diagnostic_result_v1'):\n"
           "    raise RuntimeError('Fixed192 diagnostic did not complete its new contract')\n")
    notice = {"cell_type": "markdown", "metadata": {}, "source":
        "# Conditional F009 fixed192 engineering diagnostic\n\nThe fixed P035 metadata selection is unchanged. "
        "P033 source is retained only as frozen context provenance; none of its eight scientific cells run. "
        "The temporary competition view uses a 0.5 baseline and writes diagnostic native outputs only. "
        "This is not a labeled accuracy test or a submitted candidate. Source-only packages stop before execution.\n"}
    book = {"cells": [notice, *restore_cells, base.cell(run)], "metadata": copy.deepcopy(serving["metadata"]),
            "nbformat": 4, "nbformat_minor": 5}
    for text in base.code_cells(book):
        compile(text, "F009-fixed192-carrier-cell", "exec")
    raw = (json.dumps(book, ensure_ascii=False, indent=1) + "\n").encode()
    require(len(raw) <= 950000, "Diagnostic Notebook source capacity exceeded")
    require(not any("get_ipython" in text or "_f009_parent_code" in text for text in base.code_cells(book)),
            "Diagnostic Notebook must not execute P033 scientific cells")
    return raw, archive, inventory


def prepare(output, *, slug="rsraki/f009-fixed192-diagnostic-source-20261017-v1", execute=False,
            release=None, expected_release_sha256=None):
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink(), "Fresh fixed192 source preparation directory required")
    require(re.fullmatch(r"rsraki/f009-fixed192-diagnostic-[a-z0-9-]+", slug), "Own fixed192 private slug required")
    require(type(execute) is bool and (execute or (release is None and expected_release_sha256 is None)),
            "Release arguments require explicit --execute")
    if execute:
        require(release is not None and re.fullmatch("[0-9a-f]{64}", expected_release_sha256 or ""),
                "Explicit qualified actual metadata release required")
    base = load_base()
    control = base.inspect_control()
    with base.new_support_modules() as (driver, validator):
        files, original = assemble_sources(base, control, driver)
        source_inventory = {name: raw_pin(body) for name, body in files.items()}
        require(sum(row["bytes"] for row in source_inventory.values()) <= base.MAX_SOURCE, "Source-only body cap exceeded")
        if execute:
            context, metadata = validator.prepare_context(release, expected_release_sha256,
                source_inventory=source_inventory, runtime_paths=base.runtime_paths())
        else:
            context, metadata = validator.context_template(source_inventory=source_inventory,
                                                           runtime_paths=base.runtime_paths())
        require(context["actual_entry_enabled"] is execute and type(metadata) is dict
                and "source-inventory.json" in metadata, "Real validator scope/inventory required")
        require(json.loads(metadata["source-inventory.json"]) == source_inventory, "Source inventory changed in context assembly")
        files.update(control["reg_metadata"])
        for name, body in metadata.items():
            base.safe_name(name)
            require((name.startswith("metadata/") or name == "source-inventory.json") and type(body) is bytes
                    and name.endswith((".json", ".py")) and len(body) <= 16 * 1024**2,
                    "Only bounded copied qualification metadata is allowed")
            require(name not in files or files[name] == body, "Metadata cannot replace exact source or REG grant")
            files[name] = body
        plan_name = "metadata/conditional-plan.json"
        require(plan_name not in files or files[plan_name] == control["plan_raw"], "Fixed conditional plan differs")
        files[plan_name] = control["plan_raw"]
        files["f009-context.json"] = json_bytes(context)
        files["f009-context.sha256"] = (raw_pin(files["f009-context.json"])["sha256"] + "\n").encode("ascii")
        raw, archive, inventory = make_notebook(base, files, control, context, entry_enabled=execute)
        payload = base.manual_payload(context, raw.decode(), slug) if execute else None
    manifest = {"schema_version": "f009_fixed192_diagnostic_Notebook_source_preparation_v1",
        "source": raw_pin(raw), "runtime_archive": raw_pin(archive), "runtime_files": inventory,
        "source_inventory": raw_pin(files["source-inventory.json"]), "context": raw_pin(files["f009-context.json"]),
        "source_member_count": len(source_inventory), "unchanged_serving_source_count": len(original),
        "unchanged_serving_sources": original, "fixed_selection": raw_pin(files[SELECTION_NAME]),
        "fixed_selection_runtime_path": SELECTION_NAME, "serving_builder_sha256": BASE_SHA,
        "restorer_literal_sha256": raw_pin(base.RESTORER.encode())["sha256"],
        "actual_entry_enabled": execute, "actual_metadata_release_verified": execute,
        "manual_save_payload_present": payload is not None, "parent_eight_cells_executed": False,
        "final_submission_csv_written": False, "baseline_is_artificial_point5": True,
        "checkpoint_model_MRI_binaries_in_source_archive": False,
        "accuracy_or_Public_evaluated": False, "actual192_runtime_qualified": False,
        "final_context_root_review_required_before_Save": True,
        "API_upload_Save_Submit": 0, "builder": raw_pin(Path(__file__).read_bytes())}
    staging = output.with_name(output.name + ".partial")
    parent = output.parent.resolve()
    require(output.resolve() == parent / output.name and staging.resolve() == parent / staging.name
            and not staging.exists() and not staging.is_symlink(), "Fresh same-parent owned source stage required")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        artifacts = {"source.ipynb": raw, "runtime-bundle.zip": archive, "manifest.json": json_bytes(manifest),
                     "context.json": files["f009-context.json"], "source-inventory.json": files["source-inventory.json"]}
        if payload is not None:
            artifacts["save-payload.json"] = json_bytes(payload)
        for name, body in artifacts.items():
            (staging / name).write_bytes(body)
        require(not output.exists() and not output.is_symlink() and staging.resolve() == parent / staging.name,
                "Checked fresh diagnostic source publication required")
        os.rename(staging, output)
    except BaseException as error:
        (staging / "PREPARATION_FAILED.json").write_bytes(json_bytes({"complete": False, "error_type": type(error).__name__}))
        raise
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--slug", default="rsraki/f009-fixed192-diagnostic-source-20261017-v1")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--release", type=Path)
    parser.add_argument("--expected-release-sha256")
    args = parser.parse_args(argv)
    if args.output is None:
        require(not args.execute and args.release is None and args.expected_release_sha256 is None,
                "Explicit actual source output/release required")
        print(json.dumps({"source_preparation_only": True, "input_bodies_read": 0,
                          "actual192_run": False, "API_Save_Submit": 0}))
        return 0
    manifest = prepare(args.output, slug=args.slug, execute=args.execute, release=args.release,
                       expected_release_sha256=args.expected_release_sha256)
    print(json.dumps({"source": manifest["source"], "actual_entry_enabled": manifest["actual_entry_enabled"],
                      "actual192_runtime_qualified": False, "API_Save_Submit": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
