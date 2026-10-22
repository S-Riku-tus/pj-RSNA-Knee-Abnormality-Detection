"""Offline fixed1080 fluid-sensitive package; never run, download, or upload."""
from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import importlib.util
import json
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "artifacts/kaggle/orthodiff-sag-fullweak4207-20261017-v2"
SUBSET = ROOT / "artifacts/research/20261017-orthodiff-fluid-subset-actual-v1/subset.json"
OUTPUT = ROOT / "artifacts/kaggle/orthodiff-sag-fluid1080-20261017-v3"
ID = "StudyInstanceUID"
PARENT_SHA = "15bb2d6c7c224021f95a304b08eb9e3644164ea8373f4ae6ec1c7afc77582e1f"
SUBSET_SHA = "c0e52bf58e46579d28d090cd25168130033f8b749098359ec95bdc425144c178"
AUDIT_SHA = "7fbc160b8715e69dc506b64e9187f55a2c2b67925bf6a5b0a4a95adf943c8270"
SPEC_SHA = "4326c03108ebe1d00ff833d8da4ebec891ba19d50d9bc6f7c14a3d494d2d92ff"
UID_SHA = "2fb8f009f2f2372ea0e460c174bfb373172f2477a356cd8bee783e0d0792bf7e"
RUNTIME_SHA = {
    "base": "d0b432859d6fcae23192cd31a86330dc92ab3052a1b06253f9cfb7f82b5afe90",
    "pilot": "4e8d671d6b0cf4402afac0c6c4dc3a3a007bb45878a6c40e7340159762244a1a",
    "full": "a7c2a3a6c42a26f0b8f647842e88d47e78b9cc4baf259c67b9ec8278dd268ee6",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def pin(path):
    digest, length = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
            length += len(block)
    return {"bytes": length, "sha256": digest.hexdigest()}


def read(path):
    return json.loads(Path(path).read_bytes())


def canonical(value):
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def function_text(source, name):
    node = next(item for item in ast.parse(source).body if isinstance(item, ast.FunctionDef) and item.name == name)
    return "\n".join(source.splitlines()[node.lineno - 1:node.end_lineno]) + "\n"


def science_parity(pilot, full, fluid):
    original = function_text(pilot, "read_clip")
    actual = function_text(fluid, "read_fluid_clip")
    removed = ('        classification = classify_pd(getattr(header, "SeriesDescription", ""), getattr(header, "SequenceName", ""))\n'
               '        check(classification["eligible"], "mixed_or_unknown_PD_header")\n')
    require(original.count(removed) == 1, "Exactly the two declared header statements required")
    expected = original.replace("def read_clip(", "def read_fluid_clip(", 1).replace(removed, "", 1)
    require(actual == expected, "Reader differs beyond function name and the two PD statements")
    restored = actual.replace("def read_fluid_clip(", "def read_clip(", 1).replace(
        '        geometry.append(', removed + '        geometry.append(', 1)
    require(ast.dump(ast.parse(restored), include_attributes=False) == ast.dump(ast.parse(original), include_attributes=False),
            "Restored reader AST differs")
    start = '                        x = ((x - lo) / (hi - lo + 1e-8) * 2 - 1)'
    end = '                        journal.append({"StudyInstanceUID": uid, "status": "success"'
    old_block, new_block = (text[text.index(start):text.index(end)] for text in (full, fluid))
    require(old_block == new_block, "Frozen FP32 normalization/t100/model/RNG block changed")
    record_end = '                        del x, feature, pooled, array'
    old_record, new_record = (text[text.index(end):text.index(record_end)] for text in (full, fluid))
    require(ast.dump(ast.parse(old_record.strip()), include_attributes=False) == ast.dump(ast.parse(new_record.strip()), include_attributes=False),
            "Scientific success record AST differs")
    tree = ast.parse(fluid)
    require(not any(isinstance(n, ast.Name) and n.id == "classify_pd" for n in ast.walk(tree)), "No PD-classifier replacement permitted")
    return {"read_clip_literal_copy_except_rename_and_two_header_statements": True,
            "reader_original_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "reader_new_sha256": hashlib.sha256(actual.encode()).hexdigest(),
            "removed_statements": removed.splitlines(), "restored_reader_AST_exact": True,
            "numeric_forward_block_byte_identical": True, "numeric_forward_block_sha256": hashlib.sha256(old_block.encode()).hexdigest(),
            "success_record_AST_identical": True, "old_RecordJournal_reused_by_exact_full_runtime_pin": True,
            "pilot_globals_or_classify_monkeypatched": False,
            "unchanged": ["UID/SOP/IOP/IPP/spacing", "physical central continuous16", "rescale/MONOCHROME1",
                          "float64 resize before FP32 minmax", "decoder four fixtures", "strict EMA257/schedule13",
                          "t100/mid2/FP32/globalmean256/scopedSeed/all RNG restored", "one model load", "8h/14GiB/8GiB limits"]}


def load_subset():
    require(pin(SUBSET)["sha256"] == SUBSET_SHA, "Frozen actual metadata subset differs")
    subset = read(SUBSET)
    require(subset["parent_qualification_sha256"] == AUDIT_SHA and subset["parent_source_sha256"] == PARENT_SHA,
            "Actual parent qualification/source binding differs")
    require(subset["subset_count"] == 1080 and subset["selected_metadata_candidate_count"] == 1077
            and subset["metadata_unavailable_count"] == 3 and subset["adopted"] is False
            and subset["parent_record_counts"] == {"success": 3063, "data_failure": 1144}, "Frozen parent/subset coverage differs")
    inputs = {Path(path).name: (Path(path), value) for path, value in subset["input_pins"].items()}
    require(len(inputs) == len(subset["input_pins"]) == 7, "Unique complete subset input pins required")
    for path, expected in inputs.values():
        require(pin(path) == expected, "Subset input byte pin changed: " + str(path))
    for path, expected in subset["source_pins"].items():
        require(pin(ROOT / path) == expected, "Subset preparer source changed: " + path)
    audit = read(inputs["verification.json"][0])
    require(inputs["verification.json"][1]["sha256"] == AUDIT_SHA
            and inputs["fluid_sensitive_fallback_preregistered.json"][1]["sha256"] == SPEC_SHA,
            "Actual qualification/preregistered spec SHA differs")
    spec = importlib.util.spec_from_file_location("orthodiff_fluid_metadata_offline", ROOT / "src/rsna_knee/orthodiff_fluid_fallback.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # stdlib-only pure metadata helpers; no upstream/Torch/model import.
    module.require_qualified_parent(audit, read(inputs["ORTHODIFF_FULLWEAK_SUMMARY.json"][0]), read(inputs["ORTHODIFF_FULLWEAK_STATUS.json"][0]))
    entries, seen, successes, count = {row[ID]: row for row in subset["entries"]}, set(), set(), 0
    cohort = read(inputs["cohort.json"][0])
    expected_order = [row[ID] for row in cohort["requested"]]
    with inputs["ORTHODIFF_FULLWEAK_RECORDS.jsonl"][0].open("rb") as stream:
        for count, line in enumerate(stream, start=1):
            record = json.loads(line)
            uid = record[ID]
            require(count <= 4207 and uid == expected_order[count - 1] and uid not in seen, "Actual parent journal UID order differs")
            seen.add(uid)
            eligible, _ = module.unknown_pd_family(record)
            require(eligible == (uid in entries), "Subset must exactly match the1080 unknown-family failures")
            if record["status"] == "success":
                successes.add(uid)
            if eligible:
                require(entries[uid]["parent_record_index"] == count - 1
                        and hashlib.sha256(canonical(record)).hexdigest() == entries[uid]["parent_record_canonical_sha256"],
                        "Subset canonical parent record binding differs")
    require(count == len(seen) == 4207 and len(successes) == 3063 and not successes.intersection(entries),
            "Complete journal/no parent success overlap required")
    requests = []
    for entry in subset["entries"]:
        require(entry["parent_failure_family"] == "unknown_PD_family" and entry["confirmed_PD"] is False
                and entry["route"] == "fluid_sensitive_fallback" and entry["retry_another_series_after_chosen_failure"] is False,
                "Fluid route/no retry binding differs")
        requests.append({key: entry[key] for key in (ID, "parent_record_canonical_sha256", "route", "confirmed_PD", "status", "selected_series")})
    uids = [row[ID] for row in requests]
    require(len(uids) == len(set(uids)) == 1080
            and hashlib.sha256("".join(uid + "\n" for uid in uids).encode()).hexdigest() == subset["subset_UID_order_sha256"] == UID_SHA,
            "Fixed1080 UID order differs")
    proof = {"subset_pin": pin(SUBSET), "parent_qualification_pin": inputs["verification.json"][1],
             "parent_source_sha256": PARENT_SHA, "parent_UID_order_sha256": subset["parent_UID_order_sha256"],
             "parent_journal_pin": inputs["ORTHODIFF_FULLWEAK_RECORDS.jsonl"][1],
             "frozen_fallback_spec_pin": inputs["fluid_sensitive_fallback_preregistered.json"][1],
             "parent_records_fully_consumed": count, "parent_success_UIDs": len(successes), "parent_success_intersection": 0,
             "subset_requested": 1080, "metadata_selected": 1077, "metadata_unavailable": 3,
             "runtime_embeds_groups_folds_labels_Report": False, "subset_adopted": False}
    return requests, proof


def prepare(output=OUTPUT, slug="rsraki/orthodiff-sag-fluid1080-20261017-v3"):
    output = Path(output)
    require(not output.exists() and not output.is_symlink(), "Use a fresh fluid-sensitive package")
    require(pin(PARENT / "source.ipynb")["sha256"] == PARENT_SHA, "Frozen parent source differs")
    parent = read(PARENT / "manifest.json")
    require(parent["source"] == pin(PARENT / "source.ipynb"), "Parent manifest source pin differs")
    runtimes = {name: (PARENT / (name + "-runtime.py")).read_bytes().decode() for name in RUNTIME_SHA}
    for name, source in runtimes.items():
        require(pin(PARENT / (name + "-runtime.py")) == parent["runtime_pins"][name]
                and hashlib.sha256(source.encode()).hexdigest() == RUNTIME_SHA[name], "Frozen runtime differs: " + name)
    runtimes["fluid"] = (ROOT / "scripts/orthodiffusion_fluid_fallback_runtime.py").read_bytes().decode()
    parity = science_parity(runtimes["pilot"], runtimes["full"], runtimes["fluid"])
    for name, source in runtimes.items():
        compile(ast.parse(source), name, "exec")
    requests, proof = load_subset()
    config, vendor = read(PARENT / "config.json"), read(PARENT / "vendor.json")
    require(config == parent["config"] and pin(PARENT / "vendor.json") == parent["vendor_pin"], "Frozen parent config/vendor differs")
    config = copy.deepcopy(config)
    config.update(uids=[row[ID] for row in requests], UID_order_sha256=UID_SHA,
                  predeclared_cohort_pin=pin(SUBSET), fluid_requests=requests, fluid_source_bindings=proof)
    config["fluid_protocol"] = {"route": "fluid_sensitive_fallback", "confirmed_PD": False,
        "selected_once_from_frozen_metadata_subset": True, "metadata_only_FS1_before_FS0_then_UID": True,
        "header_PD_eligibility_removed_in_separate_reader_only": True, "retry_after_data_failure": False,
        "frozen_parent_preprocess_pin": config["predeclared_preprocess_pin"],
        "remaining_geometry_pixels_encoder_pooling_and_noise": "exact frozen parent contract",
        "possible_T2_or_intermediate_contrast_domain_shift": True, "accuracy_or_adoption_verified": False}
    reverted = copy.deepcopy(config)
    for field in ("fluid_requests", "fluid_source_bindings", "fluid_protocol"):
        reverted.pop(field)
    for field in ("uids", "UID_order_sha256", "predeclared_cohort_pin"):
        reverted[field] = parent["config"][field]
    require(reverted == parent["config"], "Unlisted parent config change")
    bundle = {"runtimes": runtimes,
              "runtime_pins": {name: {"bytes": len(text.encode()), "sha256": hashlib.sha256(text.encode()).hexdigest()} for name, text in runtimes.items()},
              "config": config, "vendor": vendor}
    raw = (json.dumps(bundle, ensure_ascii=True, separators=(",", ":")) + "\n").encode()
    encoded = base64.b85encode(zlib.compress(raw, 9)).decode()
    code = ("# Fixed1080 fluid-sensitive features; no original success overwrite or score submission.\n"
            "import base64 as _ff_b,hashlib as _ff_h,json as _ff_j,zlib as _ff_z\n"
            f"_ff_raw=_ff_z.decompress(_ff_b.b85decode({encoded!r}))\n"
            f"assert _ff_h.sha256(_ff_raw).hexdigest()=={hashlib.sha256(raw).hexdigest()!r}\n"
            "_ff_bundle=_ff_j.loads(_ff_raw)\n_ff_spaces={}\n"
            "for _ff_name,_ff_text in _ff_bundle['runtimes'].items():\n"
            "    _ff_bytes=_ff_text.encode('utf-8')\n"
            "    assert {'bytes':len(_ff_bytes),'sha256':_ff_h.sha256(_ff_bytes).hexdigest()}==_ff_bundle['runtime_pins'][_ff_name]\n"
            "    _ff_spaces[_ff_name]={'__name__':'orthodiff_fluid_'+_ff_name}\n"
            "    exec(compile(_ff_bytes,'<pinned-'+_ff_name+'>','exec'),_ff_spaces[_ff_name])\n"
            "_ff_result=_ff_spaces['fluid']['run'](_ff_bundle['config'],_ff_bundle['vendor'],_ff_spaces['base'],_ff_spaces['pilot'],_ff_spaces['full'])\n"
            "if not _ff_result['complete']:\n    raise RuntimeError('Fluid-sensitive extraction incomplete; inspect separate summary/status/journal')\n")
    markdown = ("# Separate fixed1080 fluid-sensitive fallback cache\n\n"
        "Only the qualified parent's1080 unknown-PD naming failures are requested. The3063 successful parent features remain external and unchanged. "
        "1077 series were chosen once using official metadata (Sagittal/FluidSensitive1, FS1 beforeFS0, thenSeriesUID);3 unavailable requests retain featureless failures. "
        "A separate literal reader removes only its per-header PD-classification/check statements; all UID/SOP/geometry/physical central16/rescale/float64-resize checks remain. "
        "This route is not confirmedPD and may includeT2/intermediate contrasts. Exact baseline decoder fixtures, strictEMA257/13buffers, FP32 t100/mid2/globalmean256/scoped noise/RNG and single model reuse remain. "
        "Separate appendJSONL/status/summary only; no image/labels/Report/fit/upload/submission. No retry or zero features after data failure. "
        "8h wall check is cooperative at UID boundaries, with14GiB CUDA-reserved/8GiB RSS postconditions. Complete classifies all1080; it does not guarantee1080 successful features. "
        "Selection bias, coverage and weakhead utility require later qualified evaluation; no improvement or adoption is claimed.\n")
    notebook = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": markdown},
        {"cell_type": "code", "metadata": {}, "source": code, "outputs": [], "execution_count": None}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "kaggle": {"isGpuEnabled": True, "isInternetEnabled": False}}, "nbformat": 4, "nbformat_minor": 5}
    source = (json.dumps(notebook, indent=2) + "\n").encode()
    require(len(source) < 950000, "Notebook capacity exceeded")
    payload = read(PARENT / "save-payload.json")
    payload.update(slug=slug, new_title=slug.split("/")[1], text=source.decode())
    require(payload["is_private"] is True and payload["enable_gpu"] is True and payload["enable_internet"] is False
            and payload["machine_shape"] == "NvidiaTeslaT4", "Frozen private/offline/T4 contract differs")
    manifest = {"schema_version": "orthodiff_fluid_fallback1080_offline_preparation_v1",
        "source": {"bytes": len(source), "sha256": hashlib.sha256(source).hexdigest()},
        "bundle_pin": {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
        "runtime_pins": bundle["runtime_pins"], "subset_proof": proof, "science_parity": parity, "config": config,
        "vendor_pin": pin(PARENT / "vendor.json"), "immutable_parent_source": pin(PARENT / "source.ipynb"),
        "builder_pin": pin(__file__), "parent_Input_and_environment_preserved": True,
        "output_files": ["orthodiff-fluid-fallback1080/ORTHODIFF_FLUID_FALLBACK_" + name for name in ("RECORDS.jsonl", "STATUS.json", "SUMMARY.json")],
        "local_activity": {"Torch_upstream_model_import_MRI_GPU_training_network_Save_Submit": 0},
        "actual_fallback_extraction_or_head_adoption_qualified": False}
    output.mkdir(parents=True)
    values = {"source.ipynb": source, **{name + "-runtime.py": text.encode() for name, text in runtimes.items()}}
    for name, value in (("save-payload.json", payload), ("manifest.json", manifest), ("config.json", config),
                        ("vendor.json", vendor), ("science-parity.json", parity)):
        values[name] = (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode()
    for name, content in values.items():
        (output / name).write_bytes(content)
    return {"output": str(output), "source": pin(output / "source.ipynb"), "payload": pin(output / "save-payload.json"),
            "manifest": pin(output / "manifest.json"), "subset_proof": proof}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--slug", default="rsraki/orthodiff-sag-fluid1080-20261017-v3")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.slug), indent=2))


if __name__ == "__main__":
    main()
