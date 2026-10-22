"""Offline full4207 package from immutable MRI32 science; never run or upload."""
from __future__ import annotations

import argparse
import ast
import base64
import copy
import csv
import hashlib
import json
import zlib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "artifacts/kaggle/orthodiff-sag-mri32-pilot-20261017-v1"
RESEARCH = ROOT / "artifacts/research/20261017-orthodiff-fullweak-preparation-v1"
OUTPUT = ROOT / "artifacts/kaggle/orthodiff-sag-fullweak4207-20261017-v2"
MANIFEST = ROOT / "data/manifests/v1-vmohitrao-research"
ID = "StudyInstanceUID"
PILOT_SOURCE_SHA = "66fb119becbaa0cc0bd1eeb76f52402ff14906d3961f98c6b95b6e4c8c825fe3"
RUNTIME_SHA = {"base": "d0b432859d6fcae23192cd31a86330dc92ab3052a1b06253f9cfb7f82b5afe90",
               "pilot": "4e8d671d6b0cf4402afac0c6c4dc3a3a007bb45878a6c40e7340159762244a1a"}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def pin(path):
    raw = Path(path).read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def read(path):
    return json.loads(Path(path).read_bytes())


def projected_rows(path, fields):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        indices = [(name, header.index(name)) for name in fields]
        return [{name: row[index] for name, index in indices} for row in reader]


def load_cohort():
    cohort, audit, alignment = (read(RESEARCH / name) for name in ("cohort.json", "metadata-audit.json", "alignment-audit.json"))
    require(audit["passed"] is True and alignment["passed"] is True, "Metadata/alignment evidence required")
    require(pin(RESEARCH / "cohort.json") == alignment["cohort_pin"], "Aligned cohort byte pin changed")
    require(all(pin(MANIFEST / name) == value for name, value in cohort["manifest_pins"].items()), "Frozen weak/fold/Gold metadata pins differ")
    fields = (ID, "group_id", "fold", "source")
    rows = projected_rows(MANIFEST / "weak.csv", fields)
    require(rows == cohort["requested"], "Full original weak row order/mapping changed")
    uids = [row[ID] for row in rows]
    require(len(uids) == len(set(uids)) == cohort["requested_count"] == 4207, "Exactly4207 unique weak UIDs required")
    require(hashlib.sha256("".join(x + "\n" for x in uids).encode()).hexdigest() == cohort["UID_order_sha256"], "UID order hash differs")
    mapping = [(row[ID], row["group_id"], row["fold"]) for row in rows]
    require(hashlib.sha256(json.dumps(mapping, separators=(",", ":")).encode()).hexdigest() == cohort["UID_group_fold_sha256"] == alignment["UID_group_fold_sha256"], "Original UID/group/fold map differs")
    groups = defaultdict(set)
    for row in rows:
        require(row["source"] == "report_weak" and row["fold"] in {"0", "1", "2", "3", "4"}, "Weak source/fold differs")
        groups[row["group_id"]].add(row["fold"])
    require(all(len(folds) == 1 for folds in groups.values()), "Group split across folds")
    gold = projected_rows(MANIFEST / "gold.csv", (ID, "group_id"))
    require(not set(uids) & {row[ID] for row in gold} and not set(groups) & {row["group_id"] for row in gold}, "Gold UID/group overlap")
    require(dict(Counter(row["fold"] for row in rows)) == cohort["fold_counts"], "Fixed fold sizes differ")
    index = ROOT / "data/native392/v2-20261009/full-index.json"
    require(pin(index) == alignment["native_index_pin"] and set(read(index)["study_ids"]) == set(uids), "Native all4207 input alignment changed")
    control = ROOT / "artifacts/runs/f006-late-cosine-fold0-20261012-v1/epoch_012_weak.csv"
    require(pin(control) == alignment["F006_E12_predictions_pin"], "F006 E12 control prediction pin differs")
    control_ids = [row[ID] for row in projected_rows(control, (ID,))]
    require(len(control_ids) == len(set(control_ids)) == 821 and set(control_ids) == {row[ID] for row in rows if row["fold"] == "0"}, "F006 E12 exact fold0 UID set differs")
    return cohort, alignment


def science_parity(pilot, full):
    """The entire normalized-input/seed/FP32/model/RNG block is byte-identical."""
    start = '                        x = ((x - lo) / (hi - lo + 1e-8) * 2 - 1)'
    end_old = '                        state["records"].append({"StudyInstanceUID": uid, "status": "success"'
    end_new = '                        journal.append({"StudyInstanceUID": uid, "status": "success"'
    old = pilot[pilot.index(start):pilot.index(end_old)]
    new = full[full.index(start):full.index(end_new)]
    require(old == new, "Frozen normalization/seed/model/RNG numerical block differs")
    old_record = pilot[pilot.index(end_old):pilot.index('                        del x, feature, pooled, array')]
    new_record = full[full.index(end_new):full.index('                        del x, feature, pooled, array')]
    old_tree, new_tree = ast.parse(old_record.strip()), ast.parse(new_record.strip())
    require(ast.dump(old_tree.body[0].value.args[0], include_attributes=False) == ast.dump(new_tree.body[0].value.args[0], include_attributes=False), "Successful scientific record content differs")
    return {"frozen_pilot_helpers_loaded_by_exact_runtime_pin": True,
            "numerical_forward_block_byte_identical": True, "numerical_forward_block_sha256": hashlib.sha256(old.encode()).hexdigest(),
            "success_record_dictionary_AST_identical": True,
            "explicit_PD_series_geometry_continuous16_float64_resize_strictEMA_eval_RNG_changed": False,
            "operational_only_changes": ["all4207 UIDs including nonsingleton groups", "index metadata once preserving withinUID series order",
                                         "durable append-only records JSONL with incremental digest", "small status at every UID",
                                         "small flushed stdout every32 records and model-loaded/terminal events",
                                         "single terminal summary and final journal rehash", "8h cooperative/posthoc full-cohort wall budget"]}


def prepare(output=OUTPUT, slug="rsraki/orthodiff-sag-fullweak4207-20261017-v2"):
    output = Path(output)
    require(not output.exists() and not output.is_symlink(), "Use a fresh fullweak package")
    require(pin(PILOT / "source.ipynb")["sha256"] == PILOT_SOURCE_SHA, "Immutable baseline notebook differs")
    parent = read(PILOT / "manifest.json")
    require(parent["source"] == pin(PILOT / "source.ipynb"), "Baseline source manifest pin differs")
    runtimes = {name: (PILOT / ("base-runtime.py" if name == "base" else "pilot-runtime.py")).read_bytes().decode() for name in RUNTIME_SHA}
    for name, source in runtimes.items():
        require(hashlib.sha256(source.encode()).hexdigest() == RUNTIME_SHA[name], "Frozen runtime differs: " + name)
        require({"bytes": len(source.encode()), "sha256": RUNTIME_SHA[name]} == parent["runtime_pins"][name], "Runtime size/pin differs")
    runtimes["full"] = (ROOT / "scripts/orthodiffusion_fullweak_runtime.py").read_bytes().decode()
    parity = science_parity(runtimes["pilot"], runtimes["full"])
    for name, text in runtimes.items():
        compile(ast.parse(text), name, "exec")
    cohort, alignment = load_cohort()
    config, vendor = read(PILOT / "config.json"), read(PILOT / "vendor.json")
    require(config == parent["config"], "Frozen parent config differs")
    config = copy.deepcopy(config)
    config["uids"] = [row[ID] for row in cohort["requested"]]
    config["UID_order_sha256"] = cohort["UID_order_sha256"]
    config["predeclared_cohort_pin"] = pin(RESEARCH / "cohort.json")
    config["operational_resource_limits"] = {"CUDA_reserved_bytes": 14 * 1024**3, "process_RSS_bytes": 8 * 1024**3,
        "total_seconds": 28800, "UID_boundary_cooperative_clock_check": True, "hard_C_decoder_timeout": False}
    config["frozen_pilot_pins"] = {"source": parent["source"], "runtime_pins": parent["runtime_pins"],
        "config": pin(PILOT / "config.json"), "vendor": pin(PILOT / "vendor.json"),
        "preprocess": parent["config"]["predeclared_preprocess_pin"]}
    original = copy.deepcopy(config)
    for field in ("operational_resource_limits", "frozen_pilot_pins"):
        original.pop(field)
    for field in ("uids", "UID_order_sha256", "predeclared_cohort_pin"):
        original[field] = parent["config"][field]
    require(original == parent["config"], "Only cohort/control transport fields may differ")
    require(config["series_metadata_pin"] == cohort["series_metadata_pin"], "Series metadata pin differs")
    bundle = {"runtimes": runtimes,
        "runtime_pins": {name: {"bytes": len(text.encode()), "sha256": hashlib.sha256(text.encode()).hexdigest()} for name, text in runtimes.items()},
        "config": config, "vendor": vendor}
    raw = (json.dumps(bundle, ensure_ascii=True, separators=(",", ":")) + "\n").encode()
    encoded = base64.b85encode(zlib.compress(raw, 9)).decode()
    code = ("# Full4207 frozen continuous16 weak feature cache; never submit for scoring.\n"
            "import base64 as _of_b64,hashlib as _of_h,json as _of_j,zlib as _of_z\n"
            f"_of_raw=_of_z.decompress(_of_b64.b85decode({encoded!r}))\n"
            f"assert _of_h.sha256(_of_raw).hexdigest()=={hashlib.sha256(raw).hexdigest()!r}\n"
            "_of_bundle=_of_j.loads(_of_raw)\n_of_spaces={}\n"
            "for _of_name,_of_text in _of_bundle['runtimes'].items():\n"
            "    _of_bytes=_of_text.encode('utf-8')\n"
            "    assert {'bytes':len(_of_bytes),'sha256':_of_h.sha256(_of_bytes).hexdigest()}==_of_bundle['runtime_pins'][_of_name]\n"
            "    _of_spaces[_of_name]={'__name__':'orthodiff_full_'+_of_name}\n"
            "    exec(compile(_of_bytes,'<pinned-'+_of_name+'>','exec'),_of_spaces[_of_name])\n"
            "_of_result=_of_spaces['full']['run'](_of_bundle['config'],_of_bundle['vendor'],_of_spaces['base'],_of_spaces['pilot'])\n"
            "if not _of_result['complete']:\n    raise RuntimeError('Fullweak extraction incomplete; inspect summary/status/journal')\n")
    markdown = ("# Full4207 OrthoDiffusion frozen-baseline feature cache\n\n"
        "All original4207 weak UIDs in original manifest row order, including whole supplied/report groups. The original5fold mapping and reserved Gold UID/group exclusions remain local metadata. Runtime embeds onlyUIDs/protocol/pins. "
        "Exact frozen32 scientific helpers, explicit PD eligibility, series priority, physical central continuous16, float64 resize then FP32 volume minmax, strict EMA and scoped t100/mid_2/global256mean are unchanged. "
        "One frozen model serves all cases. Only durable records JSONL, small progress JSON and terminal summary JSON are exported; no images, labels, Report, training or submission. Missing cases keep failures with no feature rows. "
        "The8h wall limit is cooperative at UID boundaries and checked after execution; it does not preempt C decoders. Complete means all4207 requests classified, not all4207 features succeeded. No author-orientation equivalence, accuracy, leaderboard improvement or hidden capacity is established.\n")
    notebook = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": markdown},
        {"cell_type": "code", "metadata": {}, "source": code, "outputs": [], "execution_count": None}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "kaggle": {"isGpuEnabled": True, "isInternetEnabled": False}}, "nbformat": 4, "nbformat_minor": 5}
    source = (json.dumps(notebook, indent=2) + "\n").encode()
    require(len(source) < 950000, "Notebook source exceeds capacity")
    payload = read(PILOT / "save-payload.json")
    payload.update(slug=slug, new_title=slug.split("/")[1], text=source.decode())
    require(payload["is_private"] is True and payload["enable_gpu"] is True and payload["enable_internet"] is False
            and payload["machine_shape"] == "NvidiaTeslaT4", "Frozen private/offline/T4 environment differs")
    evidence = {"schema_version": "orthodiff_full4207_cache_offline_preparation_v1", "source": {"bytes": len(source), "sha256": hashlib.sha256(source).hexdigest()},
        "bundle_pin": {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}, "runtime_pins": bundle["runtime_pins"],
        "cohort_pin": pin(RESEARCH / "cohort.json"), "alignment_pin": pin(RESEARCH / "alignment-audit.json"),
        "cohort": {key: cohort[key] for key in ("requested_count", "UID_order_sha256", "UID_group_fold_sha256", "fold_counts", "group_count", "non_singleton_group_count", "non_singleton_study_count", "Gold_UID_overlap", "Gold_supplied_group_overlap")},
        "F006_E12_prediction_UID_exact_set_821": alignment["F006_E12_predictions_pin"],
        "native_cache_UID_exact_set_4207": alignment["native_index_pin"],
        "science_parity": parity, "config": config, "vendor_pin": pin(PILOT / "vendor.json"),
        "output_files": ["orthodiff-fullweak4207/ORTHODIFF_FULLWEAK_" + name for name in ("RECORDS.jsonl", "STATUS.json", "SUMMARY.json")],
        "local_activity": {"scientific_source_or_Torch_import_MRI_model_GPU_network_Save_Submit": 0},
        "actual_full4207_extraction_or_accuracy_qualified": False}
    output.mkdir(parents=True)
    values = {"source.ipynb": source, **{name + "-runtime.py": text.encode() for name, text in runtimes.items()}}
    for name, value in (("save-payload.json", payload), ("manifest.json", evidence), ("config.json", config), ("vendor.json", vendor)):
        values[name] = (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode()
    for name, content in values.items():
        (output / name).write_bytes(content)
    return {"output": str(output), "source": pin(output / "source.ipynb"), "payload": pin(output / "save-payload.json"), "manifest": pin(output / "manifest.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--slug", default="rsraki/orthodiff-sag-fullweak4207-20261017-v2")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.slug), indent=2))


if __name__ == "__main__":
    main()
