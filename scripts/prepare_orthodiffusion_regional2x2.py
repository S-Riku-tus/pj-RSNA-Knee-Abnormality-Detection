"""Offline pooling-only regional1024 cache pack; never run/download/upload."""
from __future__ import annotations

import argparse
import ast
import base64
import copy
import difflib
import hashlib
import json
import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "artifacts/kaggle/orthodiff-sag-fullweak4207-20261017-v2"
AUDIT = ROOT / "artifacts/research/20261017-orthodiff-fullweak-actual-audit-v1/verification.json"
SPEC = ROOT / "artifacts/research/20261017-orthodiff-regional2x2-preparation-v1/preregistered.json"
OUTPUT = ROOT / "artifacts/kaggle/orthodiff-sag-regional2x2-20261017-v1"
PARENT_SHA = "15bb2d6c7c224021f95a304b08eb9e3644164ea8373f4ae6ec1c7afc77582e1f"
AUDIT_SHA = "7fbc160b8715e69dc506b64e9187f55a2c2b67925bf6a5b0a4a95adf943c8270"
SPEC_SHA = "dccf1132a863173f4e0d7ffbfc929d44753b2fdbd5460810dd31fa33f2d4b377"
RUNTIME_SHA = {"base": "d0b432859d6fcae23192cd31a86330dc92ab3052a1b06253f9cfb7f82b5afe90",
    "pilot": "4e8d671d6b0cf4402afac0c6c4dc3a3a007bb45878a6c40e7340159762244a1a",
    "full": "a7c2a3a6c42a26f0b8f647842e88d47e78b9cc4baf259c67b9ec8278dd268ee6"}
INPUT = "rsraki/orthodiff-sag-fullweak4207-20261017-v2/1"


def require(value, message):
    if not value:
        raise ValueError(message)


def pin(path):
    digest, size = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def read(path):
    return json.loads(Path(path).read_bytes())


def science_parity(full, regional):
    start = '                        x = ((x - lo) / (hi - lo + 1e-8) * 2 - 1)'
    old_end = '                        journal.append({"StudyInstanceUID": uid, "status": "success"'
    new_end = '                        record = {"StudyInstanceUID": uid, "status": "success"'
    old, new = full[full.index(start):full.index(old_end)], regional[regional.index(start):regional.index(new_end)]
    require(old == new, "Frozen full normalization/FP32/t100/forward/globalpool/RNG block differs")
    old_record = full[full.index(old_end):full.index('                        del x, feature, pooled, array')].strip()
    new_record = regional[regional.index(new_end):regional.index('                        require_parent_success(record, parent)')].strip()
    require(ast.dump(ast.parse(old_record).body[0].value.args[0], include_attributes=False)
            == ast.dump(ast.parse(new_record).body[0].value, include_attributes=False), "Original scientific success dictionary changed")
    old_tree, new_tree = ast.parse(full), ast.parse(regional)
    # The literal per-UID preprocessing success body is unchanged; only its failure
    # behavior becomes fatal because original3063 success coverage must stay fixed.
    before = '                            selected, paths = choose_series(uid, series_by_uid[uid], image_roots[0], pydicom)'
    after = '                        except (MemoryError, SystemError):'
    require(full[full.index(before):full.index(after)] == regional[regional.index(before):regional.index(after)],
            "Original PD selector/reader/float32 conversion changed")
    require(not any(isinstance(node, (ast.Import, ast.ImportFrom)) and "fluid" in ast.unparse(node) for node in ast.walk(new_tree)),
            "FD route must remain separate")
    original_class = next(node for node in old_tree.body if isinstance(node, ast.ClassDef) and node.name == "RecordJournal")
    return {"numeric_normalization_forward_global256_RNG_block_byte_identical": True,
        "numeric_block_sha256": hashlib.sha256(old.encode()).hexdigest(), "original_success_dictionary_AST_identical": True,
        "original_PD_selector_reader_float32_body_byte_identical": True,
        "old_RecordJournal_exact_class_reused": True, "old_RecordJournal_AST_sha256": hashlib.sha256(ast.dump(original_class, include_attributes=False).encode()).hexdigest(),
        "changes": ["guard additional fixedV1 parent output before MRI", "inherit1144 failures before image/header/geometry branch",
            "recompute3063 successes; new data failure or any scientific mismatch fatal", "additional deterministic fixed2x2 mean then concat1024 after original global/RNG block",
            "regional1024 finite/FP32/hash validation; retain original256global for all-parent parity", "separate namespace/output/schema/stdout/stderr"],
        "old_base_pilot_full_vendor_science_changed": False, "FD_mixed": False}


def parent_input(config):
    require(pin(AUDIT)["sha256"] == AUDIT_SHA and pin(SPEC)["sha256"] == SPEC_SHA, "Pinned actual qualification/spec differs")
    audit, spec = read(AUDIT), read(SPEC)
    require(audit.get("passed") is True and audit.get("head_features_qualified") is True
        and audit.get("source_sha256") == PARENT_SHA and audit.get("kernel_id") == 139150600 and audit.get("version") == 1
        and audit.get("requested_count") == 4207 and audit.get("prior32_comparison", {}).get("passed") is True,
        "Actual complete fixedV1 parent qualification required")
    paths = {Path(path).name: Path(path) for path in audit["input_pins"] if Path(path).name.startswith("ORTHODIFF_FULLWEAK_")}
    names = {"ORTHODIFF_FULLWEAK_SUMMARY.json": "summary_pin", "ORTHODIFF_FULLWEAK_STATUS.json": "status_pin", "ORTHODIFF_FULLWEAK_RECORDS.jsonl": "records_pin"}
    require(set(paths) == set(names), "Exact parent3output files required")
    members = {name: audit[key] for name, key in names.items()}
    for name, path in paths.items():
        require(pin(path) == members[name], "Actual parent output pin differs: " + name)
    summary, status = read(paths["ORTHODIFF_FULLWEAK_SUMMARY.json"]), read(paths["ORTHODIFF_FULLWEAK_STATUS.json"])
    require(summary.get("complete") is True and status.get("complete") is True and "error" not in summary
        and summary["phase"] == status["phase"] == "full4207_finished" and summary["coverage"] == status["coverage"] == audit["coverage"]
        and summary["cohort_UID_hash"] == config["UID_order_sha256"] == audit["UID_order_sha256"] and summary["requested_UIDs"] == config["uids"],
        "Parent complete/order/source binding differs")
    success_ids, failure_ids, count = [], [], 0
    with paths["ORTHODIFF_FULLWEAK_RECORDS.jsonl"].open("rb") as stream:
        for count, line in enumerate(stream, start=1):
            record = json.loads(line)
            require(count <= 4207 and record["StudyInstanceUID"] == config["uids"][count - 1]
                and (json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n").encode() == line,
                "Exact original parent journal UID order/record bytes required")
            if record["status"] == "success":
                values = record["pooled_float32_values"]
                require(len(values) == 256 and list(struct.unpack("<256f", struct.pack("<256f", *values))) == values,
                    "Parent success FP32 vector differs")
                success_ids.append(record["StudyInstanceUID"])
            else:
                require(record["status"] == "data_failure" and record["feature_forward_calls"] == 0 and record["fallback_features"] is False
                    and not any(key in record for key in ("pooled", "pooled_float32_values", "mid_2", "normalized_input")), "Parent failure feature contract differs")
                failure_ids.append(record["StudyInstanceUID"])
    require(count == 4207 and len(success_ids) == 3063 and len(failure_ids) == 1144, "Exact3063/1144 partition required")
    for key, uids in (("success_UID_order_sha256", success_ids), ("failure_UID_order_sha256", failure_ids)):
        require(hashlib.sha256("".join(uid + "\n" for uid in uids).encode()).hexdigest() == audit[key], "Parent partition UID pin differs")
    for item in spec["evaluation_contract_reference"].values():
        if isinstance(item, dict) and "path" in item:
            require(pin(ROOT / item["path"]) == {key: item[key] for key in ("bytes", "sha256")}, "Frozen evaluation policy byte pin differs")
    fixed = {"Input": INPUT, "candidates": ["/kaggle/input/notebooks/rsraki/orthodiff-sag-fullweak4207-20261017-v2/orthodiff-fullweak4207",
        "/kaggle/input/orthodiff-sag-fullweak4207-20261017-v2/orthodiff-fullweak4207"], "members": members,
        "qualification_pin": pin(AUDIT), "coverage": audit["coverage"],
        "success_UID_order_sha256": audit["success_UID_order_sha256"], "failure_UID_order_sha256": audit["failure_UID_order_sha256"]}
    proof = {"preregistered_spec": pin(SPEC), "parent_qualification": pin(AUDIT), "parent_source_sha256": PARENT_SHA,
        "parent_journal_fully_consumed": count, "success_recomputed": len(success_ids), "failure_inherited_without_image_access": len(failure_ids),
        "UID_order_sha256": audit["UID_order_sha256"], "FD_availability_mixed": False,
        "dimension_only": "global256->regional1024, original global256 retained for parity", "new_model_or_preprocessing_or_head_fit": False}
    return fixed, proof


def prepare(output=OUTPUT, slug="rsraki/orthodiff-sag-regional2x2-20261017-v1"):
    output = Path(output)
    require(not output.exists() and not output.is_symlink(), "Use a fresh regional2x2 output pack")
    parent = read(PARENT / "manifest.json")
    require(pin(PARENT / "source.ipynb") == parent["source"] and parent["source"]["sha256"] == PARENT_SHA, "Immutable full parent source differs")
    runtimes = {name: (PARENT / (name + "-runtime.py")).read_bytes().decode() for name in RUNTIME_SHA}
    for name, text in runtimes.items():
        require(pin(PARENT / (name + "-runtime.py")) == parent["runtime_pins"][name]
            and hashlib.sha256(text.encode()).hexdigest() == RUNTIME_SHA[name], "Frozen runtime differs: " + name)
    runtimes["regional"] = (ROOT / "scripts/orthodiffusion_regional2x2_runtime.py").read_bytes().decode()
    parity = science_parity(runtimes["full"], runtimes["regional"])
    for name, text in runtimes.items():
        compile(ast.parse(text), name, "exec")
    config, vendor = read(PARENT / "config.json"), read(PARENT / "vendor.json")
    require(config == parent["config"] and pin(PARENT / "vendor.json") == parent["vendor_pin"], "Frozen parent config/vendor differs")
    fixed, proof = parent_input(config)
    config = copy.deepcopy(config)
    config["regional_parent_input"], config["regional_source_bindings"] = fixed, proof
    config["regional_protocol"] = {"input_mid2": [1, 256, 1, 16, 16], "pool_shape": [1, 1024], "pool_dtype": "torch.float32",
        "order": ["TL", "TR", "BL", "BR"], "region_shape_DHW": [1, 8, 8], "pool": "FP32 per-region meanDHW, concatenate along channels",
        "parent_global256_retained_for_exact_parity": True, "success_coverage_fixed": 3063, "failure_coverage_inherited": 1144,
        "new_data_failure_or_parent_parity_mismatch": "fatal whole-run disqualification", "original_preprocess_pin": config["predeclared_preprocess_pin"],
        "rawmid2_SAP_FD_or_training": False, "declared_after_failed_global256_fold1": True}
    restored = copy.deepcopy(config)
    for key in ("regional_parent_input", "regional_source_bindings", "regional_protocol"):
        restored.pop(key)
    require(restored == parent["config"], "Original config changed beyond regional control additions")
    bundle = {"runtimes": runtimes, "runtime_pins": {name: {"bytes": len(text.encode()), "sha256": hashlib.sha256(text.encode()).hexdigest()} for name, text in runtimes.items()},
        "config": config, "vendor": vendor}
    raw = (json.dumps(bundle, ensure_ascii=True, separators=(",", ":")) + "\n").encode()
    encoded = base64.b85encode(zlib.compress(raw, 9)).decode()
    code = ("# Pooling-only2x2regional1024, exact parent parity required; no scoring submission.\n"
        "import base64 as _r2_b,hashlib as _r2_h,json as _r2_j,zlib as _r2_z\n"
        f"_r2_raw=_r2_z.decompress(_r2_b.b85decode({encoded!r}))\n"
        f"assert _r2_h.sha256(_r2_raw).hexdigest()=={hashlib.sha256(raw).hexdigest()!r}\n"
        "_r2_bundle=_r2_j.loads(_r2_raw)\n_r2_spaces={}\n"
        "for _r2_name,_r2_text in _r2_bundle['runtimes'].items():\n"
        "    _r2_bytes=_r2_text.encode('utf-8')\n"
        "    assert {'bytes':len(_r2_bytes),'sha256':_r2_h.sha256(_r2_bytes).hexdigest()}==_r2_bundle['runtime_pins'][_r2_name]\n"
        "    _r2_spaces[_r2_name]={'__name__':'orthodiff_regional_'+_r2_name}\n"
        "    exec(compile(_r2_bytes,'<pinned-'+_r2_name+'>','exec'),_r2_spaces[_r2_name])\n"
        "_r2_result=_r2_spaces['regional']['run'](_r2_bundle['config'],_r2_bundle['vendor'],_r2_spaces['base'],_r2_spaces['pilot'],_r2_spaces['full'])\n"
        "if not _r2_result['complete']:\n    raise RuntimeError('Regional2x2 run unqualified; inspect separate summary/status/journal')\n")
    markdown = ("# Pooling-only fixed2x2regional1024\n\n"
        "A hypothesis declared after the failed global256 fold1 evaluation: fixed grid quadrants may retain local information diluted by global averaging. "
        "Only original3063 qualified successes are recomputed;1144 original failures are inherited unchanged without header/pixel access. "
        "Every original selected series/full geometry/16SOP/source hash/normalized-noisy-mid2-global tensor pin/global256 bits must match before recording new features; any mismatch or newly failed success makes the whole run incomplete. "
        "Exact original strictEMA257/13buffers, decoder fixtures, physical continuous16/float64resize/FP32 minmax/t100/noAMP/UIDnoise/singleforward/RNG remain. "
        "From the same mid2[1,256,1,16,16], mean allD and each8x8 quadrant inTL/TR/BL/BR order, then concatenate1024. Extra pooling changes no model or RNG. "
        "Additional fourth Input is fixedV1 parent output; other three Inputs/privateofflineT4/Docker remain. Separate appendJSONL/status/summary only, no images/rawmid2/SAP/FD/fit. "
        "The future fixedC1/scaler/seed/fold0-to1/rank20/fixed10/both-positive/report-onlyCI policy is referenced without changing current head code. Coverage/adoption/accuracy/Public improvement are not yet qualified.\n")
    notebook = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": markdown},
        {"cell_type": "code", "metadata": {}, "source": code, "outputs": [], "execution_count": None}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "kaggle": {"isGpuEnabled": True, "isInternetEnabled": False}},
        "nbformat": 4, "nbformat_minor": 5}
    source = (json.dumps(notebook, indent=2) + "\n").encode()
    require(len(source) < 950000, "Notebook source exceeds capacity")
    payload = read(PARENT / "save-payload.json")
    require(INPUT not in payload["kernel_data_sources"], "New parent input must be added exactlyonce")
    payload["kernel_data_sources"].append(INPUT)
    payload.update(slug=slug, new_title=slug.split("/")[1], text=source.decode())
    require(len(payload["dataset_data_sources"] + payload["kernel_data_sources"] + payload["competition_data_sources"]) == 4
        and payload["is_private"] is True and payload["enable_gpu"] is True and payload["enable_internet"] is False
        and payload["machine_shape"] == "NvidiaTeslaT4", "Fixed4Input/private/offline/T4 contract differs")
    manifest = {"schema_version": "orthodiff_regional2x2_offline_preparation_v1", "source": {"bytes": len(source), "sha256": hashlib.sha256(source).hexdigest()},
        "bundle_pin": {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}, "runtime_pins": bundle["runtime_pins"],
        "parent_proof": proof, "science_parity": parity, "config": config, "vendor_pin": pin(PARENT / "vendor.json"),
        "preregistered_spec": pin(SPEC), "builder_pin": pin(__file__), "original_parent_source": parent["source"],
        "output_files": ["orthodiff-regional2x2/ORTHODIFF_REGIONAL2X2_" + name for name in ("RECORDS.jsonl", "STATUS.json", "SUMMARY.json")],
        "future_head_input": "regional_2x2_float32_values1024; original pooled_float32_values256 is retained only as parent parity evidence",
        "local_activity": {"MRI_model_Torch_GPU_fit_network_Save_Submit": 0}, "actual_regional_features_or_head_adoption_qualified": False}
    output.mkdir(parents=True)
    values = {"source.ipynb": source, **{name + "-runtime.py": text.encode() for name, text in runtimes.items()},
        "parent-full-to-regional.diff": "".join(difflib.unified_diff(runtimes["full"].splitlines(keepends=True), runtimes["regional"].splitlines(keepends=True), fromfile="immutable-full-runtime.py", tofile="regional-runtime.py")).encode()}
    for name, value in (("save-payload.json", payload), ("manifest.json", manifest), ("config.json", config), ("vendor.json", vendor), ("science-parity.json", parity)):
        values[name] = (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode()
    values["preregistered.json"] = SPEC.read_bytes()
    for name, content in values.items():
        (output / name).write_bytes(content)
    return {"output": str(output), "source": pin(output / "source.ipynb"), "payload": pin(output / "save-payload.json"),
        "manifest": pin(output / "manifest.json"), "parent_proof": proof}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--slug", default="rsraki/orthodiff-sag-regional2x2-20261017-v1")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.slug), indent=2))


if __name__ == "__main__":
    main()
