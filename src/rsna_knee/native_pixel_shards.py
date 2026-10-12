"""Whole-group native392 cache shards and a strict, streaming local index builder.

The immutable pixel16 reader/array contract remains authoritative. This module
changes the export inventory only; it never reads weak/gold targets or reports.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import time
import zipfile
from collections import defaultdict
from pathlib import Path

from .contracts import ID, SERIES_COLUMNS, SERIES_ID, read_csv, sha256, uid
from .feature_contracts import json_hash, new_directory, selection_structure_hash
from .triplet_cache import (
    ARRAY_NAMES,
    PIXEL_SCHEMA,
    array_hashes,
    decode_pixel_member,
    encode_pixel_member,
    extract_pixel_study,
    pixel_contract,
    raw_pixel_bytes,
)

PLAN_SCHEMA = "native392_whole_group_shard_plan_v1"
EXPORT_SCHEMA = "native392_pixel_shard_export_v1"
INDEX_SCHEMA = "native392_pixel_archive_index_v1"
SUMMARY_NAME = "NATIVE392_PIXEL_SHARD_SUMMARY.json"
ORIGINAL_DOCKER = (
    "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"
)
EXECUTION_ENVIRONMENT = {
    "accelerator": "CPU",
    "internet": False,
    "private": True,
    "docker_image": ORIGINAL_DOCKER,
    "docker_image_pinning": "original",
    "competition": "rsna-knee-abnormality-detection",
    "decoder_dataset": "rsraki/rsna-dicom-decoders-py313-v1/2",
}
STRUCTURE_FIELDS = {ID, "group_id", "fold", "source"}
DEFAULT_BUDGET = {
    "output_disk_bytes": 20_000_000_000,
    "archive_bytes": 12 * 1024**3,
    "native_temp_bytes": 4 * 1024**3,
    "helper_metadata_reserve_bytes": 1024**3,
    "per_study_container_overhead_bytes": 1024**2,
    "max_seconds": 8 * 3600,
}


def shard_implementation_hashes():
    here = Path(__file__).resolve().parent
    return {
        name: hashlib.sha256((here / name).read_text(encoding="utf-8").replace("\r\n", "\n").encode()).hexdigest()
        for name in ("native_pixel_shards.py", "triplet_cache.py")
    }


def validate_structure(rows, gold_ids, gold_groups, n_folds):
    seen, groups = set(), defaultdict(list)
    for row in rows:
        if set(row) != STRUCTURE_FIELDS or row["source"] != "report_weak":
            raise ValueError("Only weak UID/group/fold/source structure may enter the pixel plan")
        key = uid(row[ID])
        if key in seen:
            raise ValueError("Duplicate weak UID")
        seen.add(key)
        if key in gold_ids or row["group_id"] in gold_groups:
            raise ValueError("Gold UID/group overlaps the weak pixel plan")
        if not row["group_id"] or type(row["fold"]) is not int or not 0 <= row["fold"] < n_folds:
            raise ValueError("Invalid structural group/fold")
        groups[row["group_id"]].append(row)
    if not seen or any(len({row["fold"] for row in members}) != 1 for members in groups.values()):
        raise ValueError("Empty weak inventory or a whole group crosses folds")
    return groups


def assignment(rows, seed, shard_count):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    if type(shard_count) is not int or not 1 <= shard_count <= len(groups):
        raise ValueError("Positive shard count cannot exceed whole groups")
    ranked = sorted(
        groups,
        key=lambda group: (-len(groups[group]), hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(), group),
    )
    shards = [[] for _ in range(shard_count)]
    for group in ranked:
        selected = min(range(shard_count), key=lambda index: (len(shards[index]), index))
        shards[selected].extend(groups[group])
    return [sorted(members, key=lambda row: row[ID]) for members in shards]


def make_plan(rows, gold_ids, gold_groups, *, seed, n_folds, settings, manifest_sha256, shard_count=24, budgets=None):
    rows = [dict(row) for row in rows]
    groups = validate_structure(rows, set(gold_ids), set(gold_groups), n_folds)
    budget = dict(DEFAULT_BUDGET if budgets is None else budgets)
    if set(budget) != set(DEFAULT_BUDGET) or any(type(value) is not int or value <= 0 for value in budget.values()):
        raise ValueError("Positive fixed disk/time budgets are required")
    if budget["output_disk_bytes"] > 20_000_000_000 or budget["max_seconds"] > 8 * 3600:
        raise ValueError("Do not exceed the conservative20GB/eight-hour job budgets")
    if (
        budget["archive_bytes"] + budget["native_temp_bytes"] + budget["helper_metadata_reserve_bytes"]
        > budget["output_disk_bytes"]
    ):
        raise ValueError("Archive/native/helper budgets exceed the conservative20GB output budget")
    shards = []
    for index, members in enumerate(assignment(rows, seed, shard_count)):
        raw = raw_pixel_bytes(len(members), settings)
        overhead = (len(members) + 1) * budget["per_study_container_overhead_bytes"]
        if raw + overhead >= budget["archive_bytes"]:
            raise ValueError("A whole-group shard exceeds the raw plus NPZ/ZIP overhead archive budget; add shards")
        shards.append(
            {
                "index": index,
                "name": f"native392-shard-{index + 1:02d}-of-{shard_count:02d}",
                "studies": members,
                "study_count": len(members),
                "group_count": len({row["group_id"] for row in members}),
                "raw_pixel_upper_bytes": raw,
                "container_overhead_budget_bytes": overhead,
                "cohort_sha256": selection_structure_hash(members),
            }
        )
    contract = pixel_contract(settings)
    return {
        "schema_version": PLAN_SCHEMA,
        "seed": seed,
        "n_folds": n_folds,
        "split": "train",
        "limited": False,
        "studies": sorted(rows, key=lambda row: row[ID]),
        "study_count": len(rows),
        "group_count": len(groups),
        "gold_ids": sorted(set(gold_ids)),
        "gold_groups": sorted(set(gold_groups)),
        "manifest_sha256": manifest_sha256,
        "selection_structure_sha256": selection_structure_hash(rows),
        "contract": contract,
        "fingerprint": json_hash(contract),
        "source_sha256": shard_implementation_hashes(),
        "assignment": "LPT by whole-group count; seed:group SHA256 ascending ties; minimum study load then shard index",
        "shard_count": shard_count,
        "shards": shards,
        "budgets": budget,
        "targets_used_for_selection": False,
        "patient_independence_verified": False,
    }


def validate_plan(plan):
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("split") != "train" or plan.get("limited") is not False:
        raise ValueError("Only a full training structure plan is supported")
    expected = make_plan(
        plan["studies"],
        plan["gold_ids"],
        plan["gold_groups"],
        seed=plan["seed"],
        n_folds=plan["n_folds"],
        settings=plan["contract"]["preprocess"],
        manifest_sha256=plan["manifest_sha256"],
        shard_count=plan["shard_count"],
        budgets=plan["budgets"],
    )
    if plan != expected:
        raise ValueError("Shard plan/source/contract/coverage or deterministic whole-group assignment drifted")
    return plan


def atomic_json(path, value, *, new=False):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x" if new else "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if new:
        # Atomic create without replacing another process's completed index.
        os.link(temporary, path)
        temporary.unlink()
    else:
        os.replace(temporary, path)


def array_metadata(arrays):
    return {name: {"shape": list(arrays[name].shape), "dtype": str(arrays[name].dtype)} for name in ARRAY_NAMES}


def _check_binding(binding):
    if set(binding) != {"code_zip_sha256", "code_member_sha256", "environment"}:
        raise ValueError("Explicit code ZIP/member/environment binding is required")
    if not isinstance(binding["code_zip_sha256"], str) or len(binding["code_zip_sha256"]) != 64:
        raise ValueError("Malformed code ZIP binding")
    if not binding["code_member_sha256"] or binding["environment"] != EXECUTION_ENVIRONMENT:
        raise ValueError("Foreign pixel shard code/environment binding")


def export_pixel_shard(
    root,
    output,
    plan,
    shard_index,
    *,
    reader,
    qc_reader,
    execution_binding,
    resource_fn=None,
    finalize_metadata_fn=None,
):
    """Kaggle export: one native study and one compressed ZIP entry at a time."""
    validate_plan(plan)
    _check_binding(execution_binding)
    if type(shard_index) is not int or not 0 <= shard_index < plan["shard_count"] or qc_reader is None:
        raise ValueError("A valid fixed shard and shared byte QC reader are required")
    shard = plan["shards"][shard_index]
    rows, settings, budget = shard["studies"], plan["contract"]["preprocess"], plan["budgets"]
    root, output = Path(root), Path(output)
    if (
        shutil.disk_usage(output.parent).free
        < budget["native_temp_bytes"]
        + shard["raw_pixel_upper_bytes"]
        + shard["container_overhead_budget_bytes"]
        + budget["helper_metadata_reserve_bytes"]
    ):
        raise ValueError("Insufficient free output disk for raw-bound archive and native temporary reserve")
    out = new_directory(output)
    _, series = read_csv(root / "train_series.csv", SERIES_COLUMNS)
    by_study, seen = defaultdict(list), set()
    for row in series:
        pair = (uid(row[ID]), uid(row[SERIES_ID]))
        if pair in seen:
            raise ValueError("Duplicate train-series metadata")
        seen.add(pair)
        by_study[pair[0]].append(row)
    if any(row[ID] not in by_study for row in rows):
        raise ValueError("Fixed shard study is missing train-series metadata")
    summary = {
        "schema_version": PIXEL_SCHEMA,
        "export_schema_version": EXPORT_SCHEMA,
        "status": "running",
        "complete": False,
        "for_submission": False,
        "teacher_CSV_read": False,
        "report_read": False,
        "labels_in_output": False,
        "training_performed": False,
        "gold_used": False,
        "public_score": None,
        "contract": plan["contract"],
        "fingerprint": plan["fingerprint"],
        "full_plan_sha256": json_hash(plan),
        "source_sha256": plan["source_sha256"],
        "execution_binding": execution_binding,
        "manifest_sha256": plan["manifest_sha256"],
        "selection_structure_sha256": plan["selection_structure_sha256"],
        "cohort": shard,
        "shard_index": shard_index,
        "studies": len(rows),
        "processed": 0,
        "series_csv_sha256": sha256(root / "train_series.csv"),
        "raw_pixel_upper_bytes": shard["raw_pixel_upper_bytes"],
        "archive_budget_bytes": budget["archive_bytes"],
        "files": {},
        "coverage": {},
        "resources": [],
        "preprocessing_byte_QC_count": 0,
        "preprocessing_QC_budget_skips": 0,
    }
    path = out / SUMMARY_NAME
    atomic_json(path, summary)
    began = time.perf_counter()
    final_archive = out / (shard["name"] + ".zip")
    partial_archive = final_archive.with_name(final_archive.name + ".partial")
    try:
        with zipfile.ZipFile(partial_archive, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for row in rows:
                if time.perf_counter() - began >= budget["max_seconds"]:
                    raise ValueError("Shard reached the fixed8hour processing budget; partial output cannot be indexed")
                key, start = row[ID], time.perf_counter()
                arrays, coverage = extract_pixel_study(
                    root, key, by_study[key], settings, reader=reader, qc_reader=qc_reader
                )
                encoded = encode_pixel_member(arrays, plan["fingerprint"], settings)
                if (
                    partial_archive.stat().st_size + len(encoded) + budget["per_study_container_overhead_bytes"]
                    >= budget["archive_bytes"]
                ):
                    raise ValueError("Compressed archive exceeded its fixed disk budget")
                name = f"{uid(key)}.npz"
                archive.writestr(name, encoded)
                summary["files"][name] = {
                    "bytes": len(encoded),
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                    "array_sha256": array_hashes(arrays),
                    "array_metadata": array_metadata(arrays),
                }
                summary["coverage"][key] = {
                    "series": coverage,
                    "seconds": time.perf_counter() - start,
                    "valid_triplets": int(arrays["mask"].sum()),
                }
                summary["processed"] += 1
                for detail in coverage:
                    status = detail["shared_byte_qc"]["status"]
                    summary["preprocessing_byte_QC_count"] += int(status == "passed")
                    summary["preprocessing_QC_budget_skips"] += int(status.startswith("reference_not_run"))
                if resource_fn is not None:
                    summary["resources"].append(resource_fn())
                atomic_json(path, summary)
                print(f"{shard['name']} {summary['processed']}/{len(rows)}", flush=True)
                del arrays, encoded
        if summary["processed"] != len(rows) or summary["preprocessing_byte_QC_count"] < 1:
            raise ValueError("Shard requires complete coverage and at least one native/shared byte QC")
        if partial_archive.stat().st_size >= budget["archive_bytes"]:
            raise ValueError("Closed ZIP exceeds the archive budget")
        archive_receipt = {
            "name": final_archive.name,
            "bytes": partial_archive.stat().st_size,
            "sha256": sha256(partial_archive),
        }
        finalized = {} if finalize_metadata_fn is None else finalize_metadata_fn()
        if (
            finalized.get("temporary_files_remaining") != 0
            or not finalized.get("input_csv_reads")
            or any(Path(name).name != "train_series.csv" for name in finalized.get("input_csv_reads", []))
        ):
            raise ValueError("Native temporary cleanup and train-series-only input audit must pass before completion")
        elapsed = time.perf_counter() - began
        if elapsed > budget["max_seconds"]:
            raise ValueError("Final study/ZIP/hash/QC exceeded the fixed eight-hour cap; shard cannot complete")
        os.replace(partial_archive, final_archive)  # Rename only; no second archive or extracted tree.
        summary.update(
            complete=True,
            status="pixel_shard_complete_not_for_submission",
            archive=archive_receipt,
            elapsed_seconds=elapsed,
            runtime_audit=finalized,
        )
    except Exception as error:
        summary.update(
            status="failed",
            complete=False,
            error=f"{type(error).__name__}: {error}",
            elapsed_seconds=time.perf_counter() - began,
        )
        atomic_json(path, summary)
        raise
    atomic_json(path, summary)
    return summary


def _inside(root, supplied):
    value = (root / supplied).resolve()
    if not value.is_relative_to(root.resolve()):
        raise ValueError("Cache/index paths must remain inside the chosen cache root")
    return value


def build_full_index(plan, package_metadata, summary_paths, output, *, cache_root):
    """Verify all shards and every NPZ in a streaming pass, then atomically index.

    This is an explicit local command after cache acquisition. It does not fetch
    files or unpack whole archives, and refuses a subset such as the16 pilot.
    """
    validate_plan(plan)
    binding = package_metadata["execution_binding"]
    _check_binding(binding)
    if package_metadata.get("plan_sha256") != json_hash(plan):
        raise ValueError("Package and full structural plan differ")
    output, cache_root = Path(output).resolve(), Path(cache_root).resolve()
    if output.exists() or output.with_name(output.name + ".partial").exists() or output.parent != cache_root:
        raise ValueError("Use a new index path inside the cache root")
    if len(summary_paths) != plan["shard_count"]:
        raise ValueError("Every planned shard is required; partial shard collections are forbidden")
    indexed, observed, series_sha, valid_windows = {}, set(), None, 0
    for supplied in summary_paths:
        path = Path(supplied).resolve()
        if not path.is_relative_to(cache_root):
            raise ValueError("Shard summary lies outside the cache root")
        summary = json.loads(path.read_text(encoding="utf-8"))
        index = summary.get("shard_index")
        if type(index) is not int or not 0 <= index < plan["shard_count"] or index in indexed:
            raise ValueError("Duplicate or foreign shard index")
        shard = plan["shards"][index]
        elapsed = summary.get("elapsed_seconds")
        if (
            type(elapsed) not in (int, float)
            or not math.isfinite(elapsed)
            or not 0 <= elapsed <= plan["budgets"]["max_seconds"]
        ):
            raise ValueError("Shard final elapsed time is missing, invalid or exceeds its fixed budget")
        if (
            summary.get("schema_version") != PIXEL_SCHEMA
            or summary.get("export_schema_version") != EXPORT_SCHEMA
            or summary.get("complete") is not True
            or summary.get("status") != "pixel_shard_complete_not_for_submission"
            or summary.get("for_submission") is not False
            or summary.get("processed") != shard["study_count"]
            or summary.get("studies") != shard["study_count"]
            or summary.get("cohort") != shard
            or summary.get("preprocessing_byte_QC_count", 0) < 1
            or summary.get("raw_pixel_upper_bytes") != shard["raw_pixel_upper_bytes"]
            or summary.get("archive_budget_bytes") != plan["budgets"]["archive_bytes"]
        ):
            raise ValueError("Incomplete/partial/foreign shard or missing native byte QC")
        if (
            summary.get("contract") != plan["contract"]
            or summary.get("fingerprint") != plan["fingerprint"]
            or summary.get("full_plan_sha256") != json_hash(plan)
            or summary.get("source_sha256") != plan["source_sha256"]
            or summary.get("manifest_sha256") != plan["manifest_sha256"]
            or summary.get("selection_structure_sha256") != plan["selection_structure_sha256"]
            or summary.get("execution_binding") != binding
        ):
            raise ValueError("Shard source/preprocessing/cohort/manifest/environment hash drift")
        if any(
            summary.get(flag) is not False
            for flag in ("teacher_CSV_read", "report_read", "labels_in_output", "training_performed", "gold_used")
        ):
            raise ValueError("Teacher/Report/Gold/model data is forbidden in cache exports")
        if (
            summary.get("runtime_audit", {}).get("temporary_files_remaining") != 0
            or not summary["runtime_audit"].get("input_csv_reads")
            or any(
                Path(name).name != "train_series.csv" for name in summary["runtime_audit"].get("input_csv_reads", [])
            )
        ):
            raise ValueError("Shard cleanup or input audit is incomplete")
        if any(".partial" in child.name for child in path.parent.iterdir()):
            raise ValueError("Partial output remains in the shard directory")
        current_series_sha = summary["series_csv_sha256"]
        if series_sha is not None and current_series_sha != series_sha:
            raise ValueError("Shard train-series Input metadata differs")
        series_sha = current_series_sha
        receipt = summary["archive"]
        if receipt["name"] != shard["name"] + ".zip":
            raise ValueError("Unsafe or foreign archive name")
        archive_path = _inside(path.parent, receipt["name"])
        actual_bytes = archive_path.stat().st_size
        if (
            type(receipt["bytes"]) is not int
            or not 0 < actual_bytes == receipt["bytes"] < plan["budgets"]["archive_bytes"]
        ):
            raise ValueError("Archive bytes exceed the fixed disk budget or receipt")
        if sha256(archive_path) != receipt["sha256"]:
            raise ValueError("Archive byte/hash receipt drift")
        expected = {f"{row[ID]}.npz" for row in shard["studies"]}
        if set(summary["files"]) != expected or set(summary["coverage"]) != {row[ID] for row in shard["studies"]}:
            raise ValueError("Missing/foreign UID or NPZ receipts")
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            if len(infos) != len(expected) or set(archive.namelist()) != expected:
                raise ValueError("Duplicate/missing/unsafe ZIP member inventory")
            for info in infos:
                key = uid(info.filename.removesuffix(".npz"))
                if info.filename != f"{key}.npz" or key in observed:
                    raise ValueError("UID appears more than once or an archive path is unsafe")
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError("ZIP symlink entries are forbidden")
                member = summary["files"][info.filename]
                member_cap = (
                    raw_pixel_bytes(1, plan["contract"]["preprocess"])
                    + plan["budgets"]["per_study_container_overhead_bytes"]
                )
                if not 0 < info.file_size <= member_cap:
                    raise ValueError("NPZ compressed member exceeds the single-study memory budget before read")
                if info.file_size != member["bytes"] or info.compress_type != zipfile.ZIP_STORED:
                    raise ValueError("NPZ size/container encoding differs")
                payload = archive.read(info)
                if hashlib.sha256(payload).hexdigest() != member["sha256"]:
                    raise ValueError("NPZ member hash drift")
                arrays = decode_pixel_member(payload, plan["fingerprint"], plan["contract"]["preprocess"])
                if array_hashes(arrays) != member["array_sha256"] or array_metadata(arrays) != member["array_metadata"]:
                    raise ValueError("NPZ array bytes/shape/dtype receipt drift")
                count = int(arrays["mask"].sum())
                if count != summary["coverage"][key]["valid_triplets"]:
                    raise ValueError("NPZ mask and valid-window coverage drift")
                observed.add(key)
                valid_windows += count
                del arrays, payload
        indexed[index] = {
            "summary_path": path.relative_to(output.parent).as_posix(),
            "archive_path": archive_path.relative_to(output.parent).as_posix(),
            "summary_sha256": sha256(path),
            "archive_sha256": receipt["sha256"],
            "archive_bytes": receipt["bytes"],
            "cohort_sha256": shard["cohort_sha256"],
        }
    if set(indexed) != set(range(plan["shard_count"])) or observed != {row[ID] for row in plan["studies"]}:
        raise ValueError("Full weak UID coverage is not exactly once across all whole-group shards")
    result = {
        "schema_version": INDEX_SCHEMA,
        "complete": True,
        "split": "train",
        "limited": False,
        "study_ids": sorted(observed),
        "contract": plan["contract"],
        "fingerprint": plan["fingerprint"],
        "manifest_sha256": plan["manifest_sha256"],
        "selection_structure_sha256": plan["selection_structure_sha256"],
        "full_plan_sha256": json_hash(plan),
        "export_source_sha256": plan["source_sha256"],
        "execution_binding": binding,
        "series_csv_sha256": series_sha,
        "shards": [indexed[index] for index in range(plan["shard_count"])],
        "verified": {"studies": len(observed), "arrays": len(observed) * 5, "valid_windows": valid_windows},
        "labels_or_reports_read": False,
        "archive_extraction": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result, new=True)
    return result
