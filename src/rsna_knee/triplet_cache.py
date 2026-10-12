"""Native 392px pixels for a small, label-free DINO fine-tuning resource pilot.

This is a separate pixel contract, not a frozen CLS feature cache. The original
feature reader, series selection, positions, masks and flags remain authoritative.
No training/target CSV, encoder, label, score, or submission is used here.
"""

from __future__ import annotations

import hashlib
import io
import math
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from .contracts import ID, SERIES_COLUMNS, SERIES_ID, dump_json, read_csv, sha256, uid
from .feature_contracts import implementation_hashes, json_hash, new_directory
from .feature_imaging import PLANES, read_feature_windows
from .imaging import select_series

ARRAY_NAMES = ("images", "mask", "positions", "planes", "flags")
PIXEL_SCHEMA = "native_physical_triplet_pixel_pilot_v1"


def pixel_contract(settings):
    expected = {
        "version": "physical_letterbox_triplet_v1",
        "image_size": 392,
        "max_series": 3,
        "slices_per_series": 48,
        "percentiles": [1, 99],
    }
    if settings != expected:
        raise ValueError("The pilot requires the unchanged f002 native 392px, 3x48 triplet preprocessing")
    return {
        "schema_version": PIXEL_SCHEMA,
        "preprocess": settings,
        "shared_implementation_sha256": implementation_hashes(),
        "storage": "uint8 padded physical-letterbox triplets; compressed NPZ members in one ZIP_STORED archive",
        "normalization_for_encoder": "uint8 / 255 then ImageNet mean/std; outside the pixel cache",
        "labels_in_cache": False,
        "encoder_features_in_cache": False,
    }


def raw_pixel_bytes(studies, settings):
    pixel_contract(settings)
    if type(studies) is not int or studies < 1:
        raise ValueError("Positive study count required")
    return studies * settings["max_series"] * settings["slices_per_series"] * 3 * settings["image_size"] ** 2


def validate_cohort(cohort):
    """Check only embedded IDs/groups/folds, never teacher values or reports."""
    if cohort.get("schema_version") != "fixed_weak_group_pixel_pilot_v1":
        raise ValueError("Unsupported cohort contract")
    rows = cohort["studies"]
    if len(rows) != 16 or len({uid(row[ID]) for row in rows}) != 16:
        raise ValueError("Exactly 16 unique fixed weak studies are required")
    banned_ids = set(cohort["gold_ids"])
    banned_groups = set(cohort["gold_groups"])
    groups = defaultdict(list)
    for row in rows:
        if set(row) != {ID, "group_id", "fold", "source"} or row["source"] != "report_weak":
            raise ValueError("Cohort contains unexpected fields/source; no targets or reports may be embedded")
        if row[ID] in banned_ids or row["group_id"] in banned_groups:
            raise ValueError("Gold ID/group overlap")
        if type(row["fold"]) is not int or not 0 <= row["fold"] < cohort["n_folds"]:
            raise ValueError("Invalid fold")
        if row["fold"] == cohort["held_out_fold"]:
            raise ValueError("Pilot must use only the training side of the fixed group split")
        groups[row["group_id"]].append(row)
    if set(groups) != set(cohort["complete_selected_groups"]):
        raise ValueError("Missing selected-group inventory")
    for group, members in groups.items():
        if len({row["fold"] for row in members}) != 1:
            raise ValueError("A group crosses folds")
        if sorted(row[ID] for row in members) != sorted(cohort["complete_selected_groups"][group]):
            raise ValueError("Only part of a selected group is included")
    return rows


def validate_pixels(arrays, settings):
    pixel_contract(settings)
    shape = (settings["max_series"], settings["slices_per_series"])
    expected = {
        "images": ((*shape, 3, settings["image_size"], settings["image_size"]), np.uint8),
        "mask": (shape, np.bool_),
        "positions": (shape, np.float32),
        "planes": ((shape[0],), np.int64),
        "flags": ((shape[0], 2), np.float32),
    }
    if set(arrays) != set(expected):
        raise ValueError("Malformed pixel array names")
    for name, (dims, dtype) in expected.items():
        value = arrays[name]
        if value.shape != dims or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f"Malformed pixel array: {name}")
    if not arrays["mask"].any():
        raise ValueError("An all-masked study cannot be fitted or predicted")
    if ((arrays["planes"] < 0) | (arrays["planes"] > 3)).any():
        raise ValueError("Invalid plane")
    if ((arrays["positions"] < 0) | (arrays["positions"] > 1)).any() or not np.isin(arrays["flags"], [0, 1]).all():
        raise ValueError("Invalid position/sequence flags")
    if arrays["images"][~arrays["mask"]].any() or arrays["positions"][~arrays["mask"]].any():
        raise ValueError("Padding must contain zero pixels and positions")
    return True


def array_hashes(arrays):
    return {name: hashlib.sha256(np.ascontiguousarray(arrays[name]).tobytes()).hexdigest() for name in ARRAY_NAMES}


def extract_pixel_study(
    root, key, series, settings, *, reader=read_feature_windows, qc_reader=None, qc_native_float32_limit=128 * 1024**2
):
    pixel_contract(settings)
    shape = (settings["max_series"], settings["slices_per_series"])
    arrays = {
        "images": np.zeros((*shape, 3, settings["image_size"], settings["image_size"]), np.uint8),
        "mask": np.zeros(shape, bool),
        "positions": np.zeros(shape, np.float32),
        "planes": np.full(shape[0], 3, np.int64),
        "flags": np.zeros((shape[0], 2), np.float32),
    }
    coverage = []
    for slot, row in enumerate(select_series(series, shape[0])):
        directory = Path(root) / "train_series" / uid(key) / uid(row[SERIES_ID])
        windows, locations, detail = reader(directory, settings)
        n = len(windows)
        if not 1 <= n <= shape[1] or windows.shape != (n, 3, settings["image_size"], settings["image_size"]):
            raise ValueError("Unexpected shared-reader output")
        arrays["images"][slot, :n], arrays["positions"][slot, :n], arrays["mask"][slot, :n] = windows, locations, True
        arrays["planes"][slot] = PLANES.get(row["Anatomical_Plane"].strip().lower(), 3)
        arrays["flags"][slot] = [
            float(row[name].strip().lower() in {"1", "1.0", "true", "yes"})
            for name in ("Fluid_Sensitive", "Fat_Suppression")
        ]
        qc = {"status": "not_requested"}
        if qc_reader is not None:
            native_bytes = math.prod(detail["native_shape"]) * 4
            qc = {"native_float32_bytes": native_bytes, "reference_limit_bytes": qc_native_float32_limit}
            if native_bytes <= qc_native_float32_limit:
                reference = qc_reader(directory, settings)
                # Bytes, dtype and shape must all agree; identical values alone are insufficient.
                if (
                    any(
                        a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes()
                        for a, b in zip((windows, locations), reference[:2])
                    )
                    or detail != reference[2]
                ):
                    raise ValueError("Shared preprocessing byte QC failed")
                qc.update(
                    status="passed",
                    windows_sha256=hashlib.sha256(windows.tobytes()).hexdigest(),
                    positions_sha256=hashlib.sha256(locations.tobytes()).hexdigest(),
                )
                del reference
            else:
                qc["status"] = "reference_not_run_native_volume_above_QC_budget"
        coverage.append({SERIES_ID: row[SERIES_ID], "plane": row["Anatomical_Plane"], **detail, "shared_byte_qc": qc})
    validate_pixels(arrays, settings)
    return arrays, coverage


def encode_pixel_member(arrays, fingerprint, settings):
    validate_pixels(arrays, settings)
    payload = io.BytesIO()
    np.savez_compressed(payload, **arrays, fingerprint=fingerprint)
    encoded = payload.getvalue()
    decoded = decode_pixel_member(encoded, fingerprint, settings)
    if array_hashes(decoded) != array_hashes(arrays):
        raise ValueError("Pixel payload roundtrip byte QC failed")
    return encoded


def decode_pixel_member(payload, fingerprint, settings):
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) != {*ARRAY_NAMES, "fingerprint"} or str(archive["fingerprint"].item()) != fingerprint:
            raise ValueError("Pixel payload fingerprint/member mismatch")
        arrays = {name: archive[name] for name in ARRAY_NAMES}
    validate_pixels(arrays, settings)
    return arrays


def export_pixel_pilot(
    root,
    output,
    settings,
    cohort,
    *,
    reader=read_feature_windows,
    qc_reader=None,
    resource_fn=None,
    archive_budget_bytes=1024**3,
):
    """Kaggle-only native pixel export. No train.csv/weak.csv/gold.csv is opened."""
    rows = validate_cohort(cohort)
    contract = pixel_contract(settings)
    fingerprint = json_hash(contract)
    if raw_pixel_bytes(len(rows), settings) >= archive_budget_bytes:
        raise ValueError("Pixel cohort exceeds the pilot archive budget before decoding")
    root, out = Path(root), new_directory(output)
    series_path = root / "train_series.csv"
    _, series = read_csv(series_path, SERIES_COLUMNS)
    by_study, seen = defaultdict(list), set()
    for row in series:
        pair = (uid(row[ID]), uid(row[SERIES_ID]))
        if pair in seen:
            raise ValueError("Duplicate series")
        seen.add(pair)
        by_study[pair[0]].append(row)
    selected_ids = [row[ID] for row in rows]
    if any(key not in by_study for key in selected_ids):
        raise ValueError("A fixed study has no train-series metadata")
    summary = {
        "schema_version": PIXEL_SCHEMA,
        "status": "running",
        "complete": False,
        "for_submission": False,
        "studies": len(rows),
        "processed": 0,
        "cohort": cohort,
        "contract": contract,
        "fingerprint": fingerprint,
        "series_csv_sha256": sha256(series_path),
        "teacher_CSV_read": False,
        "report_read": False,
        "training_performed": False,
        "labels_in_output": False,
        "gold_used": False,
        "raw_pixel_upper_bytes": raw_pixel_bytes(len(rows), settings),
        "archive_budget_bytes": archive_budget_bytes,
        "files": {},
        "coverage": {},
        "resources": [],
        "preprocessing_byte_QC_count": 0,
        "preprocessing_QC_budget_skips": 0,
        "public_score": None,
    }
    path = out / "NATIVE392_PIXEL_PILOT_SUMMARY.json"
    dump_json(path, summary)
    began = time.perf_counter()
    archive_path = out / "native392-pixel16.zip"
    try:
        with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for row in rows:
                start = time.perf_counter()
                key = row[ID]
                arrays, coverage = extract_pixel_study(
                    root, key, by_study[key], settings, reader=reader, qc_reader=qc_reader
                )
                encoded = encode_pixel_member(arrays, fingerprint, settings)
                name = f"{uid(key)}.npz"
                if archive_path.stat().st_size + len(encoded) + 1024**2 > archive_budget_bytes:
                    raise ValueError("Actual compressed archive exceeds the 1GiB pilot budget")
                archive.writestr(name, encoded)
                summary["files"][name] = {
                    "bytes": len(encoded),
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                    "array_sha256": array_hashes(arrays),
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
                dump_json(path, summary)
                print(f"native pixel pilot {summary['processed']}/16: {key}", flush=True)
                del arrays, encoded
        if qc_reader is not None and summary["preprocessing_byte_QC_count"] == 0:
            raise ValueError("No bounded/native shared byte QC was exercised; pilot cannot pass")
        summary.update(
            status="pixel_pilot_complete_not_for_submission",
            complete=True,
            archive={"name": archive_path.name, "bytes": archive_path.stat().st_size, "sha256": sha256(archive_path)},
            elapsed_seconds=time.perf_counter() - began,
        )
    except Exception as error:
        summary.update(
            status="failed", error=f"{type(error).__name__}: {error}", elapsed_seconds=time.perf_counter() - began
        )
        dump_json(path, summary)
        raise
    dump_json(path, summary)
    return summary


def load_archive_study(archive_path, key, summary):
    """Read one compressed member directly, without unpacking the entire cache."""
    name = f"{uid(key)}.npz"
    receipt = summary["files"].get(name)
    if receipt is None or summary.get("complete") is not True or summary.get("for_submission") is not False:
        raise ValueError("Missing study or incomplete pixel pilot")
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo(name)
        if info.file_size != receipt["bytes"]:
            raise ValueError("Pixel archive member size mismatch")
        payload = archive.read(info)
    if hashlib.sha256(payload).hexdigest() != receipt["sha256"]:
        raise ValueError("Pixel archive member hash mismatch")
    arrays = decode_pixel_member(payload, summary["fingerprint"], summary["contract"]["preprocess"])
    if array_hashes(arrays) != receipt["array_sha256"]:
        raise ValueError("Pixel array receipt mismatch")
    return arrays
