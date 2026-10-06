"""Native DICOM -> high-resolution triplets -> frozen CLS cache, shared with inference."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .contracts import ID, SERIES_COLUMNS, SERIES_ID, dump_json, index_studies, read_csv, sha256, uid, write_csv
from .feature_contracts import (
    feature_contract,
    json_hash,
    new_directory,
    selection_structure_hash,
    source_tree_hash,
    validate_encoder_provenance,
    validate_feature_config,
)
from .imaging import geometric_order, normalize_volume, select_series

PLANES = {"sagittal": 0, "coronal": 1, "axial": 2}


def letterbox(slice_, size, spacing):
    """Preserve physical FOV aspect; no anatomical crop or inferred spacing."""
    height, width = slice_.shape
    scale = size / max(height * spacing[0], width * spacing[1])
    resized_height = max(1, min(size, round(height * spacing[0] * scale)))
    resized_width = max(1, min(size, round(width * spacing[1] * scale)))
    resized = np.asarray(Image.fromarray(slice_).resize((resized_width, resized_height), Image.Resampling.BILINEAR))
    result = np.zeros((size, size), np.uint8)
    y, x = (size - resized_height) // 2, (size - resized_width) // 2
    result[y : y + resized_height, x : x + resized_width] = resized
    return result


def read_feature_windows(series_dir, settings):
    import pydicom

    paths = sorted(Path(series_dir).glob("*.dcm"))
    if not paths:
        raise ValueError("No DICOM slices")
    headers = [pydicom.dcmread(path, stop_before_pixels=True) for path in paths]
    order, method = geometric_order(headers)
    if method != "physical_position":
        raise ValueError("Frozen-feature baseline requires physical slice geometry")
    spacing = np.asarray(getattr(headers[0], "PixelSpacing", []), dtype=float)
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("Missing/invalid PixelSpacing; no invented physical scale")
    for header in headers:
        candidate = np.asarray(getattr(header, "PixelSpacing", []), dtype=float)
        if candidate.shape != (2,) or not np.allclose(candidate, spacing, atol=1e-5, rtol=1e-5):
            raise ValueError("Mixed/missing PixelSpacing")
    slices = []
    for index in order:
        dataset = pydicom.dcmread(paths[index])
        pixels = dataset.pixel_array.astype(np.float32)
        if pixels.ndim != 2:
            raise ValueError("Only single-frame 2D DICOM supported")
        pixels = pixels * float(getattr(dataset, "RescaleSlope", 1)) + float(getattr(dataset, "RescaleIntercept", 0))
        if getattr(dataset, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
            pixels = pixels.max() + pixels.min() - pixels
        slices.append(pixels)
    if any(item.shape != slices[0].shape for item in slices):
        raise ValueError("Mixed matrix sizes")
    volume = normalize_volume(np.stack(slices), settings["percentiles"])
    count = min(len(volume), settings["slices_per_series"])
    centers = np.rint(np.linspace(0, len(volume) - 1, count)).astype(int)
    needed = sorted(set(np.clip(centers[:, None] + [-1, 0, 1], 0, len(volume) - 1).ravel()))
    resized = {i: letterbox(volume[i], settings["image_size"], spacing) for i in needed}
    windows = np.stack(
        [
            np.stack([resized[i] for i in np.clip(center + np.array([-1, 0, 1]), 0, len(volume) - 1)])
            for center in centers
        ]
    )
    orient = np.asarray(headers[order[0]].ImageOrientationPatient, dtype=float)
    normal = np.cross(orient[:3], orient[3:])
    z = np.array([np.dot(np.asarray(headers[i].ImagePositionPatient, dtype=float), normal) for i in order])
    positions = (
        np.zeros(count, np.float32) if len(z) == 1 else ((z[centers] - z[0]) / (z[-1] - z[0])).astype(np.float32)
    )
    return (
        windows,
        positions,
        {
            "order": method,
            "native_shape": list(volume.shape),
            "pixel_spacing_mm": spacing.tolist(),
            "selected_centers": centers.tolist(),
            "physical_span_mm": float(z[-1] - z[0]),
            "normalization": "whole native series p1/p99 before letterbox; 2.5D adjacent slices",
        },
    )


def load_frozen_encoder(repo_dir, weights, provenance, device):
    validate_encoder_provenance(provenance)
    if (
        sha256(weights) != provenance["weights_sha256"]
        or source_tree_hash(repo_dir) != provenance["source_tree_sha256"]
    ):
        raise ValueError("Encoder source/checkpoint SHA256 mismatch")
    # Stable PyTorch attention path on Kaggle and Windows; must be set before dinov2 import.
    if any(name == "dinov2" or name.startswith("dinov2.") for name in sys.modules):
        raise ValueError("DINOv2 is already imported; use a fresh process/kernel for audited encoder loading")
    os.environ["XFORMERS_DISABLED"] = "1"
    sys.path.insert(0, str(Path(repo_dir).resolve()))
    try:
        # Import only the reviewed backbone entrypoint; unrelated hub heads/dependencies are unnecessary.
        factory = importlib.import_module("dinov2.hub.backbones").dinov2_vits14
        model = factory(pretrained=False)
    finally:
        sys.path.pop(0)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model.requires_grad_(False).eval().to(device)
    return model


@torch.inference_mode()
def encode_windows(model, windows, config, device):
    result = []
    batch = config["encoder"]["batch_size"]
    for start in range(0, len(windows), batch):
        x = torch.from_numpy(windows[start : start + batch].astype(np.float32) / 255).to(device)
        mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        with torch.autocast(
            device_type=torch.device(device).type, dtype=torch.float16, enabled=config["encoder"]["precision"] == "fp16"
        ):
            features = model((x - mean) / std)
        if features.shape != (len(x), config["encoder"]["feature_dim"]) or not torch.isfinite(features).all():
            raise ValueError("Encoder must return finite normalized CLS features")
        encoded = features.float().cpu().numpy().astype(np.float16)
        if not np.isfinite(encoded).all():
            raise ValueError("FP16 feature storage overflow")
        result.append(encoded)
    return np.concatenate(result)


def extract_study(root, split, key, series, config, encoder, device):
    p = config["preprocess"]
    shape = (p["max_series"], p["slices_per_series"])
    features = np.zeros((*shape, config["encoder"]["feature_dim"]), np.float16)
    mask, positions = np.zeros(shape, bool), np.zeros(shape, np.float32)
    planes, flags = np.full(shape[0], 3, np.int64), np.zeros((shape[0], 2), np.float32)
    coverage = []
    for slot, row in enumerate(select_series(series, shape[0])):
        windows, location, detail = read_feature_windows(
            Path(root) / f"{split}_series" / uid(key) / uid(row[SERIES_ID]), p
        )
        n = len(windows)
        features[slot, :n] = encode_windows(encoder, windows, config, device)
        positions[slot, :n], mask[slot, :n] = location, True
        planes[slot] = PLANES.get(row["Anatomical_Plane"].strip().lower(), 3)
        flags[slot] = [
            float(row[name].strip().lower() in {"1", "1.0", "true", "yes"})
            for name in ("Fluid_Sensitive", "Fat_Suppression")
        ]
        coverage.append({SERIES_ID: row[SERIES_ID], "plane": row["Anatomical_Plane"], **detail})
    if not mask.any():
        raise ValueError("No selected MRI series; no constant/fallback prediction")
    return {"features": features, "mask": mask, "positions": positions, "planes": planes, "flags": flags}, coverage


def load_features(cache_dir, key, config, fingerprint, expected_sha256=None):
    path = Path(cache_dir) / f"{uid(key)}.npz"
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise ValueError("Feature payload SHA256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["fingerprint"].item()) != fingerprint:
            raise ValueError("Feature fingerprint mismatch")
        arrays = {name: archive[name] for name in ("features", "mask", "positions", "planes", "flags")}
    p = config["preprocess"]
    shape = (p["max_series"], p["slices_per_series"])
    expected = {
        "features": ((*shape, config["encoder"]["feature_dim"]), np.float16),
        "mask": (shape, np.bool_),
        "positions": (shape, np.float32),
        "planes": ((shape[0],), np.int64),
        "flags": ((shape[0], 2), np.float32),
    }
    for name, (dims, dtype) in expected.items():
        if arrays[name].shape != dims or arrays[name].dtype != dtype or not np.isfinite(arrays[name]).all():
            raise ValueError(f"Malformed feature array: {name}")
    if not arrays["mask"].any() or (arrays["planes"] < 0).any() or (arrays["planes"] > 3).any():
        raise ValueError("Empty feature study or invalid anatomical plane")
    if (arrays["positions"] < 0).any() or (arrays["positions"] > 1).any():
        raise ValueError("Invalid normalized position")
    if not np.isin(arrays["flags"], [0, 1]).all():
        raise ValueError("Invalid sequence flags")
    return arrays


def extract_cache(
    root,
    split,
    output,
    config,
    provenance,
    repo_dir,
    weights,
    *,
    studies_csv=None,
    limit=None,
    device="cuda",
    shard_index=0,
    shard_count=1,
):
    validate_feature_config(config)
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise ValueError("Feature extraction requires CUDA")
    if split not in ("train", "test") or (limit is not None and limit < 1):
        raise ValueError("Invalid split/limit")
    if shard_count < 1 or not 0 <= shard_index < shard_count or (split == "test" and shard_count != 1):
        raise ValueError("Invalid training feature shard")
    root = Path(root)
    _, all_rows = read_csv(root / f"{split}.csv", (ID,))
    all_ids = index_studies(all_rows)
    rows = read_csv(studies_csv, (ID,))[1] if studies_csv else all_rows
    if not rows or not index_studies(rows).keys() <= all_ids.keys():
        raise ValueError("Requested study list contains unknown IDs or is empty")
    # Training exports must use the audited weak list (or a QA subset of it).
    if split == "train" and studies_csv is None:
        raise ValueError("Train extraction requires an explicit weak/pilot study manifest")
    partition = sorted(rows, key=lambda row: row[ID])[shard_index::shard_count] if shard_count > 1 else rows
    requested = partition[:limit] if limit else partition
    if not requested:
        raise ValueError("Empty requested shard")
    _, series = read_csv(root / f"{split}_series.csv", SERIES_COLUMNS)
    by_study, seen = defaultdict(list), set()
    for row in series:
        pair = (uid(row[ID]), uid(row[SERIES_ID]))
        if pair in seen or pair[0] not in all_ids:
            raise ValueError("Orphan/duplicate series")
        seen.add(pair)
        by_study[pair[0]].append(row)
    encoder = load_frozen_encoder(repo_dir, weights, provenance, device)
    contract = feature_contract(config, provenance)
    out = new_directory(output)
    metadata = {
        "schema_version": "frozen_feature_export_v1",
        "complete": False,
        "split": split,
        "limited": limit is not None,
        "requested": len(requested),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "contract": contract,
        "fingerprint": json_hash(contract),
        "studies_csv_sha256": sha256(root / f"{split}.csv"),
        "series_csv_sha256": sha256(root / f"{split}_series.csv"),
        "selection_csv_sha256": sha256(studies_csv) if studies_csv else None,
        "selection_structure_sha256": selection_structure_hash(rows) if split == "train" else None,
        "runtime": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "pydicom", "Pillow")},
        "gpu": torch.cuda.get_device_name(torch.device(device)),
        "files": {},
        "coverage": {},
    }
    dump_json(out / "export.json", metadata)
    write_csv(out / "studies.csv", (ID,), [{ID: row[ID]} for row in requested])
    metadata["study_ids_sha256"] = sha256(out / "studies.csv")
    start = time.perf_counter()
    for i, row in enumerate(requested, 1):
        began = time.perf_counter()
        arrays, coverage = extract_study(root, split, row[ID], by_study[row[ID]], config, encoder, device)
        path = out / f"{uid(row[ID])}.npz"
        np.savez_compressed(path, **arrays, fingerprint=metadata["fingerprint"])
        metadata["files"][path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        metadata["coverage"][row[ID]] = {"series": coverage, "seconds": time.perf_counter() - began}
        dump_json(out / "export.json", metadata)
        print(f"Features {i}/{len(requested)}; elapsed={time.perf_counter() - start:.1f}s", flush=True)
    metadata.update(
        complete=True,
        elapsed_seconds=time.perf_counter() - start,
        total_payload_bytes=sum(value["bytes"] for value in metadata["files"].values()),
    )
    dump_json(out / "export.json", metadata)
    return metadata


def verify_feature_cache(cache_dir, config, *, allow_limited=False):
    root = Path(cache_dir)
    metadata = json.loads((root / "export.json").read_text(encoding="utf-8"))
    contract = metadata["contract"]
    validate_encoder_provenance(contract["encoder_provenance"])
    expected = feature_contract(config, contract["encoder_provenance"])
    if contract != expected or json_hash(expected) != metadata["fingerprint"]:
        raise ValueError("Feature configuration/source/provenance mismatch")
    if metadata.get("schema_version") != "frozen_feature_export_v1" or metadata.get("complete") is not True:
        raise ValueError("Incomplete feature export")
    if metadata.get("limited") and not allow_limited:
        raise ValueError("Pilot/limited cache is not a training cache")
    if sha256(root / "studies.csv") != metadata["study_ids_sha256"]:
        raise ValueError("Feature study manifest digest mismatch")
    _, rows = read_csv(root / "studies.csv", (ID,))
    ids = index_studies(rows)
    if len(ids) != metadata["requested"] or set(metadata["files"]) != {f"{key}.npz" for key in ids}:
        raise ValueError("Feature export coverage mismatch")
    for key in ids:
        entry = metadata["files"][f"{key}.npz"]
        if (root / f"{key}.npz").stat().st_size != entry["bytes"]:
            raise ValueError("Feature payload size mismatch")
        load_features(root, key, config, metadata["fingerprint"], entry["sha256"])
    return metadata, ids


def merge_feature_shards(cache_dirs, studies_csv, output, config):
    """Local verified copy only: assemble exact disjoint weak UID coverage without re-encoding."""
    _, rows = read_csv(studies_csv, (ID, "group_id", "fold"))
    expected = index_studies(rows)
    checked = [(*verify_feature_cache(path, config), Path(path)) for path in cache_dirs]
    if not checked:
        raise ValueError("No feature shards")
    first = checked[0][0]
    combined = {}
    for metadata, ids, path in checked:
        for name in ("contract", "studies_csv_sha256", "series_csv_sha256", "selection_structure_sha256"):
            if metadata[name] != first[name]:
                raise ValueError("Feature shards have different source/config/selection")
        if metadata["split"] != "train" or combined.keys() & ids.keys():
            raise ValueError("Overlapping or non-training feature shards")
        combined.update({key: (path, metadata) for key in ids})
    if combined.keys() != expected.keys() or first["selection_structure_sha256"] != selection_structure_hash(rows):
        raise ValueError("Feature shards do not exactly cover the fixed weak selection")
    out = new_directory(output)
    result = {
        **first,
        "complete": False,
        "limited": False,
        "requested": len(rows),
        "shard_count": 1,
        "shard_index": 0,
        "files": {},
        "coverage": {},
        "selection_csv_sha256": sha256(studies_csv),
        "merged_exports": [sha256(path / "export.json") for _, _, path in checked],
    }
    dump_json(out / "export.json", result)
    write_csv(out / "studies.csv", (ID,), [{ID: row[ID]} for row in rows])
    result["study_ids_sha256"] = sha256(out / "studies.csv")
    for key, (path, metadata) in combined.items():
        filename = f"{uid(key)}.npz"
        shutil.copyfile(path / filename, out / filename)
        result["files"][filename] = metadata["files"][filename]
        result["coverage"][key] = metadata.get("coverage", {}).get(key, {})
    result.update(
        complete=True,
        total_payload_bytes=sum(item["bytes"] for item in result["files"].values()),
        elapsed_seconds=sum(metadata.get("elapsed_seconds", 0) for metadata, _, _ in checked),
    )
    dump_json(out / "export.json", result)
    verify_feature_cache(out, config)
    return result
