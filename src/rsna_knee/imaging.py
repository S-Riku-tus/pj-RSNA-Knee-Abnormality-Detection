"""Bounded 2.5D cache shared by training and inference. Never downloads images."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .contracts import (
    ID,
    SERIES_COLUMNS,
    SERIES_ID,
    dump_json,
    index_studies,
    preprocess_fingerprint,
    read_csv,
    sha256,
    uid,
)


def geometric_order(headers):
    """Sort all slices on the same normal; use InstanceNumber only when geometry is absent."""
    geometry = [hasattr(h, "ImageOrientationPatient") and hasattr(h, "ImagePositionPatient") for h in headers]
    if all(geometry):
        orientation = np.asarray(headers[0].ImageOrientationPatient, dtype=float)
        normal = np.cross(orientation[:3], orientation[3:])
        if not np.isfinite(normal).all() or np.linalg.norm(normal) < 0.9:
            raise ValueError("Invalid DICOM slice normal")
        if any(not np.allclose(h.ImageOrientationPatient, orientation, atol=1e-3) for h in headers):
            raise ValueError("Mixed orientation in a series")
        positions = [float(np.dot(np.asarray(h.ImagePositionPatient, dtype=float), normal)) for h in headers]
        method = "physical_position"
    elif not any(geometry) and all(hasattr(h, "InstanceNumber") for h in headers):
        positions = [float(h.InstanceNumber) for h in headers]
        method = "instance_number_fallback"
    else:
        raise ValueError("Incomplete geometry; slice order is ambiguous")
    if not np.isfinite(positions).all() or len(set(positions)) != len(positions):
        raise ValueError("Nonfinite or duplicate slice positions")
    return np.argsort(positions).tolist(), method


def normalize_volume(volume, percentiles):
    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError("Expected finite single-frame slice volume")
    lo, hi = np.percentile(volume, percentiles)
    if hi <= lo:
        raise ValueError("Constant-intensity series")
    return np.rint(np.clip((volume - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def select_series(rows, maximum):
    def priority(row):
        yes = {"1", "1.0", "true", "yes"}
        return (row["Fluid_Sensitive"].lower() not in yes, row["Fat_Suppression"].lower() not in yes, row[SERIES_ID])

    ordered = sorted(rows, key=priority)
    selected, planes = [], set()
    for row in ordered:
        plane = row["Anatomical_Plane"].strip().lower()
        if plane not in planes:
            selected.append(row)
            planes.add(plane)
    selected = selected[:maximum]
    for row in ordered:
        if len(selected) >= maximum:
            break
        if row not in selected:
            selected.append(row)
    return selected


def read_windows(series_dir, config):
    import pydicom
    from PIL import Image

    paths = sorted(Path(series_dir).glob("*.dcm"))
    if not paths:
        raise ValueError("No DICOM slices in series")
    headers = [pydicom.dcmread(p, stop_before_pixels=True) for p in paths]
    order, method = geometric_order(headers)
    slices = []
    for index in order:
        dataset = pydicom.dcmread(paths[index])
        pixels = dataset.pixel_array.astype(np.float32)
        if pixels.ndim != 2:
            raise ValueError("Multiframe DICOM unsupported by v1; inspect series explicitly")
        pixels = pixels * float(getattr(dataset, "RescaleSlope", 1)) + float(getattr(dataset, "RescaleIntercept", 0))
        if getattr(dataset, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
            pixels = pixels.max() + pixels.min() - pixels
        slices.append(pixels)
    shape = slices[0].shape
    if any(s.shape != shape for s in slices):
        raise ValueError("Mixed matrix size in series")
    volume = normalize_volume(np.stack(slices), config["percentiles"])
    size = config["image_size"]
    resized = np.stack([np.asarray(Image.fromarray(s).resize((size, size), Image.Resampling.BILINEAR)) for s in volume])
    centers = np.rint(np.linspace(0, len(volume) - 1, config["windows_per_series"])).astype(int)
    windows = np.stack([resized[np.clip([i - 1, i, i + 1], 0, len(volume) - 1)] for i in centers])
    return windows, method


def load_cached(cache_dir, study, config):
    with np.load(Path(cache_dir) / f"{uid(study)}.npz", allow_pickle=False) as archive:
        fingerprint = str(archive["fingerprint"].item())
        images, mask = archive["images"], archive["mask"]
    p = config["preprocess"]
    expected = (p["max_series"] * p["windows_per_series"], 3, p["image_size"], p["image_size"])
    if fingerprint != preprocess_fingerprint(config) or images.shape != expected:
        raise ValueError("Cache/preprocess mismatch; use matching config or rebuild into a new directory")
    if images.dtype != np.uint8 or mask.dtype != np.bool_ or mask.shape != expected[:1] or not mask.any():
        raise ValueError("Malformed cache or study without any valid MRI windows")
    return images, mask


def build_cache(data_root, split, cache_dir, config, limit=None):
    import pydicom

    root, out = Path(data_root), Path(cache_dir)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    _, studies = read_csv(root / f"{split}.csv", (ID,))
    index = index_studies(studies)
    _, series = read_csv(root / f"{split}_series.csv", SERIES_COLUMNS)
    by_study, pairs = defaultdict(list), set()
    for row in series:
        pair = (uid(row[ID]), uid(row[SERIES_ID]))
        if pair[0] not in index or pair in pairs:
            raise ValueError("Orphan or duplicate MRI series")
        pairs.add(pair)
        by_study[row[ID]].append(row)
    metadata = {
        "schema_version": 1,
        "split": split,
        "preprocess": config["preprocess"],
        "fingerprint": preprocess_fingerprint(config),
        "studies_csv_sha256": sha256(root / f"{split}.csv"),
        "series_csv_sha256": sha256(root / f"{split}_series.csv"),
        "note": "DICOM contents must be immutable. Delete/rebuild cache if raw images change.",
    }
    meta_path = out / "cache.json"
    if out.exists() and any(out.iterdir()) and not meta_path.exists():
        raise ValueError("Nonempty cache directory without a manifest")
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")) != metadata:
        raise ValueError("Cache input/config differs; use a new cache directory")
    out.mkdir(parents=True, exist_ok=True)
    dump_json(meta_path, metadata)
    # Keep diagnostics across resume; a cached study can have warnings worth reviewing.
    log_path = out / "coverage.json"
    coverage = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
    failed = 0
    p = config["preprocess"]
    n = p["max_series"] * p["windows_per_series"]
    requested = studies[:limit] if limit else studies
    for number, study in enumerate(requested, 1):
        key = study[ID]
        destination = out / f"{key}.npz"
        if destination.exists():
            load_cached(out, key, config)
            continue
        images = np.zeros((n, 3, p["image_size"], p["image_size"]), dtype=np.uint8)
        mask = np.zeros(n, dtype=bool)
        details = {"series": [], "errors": []}
        for slot, row in enumerate(select_series(by_study[key], p["max_series"])):
            try:
                windows, order_method = read_windows(root / f"{split}_series" / key / row[SERIES_ID], p)
                start = slot * p["windows_per_series"]
                images[start : start + len(windows)] = windows
                mask[start : start + len(windows)] = True
                details["series"].append({"uid": row[SERIES_ID], "order": order_method})
            except (ValueError, RuntimeError, OSError, pydicom.errors.InvalidDicomError) as error:
                details["errors"].append({"uid": row[SERIES_ID], "error_type": type(error).__name__})
        details["valid_windows"] = int(mask.sum())
        coverage[key] = details
        if mask.any():
            temporary = out / f"{key}.tmp.npz"
            np.savez_compressed(temporary, images=images, mask=mask, fingerprint=metadata["fingerprint"])
            temporary.replace(destination)
        else:
            failed += 1
        dump_json(log_path, coverage)
        if number % 25 == 0:
            print(f"Cache: {number}/{len(requested)} studies", flush=True)
    if failed:
        raise ValueError(
            f"{failed} studies have no valid windows. Inspect private coverage.json; no constant predictions are substituted."
        )
    return {
        "requested": len(requested),
        "cache_dir": str(out),
        "studies_with_warnings": sum(bool(coverage.get(s[ID], {}).get("errors")) for s in requested),
        "debug_limit": limit,
        "fingerprint": metadata["fingerprint"],
    }
