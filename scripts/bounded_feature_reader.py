"""Identical frozen MRI windows using disk-backed native and percentile arrays.

The frozen core implementation and checkpoint contract remain unchanged. This
reader changes where intermediate pixels live, preserving global percentiles,
slice ordering, integer quantization and the existing physical letterbox. It
never skips a series, invents geometry, or downloads data.

File-backed percentile partitioning may still page in a native-sized working
set. This reduces anonymous allocations; it does not promise constant peak RSS.
"""

from __future__ import annotations

import mmap
import shutil
import tempfile
from pathlib import Path

COPY_CHUNK_BYTES = 8 * 1024**2
DISK_RESERVE_BYTES = 256 * 1024**2


def _release_file_pages(array):
    """Allow Linux to reclaim flushed mapped pages; portable no-op elsewhere."""
    mapping = getattr(array, "_mmap", None)
    if mapping is not None and hasattr(mapping, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
        array.flush()
        try:
            mapping.madvise(mmap.MADV_DONTNEED)
        except (OSError, ValueError):
            # Reclaim advice is optional. Valid images must remain readable on
            # filesystems/runtimes that do not support this particular hint.
            pass


def _close_mapping(array):
    mapping = getattr(array, "_mmap", None)
    if mapping is not None and not mapping.closed:
        mapping.close()


def allocation_plan(shape, *, copy_chunk_bytes=COPY_CHUNK_BYTES, disk_reserve_bytes=DISK_RESERVE_BYTES):
    """Byte budget only; computes no pixels and does not allocate the shape."""
    import math

    if len(shape) != 3 or any(type(value) is not int or value < 1 for value in shape):
        raise ValueError("Expected positive native volume shape")
    if type(copy_chunk_bytes) is not int or copy_chunk_bytes < 4:
        raise ValueError("Copy chunk must hold at least one float32 pixel")
    if type(disk_reserve_bytes) is not int or disk_reserve_bytes < 0:
        raise ValueError("Invalid temporary disk reserve")
    native_bytes = math.prod(shape) * 4
    return {
        "native_shape": list(shape),
        "native_file_bytes": native_bytes,
        "quantile_file_bytes": native_bytes,
        "minimum_free_disk_bytes": native_bytes * 2 + disk_reserve_bytes,
        "copy_chunk_bytes": (copy_chunk_bytes // 4) * 4,
        "maximum_normalization_input_slice_voxels": shape[1] * shape[2],
        "anonymous_native_volume_allocated": False,
        "percentile_partition": "inplace on scratch memmap; file-backed resident pages are not bounded to one slice",
    }


def make_bounded_reader(
    workspace, *, copy_chunk_bytes=COPY_CHUNK_BYTES, disk_reserve_bytes=DISK_RESERVE_BYTES, observer=None
):
    """Bind a writable workspace while retaining read_feature_windows' API."""
    workspace = Path(workspace)

    def read_feature_windows(series_dir, settings):
        return read_bounded_feature_windows(
            series_dir,
            settings,
            workspace=workspace,
            copy_chunk_bytes=copy_chunk_bytes,
            disk_reserve_bytes=disk_reserve_bytes,
            observer=observer,
        )

    return read_feature_windows


def read_bounded_feature_windows(
    series_dir,
    settings,
    *,
    workspace,
    copy_chunk_bytes=COPY_CHUNK_BYTES,
    disk_reserve_bytes=DISK_RESERVE_BYTES,
    observer=None,
):
    import numpy as np
    import pydicom

    from rsna_knee.feature_imaging import letterbox
    from rsna_knee.imaging import geometric_order

    workspace = Path(workspace)
    if not workspace.is_dir():
        raise ValueError("Temporary native-volume workspace must already exist")
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

    def read_pixels(index):
        dataset = pydicom.dcmread(paths[index])
        pixels = dataset.pixel_array.astype(np.float32)
        if pixels.ndim != 2:
            raise ValueError("Only single-frame 2D DICOM supported")
        pixels = pixels * float(getattr(dataset, "RescaleSlope", 1)) + float(getattr(dataset, "RescaleIntercept", 0))
        if getattr(dataset, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
            pixels = pixels.max() + pixels.min() - pixels
        # The original normalizer checks this after stacking. Checking each slice
        # early has the same accepted input set and avoids persisting invalid data.
        if not np.isfinite(pixels).all():
            raise ValueError("Expected finite single-frame slice volume")
        return pixels

    first_pixels = read_pixels(order[0])
    shape = (len(order), *first_pixels.shape)
    plan = allocation_plan(shape, copy_chunk_bytes=copy_chunk_bytes, disk_reserve_bytes=disk_reserve_bytes)
    available = shutil.disk_usage(workspace).free
    if available < plan["minimum_free_disk_bytes"]:
        raise OSError(
            f"Insufficient temporary disk for exact global MRI normalization: "
            f"free={available}, required={plan['minimum_free_disk_bytes']}"
        )
    if observer is not None:
        observer("allocation_plan", {**plan, "available_disk_bytes": available})
    native = scratch = None
    with tempfile.TemporaryDirectory(prefix="f002-native-", dir=workspace) as temporary:
        try:
            native = np.memmap(Path(temporary) / "native.float32", dtype=np.float32, mode="w+", shape=shape)
            native[0] = first_pixels
            del first_pixels
            dirty_bytes = shape[1] * shape[2] * 4
            for slot, index in enumerate(order[1:], 1):
                pixels = read_pixels(index)
                if pixels.shape != shape[1:]:
                    raise ValueError("Mixed matrix sizes")
                native[slot] = pixels
                del pixels
                dirty_bytes += shape[1] * shape[2] * 4
                if dirty_bytes >= plan["copy_chunk_bytes"]:
                    _release_file_pages(native)
                    dirty_bytes = 0
            native.flush()
            _release_file_pages(native)
            scratch = np.memmap(Path(temporary) / "quantile.float32", dtype=np.float32, mode="w+", shape=shape)
            source, destination = native.reshape(-1), scratch.reshape(-1)
            chunk = plan["copy_chunk_bytes"] // 4
            for start in range(0, source.size, chunk):
                destination[start : start + chunk] = source[start : start + chunk]
                if observer is not None:
                    observer("quantile_copy", {"copied_bytes": min(chunk, source.size - start) * 4})
                _release_file_pages(native)
                _release_file_pages(scratch)
            del source, destination
            scratch.flush()
            _release_file_pages(native)
            _release_file_pages(scratch)
            if observer is not None:
                observer("global_percentiles", {"native_bytes": plan["native_file_bytes"], "overwrite_input": True})
            # Same NumPy percentile default/interpolation and all native voxels.
            # Only the expendable scratch map is partitioned in place.
            lo, hi = np.percentile(scratch, settings["percentiles"], overwrite_input=True)
            _close_mapping(scratch)
            scratch = None
            if hi <= lo:
                raise ValueError("Constant-intensity series")
            count = min(shape[0], settings["slices_per_series"])
            centers = np.rint(np.linspace(0, shape[0] - 1, count)).astype(int)
            needed = sorted(set(np.clip(centers[:, None] + [-1, 0, 1], 0, shape[0] - 1).ravel()))
            resized = {}
            for index in needed:
                # Identical arithmetic/dtypes to normalize_volume, one native
                # slice at a time, using the unchanged global float64 lo/hi.
                normalized = np.rint(np.clip((native[index] - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
                resized[index] = letterbox(normalized, settings["image_size"], spacing)
                if observer is not None:
                    observer(
                        "normalize_slice",
                        {"native_slice_shape": list(shape[1:]), "normalized_dtype": str(normalized.dtype)},
                    )
                del normalized
            _release_file_pages(native)
            windows = np.stack(
                [
                    np.stack([resized[index] for index in np.clip(center + np.array([-1, 0, 1]), 0, shape[0] - 1)])
                    for center in centers
                ]
            )
            orient = np.asarray(headers[order[0]].ImageOrientationPatient, dtype=float)
            normal = np.cross(orient[:3], orient[3:])
            z = np.array(
                [np.dot(np.asarray(headers[index].ImagePositionPatient, dtype=float), normal) for index in order]
            )
            positions = (
                np.zeros(count, np.float32)
                if len(z) == 1
                else ((z[centers] - z[0]) / (z[-1] - z[0])).astype(np.float32)
            )
            return (
                windows,
                positions,
                {
                    "order": method,
                    "native_shape": list(shape),
                    "pixel_spacing_mm": spacing.tolist(),
                    "selected_centers": centers.tolist(),
                    "physical_span_mm": float(z[-1] - z[0]),
                    "normalization": "whole native series p1/p99 before letterbox; 2.5D adjacent slices",
                },
            )
        finally:
            _close_mapping(scratch)
            _close_mapping(native)
