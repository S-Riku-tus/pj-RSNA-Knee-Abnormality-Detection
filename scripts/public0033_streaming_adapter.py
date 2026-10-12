"""Prepare a separate low-RAM Meniscus10 branch without changing Apex inputs.

Pixel operations follow the cache consumed by the public Tri-Specialist V2
Notebook (script 355742320), not Apex's canonical ConvNeXt/CoAtNet volumes.
Source attribution:
https://www.kaggle.com/code/sujanmajhisuzan/rsna-knee-tri-specialist-superstack
https://www.kaggle.com/datasets/renta0426/rsna-knee-public0033-meniscus-bag-v1/versions/2

The released checkpoint/runtime use Apache-2.0; upstream DINOv2 and competition
data retain their conditions. This adapter is new project code. Public0033 is
a weak fullfit checkpoint, never a fold-safe OOF model. No downloads, real
inference, training, remote execution or submissions occur on import.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

LABELS = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)
MENISCUS = ("Medial Meniscus", "Lateral Meniscus")
SLOTS = (
    ("SAG_FLUID_FS", "Sagittal", True, True),
    ("COR_FLUID_FS", "Coronal", True, True),
    ("AX_FLUID_FS", "Axial", True, True),
    ("SAG_FLUID_NOFS", "Sagittal", True, False),
    ("COR_T1", "Coronal", False, False),
    ("SAG_T1", "Sagittal", False, False),
)
CACHE_SHAPE = (6, 12, 336, 336)
WINDOW_STARTS = tuple(range(10))
PINNED_RUNTIME_SHA256 = "9541f82a993d7dc2942ca2a5112e110471285f8d80701846079a3dc708761683"
HEADER_TAGS = (
    "SeriesDescription",
    "SequenceName",
    "ScanOptions",
    "ScanningSequence",
    "RepetitionTime",
    "EchoTime",
    "Laterality",
    "PixelSpacing",
    "Rows",
    "Columns",
    "RescaleSlope",
    "RescaleIntercept",
    "ImagePositionPatient",
    "ImageOrientationPatient",
)
FATSAT_OPTIONS = {"FS", "FATSAT", "FAT_SAT", "FSAT"}
FATSAT_PATTERN = re.compile(
    r"\bfs\b|fatsat|fat sat|\bstir\b|\bspair\b|\bspir\b|\bwe\b|water excit|\btirm\b|\bsting\b|"
    r"\bfatsup\b|smart fat|\bwater\b"
)


def _numeric(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return float("nan")


def _vector(value, length):
    import numpy as np

    if not isinstance(value, str):
        return None
    try:
        parts = np.asarray([float(piece) for piece in value.split("|")])
    except ValueError:
        return None
    return parts if len(parts) >= length else None


def classify_header(header):
    """Reference recovered-slot classification from source DICOM text/sequence."""
    description = str(header.get("SeriesDescription") or "") + " " + str(header.get("SequenceName") or "")
    description = re.sub(r"[_\-.]", " ", description.lower())
    options = str(header.get("ScanOptions") or "").upper().split("|")
    fat_sat = bool(FATSAT_PATTERN.search(description)) or any(option.strip() in FATSAT_OPTIONS for option in options)
    t1 = bool(re.search(r"\bt1\b|\bt1w\b", description))
    t2 = bool(re.search(r"\bt2\b|\bt2w\b", description))
    pdw = bool(re.search(r"\bpd\b|\bpdw\b|proton|\bdp\b|dens", description))
    tr, te = _numeric(header.get("RepetitionTime")), _numeric(header.get("EchoTime"))
    gradient = "GR" in str(header.get("ScanningSequence") or "").upper()
    weight = (
        "T1"
        if t1 and not t2 and not pdw
        else "T2"
        if t2 and not pdw
        else "PD"
        if pdw
        else "GRE"
        if gradient
        else "T1"
        if tr < 800
        else "T2"
        if te > 60
        else "PD"
        if tr >= 800
        else "UNK"
    )
    spacing = _numeric(str(header.get("PixelSpacing") or "").split("|")[0])
    return {**header, "fatsat": fat_sat, "fluid": weight in ("PD", "T2"), "weight": weight, "px": spacing}


def choose_recovered_slots(records):
    """Select longest eligible series, preserving reference numpy quicksort ties."""
    import numpy as np

    selected = {}
    for name, plane, fluid, fat_sat in SLOTS:
        eligible = [
            row for row in records if row.get("plane") == plane and row["fatsat"] == fat_sat and row["fluid"] == fluid
        ]
        if eligible:
            counts = np.asarray([row["n_slices"] for row in eligible], dtype=np.int64)
            # pandas descending nargsort reverses source values/indices before
            # quicksort, then reverses its indexer. Do not introduce a new tie rule.
            indexer = np.arange(len(counts))[::-1][counts[::-1].argsort(kind="quicksort")][::-1]
            selected[name] = eligible[int(indexer[0])]
    return selected


def laterality_from_headers(records):
    """Source first explicit L/R tag; otherwise median image-centre LPS x."""
    import numpy as np

    explicit = [str(row.get("Laterality") or "").strip().upper() for row in records]
    explicit = [value[0] for value in explicit if value and value[0] in ("L", "R")]
    if explicit:
        return explicit[0]
    centre_x = []
    for row in records:
        ipp, iop, spacing = (
            _vector(row.get(key), length)
            for key, length in (
                ("ImagePositionPatient", 3),
                ("ImageOrientationPatient", 6),
                ("PixelSpacing", 2),
            )
        )
        rows, columns = row.get("Rows"), row.get("Columns")
        if ipp is None or iop is None or spacing is None or not rows or not columns:
            continue
        try:
            centre = ipp[:3] + iop[:3] * spacing[1] * float(columns) / 2 + iop[3:6] * spacing[0] * float(rows) / 2
        except (ValueError, TypeError):
            continue
        centre_x.append(float(centre[0]))
    if not centre_x:
        return None
    middle = float(np.median(centre_x))
    return None if abs(middle) < 20 else "R" if middle < 0 else "L"


def _safe_child(root, identifier):
    identifier = str(identifier)
    path = root / identifier
    if not identifier or path.resolve().parent != root.resolve() or path.name != identifier:
        raise ValueError("study/series path must be a single child identifier")
    return path


def make_dicom_study_builder(series_rows, test_series_root):
    """Construct independent Meniscus raw-DICOM input, lazily one requested study.

    Only competition series UID/plane metadata are consumed; reports and training
    labels are unnecessary. This closure performs no IO until called. It matches
    the public cache's midpoint header classification, whole-selected-series
    projection order, central12 sampling and native/laterality pixel operations.
    Decoder availability and real data runtime still require Kaggle profiling.
    """
    import numpy as np
    import pydicom

    root = Path(test_series_root).resolve()
    plane_map = {str(row["SeriesInstanceUID"]): row["Anatomical_Plane"] for row in series_rows}

    def probe(path):
        files = sorted(entry.name for entry in os.scandir(path) if entry.name.endswith(".dcm"))
        record = {"dir": path, "files": files, "n_slices": len(files), "plane": plane_map.get(path.name)}
        if files:
            header = pydicom.dcmread(path / files[len(files) // 2], stop_before_pixels=True, force=True)
            for key in HEADER_TAGS:
                value = getattr(header, key, None)
                if value is None:
                    record[key] = None
                elif isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
                    record[key] = "|".join(map(str, value))
                else:
                    record[key] = str(value)
        return classify_header(record)

    def order(record):
        keyed = []
        for filename in record["files"]:
            dataset = None
            try:
                try:
                    dataset = pydicom.dcmread(record["dir"] / filename, stop_before_pixels=True)
                except Exception:
                    dataset = pydicom.dcmread(
                        record["dir"] / filename,
                        stop_before_pixels=True,
                        force=True,
                        specific_tags=[(32, 50), (32, 55), (32, 19)],
                    )
                iop, ipp = (
                    np.asarray(dataset.ImageOrientationPatient, dtype=float),
                    np.asarray(dataset.ImagePositionPatient, dtype=float),
                )
                position = float(np.dot(ipp, np.cross(iop[:3], iop[3:])))
            except Exception:
                try:
                    position = float(dataset.InstanceNumber)
                except Exception:
                    position = None
            keyed.append((position, filename))
        if any(position is None for position, _ in keyed):
            return sorted(
                record["files"],
                key=lambda filename: tuple(
                    int(piece) if piece.isdigit() else piece.lower() for piece in re.split(r"(\d+)", filename)
                ),
            )
        return [filename for _, filename in sorted(keyed, key=lambda value: value[0])]

    def build(uid):
        study = _safe_child(root, uid)
        if not study.is_dir():
            raise FileNotFoundError("requested study has no DICOM directory")
        records = []
        for entry in os.scandir(study):
            if entry.is_dir():
                try:
                    records.append(probe(_safe_child(study, entry.name)))
                except Exception:
                    # The public probe flags unreadable metadata and that series
                    # is not eligible. It must not borrow a different plane/slot.
                    continue
        chosen = choose_recovered_slots(records)
        laterality = laterality_from_headers(records)
        slots = {}
        for name, record in chosen.items():
            ordered = order(record)

            def read_pixels(index, directory=record["dir"], files=ordered):
                dataset = pydicom.dcmread(directory / files[index], force=True)
                array = dataset.pixel_array.astype(np.float32)
                return array * float(getattr(dataset, "RescaleSlope", 1) or 1) + float(
                    getattr(dataset, "RescaleIntercept", 0) or 0
                )

            slots[name] = {"file_count": len(ordered), "read_pixels": read_pixels, "pixel_spacing": record["px"]}
        return prepare_study(slots, laterality=laterality)

    return build


def selected_indices(file_count):
    """Reference floor/unique/pad-last sampling over the central 20--80%."""
    import numpy as np

    if file_count <= 0:
        raise ValueError("slot must contain at least one source frame")
    lo, hi = int(0.2 * (file_count - 1)), int(0.8 * (file_count - 1))
    indices = np.unique(np.linspace(lo, hi, 12).astype(int)) if hi > lo else np.array([file_count // 2])
    while len(indices) < 12:
        indices = np.append(indices, indices[-1])
    return indices[:12]


def prepare_slot(file_count, read_pixels, pixel_spacing, *, plane, laterality=None):
    """Decode only selected native frames; reproduce the reference healthy pixels.

    ``read_pixels(index)`` supplies rescaled float32 pixels in the source orientation.
    It must not apply Apex's MONOCHROME1 inversion, canonical rotation, resampling,
    normalization, or crop. A fully unavailable slot returns None and is masked.
    """
    import numpy as np
    import torch
    import torch.nn.functional as functional

    if file_count <= 0:
        return None
    indices = selected_indices(file_count)
    decoded = {}
    for index in sorted(set(map(int, indices))):
        try:
            pixels = np.asarray(read_pixels(index), dtype=np.float32)
            if pixels.ndim != 2 or not np.isfinite(pixels).all():
                raise ValueError("expected finite 2-D rescaled pixels")
            decoded[index] = pixels
        except Exception:
            decoded[index] = None
    planes = [decoded[int(index)] for index in indices]
    good = [index for index, value in enumerate(planes) if value is not None]
    if not good:
        return None
    shape = Counter(planes[index].shape for index in good).most_common(1)[0][0]
    for index in good:
        if planes[index].shape != shape:
            planes[index] = None
    good = [index for index in good if planes[index] is not None]
    for index, value in enumerate(planes):
        if value is None:
            planes[index] = planes[min(good, key=lambda candidate: abs(candidate - index))]
    volume = np.stack(planes)
    if pixel_spacing and np.isfinite(pixel_spacing) and pixel_spacing > 0:
        want = int(round(130.0 / pixel_spacing))
        height, width = shape
        if 16 < want < min(height, width):
            centre_y, centre_x, half = height // 2, width // 2, want // 2
            volume = volume[:, max(0, centre_y - half) : centre_y + half, max(0, centre_x - half) : centre_x + half]
    low, high = np.percentile(volume, [1, 99])
    volume = np.clip((volume - low) / max(high - low, 1e-6), 0, 1)
    tensor = torch.from_numpy(np.ascontiguousarray(volume)).unsqueeze(0)
    tensor = functional.interpolate(tensor, size=(336, 336), mode="bilinear", align_corners=False)
    tensor = (tensor.squeeze(0) * 255).round().clamp(0, 255).to(torch.uint8)
    if laterality == "R":
        tensor = torch.flip(tensor, dims=[-1] if plane in ("Coronal", "Axial") else [0])
    return tensor.numpy()


def prepare_study(slots, *, laterality=None):
    """Build only one exact uint8 study; caller supplies audited slot/header choices.

    ``slots`` maps the six recovered names to dictionaries containing file_count,
    read_pixels and pixel_spacing. Slice ordering, series classification and
    laterality provenance remain explicit caller responsibilities. This function
    does not guess those decisions from Apex's already processed arrays.
    """
    import numpy as np

    unknown = set(slots) - {record[0] for record in SLOTS}
    if unknown:
        raise ValueError(f"unknown recovered slots: {sorted(unknown)}")
    cache = np.zeros(CACHE_SHAPE, dtype=np.uint8)
    mask = np.zeros(6, dtype=np.float32)
    for index, (name, plane, _, _) in enumerate(SLOTS):
        if name not in slots:
            continue
        record = slots[name]
        pixels = prepare_slot(
            record["file_count"],
            record["read_pixels"],
            record["pixel_spacing"],
            plane=plane,
            laterality=laterality,
        )
        if pixels is not None:
            cache[index] = pixels
            mask[index] = 1
    if not bool(mask.sum()):
        raise ValueError("no valid Meniscus10 slot: preserve incumbent instead of inventing a prediction")
    return cache, mask


@contextmanager
def memmap_cache(study_ids, build_study, *, scratch):
    """Yield the stock runtime's exact dense API while building one study at a time.

    The cohort pixels live on scratch disk. Only one study and the decoder's selected
    frames need anonymous RAM during construction. The stock runtime remains
    unchanged and uses its own FP32/T4x2/batch32 checks. Its peak GPU memory and page
    cache behaviour must still be profiled on Kaggle. UIDs returned here are sorted;
    the final overlay separately restores incumbent row order.
    """
    import numpy as np

    ids = list(study_ids)
    if len(ids) < 2 or any(not isinstance(uid, str) or not uid for uid in ids) or len(set(ids)) != len(ids):
        raise ValueError("expected at least two unique, nonempty study UIDs")
    ordered = sorted(ids)
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    required = len(ordered) * math.prod(CACHE_SHAPE)
    if shutil.disk_usage(scratch).free < required + (512 << 20):
        raise OSError("insufficient temporary disk for complete Meniscus10 cache")
    array = None
    with tempfile.TemporaryDirectory(prefix="meniscus10-pixels-", dir=scratch) as temporary:
        cache_path = Path(temporary) / "cache.npy"
        array = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.uint8, shape=(len(ordered), *CACHE_SHAPE))
        masks = np.zeros((len(ordered), 6), dtype=np.float32)
        try:
            for index, uid in enumerate(ordered):
                pixels, mask = build_study(uid)
                if pixels.shape != CACHE_SHAPE or pixels.dtype != np.uint8:
                    raise ValueError("Meniscus10 study shape/dtype differs from checkpoint input")
                if mask.shape != (6,) or mask.dtype != np.float32 or not np.isfinite(mask).all():
                    raise ValueError("Meniscus10 slot mask shape/dtype/finite contract failed")
                if np.any((mask < 0) | (mask > 1)) or not bool(mask.sum()):
                    raise ValueError("Meniscus10 study mask is unavailable or out of range")
                array[index] = pixels
                masks[index] = mask
                del pixels, mask
            array.flush()
            yield array, masks, ordered
        finally:
            if array is not None:
                array.flush()
                array._mmap.close()


def _read_csv(path, labels):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["StudyInstanceUID", *labels]:
            raise ValueError("submission/raw CSV column contract differs")
        rows = list(reader)
    if not rows or any(not row["StudyInstanceUID"] for row in rows):
        raise ValueError("empty submission/raw CSV or empty UID")
    if len({row["StudyInstanceUID"] for row in rows}) != len(rows):
        raise ValueError("duplicate prediction UIDs")
    for row in rows:
        for label in labels:
            value = float(row[label])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("prediction outside finite [0,1]")
    return rows


def _rank_percentile(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranked = [0.0] * len(values)
    offset = 0
    while offset < len(order):
        end = offset + 1
        while end < len(order) and values[order[end]] == values[order[offset]]:
            end += 1
        percentile = ((offset + 1 + end) / 2) / len(values)
        for index in order[offset:end]:
            ranked[index] = percentile
        offset = end
    return ranked


def fixed_meniscus_overlay(incumbent, branch, output):
    """Prespecified 90:10 global-rank hypothesis, only the two Meniscus columns.

    The incumbent file stays byte-identical. No weak/gold labels or score feedback
    are read. This does not inherit the original author's exact0.936 route or claim
    its score; Apex+Meniscus10 is a new unmeasured candidate.
    """
    incumbent, branch, output = map(Path, (incumbent, branch, output))
    if output.exists() or output.resolve() in (incumbent.resolve(), branch.resolve()):
        raise FileExistsError("overlay requires a new output path")
    before = hashlib.sha256(incumbent.read_bytes()).hexdigest()
    base = _read_csv(incumbent, LABELS)
    raw = _read_csv(branch, MENISCUS)
    indexed = {row["StudyInstanceUID"]: row for row in raw}
    if set(indexed) != {row["StudyInstanceUID"] for row in base}:
        raise ValueError("branch UID coverage differs from incumbent")
    result = [dict(row) for row in base]
    for label in MENISCUS:
        rank_base = _rank_percentile([float(row[label]) for row in base])
        rank_branch = _rank_percentile([float(indexed[row["StudyInstanceUID"]][label]) for row in base])
        mixed = [0.9 * left + 0.1 * right for left, right in zip(rank_base, rank_branch)]
        for row, value in zip(result, _rank_percentile(mixed)):
            row[label] = format(value, ".17g")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["StudyInstanceUID", *LABELS])
        writer.writeheader()
        writer.writerows(result)
    if hashlib.sha256(incumbent.read_bytes()).hexdigest() != before:
        raise RuntimeError("incumbent unexpectedly changed")
    return {
        "schema": "apex_meniscus10_fixed_overlay_v1",
        "incumbent_sha256": before,
        "branch_sha256": hashlib.sha256(branch.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "study_count": len(base),
        "targets": list(MENISCUS),
        "branch_rank_weight": 0.1,
        "other_target_strings_preserved": True,
        "labels_or_score_feedback_read": False,
        "public_improvement_verified": False,
    }


def save_receipt(path, receipt):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def load_pinned_runtime(bundle_root):
    """Import only the separately mounted, pinned public runtime; download nothing."""
    bundle_root = Path(bundle_root)
    path = bundle_root / "public0033_runtime.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != PINNED_RUNTIME_SHA256:
        raise ValueError("public0033 runtime SHA256 differs from audited Dataset V2")
    spec = importlib.util.spec_from_file_location("public0033_pinned_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_separate_branch(*, bundle_root, series_rows, test_series_root, study_ids, output_directory, scratch):
    """Run the stock branch into a fresh directory, leaving Apex CSV untouched.

    This explicit caller entrypoint requires existing Kaggle Inputs and T4x2. No
    code starts remotely or submits files. The output's fixed rank overlay is a
    separate step after full branch success; a failed branch leaves no new candidate.
    """
    output_directory = Path(output_directory)
    if output_directory.exists():
        raise FileExistsError("branch output must use a new directory")
    module = load_pinned_runtime(bundle_root)
    module.verify_bundle(Path(bundle_root))
    module._require_t4x2()
    build_study = make_dicom_study_builder(series_rows, test_series_root)
    output_directory.mkdir(parents=True, exist_ok=False)
    paths = {
        "bag_raw_csv": str(output_directory / "public0033_bag_raw.csv"),
        "receipt_json": str(output_directory / "public0033_cached_inference_receipt.json"),
        "work_dir": str(output_directory),
    }
    with memmap_cache(study_ids, build_study, scratch=scratch) as (cache, mask, ordered):
        return module.run_cached_inference(cache, mask, ordered, paths)
