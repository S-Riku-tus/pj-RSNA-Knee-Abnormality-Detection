"""Read exactly one header per sagittal series in fixed32; never read pixels."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import time
import traceback
from collections.abc import Sequence
from pathlib import Path

TAGS = (
    "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "SeriesDescription", "SequenceName",
    "RepetitionTime", "EchoTime", "ScanningSequence", "SequenceVariant", "ScanOptions", "MRAcquisitionType",
    "FlipAngle", "EchoTrainLength", "MagneticFieldStrength", "InversionTime", "ImageType",
    "Manufacturer", "ManufacturerModelName", "SoftwareVersions", "Rows", "Columns", "NumberOfFrames",
    "PixelSpacing", "SliceThickness", "SpacingBetweenSlices", "ImageOrientationPatient", "ImagePositionPatient",
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pin(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": sha(raw)}


def plain(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Sequence):
        return [plain(item) for item in value]
    return str(value)


def save(path, value):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes((json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n").encode())
    os.replace(temporary, path)


def read_first_header(uid, row, image_root, reader):
    sid = row["SeriesInstanceUID"]
    paths = sorted((image_root / uid / sid).glob("*.dcm"))
    entry = {"SeriesInstanceUID": sid, "metadata": {key: row[key] for key in ("Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression")},
             "DICOM_file_count": len(paths), "header_only": True, "pixel_decode_calls": 0}
    if not paths:
        return {**entry, "status": "header_failure", "reason": "no_DICOM_files"}, 0
    path = paths[0]
    try:
        header = reader(path, stop_before_pixels=True, specific_tags=list(TAGS))
        values = {name: plain(getattr(header, name, None)) for name in TAGS}
        require(str(values["StudyInstanceUID"]) == uid and str(values["SeriesInstanceUID"]) == sid, "Header UID differs from official metadata/path")
        raw_values = (json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        entry.update(status="header_success", selected_filename=path.name, selected_file_bytes_stat_only=path.stat().st_size,
                     tags=values, header_projection_sha256=sha(raw_values),
                     TransferSyntaxUID=str(getattr(getattr(header, "file_meta", None), "TransferSyntaxUID", "")),
                     protocol="lexicographically first *.dcm per official Sagittal series; no pixel data/tag, no full-file read/hash")
    except Exception as error:
        entry.update(status="header_failure", reason_type=type(error).__name__, reason=str(error))
    return entry, 1


def run(config, output=Path("/kaggle/working/ORTHODIFF_HEADER32.json")):
    output = Path(output)
    require(not output.exists(), "Fresh metadata diagnostic output required")
    started = time.monotonic()
    state = {"schema_version": "orthodiff_sagittal_same32_header_only_v1", "complete": False,
             "UID_order_sha256": config["UID_order_sha256"], "requested_UIDs": config["uids"],
             "parent_pilot_receipt_pin": config["parent_pilot_receipt_pin"], "records": [], "header_read_calls": 0,
             "scope": {"pixel_decode_calls": 0, "pixel_data_requested": False, "Torch_model_GPU_training": 0,
                       "Gold_or_weak_targets": 0, "Report": 0, "network_Save_Submit": 0,
                       "PD_eligibility_modified": False, "TR_TE_classification_applied": False}}
    try:
        require(len(config["uids"]) == len(set(config["uids"])) == 32, "Same32 unique UID list required")
        require(sha("".join(uid + "\n" for uid in config["uids"]).encode()) == config["UID_order_sha256"], "Frozen UID order differs")
        roots = [Path(value) for value in config["competition_candidates"] if Path(value).is_dir()]
        require(len(roots) == 1, "Exactly one competition mount required")
        root = roots[0]
        require(pin(root / "train_series.csv") == config["series_metadata_pin"], "Original official series metadata SHA/size differs")
        with (root / "train_series.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if row["StudyInstanceUID"] in set(config["uids"])
                    and row["Anatomical_Plane"].strip().casefold() == "sagittal"]
        require({row["StudyInstanceUID"] for row in rows} == set(config["uids"]), "A fixed UID has no sagittal metadata")
        require(len({(row["StudyInstanceUID"], row["SeriesInstanceUID"]) for row in rows}) == len(rows), "Duplicate official series row")
        images = [path for path in (root / "train_series", root / "train_images") if path.is_dir()]
        require(len(images) == 1, "Unique official training DICOM directory required")
        import pydicom

        state["environment"] = {"python": platform.python_version(), "pydicom": pydicom.__version__, "pydicom_path": pydicom.__file__}
        state["series_metadata_pin"] = config["series_metadata_pin"]
        for uid in config["uids"]:
            entries = []
            for row in sorted((row for row in rows if row["StudyInstanceUID"] == uid), key=lambda row: row["SeriesInstanceUID"]):
                entry, calls = read_first_header(uid, row, images[0], pydicom.dcmread)
                state["header_read_calls"] += calls
                entries.append(entry)
            state["records"].append({"StudyInstanceUID": uid, "sagittal_series": entries})
            state["elapsed_seconds"] = time.monotonic() - started
            save(output, state)
        all_series = [entry for row in state["records"] for entry in row["sagittal_series"]]
        state["coverage"] = {"requested_studies": 32, "recorded_studies": len(state["records"]), "sagittal_series": len(all_series),
                             "header_success": sum(entry["status"] == "header_success" for entry in all_series),
                             "header_failure": sum(entry["status"] != "header_success" for entry in all_series)}
        state["complete"] = True
    except Exception as error:
        state["complete"] = False
        state["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    state["elapsed_seconds"] = time.monotonic() - started
    save(output, state)
    print("ORTHODIFF_HEADER32 " + json.dumps({"complete": state["complete"], "coverage": state.get("coverage"), "receipt_pin": pin(output)}), flush=True)
    return state
