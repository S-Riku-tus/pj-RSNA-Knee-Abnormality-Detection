"""Fixed historical192 metadata fixture; no decoder, model, labels or API.

The public path is Kaggle-only, explicit-execution and fixed-pin. Artificial
tests use the private metadata path and an injected empty-directory link mock;
their receipts can never qualify an actual fixture.
"""
from __future__ import annotations

import atexit
import csv
import hashlib
import io
import json
import os
import secrets
import shutil
import tempfile
from pathlib import Path

ID = "StudyInstanceUID"
SERIES_COLUMNS = (ID, "SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane")
TARGETS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
           "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")
SELECTION_SHA256 = "2f004a7cf359f62bc1fa56926f47bd63e5bc909388a304dcbc5080043bea54b0"
ORDER_SHA256 = "2815c9fc459f5a02ade7d305e9acb17ca3b2e4028125b923d9e931b02abfd7c4"
SERIES_SHA256 = "573c1d80772bf41211c91b149c95677385a1c22d63f485c347f1b46c0177aef3"
SCHEMA = "f009_fixed192_metadata_fixture_v1"
COUNT = 192
PREFIX = "f009-fixed192-"
WORKING = Path("/kaggle/working")


def _require(condition, message):
    if condition is not True:
        raise ValueError(message)


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _pin(path):
    raw = Path(path).read_bytes()
    return {"bytes": len(raw), "sha256": _digest(raw)}


def _read(path, cap):
    path = Path(path)
    _require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= cap,
             "Bounded regular metadata file required")
    raw = path.read_bytes()
    _require(len(raw) <= cap, "Metadata grew beyond the fixed cap")
    return raw


def _order_sha(uids):
    return _digest(json.dumps(uids, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _safe_id(value):
    return type(value) is str and bool(value) and value not in (".", "..") and not any(
        char in value for char in ("/", "\\", "\x00", "\n", "\r"))


def _write_csv(path, fields, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def _actual_link(link, target):
    link.symlink_to(target, target_is_directory=True)


def _cleanup_owner(temporary, parent, token, marker_raw, root_identity, link, image_root,
                   *, fixture_only, receipt):
    """Check exact ownership before unlink/rmtree; never traverse the MRI link."""
    _require(not temporary.is_symlink(), "Refusing symlinked fixture root cleanup")
    _require(temporary.exists(), "Owned fixture disappeared before positively verified cleanup")
    _require(temporary.is_dir() and temporary.resolve(strict=True) == temporary
             and temporary.parent == parent and parent.resolve(strict=True) == parent
             and temporary.name.startswith(PREFIX) and temporary.name == receipt["temporary_name"],
             "Refusing escaped or replaced fixture root cleanup")
    identity = temporary.stat()
    _require((identity.st_dev, identity.st_ino) == root_identity,
             "Refusing replaced fixture directory identity")
    marker = temporary / ".fixture-owner.json"
    _require(marker.is_file() and not marker.is_symlink() and marker.stat().st_size == len(marker_raw)
             and marker.read_bytes() == marker_raw and json.loads(marker_raw)["owner_token"] == token,
             "Fixture ownership token/marker differs; no recursive delete performed")
    if link.is_symlink():
        _require(link.resolve(strict=True) == image_root, "Fixture MRI link target changed")
        link.unlink()  # Unlink first. rmtree never traverses the external MRI target.
    elif link.exists():
        _require(fixture_only and link.is_dir() and not any(link.iterdir()),
                 "Only an empty artificial link placeholder may be removed")
    shutil.rmtree(temporary)
    _require(not temporary.exists() and not temporary.is_symlink(), "Owned fixture cleanup incomplete")
    receipt.update(cleanup_complete=True, fixture_root_removed=True, root_already_absent=False)


def _prepare(competition_root, selection_path, expected_selection_sha256, temporary_parent, *,
             expected_order_sha256, expected_series_sha256, fixture_only=False, link_helper=None):
    _require(type(fixture_only) is bool, "Explicit fixture scope required")
    if not fixture_only:
        _require((expected_selection_sha256, expected_order_sha256, expected_series_sha256)
                 == (SELECTION_SHA256, ORDER_SHA256, SERIES_SHA256) and link_helper is None,
                 "Actual route requires fixed three pins and original link helper")
        _require(os.name == "posix", "Actual fixed192 fixture is Kaggle POSIX only")
    else:
        _require(callable(link_helper), "Artificial fixtures require an explicit no-MRI link mock")
    competition = Path(competition_root).resolve(strict=True)
    parent_path = Path(temporary_parent)
    _require(parent_path.is_dir() and not parent_path.is_symlink(), "Explicit existing temporary parent required")
    parent = parent_path.resolve(strict=True)
    working = WORKING.resolve()
    _require(not parent.is_relative_to(working) and not working.is_relative_to(parent),
             "Fixture parent must be outside Kaggle Output working and its ancestors")
    _require(parent != competition and not parent.is_relative_to(competition), "Fixture must be separate from competition Input")
    if not fixture_only:
        _require(competition.is_relative_to(Path("/kaggle/input").resolve())
                 and not parent.is_relative_to(Path("/kaggle/input").resolve()),
                 "Actual competition Input and external temporary roots required")
    selection_path = Path(selection_path)
    _require(selection_path.suffix == ".json", "Selection must be metadata JSON")
    raw = _read(selection_path, 128*1024)
    _require(_digest(raw) == expected_selection_sha256, "Frozen selection source SHA differs")
    selection = json.loads(raw, parse_constant=lambda _: _require(False, "Nonfinite selection"))
    uids = selection["UIDs"]
    _require(type(uids) is list and len(uids) == len(set(uids)) == selection["studies"] == COUNT
             and all(_safe_id(uid) for uid in uids) and _order_sha(uids) == expected_order_sha256,
             "Exact ordered192 unique UIDs required; no reselection or replacement")
    _require(selection["train_series_csv_sha256"] == expected_series_sha256
             and selection["train_csv_Report_labels_read"] is False
             and selection["Gold_labels_fit_select_or_evaluate"] is False
             and selection["old_native64_UID_overlap"] == 0,
             "Frozen metadata-only cohort contract differs")
    series_path = competition / "train_series.csv"
    series_raw = _read(series_path, 64*1024**2)
    _require(_digest(series_raw) == expected_series_sha256, "Frozen train_series metadata SHA differs")
    reader = csv.DictReader(io.StringIO(series_raw.decode("utf-8-sig"), newline=""))
    fields = reader.fieldnames or []
    _require(len(fields) == len(set(fields)) == len(SERIES_COLUMNS) and set(fields) == set(SERIES_COLUMNS),
             "Only the five original series metadata columns are allowed")
    grouped, seen = {uid: [] for uid in uids}, set()
    for source in reader:
        _require(set(source) == set(SERIES_COLUMNS) and all(type(value) is str for value in source.values()),
                 "Malformed series metadata row")
        uid = source[ID]
        if uid in grouped:
            sid = source["SeriesInstanceUID"]
            _require(_safe_id(sid) and (uid, sid) not in seen, "Duplicate/unsafe selected series metadata")
            seen.add((uid, sid))
            grouped[uid].append([source[name] for name in SERIES_COLUMNS])
    _require(all(grouped[uid] for uid in uids), "A fixed selected UID is missing; no substitute permitted")
    image_path = competition / "train_series"
    _require(image_path.is_dir() and not image_path.is_symlink(), "Original MRI directory required; no image enumeration")
    image_root = image_path.resolve(strict=True)
    _require(not image_root.is_relative_to(working), "MRI link cannot point into Output working")
    temporary = Path(tempfile.mkdtemp(prefix=PREFIX, dir=parent)).resolve(strict=True)
    _require(temporary.parent == parent and temporary.name.startswith(PREFIX) and not temporary.is_symlink(),
             "Exclusive owned temporary fixture required")
    identity = temporary.stat()
    root_identity = (identity.st_dev, identity.st_ino)
    token = secrets.token_hex(16)
    marker_raw = (json.dumps({"owner_token": token, "root": str(temporary), "parent": str(parent)}, sort_keys=True)+"\n").encode()
    (temporary / ".fixture-owner.json").write_bytes(marker_raw)
    virtual = temporary / "rsna-knee-abnormality-detection"
    link = virtual / "test_series"
    receipt = {"schema_version": SCHEMA, "fixture_complete": False, "actual_fixture_created": False,
        "artificial_fixture_only": fixture_only, "selection_pin": {"bytes": len(raw), "sha256": _digest(raw)},
        "ordered_UID_sha256": _order_sha(uids), "ordered_UID_hash_encoding": "compact UTF8 JSON list",
        "UIDs": uids, "studies": COUNT, "train_series_csv_sha256": _digest(series_raw),
        "series_rows": len(seen), "reselection_or_replacement": False, "new_selection_seed_used": False,
        "old_native64_UID_overlap_from_frozen_record": 0, "train_csv_Report_labels_read": False,
        "Gold_image_UID_presence_audited": False, "Gold_labels_fit_select_or_evaluate": False,
        "MRI_decode_calls": 0, "MRI_directory_enumerated": False, "model_GPU_API_calls": 0,
        "MRI_link_under_Output_working": False, "P033_predictions_used": False, "submission_CSV_written": False,
        "baseline_purpose": "finite0.5 placeholder; engineering only, no labels/rank/accuracy",
        "temporary_root": str(temporary), "temporary_parent": str(parent), "temporary_name": temporary.name,
        "owner_token": token, "virtual_competition_root": str(virtual), "MRI_source_root": str(image_root),
        "cleanup_complete": False, "fixture_root_removed": False}

    cleanup_done = False

    def cleanup():
        nonlocal cleanup_done
        if cleanup_done:
            return
        receipt["cleanup_complete"] = False  # Output only, never caller-controlled authorization.
        _cleanup_owner(temporary, parent, token, marker_raw, root_identity, link, image_root,
                       fixture_only=fixture_only, receipt=receipt)
        cleanup_done = True
        atexit.unregister(cleanup)

    atexit.register(cleanup)
    try:
        virtual.mkdir(exist_ok=False)
        (link_helper if fixture_only else _actual_link)(link, image_root)
        if not fixture_only:
            _require(link.is_symlink() and link.resolve(strict=True) == image_root, "Actual MRI directory link differs")
        _write_csv(virtual / "test.csv", (ID,), ([uid] for uid in uids))
        _write_csv(virtual / "test_series.csv", SERIES_COLUMNS, (row for uid in uids for row in grouped[uid]))
        baseline = temporary / "diagnostic-baseline.csv"
        for path in (virtual / "sample_submission.csv", baseline):
            _write_csv(path, (ID, *TARGETS), ([uid, *(["0.5"]*12)] for uid in uids))
        receipt.update(fixture_complete=True, actual_fixture_created=not fixture_only,
            MRI_symlink_created=link.is_symlink(), injected_artificial_link_helper=fixture_only,
            generated_csv_pins={path.relative_to(temporary).as_posix(): _pin(path) for path in (
                virtual / "test.csv", virtual / "test_series.csv", virtual / "sample_submission.csv", baseline)})
        return temporary, baseline, receipt, cleanup
    except BaseException:
        cleanup()
        raise


def prepare_fixed192_fixture(competition_root, selection_path, expected_selection_sha256,
                            temporary_parent, execute=False):
    """Public actual route; default refuses before opening any metadata body."""
    _require(execute is True, "Explicit fixed192 fixture execution required")
    _require(expected_selection_sha256 == SELECTION_SHA256, "Fixed saved192 selection pin required")
    return _prepare(competition_root, selection_path, expected_selection_sha256, temporary_parent,
                    expected_order_sha256=ORDER_SHA256, expected_series_sha256=SERIES_SHA256,
                    fixture_only=False)
