"""CSV contracts, identifiers, and reproducibility helpers (standard library only)."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

ID = "StudyInstanceUID"
SERIES_ID = "SeriesInstanceUID"
TARGETS = (
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
SUBMISSION_COLUMNS = (ID, *TARGETS)
SERIES_COLUMNS = (ID, SERIES_ID, "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane")


def read_csv(path, required=()):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        missing = set(required) - set(fields)
        if missing or len(fields) != len(set(fields)):
            raise ValueError(f"Invalid CSV header in {path}; missing={sorted(missing)}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed CSV row in {path}")
    return fields, rows


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def uid(value):
    # IDs remain strings. Accept synthetic IDs too, but never paths.
    if not value or value.strip() != value or any(c in value for c in "/\\:") or value in (".", ".."):
        raise ValueError("Empty or unsafe study/series identifier")
    return value


def index_studies(rows):
    result = {}
    for row in rows:
        key = uid(row[ID])
        if key in result:
            raise ValueError("Duplicate StudyInstanceUID")
        result[key] = row
    return result


def probability(value, *, missing=False, binary=False):
    if missing and value.strip().lower() in ("", "nan", "na", "null"):
        return None
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1 or (binary and number not in (0, 1)):
        raise ValueError("Expected a finite probability in [0, 1]" + (" (binary)" if binary else ""))
    return number


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported config schema")
    p, t = config["preprocess"], config["train"]
    for value in (
        config["n_folds"],
        p["image_size"],
        p["max_series"],
        p["windows_per_series"],
        t["epochs"],
        t["batch_size"],
        t["accumulation_steps"],
    ):
        if type(value) is not int or value < 1:
            raise ValueError("Counts must be positive integers")
    if config["n_folds"] < 2 or p["image_size"] < 32:
        raise ValueError("Need >=2 folds and image_size >=32")
    lo, hi = p["percentiles"]
    if not 0 <= lo < hi <= 100 or config["model"]["backbone"] != "resnet18":
        raise ValueError("Invalid normalization or unsupported backbone")
    return config


def preprocess_fingerprint(config):
    implementation = Path(__file__).with_name("imaging.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    payload = {"settings": config["preprocess"], "implementation": hashlib.sha256(implementation.encode()).hexdigest()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def source_hashes():
    return {
        p.name: hashlib.sha256(p.read_text(encoding="utf-8").replace("\r\n", "\n").encode()).hexdigest()
        for p in sorted(Path(__file__).parent.glob("*.py"))
    }


def validate_submission(path, test_path, sample_path=None):
    fields, predictions = read_csv(path, SUBMISSION_COLUMNS)
    if tuple(fields) != SUBMISSION_COLUMNS:
        raise ValueError("Submission column order must match the competition exactly")
    _, test = read_csv(test_path, (ID,))
    expected, actual = index_studies(test), index_studies(predictions)
    if not expected or expected.keys() != actual.keys():
        raise ValueError("Submission study IDs must match test.csv exactly")
    if [r[ID] for r in test] != [r[ID] for r in predictions]:
        raise ValueError("Submission row order must follow test.csv")
    if sample_path:
        header, _ = read_csv(sample_path, SUBMISSION_COLUMNS)
        if tuple(header) != SUBMISSION_COLUMNS:
            raise ValueError("Live sample submission differs; update the contract before proceeding")
    for row in predictions:
        for target in TARGETS:
            probability(row[target])
    return {"studies": len(predictions), "targets": len(TARGETS), "valid": True}
