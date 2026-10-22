"""Prepare metadata only after actual parent qualification; never MRI/GPU/fit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rsna_knee import orthodiff_fluid_fallback as F  # noqa: E402

SPEC = ROOT / "artifacts/research/20261017-orthodiff-pooling-contrast-audit-v1/fluid_sensitive_fallback_preregistered.json"
COHORT = ROOT / "artifacts/research/20261017-orthodiff-fullweak-preparation-v1/cohort.json"
SERIES = ROOT / "data/exports/rsraki-rsna-knee-sv354838181/rsna-cache-v1/raw/train_series.csv"
COHORT_SHA = "86d976f55bb67bd3a738679625c9281aead06b0c7f01d7c894ffe870ead65246"
SERIES_SHA = "573c1d80772bf41211c91b149c95677385a1c22d63f485c347f1b46c0177aef3"


def file_pin(path, maximum=4 * 1024**2):
    path = Path(path)
    F.require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= maximum, "Bounded regular input required")
    h, length = hashlib.sha256(), 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            h.update(block)
            length += len(block)
            F.require(length <= maximum, "Input grew beyond bound")
    return {"bytes": length, "sha256": h.hexdigest()}


def prepare(args):
    F.require(not args.output.exists(), "Fresh metadata subset output required")
    pins = {}

    def read(path, expected=None):
        value = file_pin(path)
        if expected is not None:
            F.require(value == {key: expected[key] for key in ("bytes", "sha256")}, "Positive parent input pin differs")
        pins[str(path.resolve())] = value
        return F.parse_json(path.read_bytes())

    F.require(file_pin(args.qualification)["sha256"] == args.expected_qualification_sha256, "Expected independent parent qualification SHA differs")
    audit = read(args.qualification)
    # Stop before reading other data or creating any output if actual parent is absent.
    F.require(audit.get("passed") is True and audit.get("head_features_qualified") is True
              and audit.get("schema_version") == "orthodiff_fullweak4207_actual_qualification_v1",
              "Actual complete parent qualification required; preparation evidence is insufficient")
    summary = read(args.summary, audit["summary_pin"])
    status = read(args.status, audit["status_pin"])
    F.require_qualified_parent(audit, summary, status)
    records_pin = file_pin(args.records, 128 * 1024**2)
    F.require(records_pin == audit["records_pin"]
              and summary["journal"]["sha256"] == status["journal"]["sha256"] == records_pin["sha256"]
              and summary["journal"]["bytes"] == status["journal"]["bytes"] == records_pin["bytes"], "Complete parent journal pin differs")
    pins[str(args.records.resolve())] = records_pin
    F.require(file_pin(args.spec)["sha256"] == F.SPEC_SHA and file_pin(args.cohort)["sha256"] == COHORT_SHA
              and file_pin(args.series_metadata)["sha256"] == SERIES_SHA, "Predeclared spec/cohort/official metadata differs")
    spec, cohort = read(args.spec), read(args.cohort)
    F.require(spec["schema_version"] == "orthodiff_fluid_sensitive_fallback_future_one_factor_v1"
              and cohort["Gold_UID_overlap"] == cohort["Gold_supplied_group_overlap"] == 0
              and cohort["target_values_Report_or_Gold_label_values_retained"] is False, "Reserved Gold exclusion metadata required")
    uids = [row[F.ID] for row in cohort["requested"]]
    F.require(len(uids) == len(set(uids)) == 4207 and uids == summary["requested_UIDs"]
              and hashlib.sha256("".join(uid + "\n" for uid in uids).encode()).hexdigest() == F.UID_SHA,
              "Original complete4207 UID order required")
    pins[str(args.series_metadata.resolve())] = file_pin(args.series_metadata)
    wanted = set(uids)
    with args.series_metadata.open(encoding="utf-8-sig", newline="") as handle:
        fields = (F.ID, "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression")
        series = [{key: row[key] for key in fields} for row in csv.DictReader(handle) if row[F.ID] in wanted]

    consumed_digest, consumed_bytes = hashlib.sha256(), 0

    def records():
        nonlocal consumed_bytes
        with args.records.open("rb") as handle:
            for line in handle:
                F.require(line.endswith(b"\n") and 0 < len(line) <= 256 * 1024 and line.strip(), "Bounded complete parent JSONL required")
                consumed_digest.update(line)
                consumed_bytes += len(line)
                F.require(consumed_bytes <= 128 * 1024**2, "Parent journal grew beyond bound")
                yield F.parse_json(line)

    result = F.build_metadata_subset(records(), cohort["requested"], series)
    F.require({"bytes": consumed_bytes, "sha256": consumed_digest.hexdigest()} == records_pin, "Consumed parent journal pin changed")
    F.require(result["parent_record_counts"].get("success", 0) == audit["coverage"]["success"]
              and result["parent_record_counts"].get("data_failure", 0) == audit["coverage"]["failure"], "Parent qualified status counts differ")
    result.update(parent_qualification_sha256=args.expected_qualification_sha256, parent_source_sha256=F.SOURCE_SHA,
                  parent_UID_order_sha256=F.UID_SHA, input_pins=pins,
                  source_pins={str(path.relative_to(ROOT)): file_pin(path) for path in (Path(__file__), ROOT / "src/rsna_knee/orthodiff_fluid_fallback.py")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("qualification", "records", "summary", "status", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-qualification-sha256", required=True)
    parser.add_argument("--spec", type=Path, default=SPEC)
    parser.add_argument("--cohort", type=Path, default=COHORT)
    parser.add_argument("--series-metadata", type=Path, default=SERIES)
    args = parser.parse_args()
    value = prepare(args)
    print(json.dumps({key: value[key] for key in ("subset_count", "selected_metadata_candidate_count", "metadata_unavailable_count", "adopted")}))


if __name__ == "__main__":
    main()
