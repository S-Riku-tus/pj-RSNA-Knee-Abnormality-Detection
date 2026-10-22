"""Qualified regional1024 transport, with immutable parent3063/1144 parity."""
from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path

SOURCE_SHA = "52842d703bcf05c762a6bf44a2c520fbdd47589239f553cc5aaabbaa7220024e"
RUNTIME_SHA = "4b24207e9160a87bce2ee6fce3016ea04f19639eee1ef48f11c5d2278d5272f9"
UID_SHA = "911f91fbc914213db045780675d6518a8cc4a49808599bbb74f2f247bc30eb82"
PARENT_BUNDLE_SHA = "525094b583c7cd821769fac56b9e64a524d5ef185a4a9eb689f91480b31ce83c"
PARENT_AUDIT_SHA = "7fbc160b8715e69dc506b64e9187f55a2c2b67925bf6a5b0a4a95adf943c8270"
PARENT_SOURCE_SHA = "15bb2d6c7c224021f95a304b08eb9e3644164ea8373f4ae6ec1c7afc77582e1f"
STEM = "ORTHODIFF_REGIONAL2X2_"
SCIENTIFIC_FIELDS = ("StudyInstanceUID", "status", "metadata", "noise_seed", "normalized_input",
                     "normalization_min_max_float32", "noisy_input", "mid_2", "pooled",
                     "pooled_float32_values", "forward_counts", "all_RNG_restored")


def require(value, message):
    if not value:
        raise ValueError(message)


def pin(path, maximum=128 * 1024**2):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= maximum, "Bounded regular regional artifact required")
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            size += len(block)
            require(size <= maximum, "Regional artifact grew beyond bound")
            digest.update(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def checked(path, expected):
    require(pin(path) == {key: expected[key] for key in ("bytes", "sha256")}, "Regional positive file pin differs")
    return Path(path)


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "Duplicate regional JSON field")
        result[key] = value
    return result


def parse(raw):
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON token")))


def read(path):
    pin(path, 4 * 1024**2)
    return parse(Path(path).read_bytes())


def line_bytes(record):
    return (json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n").encode()


def vector(record, regional=True):
    dimension, key, info = (1024, "regional_2x2_float32_values", "regional_2x2") if regional else (256, "pooled_float32_values", "pooled")
    values = record.get(key)
    require(isinstance(values, list) and len(values) == dimension and all(type(x) in (float, int) and math.isfinite(x) for x in values), "Finite exact-dimension regional values required")
    raw = struct.pack("<" + str(dimension) + "f", *values)
    require(list(struct.unpack("<" + str(dimension) + "f", raw)) == values and record.get(info) == {
        "shape": [1, dimension], "dtype": "torch.float32", "bytes": 4 * dimension, "sha256": hashlib.sha256(raw).hexdigest()}, "Regional FP32 vector bits/SHA differ")
    return values


def compare_parent(record, parent, parent_line):
    """Regional success must preserve all parent science; failures preserve bytes."""
    if parent.get("status") == "data_failure":
        require(record == parent and line_bytes(record) == parent_line and record.get("feature_forward_calls") == 0
                and record.get("fallback_features") is False and not any(key in record for key in (
                    "regional_2x2", "regional_2x2_float32_values", "pooled", "pooled_float32_values", "mid_2", "normalized_input")),
                "Parent failure changed or fabricated regional features")
        return
    require(parent.get("status") == record.get("status") == "success" and all(record.get(key) == parent.get(key) for key in SCIENTIFIC_FIELDS),
            "Parent3063 scientific evidence differs")
    a, b = vector(record, regional=False), vector(parent, regional=False)
    require(struct.pack("<256f", *a) == struct.pack("<256f", *b), "Parent global256 FP32 bits changed")
    require(record.get("parent_record_line_sha256") == hashlib.sha256(parent_line).hexdigest()
            and record.get("parent_scientific_parity_passed") is True and record.get("regional_pooling_seconds", -1) >= 0,
            "Parent line/parity or regional latency evidence differs")
    vector(record)


def parent_evidence(path):
    require(pin(path)["sha256"] == PARENT_BUNDLE_SHA, "Immutable original parent bundle required")
    bundle = read(path)
    require(bundle.get("schema_version") == "orthodiff_fullweak4207_qualified_feature_bundle_v1" and bundle.get("qualified") is True
            and bundle.get("source_sha256") == PARENT_SOURCE_SHA and bundle.get("UID_order_sha256") == UID_SHA
            and bundle.get("requested_count") == 4207 and bundle["coverage"]["success"] == 3063 and bundle["coverage"]["failure"] == 1144,
            "Original explicit-PD parent partition required")
    reference = bundle["qualification"]
    checked(reference["path"], reference)
    require(reference["sha256"] == PARENT_AUDIT_SHA, "Original parent audit pin required")
    audit = read(reference["path"])
    require(audit.get("passed") is audit.get("head_features_qualified") is True and audit.get("prior32_comparison", {}).get("passed") is True,
            "Parent actual qualification required")
    require(set(bundle["files"]) == {"records", "summary", "status"}, "Parent raw3 required")
    for name, expected in bundle["files"].items():
        checked(expected["path"], expected)
        require({key: expected[key] for key in ("bytes", "sha256")} == audit[name + "_pin"], "Parent raw/audit pin differs")
    summary, status = (read(bundle["files"][name]["path"]) for name in ("summary", "status"))
    require(summary.get("complete") is status.get("complete") is True and summary.get("phase") == status.get("phase") == "full4207_finished"
            and summary["coverage"] == status["coverage"] == bundle["coverage"], "Parent complete terminal required")
    uids = summary["requested_UIDs"]
    require(len(uids) == len(set(uids)) == 4207 and hashlib.sha256("".join(uid + "\n" for uid in uids).encode()).hexdigest() == UID_SHA,
            "Exact parent4207 UID order required")
    return bundle, audit, summary


def audit_evidence(path, expected_sha, parent_path):
    require(isinstance(expected_sha, str) and len(expected_sha) == 64 and pin(path)["sha256"] == expected_sha, "Expected regional audit SHA required")
    audit = read(path)
    require(audit.get("schema_version") == "orthodiff_regional2x2_actual_qualification_v1"
            and audit.get("passed") is audit.get("head_features_qualified") is True
            and audit.get("source_sha256") == SOURCE_SHA and audit.get("UID_order_sha256") == UID_SHA
            and audit.get("requested_count") == 4207 and audit.get("feature_dimension") == 1024
            and audit["coverage"]["requested"] == audit["coverage"]["recorded"] == 4207
            and audit["coverage"]["success"] == 3063 and audit["coverage"]["failure"] == 1144
            and audit.get("parent_bundle_pin") == pin(parent_path)
            and audit.get("parent_qualification_pin", {}).get("sha256") == PARENT_AUDIT_SHA
            and audit.get("parent_comparison", {}).get("passed") is True
            and audit["parent_comparison"]["successful_scientific_parity"] == 3063
            and audit["parent_comparison"]["failure_records_byte_exact"] == 1144
            and audit.get("FD_availability_mixed") is False,
            "Actual regional1024 qualification is absent or differs")
    return audit


def raw_evidence(audit, records, summary, status):
    files = {}
    for name, path, suffix in (("records", records, "RECORDS.jsonl"), ("summary", summary, "SUMMARY.json"), ("status", status, "STATUS.json")):
        require(Path(path).name == STEM + suffix, "Regional artifact filename differs")
        checked(path, audit[name + "_pin"])
        files[name] = {"path": str(Path(path).resolve()), **pin(path)}
    a, b = read(summary), read(status)
    require(a.get("complete") is b.get("complete") is True and a.get("phase") == b.get("phase") == "regional4207_finished"
            and a["coverage"] == b["coverage"] == audit["coverage"]
            and a["journal"]["sha256"] == b["journal"]["sha256"] == files["records"]["sha256"], "Regional terminal/raw3 hash differs")
    return files, a


def seal_regional_bundle(parent_path, audit_path, expected_audit_sha, records, summary, status, output):
    require(not Path(output).exists(), "New regional bundle output required")
    parent, _, parent_summary = parent_evidence(parent_path)
    audit = audit_evidence(audit_path, expected_audit_sha, parent_path)
    files, actual_summary = raw_evidence(audit, records, summary, status)
    require(actual_summary["requested_UIDs"] == parent_summary["requested_UIDs"], "Regional and parent4207 order differs")
    value = {"schema_version": "orthodiff_regional2x2_qualified_feature_bundle_v1", "qualified": True,
             "source_sha256": SOURCE_SHA, "UID_order_sha256": UID_SHA, "requested_count": 4207, "feature_dimension": 1024,
             "coverage": audit["coverage"], "files": files,
             "qualification": {"path": str(Path(audit_path).resolve()), **pin(audit_path)},
             "parent": {"path": str(Path(parent_path).resolve()), **pin(parent_path)}, "parent_qualification": parent["qualification"],
             "parent_comparison": audit["parent_comparison"], "FD_availability_mixed": False,
             "scope": "Qualified regional1024 transport only, exact original3063 successful and1144 failed UID/group/fold partition; no fit or accuracy claim"}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return pin(output)


def load_qualified_regional_features(path, expected_bundle_sha):
    require(pin(path)["sha256"] == expected_bundle_sha, "Explicit expected regional bundle SHA required")
    bundle = read(path)
    require(bundle.get("schema_version") == "orthodiff_regional2x2_qualified_feature_bundle_v1" and bundle.get("qualified") is True
            and bundle.get("source_sha256") == SOURCE_SHA and bundle.get("UID_order_sha256") == UID_SHA
            and bundle.get("requested_count") == 4207 and bundle.get("feature_dimension") == 1024 and bundle.get("FD_availability_mixed") is False,
            "Qualified standalone regional1024 bundle required")
    parent_ref, reference = bundle["parent"], bundle["qualification"]
    checked(parent_ref["path"], parent_ref)
    parent, _, parent_summary = parent_evidence(parent_ref["path"])
    checked(reference["path"], reference)
    audit = audit_evidence(reference["path"], reference["sha256"], parent_ref["path"])
    files, summary = raw_evidence(audit, *(bundle["files"][name]["path"] for name in ("records", "summary", "status")))
    require(files == bundle["files"] and bundle["coverage"] == audit["coverage"] == parent["coverage"]
            and bundle["parent_qualification"] == parent["qualification"] and bundle["parent_comparison"] == audit["parent_comparison"]
            and summary["requested_UIDs"] == parent_summary["requested_UIDs"], "Regional bundle/parent/actual audit refs differ")
    uids, features, failures, strata, count = summary["requested_UIDs"], {}, {}, {}, 0
    with Path(parent["files"]["records"]["path"]).open("rb") as old, Path(files["records"]["path"]).open("rb") as new:
        for count, line in enumerate(new, 1):
            parent_line = old.readline()
            require(count <= 4207 and line.endswith(b"\n") and len(line) <= 256 * 1024 and parent_line, "Regional record count/line bounds differ")
            record, previous = parse(line), parse(parent_line)
            uid = record.get("StudyInstanceUID")
            require(uid == uids[count - 1] == previous["StudyInstanceUID"], "Regional original4207 UID order differs")
            compare_parent(record, previous, parent_line)
            if record["status"] == "success":
                features[uid] = vector(record).copy()
                strata[uid] = previous["metadata"]["stratum"]
            else:
                failures[uid] = {"reason_type": previous["reason_type"], "reason": previous["reason"]}
        require(not old.readline(), "Extra parent records")
    require(count == 4207 and len(features) == 3063 and len(failures) == 1144 and set(features).isdisjoint(failures)
            and set(features) | set(failures) == set(uids), "Regional original3063/1144 partition changed")
    return bundle, features, failures, strata, uids
