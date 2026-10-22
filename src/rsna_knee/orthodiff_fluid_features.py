"""Hash-bound parent plus fluid-success feature transport; never fits a head."""
from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path

from .orthodiff_fluid_fallback import parse_json, require

PARENT_BUNDLE_SHA = "525094b583c7cd821769fac56b9e64a524d5ef185a4a9eb689f91480b31ce83c"
PARENT_AUDIT_SHA = "7fbc160b8715e69dc506b64e9187f55a2c2b67925bf6a5b0a4a95adf943c8270"
PARENT_SOURCE_SHA = "15bb2d6c7c224021f95a304b08eb9e3644164ea8373f4ae6ec1c7afc77582e1f"
PARENT_UID_SHA = "911f91fbc914213db045780675d6518a8cc4a49808599bbb74f2f247bc30eb82"
CHILD_SOURCE_SHA = "4b8237aa96f44e1a4365180bba5fd2b65782549ad0f0833a6d7e4213a01209b0"
CHILD_UID_SHA = "2fb8f009f2f2372ea0e460c174bfb373172f2477a356cd8bee783e0d0792bf7e"
SUBSET_SHA = "c0e52bf58e46579d28d090cd25168130033f8b749098359ec95bdc425144c178"
CHILD_STEM = "ORTHODIFF_FLUID_FALLBACK_"


def pin(path, maximum=128 * 1024**2):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= maximum, "Bounded regular feature artifact required")
    h, length = hashlib.sha256(), 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            h.update(block)
            length += len(block)
            require(length <= maximum, "Feature input grew beyond bound")
    return {"bytes": length, "sha256": h.hexdigest()}


def checked(path, expected, maximum=128 * 1024**2):
    require(pin(path, maximum) == {key: expected[key] for key in ("bytes", "sha256")}, "Positive feature input pin differs")
    return Path(path)


def read(path):
    pin(path, 4 * 1024**2)
    return parse_json(Path(path).read_bytes())


def uid_hash(uids):
    return hashlib.sha256("".join(uid + "\n" for uid in uids).encode()).hexdigest()


def feature_values(record):
    values = record.get("pooled_float32_values")
    require(isinstance(values, list) and len(values) == 256 and all(type(value) in (float, int) and math.isfinite(value) for value in values), "Finite256 feature vector required")
    raw = struct.pack("<256f", *values)
    require(list(struct.unpack("<256f", raw)) == values and record.get("pooled") == {
        "shape": [1, 256], "dtype": "torch.float32", "bytes": 1024, "sha256": hashlib.sha256(raw).hexdigest()}, "Feature FP32 bits/SHA differ")
    return values


def parent_bundle(path, expected_sha):
    require(expected_sha == PARENT_BUNDLE_SHA and pin(path)["sha256"] == expected_sha, "The immutable qualified parent bundle is required")
    bundle = read(path)
    require(bundle.get("schema_version") == "orthodiff_fullweak4207_qualified_feature_bundle_v1" and bundle.get("qualified") is True
            and bundle.get("source_sha256") == PARENT_SOURCE_SHA and bundle.get("UID_order_sha256") == PARENT_UID_SHA
            and bundle.get("requested_count") == 4207 and bundle["coverage"]["success"] == 3063, "Parent qualification identity differs")
    qualification = bundle["qualification"]
    require(qualification["sha256"] == PARENT_AUDIT_SHA, "Exact parent actual audit required")
    audit = read(checked(qualification["path"], qualification))
    require(audit.get("passed") is audit.get("head_features_qualified") is True and audit.get("prior32_comparison", {}).get("passed") is True,
            "Actual parent/old32 qualification is absent")
    for name, item in bundle["files"].items():
        checked(item["path"], item)
        require({key: item[key] for key in ("bytes", "sha256")} == audit[name + "_pin"], "Parent raw/audit pin differs")
    return bundle, audit


def qualified_child(path, expected_sha, parent_pin):
    require(isinstance(expected_sha, str) and len(expected_sha) == 64 and pin(path)["sha256"] == expected_sha, "Expected actual child qualification SHA required")
    audit = read(path)
    require(audit.get("schema_version") == "orthodiff_fluid1080_actual_qualification_v1"
            and audit.get("passed") is audit.get("child_features_qualified") is True
            and audit.get("source_sha256") == CHILD_SOURCE_SHA and audit.get("UID_order_sha256") == CHILD_UID_SHA
            and audit.get("requested_count") == 1080 and audit["coverage"]["requested"] == audit["coverage"]["recorded"] == 1080
            and audit["coverage"]["success"] + audit["coverage"]["failure"] == 1080
            and audit["parent_bundle_pin"] == parent_pin and audit["parent_qualification_pin"]["sha256"] == PARENT_AUDIT_SHA
            and audit["subset_pin"]["sha256"] == SUBSET_SHA
            and audit["parent_success_unchanged"] is True and audit["parent_success_overwrite_count"] == 0,
            "Actual complete fixed1080 fluid qualification is required")
    return audit


def child_files(audit, records, summary, status):
    files = {}
    for name, path, suffix in (("records", records, "RECORDS.jsonl"), ("summary", summary, "SUMMARY.json"), ("status", status, "STATUS.json")):
        require(Path(path).name == CHILD_STEM + suffix, "Unexpected child artifact filename")
        checked(path, audit[name + "_pin"])
        files[name] = {"path": str(Path(path).resolve()), **pin(path)}
    a, b = read(summary), read(status)
    require(a.get("complete") is b.get("complete") is True and a.get("phase") == b.get("phase") == "fluid1080_finished"
            and a["coverage"] == b["coverage"] == audit["coverage"] and a["route"] == "fluid_sensitive_fallback"
            and a["confirmed_PD"] is False and a["journal"]["sha256"] == b["journal"]["sha256"] == files["records"]["sha256"],
            "Complete child terminal/hash binding differs")
    uids = a["requested_UIDs"]
    require(len(uids) == len(set(uids)) == 1080 and uid_hash(uids) == CHILD_UID_SHA, "Fixed child UID order required")
    return files, uids


def seal_composite_bundle(parent_path, expected_parent_sha, child_audit_path, expected_child_sha, records, summary, status, output):
    """Only seal after actual parent/child qualification. No labels, images or fit."""
    output = Path(output)
    require(not output.exists(), "New composite bundle output required")
    parent, _ = parent_bundle(parent_path, expected_parent_sha)
    audit = qualified_child(child_audit_path, expected_child_sha, pin(parent_path))
    files, _ = child_files(audit, records, summary, status)
    value = {"schema_version": "orthodiff_parent_fluid_composite_bundle_v1", "qualified": True,
             "parent": {"path": str(Path(parent_path).resolve()), **pin(parent_path)}, "parent_qualification": parent["qualification"],
             "child_qualification": {"path": str(Path(child_audit_path).resolve()), **pin(child_audit_path)}, "child_files": files,
             "parent_source_sha256": PARENT_SOURCE_SHA, "child_source_sha256": CHILD_SOURCE_SHA, "UID_order_sha256": PARENT_UID_SHA,
             "requested_count": 4207, "parent_success_count": 3063, "child_coverage": audit["coverage"],
             "combined_success_count": 3063 + audit["coverage"]["success"], "combined_failure_count": 1144 - audit["coverage"]["success"],
             "merge_proof": {"parent_success_features_byte_exact": True, "parent_success_overwrite_count": 0,
                             "child_success_additions_only": True, "same4207_UID_group_fold_structure": True,
                             "unrecovered_parent_failures_retained": True, "parent_PD_and_child_fluid_strata_distinct": True},
             "scope": "Qualified feature availability transport only; no head fitting, candidate adoption or accuracy claim"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return pin(output)


def merge_child_records(parent_features, parent_failures, parent_strata, uids, records, child_uids):
    """Pure non-overwriting merge; failures keep their original parent reasons."""
    require(len(uids) == len(set(uids)) and set(parent_features).isdisjoint(parent_failures)
            and set(parent_features) | set(parent_failures) == set(uids), "Parent feature partition differs")
    require(len(child_uids) == len(set(child_uids)) and set(child_uids).issubset(parent_failures),
            "Child UID duplicate/unknown/parent-success overwrite")
    features, failures, strata = dict(parent_features), dict(parent_failures), dict(parent_strata)
    original_bits = {uid: struct.pack("<256f", *values) for uid, values in parent_features.items()}
    count, successes = 0, 0
    for count, record in enumerate(records, start=1):
        require(count <= len(child_uids) and record.get("StudyInstanceUID") == child_uids[count - 1]
                and record["StudyInstanceUID"] in parent_failures and record["StudyInstanceUID"] not in parent_features,
                "Child UID duplicate/unknown/parent-success overwrite")
        uid = record["StudyInstanceUID"]
        if record.get("status") == "success":
            meta = record["metadata"]
            require(meta["route"] == "fluid_sensitive_fallback" and meta["confirmed_PD"] is False
                    and meta["stratum"] in {"fluid_sensitive_FS", "fluid_sensitive_nonFS"}
                    and record["all_RNG_restored"] is True and record["forward_counts"] == {"denoiser": 1, "decoder_blocks": 0, "final_out": 0},
                    "Child route/stratum/forward contract differs")
            features[uid] = feature_values(record).copy()
            del failures[uid]
            strata[uid] = meta["stratum"]
            successes += 1
        else:
            require(record.get("status") == "data_failure" and record.get("route") == "fluid_sensitive_fallback"
                    and record.get("confirmed_PD") is False and record.get("feature_forward_calls") == 0 and record.get("fallback_features") is False
                    and not any(key in record for key in ("pooled", "pooled_float32_values", "mid_2", "normalized_input")), "Child failure must not fabricate features")
    require(count == len(child_uids) and len(child_uids) == len(set(child_uids)), "Incomplete/duplicate child UID coverage")
    require(all(struct.pack("<256f", *features[uid]) == bits for uid, bits in original_bits.items())
            and len(features) == len(parent_features) + successes and set(features).isdisjoint(failures)
            and set(features) | set(failures) == set(uids), "Parent bytes or composite partition changed")
    return features, failures, strata


def load_composite_features(path, expected_bundle_sha):
    """Parent-compatible five-tuple; NumPy enters only through existing parent H."""
    require(pin(path)["sha256"] == expected_bundle_sha, "Expected sealed composite SHA is mandatory")
    bundle = read(path)
    require(bundle.get("schema_version") == "orthodiff_parent_fluid_composite_bundle_v1" and bundle.get("qualified") is True
            and bundle.get("requested_count") == 4207 and bundle.get("UID_order_sha256") == PARENT_UID_SHA
            and bundle.get("parent_source_sha256") == PARENT_SOURCE_SHA and bundle.get("child_source_sha256") == CHILD_SOURCE_SHA,
            "Qualified composite feature bundle required")
    parent_ref, child_ref = bundle["parent"], bundle["child_qualification"]
    checked(parent_ref["path"], parent_ref)
    parent, _ = parent_bundle(parent_ref["path"], PARENT_BUNDLE_SHA)
    checked(child_ref["path"], child_ref)
    audit = qualified_child(child_ref["path"], child_ref["sha256"], pin(parent_ref["path"]))
    files, child_uids = child_files(audit, *(bundle["child_files"][key]["path"] for key in ("records", "summary", "status")))
    require(files == bundle["child_files"] and bundle["parent_qualification"] == parent["qualification"], "Composite refs/pins differ")
    from . import orthodiff_weak_head as H  # Parent H is immutable, no fit function is called.
    require(pin(Path(H.__file__))["sha256"] == "e1149f8e88eeba0f32418e8ee4986cd41ef5aae208a31142cd907689958d78e7", "Original parent loader source changed")
    _, features, failures, strata, uids = H.load_qualified_features(parent_ref["path"], PARENT_BUNDLE_SHA)

    def child_records():
        with Path(files["records"]["path"]).open("rb") as handle:
            for line in handle:
                require(line.endswith(b"\n") and 0 < len(line) <= 256 * 1024, "Bounded complete child JSONL line required")
                yield parse_json(line)

    before = len(features)
    features, failures, strata = merge_child_records(features, failures, strata, uids, child_records(), child_uids)
    require(before == 3063 and len(features) == bundle["combined_success_count"] == 3063 + audit["coverage"]["success"]
            and len(failures) == bundle["combined_failure_count"] and uid_hash(uids) == PARENT_UID_SHA, "Composite final counts/order differ")
    return bundle, features, failures, strata, uids
