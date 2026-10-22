"""Metadata-only, prospective fallback selection; no image/model/label code."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict

ID = "StudyInstanceUID"
TRUE_VALUES = frozenset({"1", "1.0", "true", "yes"})
FALSE_VALUES = frozenset({"0", "0.0", "false", "no"})
FAMILY_PREFIX = "no_explicit_noncontradictory_PD_sagittal_series:"
SOURCE_SHA = "15bb2d6c7c224021f95a304b08eb9e3644164ea8373f4ae6ec1c7afc77582e1f"
UID_SHA = "911f91fbc914213db045780675d6518a8cc4a49808599bbb74f2f247bc30eb82"
SPEC_SHA = "4326c03108ebe1d00ff833d8da4ebec891ba19d50d9bc6f7c14a3d494d2d92ff"


def require(value, message):
    if not value:
        raise ValueError(message)


def unique_pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "Duplicate JSON field")
        result[key] = value
    return result


def parse_json(raw):
    return json.loads(raw, object_pairs_hook=unique_pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON token")))


def metadata_bool(value):
    normalized = str(value).strip().casefold()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError("Unknown/missing boolean metadata flag")


def select_fluid_sensitive(uid, rows):
    """Choose once from official metadata, before headers/geometry/pixels."""
    seen, candidates, rejected = set(), [], []
    for row in rows:
        require(row.get(ID) == uid and row.get("SeriesInstanceUID"), "Series metadata UID mismatch")
        sid = row["SeriesInstanceUID"]
        require(sid not in seen, "Duplicate series metadata UID")
        seen.add(sid)
        if str(row.get("Anatomical_Plane", "")).strip().casefold() != "sagittal":
            continue
        values = {key: row.get(key) for key in ("Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression")}
        try:
            fluid = metadata_bool(values["Fluid_Sensitive"])
            suppressed = metadata_bool(values["Fat_Suppression"])
        except ValueError:
            rejected.append({"SeriesInstanceUID": sid, "metadata": values, "reason": "unknown_boolean_flag"})
            continue
        if not fluid:
            rejected.append({"SeriesInstanceUID": sid, "metadata": values, "reason": "not_fluid_sensitive"})
            continue
        candidates.append({"SeriesInstanceUID": sid, "metadata": values, "Fluid_Sensitive": True, "Fat_Suppression": suppressed})
    candidates.sort(key=lambda row: (not row["Fat_Suppression"], row["SeriesInstanceUID"]))
    return {"route": "fluid_sensitive_fallback", "confirmed_PD": False,
            "status": "metadata_candidate_selected" if candidates else "metadata_unavailable",
            "selected_series": candidates[0] if candidates else None,
            "ordered_eligible_series_UIDs": [row["SeriesInstanceUID"] for row in candidates],
            "rejected_series": sorted(rejected, key=lambda row: row["SeriesInstanceUID"]),
            "selection_before_headers_geometry_pixels": True,
            "retry_another_series_after_chosen_failure": False}


def unknown_pd_family(record):
    """Naming-loss family only; never success or a selected-series failure."""
    require(record.get("status") in {"success", "data_failure"}, "Unexpected parent status")
    if record["status"] == "success":
        return False, "parent_success"
    if record.get("selected_series") is not None:
        return False, "parent_selected_series_data_failure"
    if record.get("reason_type") != "PilotDataError" or not str(record.get("reason", "")).startswith(FAMILY_PREFIX):
        return False, "parent_other_failure"
    require(record.get("feature_forward_calls") == 0 and record.get("fallback_features") is False
            and not any(key in record for key in ("pooled", "pooled_float32_values", "mid_2", "normalized_input")),
            "Parent failure feature/forward contract differs")
    rejected = parse_json(record["reason"][len(FAMILY_PREFIX):])
    require(isinstance(rejected, list), "Parent rejected candidates must be an array")
    ids = [row.get("SeriesInstanceUID") for row in rejected]
    require(all(ids) and len(ids) == len(set(ids)), "Duplicate/empty parent rejected series UID")
    reasons = [row.get("reason") for row in rejected]
    eligible = bool(reasons) and "unknown_PD" in reasons and all(reason in {"unknown_PD", "T1_T2_contradiction"} for reason in reasons)
    return eligible, "unknown_PD_family" if eligible else "parent_not_unknown_PD_only_naming_family"


def require_qualified_parent(audit, summary, status):
    """Never accepts a partial extraction or a static preparation qualification."""
    require(audit.get("schema_version") == "orthodiff_fullweak4207_actual_qualification_v1"
            and audit.get("passed") is True and audit.get("head_features_qualified") is True
            and audit.get("source_sha256") == SOURCE_SHA and audit.get("UID_order_sha256") == UID_SHA
            and audit.get("requested_count") == 4207 and audit.get("prior32_comparison", {}).get("passed") is True,
            "Actual complete4207 and prior32 independent qualification required")
    require(summary.get("complete") is True and status.get("complete") is True and "error" not in summary
            and summary.get("phase") == status.get("phase") == "full4207_finished"
            and summary.get("coverage") == status.get("coverage") == audit.get("coverage")
            and audit["coverage"]["requested"] == audit["coverage"]["recorded"] == 4207,
            "Parent extraction is not completely classified and qualified")


def build_metadata_subset(records, cohort_rows, series_rows):
    """Pure metadata projection; caller must enforce the actual-parent guard."""
    uids = [row[ID] for row in cohort_rows]
    require(len(uids) == len(set(uids)) and uids, "Unique nonempty original cohort required")
    structure = {row[ID]: row for row in cohort_rows}
    metadata, seen_series = defaultdict(list), set()
    for row in series_rows:
        require(row.get(ID) in structure and row.get("SeriesInstanceUID"), "Unknown UID in scoped metadata")
        key = (row[ID], row["SeriesInstanceUID"])
        require(key not in seen_series, "Duplicate official metadata series")
        seen_series.add(key)
        metadata[row[ID]].append(row)
    entries, excluded, counts = [], Counter(), Counter()
    seen = set()
    for index, record in enumerate(records):
        uid = record.get(ID)
        require(index < len(uids) and uid == uids[index] and uid not in seen, "Unknown/duplicate/out-of-order parent UID")
        seen.add(uid)
        eligible, reason = unknown_pd_family(record)
        counts[record["status"]] += 1
        if not eligible:
            excluded[reason] += 1
            continue
        row = structure[uid]
        require(row.get("source") == "report_weak" and row.get("fold") in {"0", "1", "2", "3", "4"}, "Original group/fold/source metadata required")
        selection = select_fluid_sensitive(uid, metadata.get(uid, []))
        encoded = (json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()
        entries.append({ID: uid, "group_id": row["group_id"], "fold": row["fold"], "source": row["source"],
                        "parent_record_index": index, "parent_failure_family": reason,
                        "parent_record_canonical_sha256": hashlib.sha256(encoded).hexdigest(), **selection})
    require(seen == set(uids), "Incomplete parent record coverage")
    selected = [row for row in entries if row["status"] == "metadata_candidate_selected"]
    return {"schema_version": "orthodiff_fluid_sensitive_metadata_subset_preparation_v1", "adopted": False,
            "fallback_features_qualified": False, "actual_MRI_forward_or_fit_performed": False,
            "parent_record_counts": dict(counts), "subset_count": len(entries), "selected_metadata_candidate_count": len(selected),
            "metadata_unavailable_count": len(entries) - len(selected), "excluded_parent_family_counts": dict(excluded),
            "subset_UID_order_sha256": hashlib.sha256("".join(row[ID] + "\n" for row in entries).encode()).hexdigest(),
            "selected_UID_order_sha256": hashlib.sha256("".join(row[ID] + "\n" for row in selected).encode()).hexdigest(),
            "entries": entries, "scope": {"pixel_header_geometry_reads": 0, "targets_Report_Gold_labels": 0, "model_GPU_network_fit_Save_Submit": 0},
            "contract": {"confirmed_PD_claim": False, "selection_before_pixel_header_geometry": True, "no_chosen_series_retry": True,
                         "no_parent_success_overwrite_or_new_UID": True, "original_group_fold_unchanged": True,
                         "no_replacement_for_desired_coverage": True}}
