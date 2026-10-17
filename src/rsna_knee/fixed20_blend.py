"""Local-only fixed20 integration; no model loading or external operations.

For each supported target with all N native scores present, calculate
``f32(f32(f32(.8) * f32(parent)) + f32(f32(.2) * f32(rank_N(native))))``.
The parent and result are NEVER reranked. Native inputs are quantized to FP32
before full-N, ascending, one-based average-tie ranks are computed. Each rank
is then rounded to FP32. This specifies separate multiplies and addition,
without a fused multiply-add, as in separate NumPy float32 ufunc operations.

One missing native score disables the entire target column. Missing means
None (or an empty CSV cell), or an explicit failure mask. NaN, infinity,
malformed values and schema errors always raise, even on masked cells.
Excluded and fallback columns, and every cell at alpha=0, retain parent text.
This numeric contract does not qualify a model, its inputs, or a submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections.abc import Mapping
from pathlib import Path

from .contracts import ID, SUBMISSION_COLUMNS, TARGETS, dump_json, read_csv, sha256, uid, validate_submission, write_csv

EXCLUDED_TARGETS = ("MCL", "Synovitis")
BLENDED_TARGETS = tuple(target for target in TARGETS if target not in EXCLUDED_TARGETS)
P029M_COMPONENT_WEIGHTS = {"Raptor384": 0.325, "Raptor224": 0.175, "CompactR6": 0.5}
FORMULA = "float32(0.8 * parent_s + 0.2 * full_N_average_tie_rank(native_mean))"


def float32(value):
    """Round once to IEEE-754 binary32, returning an exactly representing float."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def percentile_average_ranks(values):
    """Full-population average-tie ranks/N; inputs must already be validated."""
    size = len(values)
    if not size:
        raise ValueError("Cannot rank an empty population")
    order = sorted(range(size), key=values.__getitem__)
    ranks = [0.0] * size
    begin = 0
    while begin < size:
        end = begin + 1
        while end < size and values[order[end]] == values[order[begin]]:
            end += 1
        rank = (begin + end + 1) / (2 * size)
        for index in order[begin:end]:
            ranks[index] = rank
        begin = end
    return ranks


def _probability(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("Expected a finite probability in [0, 1]")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError("Expected a finite probability in [0, 1]") from exc
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError("Expected a finite probability in [0, 1]")
    return number


def _load_rows(rows, *, native):
    result, numeric = {}, {}
    for row in rows:
        if not isinstance(row, Mapping) or tuple(row) != SUBMISSION_COLUMNS:
            raise ValueError("Column order must match SUBMISSION_COLUMNS exactly")
        key = row[ID]
        if not isinstance(key, str) or any(ord(char) < 32 or ord(char) == 127 for char in key):
            raise ValueError("Invalid StudyInstanceUID")
        uid(key)
        if key in result:
            raise ValueError("Duplicate StudyInstanceUID")
        result[key], numeric[key] = {ID: key}, {}
        for target in TARGETS:
            value = row[target]
            missing = native and (value is None or value == "")
            numeric[key][target] = None if missing else _probability(value)
            result[key][target] = None if missing else str(value)
    if not result:
        raise ValueError("Prediction inputs must contain at least one study")
    return result, numeric


def _failures(mask, keys):
    failures = {key: set() for key in keys}
    if mask is None:
        return failures
    if not isinstance(mask, Mapping) or set(mask) - set(keys):
        raise ValueError("Failure mask must map known UIDs to booleans or target boolean mappings")
    for key, value in mask.items():
        if type(value) is bool:
            if value:
                failures[key].update(TARGETS)
        elif isinstance(value, Mapping) and not set(value) - set(TARGETS):
            if any(type(flag) is not bool for flag in value.values()):
                raise ValueError("Failure mask flags must be booleans; True means failed")
            failures[key].update(target for target, failed in value.items() if failed)
        else:
            raise ValueError("Failure mask must contain booleans or target boolean mappings")
    return failures


def _uid_hash(keys):
    payload = json.dumps(list(keys), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def blend_rows(parent_rows, native_rows, *, failure_mask=None, alpha=0.2):
    """Return (submission rows, receipt) in parent UID order, with UID alignment.

    Rows must use exact SUBMISSION_COLUMNS in order. Native rows may contain
    None or empty strings for explicitly unscored cells. ``failure_mask`` is a
    UID -> bool or UID -> {target: bool} mapping; True denotes failure and
    omitted mask entries denote no explicit failure. All provided numbers are
    validated before masking. Only alpha=.2 (candidate) and 0 (identity check)
    are accepted; alpha=0 still validates all input contracts.
    """
    if isinstance(alpha, bool) or alpha not in (0, 0.2):
        raise ValueError("Only fixed alpha=0.2 or identity-check alpha=0 is supported")
    parent, parent_values = _load_rows(parent_rows, native=False)
    native, native_values = _load_rows(native_rows, native=True)
    if parent.keys() != native.keys():
        raise ValueError("Parent and native UID sets must match exactly")
    keys = list(parent)
    failures = _failures(failure_mask, keys)
    output = [dict(parent[key]) for key in keys]
    per_target, fallback_targets = {}, []
    active_count = 0
    total_numeric_changes = total_text_changes = total_rank_changes = 0
    for target in TARGETS:
        failed = [key for key in keys if native_values[key][target] is None or target in failures[key]]
        if target in EXCLUDED_TARGETS:
            status = "parent_only_excluded"
        elif alpha == 0:
            status = "parent_only_alpha_zero"
        elif failed:
            status = "parent_only_incomplete_native_column"
            fallback_targets.append(target)
        else:
            status = "blended_full_N"
            active_count += 1
            values = [float32(native_values[key][target]) for key in keys]
            ranks = percentile_average_ranks(values)
            for index, (key, rank) in enumerate(zip(keys, ranks)):
                left = float32(float32(1 - alpha) * float32(parent_values[key][target]))
                right = float32(float32(alpha) * float32(rank))
                value = float32(left + right)
                _probability(value)
                output[index][target] = repr(value)
        blended = status == "blended_full_N"
        parent_weight, native_weight = (1 - alpha, alpha) if blended else (1.0, 0.0)
        old = [parent_values[key][target] for key in keys]
        new = [float(row[target]) for row in output]
        deltas = [abs(left - right) for left, right in zip(old, new)]
        numeric_changes = sum(left != right for left, right in zip(old, new))
        text_changes = sum(parent[key][target] != row[target] for key, row in zip(keys, output))
        rank_changes = sum(
            left != right for left, right in zip(percentile_average_ranks(old), percentile_average_ranks(new))
        )
        total_numeric_changes += numeric_changes
        total_text_changes += text_changes
        total_rank_changes += rank_changes
        per_target[target] = {
            "status": status,
            "native_success_count": len(keys) - len(failed),
            "native_failure_count": len(failed),
            "native_failed_UID_sha256": _uid_hash(sorted(failed)),
            "effective_weights": {"parent_s": parent_weight, "native_rank": native_weight},
            "conditional_P029M_component_weights_nominal": {
                **{name: round(weight * parent_weight, 12) for name, weight in P029M_COMPONENT_WEIGHTS.items()},
                "F006_native_rank": native_weight,
            },
            "numeric_changed_cell_count": numeric_changes,
            "text_changed_cell_count": text_changes,
            "rank_changed_cell_count": rank_changes,
            "max_abs_delta_from_parent": max(deltas),
        }
    all_disabled = active_count == 0
    if all_disabled:
        comparison = "not_established_all_native_disabled"
    elif total_rank_changes == 0:
        comparison = "ranking_noop_under_auc"
    else:
        comparison = "changed_rankings_unscored"
    receipt = {
        "schema_version": "fixed20_fullN_column_fallback_v1",
        "stage": "local_csv_integration_only",
        "formula": FORMULA,
        "alpha_requested": alpha,
        "row_count": len(keys),
        "columns": list(SUBMISSION_COLUMNS),
        "ordered_parent_UID_sha256": _uid_hash(keys),
        "arithmetic": {
            "dtype": "IEEE-754 binary32; round-to-nearest, ties-to-even",
            "native_inputs": "quantize to float32 before ranking; no averaging is performed here",
            "rank": "ascending one-based average-tie rank / full N; then quantize to float32",
            "operations": "float32 coefficients, inputs, separate multiplies, then float32 addition; no FMA",
            "parent_reranked": False,
            "output_reranked": False,
            "serialization": "repr(exactly representing Python float of binary32 result)",
            "parent_component_weights": "nominal algebra only, conditional on externally verified P029M parent",
            "flattened_component_sum_bitwise_equivalent": False,
        },
        "failure_policy": "any unscored native cell disables its entire target column; no success-subset ranks",
        "excluded_targets": list(EXCLUDED_TARGETS),
        "fallback_targets": fallback_targets,
        "blended_target_count": active_count,
        "all_native_disabled": all_disabled,
        "numeric_noop": total_numeric_changes == 0,
        "text_noop": total_text_changes == 0,
        "rank_noop": total_rank_changes == 0,
        "accuracy_comparison_status": comparison,
        "parent_to_final": {
            "numeric_changed_cell_count": total_numeric_changes,
            "text_changed_cell_count": total_text_changes,
            "rank_changed_cell_count": total_rank_changes,
        },
        "per_target": per_target,
        "qualification": {"model_provenance_verified": False, "Kaggle_execution_verified": False, "scored": False},
    }
    return output, receipt


def _load_failure_csv(path):
    fields, rows = read_csv(path)
    if tuple(fields) not in ((ID,), SUBMISSION_COLUMNS):
        raise ValueError("Failure CSV must contain only UID, or exact submission columns with 0/1 failure flags")
    mask = {}
    for row in rows:
        key = uid(row[ID])
        if key in mask:
            raise ValueError("Duplicate failure UID")
        if tuple(fields) == (ID,):
            mask[key] = True
        else:
            if any(row[target] not in ("0", "1") for target in TARGETS):
                raise ValueError("Failure CSV flags must be exactly 0 or 1; 1 means failed")
            mask[key] = {target: row[target] == "1" for target in TARGETS}
    return mask


def blend_csv(
    parent_csv, native_mean_csv, output_dir, *, failure_mask_csv=None, alpha=0.2, test_csv=None, sample_csv=None
):
    """Write submission.csv and receipt into a fresh directory, never uploading."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {output_dir}")
    inputs = {}
    loaded = []
    for name, path in (("parent", parent_csv), ("native_mean", native_mean_csv)):
        fields, rows = read_csv(path)
        if tuple(fields) != SUBMISSION_COLUMNS:
            raise ValueError("Column order must match SUBMISSION_COLUMNS exactly")
        loaded.append(rows)
        inputs[name] = {"path": str(Path(path).resolve()), "sha256": sha256(path)}
    mask = None if failure_mask_csv is None else _load_failure_csv(failure_mask_csv)
    if failure_mask_csv is not None:
        inputs["failure_mask"] = {"path": str(Path(failure_mask_csv).resolve()), "sha256": sha256(failure_mask_csv)}
    if sample_csv is not None and test_csv is None:
        raise ValueError("sample_csv requires test_csv")
    if test_csv is not None:
        validate_submission(parent_csv, test_csv, sample_csv)
        for name, path in (("test", test_csv), ("sample", sample_csv)):
            if path is not None:
                inputs[name] = {"path": str(Path(path).resolve()), "sha256": sha256(path)}
    rows, receipt = blend_rows(*loaded, failure_mask=mask, alpha=alpha)
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "submission.csv"
    if receipt["text_noop"]:
        output_path.write_bytes(Path(parent_csv).read_bytes())
    else:
        write_csv(output_path, SUBMISSION_COLUMNS, rows)
    receipt["inputs"] = inputs
    receipt["output"] = {"path": str(output_path.resolve()), "sha256": sha256(output_path)}
    if test_csv is not None:
        receipt["submission_contract"] = validate_submission(output_path, test_csv, sample_csv)
    receipt_path = output_dir / "receipt.json"
    dump_json(receipt_path, receipt)
    (output_dir / "receipt.sha256").write_text(f"{sha256(receipt_path)}  receipt.json\n", encoding="utf-8")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description="ローカルCSVだけで固定20%を統合。アップロード・提出は実行しません。")
    parser.add_argument("--parent-csv", type=Path, required=True)
    parser.add_argument("--native-mean-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--failure-mask-csv", type=Path)
    parser.add_argument("--alpha", type=float, choices=(0.0, 0.2), default=0.2)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--sample-csv", type=Path)
    args = parser.parse_args(argv)
    receipt = blend_csv(**vars(args))
    print(json.dumps({key: receipt[key] for key in ("stage", "all_native_disabled", "accuracy_comparison_status")}))


if __name__ == "__main__":
    main()
