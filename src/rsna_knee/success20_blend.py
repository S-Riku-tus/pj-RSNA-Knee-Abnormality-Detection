"""Fixed20 integration with native failures isolated to their own study cells.

This is a new contract, not the full-N/whole-column fallback contract. For a
partially available target, native average-tie ranks use only its successful
studies and denominator N_valid. Failed cells retain the exact parent text.
The parent and output are never reranked. With complete native coverage the
numeric and text results match fixed20_blend exactly. No model or network I/O.
"""

from __future__ import annotations

from pathlib import Path

from . import fixed20_blend as original
from .contracts import SUBMISSION_COLUMNS, TARGETS, dump_json, read_csv, sha256, validate_submission, write_csv


def blend_rows(parent_rows, native_rows, *, failure_mask=None, alpha=0.2):
    """Retain failed study cells; blend supported successful target cells.

    Partial coverage requires two successful studies with distinct FP32 scores.
    No coverage percentage threshold or model weight search is introduced.
    Native scores and all provided numbers must satisfy the original strict
    schema even when masked. alpha=0 is the original exact identity check.
    """
    # The original validator checks alpha, schema, UID alignment, masked
    # nonfinite values and all twelve labels before any output is produced.
    parent_rows, native_rows = list(parent_rows), list(native_rows)
    output, receipt = original.blend_rows(parent_rows, native_rows, failure_mask=failure_mask, alpha=alpha)
    parent, parent_values = original._load_rows(parent_rows, native=False)
    native, native_values = original._load_rows(native_rows, native=True)
    keys = list(parent)
    failures = original._failures(failure_mask, keys)
    changed = {key: row for key, row in zip(keys, output)}

    for target in TARGETS:
        entry = receipt["per_target"][target]
        failed = [key for key in keys if native_values[key][target] is None or target in failures[key]]
        failed_set = set(failed)
        valid = [key for key in keys if key not in failed_set]
        entry["native_rank_population_count"] = len(valid)
        entry["failed_cells_retain_parent_text"] = True
        if target in original.EXCLUDED_TARGETS or alpha == 0 or not failed:
            entry["rank_basis"] = "full N" if entry["status"] == "blended_full_N" else "no native contribution"
            continue
        if len(valid) < 2:
            entry["status"] = "parent_only_fewer_than_two_successes"
            entry["rank_basis"] = "no native contribution"
            continue

        values = [original.float32(native_values[key][target]) for key in valid]
        if len(set(values)) < 2:
            entry["status"] = "parent_only_constant_partial_native"
            entry["rank_basis"] = "no native contribution"
            continue
        ranks = original.percentile_average_ranks(values)
        for key, rank in zip(valid, ranks):
            left = original.float32(original.float32(1 - alpha) * original.float32(parent_values[key][target]))
            right = original.float32(original.float32(alpha) * original.float32(rank))
            changed[key][target] = repr(original.float32(left + right))
        entry.update(
            status="blended_success_subset",
            rank_basis="ascending one-based average-tie rank / N_valid; then float32",
            effective_weights={
                "successful_cells": {"parent_s": 1 - alpha, "native_rank": alpha},
                "failed_cells": {"parent_s": 1.0, "native_rank": 0.0},
            },
            conditional_P029M_component_weights_nominal={
                "successful_cells": {**{name: round(weight * (1 - alpha), 12)
                                        for name, weight in original.P029M_COMPONENT_WEIGHTS.items()},
                                     "F006_native_rank": alpha},
                "failed_cells": {**original.P029M_COMPONENT_WEIGHTS, "F006_native_rank": 0.0},
            },
        )

    totals = {"numeric_changed_cell_count": 0, "text_changed_cell_count": 0, "rank_changed_cell_count": 0}
    active = 0
    fallback = []
    native_cells = 0
    for target in TARGETS:
        entry = receipt["per_target"][target]
        old = [parent_values[key][target] for key in keys]
        new = [float(row[target]) for row in output]
        counts = {
            "numeric_changed_cell_count": sum(a != b for a, b in zip(old, new)),
            "text_changed_cell_count": sum(parent[key][target] != row[target] for key, row in zip(keys, output)),
            "rank_changed_cell_count": sum(a != b for a, b in zip(
                original.percentile_average_ranks(old), original.percentile_average_ranks(new))),
        }
        entry.update(counts, max_abs_delta_from_parent=max(abs(a - b) for a, b in zip(old, new)))
        for name, count in counts.items():
            totals[name] += count
        blended = entry["status"] in ("blended_full_N", "blended_success_subset")
        active += int(blended)
        native_cells += entry["native_success_count"] if blended else 0
        if target not in original.EXCLUDED_TARGETS and not blended:
            fallback.append(target)
    receipt.update(
        schema_version="fixed20_success_subset_cell_fallback_v1",
        formula="float32(0.8 * parent_s + 0.2 * native_average_tie_rank_over_successes); failed cells = parent text",
        failure_policy="failed cells retain parent; partial target requires N_valid >= 2 and two distinct FP32 scores; no imputation",
        fallback_targets=fallback, blended_target_count=active, all_native_disabled=active == 0,
        native_blended_cell_count=native_cells,
        numeric_noop=totals["numeric_changed_cell_count"] == 0,
        text_noop=totals["text_changed_cell_count"] == 0,
        rank_noop=totals["rank_changed_cell_count"] == 0,
        parent_to_final=totals,
        accuracy_comparison_status=("not_established_all_native_disabled" if not active else
                                    "ranking_noop_under_auc" if not totals["rank_changed_cell_count"] else
                                    "changed_rankings_unscored"),
    )
    receipt["arithmetic"]["rank"] = "full N if complete; otherwise N_valid successful cells only; then float32"
    receipt["selection_bias_note"] = (
        "Reader failures can be nonrandom. N_valid ranks estimate a different empirical population "
        "from the parent. Parent/failed/output cells are not reranked; this does not establish an accuracy gain."
    )
    return output, receipt


def blend_csv(parent_csv, native_mean_csv, output_dir, *, failure_mask_csv=None, alpha=0.2,
              test_csv=None, sample_csv=None):
    """Validate inputs first, then create a fresh local CSV/receipt directory."""
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing run: {output_dir}")
    loaded, inputs = [], {}
    for name, path in (("parent", parent_csv), ("native_mean", native_mean_csv)):
        fields, rows = read_csv(path)
        if tuple(fields) != SUBMISSION_COLUMNS:
            raise ValueError("Column order must match SUBMISSION_COLUMNS exactly")
        loaded.append(rows)
        inputs[name] = {"path": str(Path(path).resolve()), "sha256": sha256(path)}
    mask = None if failure_mask_csv is None else original._load_failure_csv(failure_mask_csv)
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
    path = output_dir / "submission.csv"
    if receipt["text_noop"]:
        path.write_bytes(Path(parent_csv).read_bytes())
    else:
        write_csv(path, SUBMISSION_COLUMNS, rows)
    receipt["inputs"] = inputs
    receipt["output"] = {"path": str(path.resolve()), "sha256": sha256(path)}
    if test_csv is not None:
        receipt["submission_contract"] = validate_submission(path, test_csv, sample_csv)
    receipt_path = output_dir / "receipt.json"
    dump_json(receipt_path, receipt)
    (output_dir / "receipt.sha256").write_text(f"{sha256(receipt_path)}  receipt.json\n", encoding="utf-8")
    return receipt
