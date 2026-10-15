"""Preview the fixed20 CSV contract using only the Python standard library.

This is an offline numerical reference, not a Kaggle serving implementation.
It does not verify model lineage, FP32/native means, inference, or Public scores.
All N input rows participate in ranks before explicit failed UIDs are restored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

from rsna_knee.contracts import (
    ID,
    SUBMISSION_COLUMNS,
    TARGETS,
    dump_json,
    index_studies,
    probability,
    read_csv,
    sha256,
    write_csv,
)

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "experiments/F007-visible-pass-F006-bothfold-native64-pass-20261014.json"
UNSUPPORTED = ("MCL", "Synovitis")
FORMULA = "rank_N(0.8 * rank_N(parent) + 0.2 * rank_N(native_mean))"


def percentile_average_ranks(values):
    """Return ascending, one-based average-tie ranks divided by the full N.

    This is the mathematical rank convention of pandas rank(method='average',
    ascending=True, pct=True). No pandas/runtime equivalence is established.
    """
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("Ranks require nonempty finite values")
    size = len(values)
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


def _load_csv(path, expected_columns):
    """Bind the read CSV to its content hash and reject changes during reading."""
    path = Path(path).resolve()
    before = sha256(path)
    fields, rows = read_csv(path, expected_columns)
    if tuple(fields) != tuple(expected_columns):
        raise ValueError(f"CSV column order must be exactly {tuple(expected_columns)!r}: {path}")
    indexed = index_studies(rows)
    if sha256(path) != before:
        raise ValueError(f"Input CSV changed during reading: {path}")
    return rows, indexed, {"path": str(path), "sha256": before, "bytes": path.stat().st_size}


def _load_probabilities(path):
    rows, indexed, descriptor = _load_csv(path, SUBMISSION_COLUMNS)
    if not rows:
        raise ValueError("Probability CSV must contain at least one study")
    values = {key: {target: probability(row[target]) for target in TARGETS} for key, row in indexed.items()}
    return rows, indexed, values, descriptor


def _uid_hash(uids):
    # JSON prevents ambiguity for synthetic identifiers containing newlines.
    payload = json.dumps(list(uids), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def preview(parent_csv, native_mean_csv, output_dir, *, failed_uids_csv=None):
    """Write one new preview directory after validating both complete inputs.

    An optional failed-UID CSV has exactly the StudyInstanceUID column. Even
    failed rows must have finite, complete native probabilities: this tool takes
    a supplied full-N numerical table and does not simulate inference failures.
    """
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output directory must be new: {output}")
    parent_rows, parent, parent_values, parent_descriptor = _load_probabilities(parent_csv)
    _, native, native_values, native_descriptor = _load_probabilities(native_mean_csv)
    if parent.keys() != native.keys():
        raise ValueError("Parent and native_mean CSV UID sets must match exactly")
    failed = {}
    failed_descriptor = None
    if failed_uids_csv is not None:
        _, failed, failed_descriptor = _load_csv(failed_uids_csv, (ID,))
        if failed.keys() - parent.keys():
            raise ValueError("Failed UID CSV contains a study absent from the probability inputs")

    ordered_uids = [row[ID] for row in parent_rows]
    # Copy strings first so unsupported columns and failed rows keep every
    # original cell string, including its decimal/exponent/whitespace spelling.
    result = [dict(row) for row in parent_rows]
    for target in TARGETS:
        if target in UNSUPPORTED:
            continue
        parent_rank = percentile_average_ranks([parent_values[key][target] for key in ordered_uids])
        native_rank = percentile_average_ranks([native_values[key][target] for key in ordered_uids])
        mixed = [0.8 * left + 0.2 * right for left, right in zip(parent_rank, native_rank)]
        candidate_rank = percentile_average_ranks(mixed)
        for row, value in zip(result, candidate_rank):
            if row[ID] not in failed:
                row[target] = repr(value)

    provenance = {
        "tracked_design_summary": {"path": str(DESIGN.relative_to(ROOT)), "sha256": sha256(DESIGN)},
        "preview_source_sha256": sha256(__file__),
        "shared_contracts_sha256": sha256(ROOT / "src/rsna_knee/contracts.py"),
    }
    # A new directory is the final ownership check; concurrent existing output
    # creation is rejected by mkdir rather than overwriting an earlier run.
    output.mkdir(parents=True, exist_ok=False)
    candidate_path = output / "preview.csv"
    write_csv(candidate_path, SUBMISSION_COLUMNS, result)
    manifest = {
        "schema_version": "fixed20_offline_csv_preview_v1",
        "stage": "offline_preview_only",
        "created_at_UTC": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "inputs": {"parent": parent_descriptor, "native_mean": native_descriptor, "failed_uids": failed_descriptor},
        "studies": len(ordered_uids),
        "targets": list(TARGETS),
        "ordered_parent_UID_sha256": _uid_hash(ordered_uids),
        "explicit_failed_UID_count": len(failed),
        "sorted_explicit_failed_UID_sha256": _uid_hash(sorted(failed)),
        "numerical_contract": {
            "formula": FORMULA,
            "fixed_weights": {"parent": 0.8, "native_mean": 0.2},
            "rank": "ascending one-based average ties divided by full N; applied per target at both stages",
            "arithmetic": "Python float; no FP32/runtime equivalence established",
            "rank_population": "all N rows, including explicit failed UIDs, before restoration",
            "output_order": "parent CSV row order; native_mean aligned by UID",
            "unchanged_parent_columns": list(UNSUPPORTED),
            "failed_UID_restoration": "all 12 original parent cell strings; no rerank after restoration",
            "string_preservation_scope": "CSV cell values, not whole-file bytes or original quoting",
        },
        "provenance": provenance,
        "output": {"file": candidate_path.name, "sha256": sha256(candidate_path), "bytes": candidate_path.stat().st_size},
        "qualification": {
            "Kaggle_serving_qualified": False,
            "existing_inference_equivalence_verified": False,
            "parent_or_F006_model_lineage_verified": False,
            "native_mean_aggregation_verified": False,
            "GPU_or_MRI_inference_performed": False,
            "hidden_completion_verified": False,
            "Public_score_verified": False,
            "training_network_upload_or_submission_performed": False,
        },
        "limitations": [
            "Offline numerical preview only; this CSV and manifest issue no Kaggle execution or accuracy qualification.",
            "The full raw bounded design and original combined runtime are unavailable in this working copy.",
            "Input CSVs are validated numerically; model identity, calibration, provenance and failure detection are not proven.",
            "No coefficient search; weights are fixed at 0.8 and 0.2.",
        ],
    }
    manifest_path = output / "manifest.json"
    dump_json(manifest_path, manifest)
    manifest_hash = sha256(manifest_path)
    (output / "manifest.sha256").write_text(f"{manifest_hash}  manifest.json\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="固定20%のオフラインCSVプレビュー。Kaggle本推論との等価・提出資格・Public改善は未検証。"
    )
    parser.add_argument("--parent-csv", required=True, type=Path, help="親モデルの全N検査・12所見の確率CSV")
    parser.add_argument("--native-mean-csv", required=True, type=Path, help="同じ全UIDのnative平均確率CSV")
    parser.add_argument("--failed-uids-csv", type=Path, help="任意。StudyInstanceUID列だけの明示失敗UID一覧")
    parser.add_argument("--output-dir", required=True, type=Path, help="新規ディレクトリのみ。既存runを上書きしない")
    args = parser.parse_args(argv)
    manifest = preview(
        args.parent_csv, args.native_mean_csv, args.output_dir, failed_uids_csv=args.failed_uids_csv
    )
    print(
        json.dumps(
            {
                "stage": manifest["stage"],
                "studies": manifest["studies"],
                "output_dir": str(args.output_dir),
                "Kaggle_serving_qualified": False,
                "existing_inference_equivalence_verified": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
