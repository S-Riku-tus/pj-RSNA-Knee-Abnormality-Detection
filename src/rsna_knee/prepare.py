"""Audit supplied metadata and build leakage-aware weak/gold manifests."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .contracts import (
    ID,
    SERIES_COLUMNS,
    SERIES_ID,
    TARGETS,
    dump_json,
    index_studies,
    probability,
    read_csv,
    sha256,
    uid,
    write_csv,
)


def audit(data_root, output):
    root = Path(data_root)
    _, rows = read_csv(root / "train.csv", (ID, "Report", *TARGETS))
    studies = index_studies(rows)
    _, series = read_csv(root / "train_series.csv", SERIES_COLUMNS)
    pairs = set()
    for row in series:
        key = (uid(row[ID]), uid(row[SERIES_ID]))
        if row[ID] not in studies or key in pairs:
            raise ValueError("Orphan or duplicate series row")
        pairs.add(key)
    label_counts, complete, partial = Counter(), 0, 0
    reports = Counter()
    for row in rows:
        values = [probability(row[t], missing=True, binary=True) for t in TARGETS]
        n = sum(v is not None for v in values)
        complete += n == 12
        partial += 0 < n < 12
        for t, v in zip(TARGETS, values):
            label_counts[t] += v is not None
        normalized = " ".join(row["Report"].split()).casefold()
        if normalized:
            reports[hashlib.sha256(normalized.encode()).hexdigest()] += 1
    result = {
        "studies": len(rows),
        "series": len(series),
        "complete_official_labels": complete,
        "partial_official_labels": partial,
        "observed_by_target": dict(label_counts),
        "studies_without_series": len(set(studies) - {s for s, _ in pairs}),
        "empty_reports": sum(not r["Report"].strip() for r in rows),
        "duplicate_report_groups": sum(n > 1 for n in reports.values()),
        "planes": dict(Counter(r["Anatomical_Plane"] for r in series)),
        "csv_sha256": {name: sha256(root / name) for name in ("train.csv", "train_series.csv")},
        "note": "CSV-only audit; no DICOM decode. No report text or study IDs in this summary.",
    }
    dump_json(output, result)
    return result


def label_template(train_path, output):
    _, rows = read_csv(train_path, (ID, *TARGETS))
    index_studies(rows)
    empty = [
        {ID: r[ID], **{t: "" for t in TARGETS}}
        for r in rows
        if all(probability(r[t], missing=True, binary=True) is None for t in TARGETS)
    ]
    write_csv(output, (ID, *TARGETS), empty)
    return {"rows": len(empty), "note": "Empty template only; populate from a reviewed label source before prepare."}


def prepare(train_path, weak_path, provenance_path, output_dir, *, n_folds=5, seed=20261002, groups_path=None):
    if n_folds < 2:
        raise ValueError("Need at least 2 folds")
    _, rows = read_csv(train_path, (ID, "Report", *TARGETS))
    studies = index_studies(rows)
    _, weak_rows = read_csv(weak_path, (ID, *TARGETS))
    weak = index_studies(weak_rows)
    if weak.keys() - studies.keys():
        raise ValueError("Weak labels contain unknown study IDs")
    provenance = json.loads(Path(provenance_path).read_text(encoding="utf-8-sig"))
    for key in ("source_url", "source_version", "license", "method", "created_at", "gold_used_for_tuning"):
        if key not in provenance or provenance[key] in (None, "", "TODO", "UNCONFIRMED"):
            raise ValueError(f"Complete label provenance first: {key}")
    if type(provenance["gold_used_for_tuning"]) is not bool:
        raise ValueError("gold_used_for_tuning must be a boolean")

    # Transitive grouping of identical report text and optional patient/site/duplicate mapping.
    parent = {key: key for key in studies}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def join(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    report_groups = {}
    for row in rows:
        report = " ".join(row["Report"].split()).casefold()
        if report:
            digest = hashlib.sha256(report.encode()).hexdigest()
            if digest in report_groups:
                join(row[ID], report_groups[digest])
            else:
                report_groups[digest] = row[ID]
    if groups_path:
        _, group_rows = read_csv(groups_path, (ID, "group_id"))
        groups = index_studies(group_rows)
        if groups.keys() != studies.keys():
            raise ValueError("Group mapping must cover every training study exactly once")
        seen = {}
        for key, row in groups.items():
            group = row["group_id"].strip()
            if not group:
                raise ValueError("Empty group_id")
            if group in seen:
                join(key, seen[group])
            else:
                seen[group] = key

    official = {
        key
        for key, row in studies.items()
        if any(probability(row[t], missing=True, binary=True) is not None for t in TARGETS)
    }
    reserved_groups = {find(key) for key in official}
    manifest, gold, ignored = [], [], Counter()
    for key, row in studies.items():
        group_hash = hashlib.sha256(find(key).encode()).hexdigest()
        if key in official:
            gold.append(
                {
                    ID: key,
                    **{t: row[t] for t in TARGETS},
                    "source": "official_image_review",
                    "group_id": group_hash,
                    "fold": -1,
                }
            )
            continue
        if find(key) in reserved_groups:
            ignored["shares_group_with_official"] += 1
            continue
        if key not in weak:
            ignored["missing_weak_row"] += 1
            continue
        values = {t: probability(weak[key][t], missing=True) for t in TARGETS}
        if all(v is None for v in values.values()):
            ignored["all_targets_missing"] += 1
            continue
        fold = int(hashlib.sha256(f"{seed}:{group_hash}".encode()).hexdigest(), 16) % n_folds
        manifest.append(
            {
                ID: key,
                **{t: "" if values[t] is None else values[t] for t in TARGETS},
                "source": "report_weak",
                "group_id": group_hash,
                "fold": fold,
            }
        )
    if not manifest:
        raise ValueError("No usable weak labels. An empty template is not training data.")
    if len({r["fold"] for r in manifest}) < 2:
        raise ValueError("Usable weak labels occupy fewer than 2 folds")
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty manifest directory; preserve previous experiments")
    fields = (ID, *TARGETS, "source", "group_id", "fold")
    write_csv(out / "weak.csv", fields, manifest)
    write_csv(out / "gold.csv", fields, gold)
    summary = {
        "schema_version": 1,
        "seed": seed,
        "n_folds": n_folds,
        "weak_studies": len(manifest),
        "gold_studies": len(gold),
        "fold_counts": dict(Counter(r["fold"] for r in manifest)),
        "excluded": dict(ignored),
        "label_provenance": provenance,
        "inputs_sha256": {
            "train": sha256(train_path),
            "weak": sha256(weak_path),
            "provenance": sha256(provenance_path),
        },
        "grouping": "duplicate_report + supplied_groups" if groups_path else "duplicate_report + study",
        "patient_independence_verified": False,
        "gold_independence": "tuned: exploratory only"
        if provenance["gold_used_for_tuning"]
        else "reserved from this training pipeline; audit label-source exposure separately",
    }
    if groups_path:
        summary["inputs_sha256"]["groups"] = sha256(groups_path)
    dump_json(out / "manifest.json", summary)
    return summary
