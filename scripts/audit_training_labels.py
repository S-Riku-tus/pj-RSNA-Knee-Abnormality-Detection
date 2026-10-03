"""CSV-only label conversion and independent manifest audit; no downloads or training."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict, deque
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


def new_output(path):
    path = Path(path)
    if path.exists():
        raise ValueError("Use a new output path; preserve previous audits")
    return path


def target_summary(rows):
    result = {}
    for target in TARGETS:
        values = [probability(row[target], missing=True) for row in rows]
        observed = [v for v in values if v is not None]
        result[target] = {
            "observed": len(observed),
            "missing": len(rows) - len(observed),
            "missing_fraction": (len(rows) - len(observed)) / len(rows) if rows else None,
            "exact_zero": observed.count(0),
            "exact_one": observed.count(1),
            "soft": sum(0 < v < 1 for v in observed),
            "mean": sum(observed) / len(observed) if observed else None,
            "minimum": min(observed) if observed else None,
            "maximum": max(observed) if observed else None,
        }
    return result


def convert_pilkwang(source, output, audit_output):
    """Published api_labeler.frame schema: score, __conf, __verdict per target."""
    output, audit_output = new_output(output), new_output(audit_output)
    if output.resolve() == audit_output.resolve():
        raise ValueError("CSV and audit outputs must differ")
    required = (ID, *(column for t in TARGETS for column in (t, t + "__conf", t + "__verdict")))
    _, rows = read_csv(source, required)
    index_studies(rows)
    converted, masked = [], Counter()
    for row in rows:
        result = {ID: row[ID]}
        for target in TARGETS:
            score = probability(row[target])
            probability(row[target + "__conf"])
            verdict = row[target + "__verdict"]
            if verdict not in ("YES", "NO", "UNK"):
                raise ValueError("Unexpected verdict; review the source schema before conversion")
            result[target] = "" if verdict == "UNK" else str(score)
            masked[target] += verdict == "UNK"
        converted.append(result)
    if not converted:
        raise ValueError("Empty source labels")
    # Validate everything before writing; preserve explicit 0/1 and soft scores.
    write_csv(output, SUBMISSION_COLUMNS, converted)
    summary = {
        "schema_version": 1,
        "method": "pilkwang api_labeler.frame: preserve scores for YES/NO; mask UNK as empty",
        "source_sha256": sha256(source),
        "output_sha256": sha256(output),
        "studies": len(converted),
        "masked_unknown_by_target": dict(masked),
        "per_target": target_summary(converted),
        "gold_exposure": "not established by conversion; audit source provenance before prepare",
    }
    dump_json(audit_output, summary)
    return summary


def training_rows(train_path):
    _, rows = read_csv(train_path, (ID, "Report", *TARGETS))
    studies = index_studies(rows)
    official = {
        key
        for key, row in studies.items()
        if any(probability(row[t], missing=True, binary=True) is not None for t in TARGETS)
    }
    return studies, official


def import_vmohitrao(source_dir, source_manifest, source_readme, train_path, output_dir, version):
    """Verify the reviewed v3 source contract and prepare a research candidate, never train."""
    if version != 3:
        raise ValueError("Only the audited Dataset Version 3 contract is supported")
    out = new_output(output_dir)
    source = Path(source_dir)
    metadata = json.loads(Path(source_manifest).read_text(encoding="utf-8-sig"))
    studies, official = training_rows(train_path)
    if metadata["source_train_sha256"] != sha256(train_path):
        raise ValueError("Source labels and cache originate from different train.csv files")
    names = ("weak_labels.csv", "label_weights.csv", "label_states.csv", "label_provenance.csv")
    tables = {}
    for name in names:
        path = source / name
        if sha256(path) != metadata["artifact_sha256"][name]:
            raise ValueError("Source artifact hash mismatch")
        required = (ID, "report_group") if name == "label_provenance.csv" else SUBMISSION_COLUMNS
        _, rows = read_csv(path, required)
        tables[name] = index_studies(rows)
        if tables[name].keys() != studies.keys() - official:
            raise ValueError("Source must cover exactly the non-gold studies")
    if metadata["expert_studies_excluded"] != len(official):
        raise ValueError("Source gold exclusion count differs")
    if metadata["generated_study_rows"] != len(tables["weak_labels.csv"]):
        raise ValueError("Source label count differs")
    gold_declaration = (
        f"All {len(official)} expert-annotated studies were excluded from report extraction and prompt development."
    )
    if gold_declaration not in Path(source_readme).read_text(encoding="utf-8-sig"):
        raise ValueError("The reviewed gold-exclusion declaration is absent from source README")
    state_counts = {target: Counter() for target in TARGETS}
    for key, labels in tables["weak_labels.csv"].items():
        if not tables["label_provenance.csv"][key]["report_group"].strip():
            raise ValueError("Empty source report group")
        for target in TARGETS:
            state = tables["label_states.csv"][key][target]
            if state not in ("P", "N", "B", "U", "M"):
                raise ValueError("Unrecognized source label state")
            value = probability(labels[target], missing=True, binary=True)
            weight = probability(tables["label_weights.csv"][key][target], binary=True)
            expected = 1 if state == "P" else 0 if state in ("N", "B") else None
            if value != expected or weight != int(expected is not None):
                raise ValueError("Source state/value/observation-mask mismatch")
            state_counts[target][state] += 1
    if {t: dict(counts) for t, counts in state_counts.items()} != metadata["states"]:
        raise ValueError("Source state distribution differs from published manifest")
    # The declaration was manually read during source audit; preserve its exact evidence.
    provenance = {
        "source_url": "https://www.kaggle.com/datasets/vmohitrao/rsna-knee-report-labels/versions/3",
        "source_version": "Dataset Version 3",
        "license": "CC BY-NC 4.0",
        "method": "Report-only LLM extraction and blind text review; P=1, N/B=0, U/M=missing",
        "created_at": metadata["api_usage"]["generated_at_utc"],
        "gold_used_for_tuning": False,
        "gold_usage_basis": "Author README declares all 58 expert studies excluded from extraction and prompt development",
        "gold_usage_evidence_sha256": sha256(source_readme),
        "source_manifest_sha256": sha256(source_manifest),
        "source_artifact_sha256": {name: sha256(source / name) for name in names},
        "models": metadata["model"],
        "adoption_scope": "Non-commercial research candidate; Kaggle submission eligibility not established",
        "notes": "Gold non-use is an author declaration, not an independent audit of their private prompt-development logs",
    }
    out.mkdir(parents=True)
    shutil.copyfile(source / "weak_labels.csv", out / "weak_labels.csv")
    shutil.copyfile(source_manifest, out / "source-manifest.json")
    shutil.copyfile(source_readme, out / "SOURCE-README.md")
    groups = [
        {
            ID: key,
            "group_id": "source_report:" + tables["label_provenance.csv"][key]["report_group"]
            if key not in official
            else "official_study:" + key,
        }
        for key in studies
    ]
    write_csv(out / "groups.csv", (ID, "group_id"), groups)
    dump_json(out / "provenance.json", provenance)
    return audit_labels(train_path, out / "weak_labels.csv", out / "label-audit.json")


def audit_labels(train_path, labels_path, output):
    output = new_output(output)
    studies, official = training_rows(train_path)
    fields, rows = read_csv(labels_path, SUBMISSION_COLUMNS)
    if tuple(fields) != SUBMISSION_COLUMNS:
        raise ValueError("Canonical label CSV must contain exactly the 13 ordered columns")
    labels = index_studies(rows)
    if labels.keys() - studies.keys():
        raise ValueError("Labels contain unknown study IDs")
    stats = target_summary(rows)
    matched_rows = 0
    for key in official & labels.keys():
        official_values = {t: probability(studies[key][t], missing=True, binary=True) for t in TARGETS}
        matches = [
            probability(labels[key][t], missing=True) == value
            for t, value in official_values.items()
            if value is not None
        ]
        matched_rows += bool(matches) and all(matches)
    all_missing = sum(all(probability(r[t], missing=True) is None for t in TARGETS) for r in rows)
    result = {
        "schema_version": 1,
        "inputs_sha256": {"train": sha256(train_path), "weak": sha256(labels_path)},
        "studies": len(rows),
        "official_studies_present": len(official & labels.keys()),
        "official_rows_matching_all_observed_labels": matched_rows,
        "official_match_note": "Equality is a copy-screening indicator, not proof of copying or gold independence",
        "non_official_studies_present": len(labels.keys() - official),
        "studies_without_label_row": len(studies.keys() - labels.keys()),
        "all_targets_missing": all_missing,
        "per_target": stats,
        "note": "Source eligibility and grouping require separate provenance and manifest audits; no AUC calculated",
    }
    dump_json(output, result)
    return result


def expected_components(studies, groups_path):
    """Graph traversal audits report/supplied-group closure, including excluded bridge rows."""
    neighbors = {key: set() for key in studies}
    pools = defaultdict(list)
    for key, row in studies.items():
        report = " ".join(row["Report"].split()).casefold()
        if report:
            pools[("report", report)].append(key)
    if groups_path:
        _, rows = read_csv(groups_path, (ID, "group_id"))
        groups = index_studies(rows)
        if groups.keys() != studies.keys():
            raise ValueError("Supplied groups must cover every study")
        for key, row in groups.items():
            group = row["group_id"].strip()
            if not group:
                raise ValueError("Empty supplied group")
            pools[("supplied", group)].append(key)
    for members in pools.values():
        for key in members[1:]:
            neighbors[members[0]].add(key)
            neighbors[key].add(members[0])
    remaining, components = set(studies), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        visited, queue = {start}, deque([start])
        while queue:
            for key in neighbors[queue.popleft()] - visited:
                visited.add(key)
                remaining.remove(key)
                queue.append(key)
        components.append(visited)
    return components


def audit_folds(train_path, weak_path, provenance_path, manifest_dir, output, config_path, groups_path=None):
    output = new_output(output)
    root = Path(manifest_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8-sig"))
    config = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    inputs = {"train": train_path, "weak": weak_path, "provenance": provenance_path}
    if groups_path:
        inputs["groups"] = groups_path
    if set(inputs) != set(manifest["inputs_sha256"]):
        raise ValueError("Supply every original manifest input, including groups if used")
    if any(sha256(path) != manifest["inputs_sha256"][key] for key, path in inputs.items()):
        raise ValueError("Manifest input hashes do not match")
    provenance = json.loads(Path(provenance_path).read_text(encoding="utf-8-sig"))
    if provenance != manifest["label_provenance"] or type(provenance.get("gold_used_for_tuning")) is not bool:
        raise ValueError("Manifest label provenance differs or has unreviewed gold exposure")
    if any(config[key] != manifest[key] for key in ("seed", "n_folds")):
        raise ValueError("Config seed/fold count differs from manifest")
    studies, official = training_rows(train_path)
    _, weak_source_rows = read_csv(weak_path, SUBMISSION_COLUMNS)
    weak_source = index_studies(weak_source_rows)
    required = (*SUBMISSION_COLUMNS, "source", "group_id", "fold")
    _, weak_rows = read_csv(root / "weak.csv", required)
    _, gold_rows = read_csv(root / "gold.csv", required)
    weak, gold = index_studies(weak_rows), index_studies(gold_rows)
    if not weak or gold.keys() != official or weak.keys() & official or weak.keys() - studies.keys():
        raise ValueError("Gold reservation or weak identity invalid")
    by_fold = {fold: [] for fold in range(manifest["n_folds"])}
    for key, row in weak.items():
        fold = int(row["fold"])
        if fold not in by_fold or row["source"] != "report_weak" or not row["group_id"]:
            raise ValueError("Invalid weak fold/source/group")
        if key not in weak_source:
            raise ValueError("Manifest weak row absent from its label source")
        values = [probability(row[t], missing=True) for t in TARGETS]
        if all(v is None for v in values):
            raise ValueError("Manifest includes a study without observed targets")
        if any(v != probability(weak_source[key][t], missing=True) for t, v in zip(TARGETS, values)):
            raise ValueError("Manifest weak values differ from source labels")
        by_fold[fold].append(row)
    for key, row in gold.items():
        if row["fold"] != "-1" or row["source"] != "official_image_review" or not row["group_id"]:
            raise ValueError("Invalid gold fold/source/group")
        if any(
            probability(row[t], missing=True, binary=True) != probability(studies[key][t], missing=True, binary=True)
            for t in TARGETS
        ):
            raise ValueError("Gold values differ from official CSV")
    groups = defaultdict(set)
    for row in weak_rows + gold_rows:
        groups[row["group_id"]].add(row["fold"])
    if any(len(folds) != 1 for folds in groups.values()):
        raise ValueError("Manifest group crosses folds or overlaps gold")
    for component in expected_components(studies, groups_path):
        members = component & weak.keys()
        if members and component & official:
            raise ValueError("Weak study connected to gold through report or supplied groups")
        if len({weak[key]["fold"] for key in members}) > 1:
            raise ValueError("Report/supplied group crosses folds")
    counts = {str(fold): len(rows) for fold, rows in by_fold.items()}
    recorded_counts = {str(key): value for key, value in manifest["fold_counts"].items()}
    if counts != {str(f): recorded_counts.get(str(f), 0) for f in by_fold}:
        raise ValueError("Manifest fold counts differ from CSV")
    if len(weak) != manifest["weak_studies"] or len(gold) != manifest["gold_studies"]:
        raise ValueError("Manifest study counts differ from CSV")
    warnings, distributions = [], {}
    for fold, validation in by_fold.items():
        training = [row for other, rows in by_fold.items() if other != fold for row in rows]
        splits = {"train": target_summary(training), "valid": target_summary(validation)}
        distributions[str(fold)] = {"train_studies": len(training), "valid_studies": len(validation), **splits}
        for name, stats in splits.items():
            for target, stat in stats.items():
                if not stat["observed"]:
                    warnings.append(f"fold {fold} {name}: no observed labels for {target}")
    result = {
        "schema_version": 1,
        "integrity_valid": True,
        "ready_for_training": not warnings,
        "readiness_scope": "CSV integrity and label coverage only; image quality and source eligibility require review",
        "warnings": warnings,
        "weak_studies": len(weak),
        "gold_studies": len(gold),
        "fold_counts": counts,
        "per_fold": distributions,
        "inputs_sha256": manifest["inputs_sha256"],
        "manifest_sha256": {name: sha256(root / name) for name in ("manifest.json", "weak.csv", "gold.csv")},
        "gold_used_for_tuning": provenance["gold_used_for_tuning"],
        "gold_evaluation": "exploratory"
        if provenance["gold_used_for_tuning"]
        else "source declaration; exposure audit required",
        "patient_independence_verified": False,
        "note": "No stratification claim; distributions and image quality still need human review. No identifiers in audit.",
    }
    dump_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    convert = sub.add_parser("convert-pilkwang")
    convert.add_argument("--source", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--audit-output", required=True)
    importer = sub.add_parser("import-vmohitrao")
    importer.add_argument("--source-dir", required=True)
    importer.add_argument("--source-manifest", required=True)
    importer.add_argument("--source-readme", required=True)
    importer.add_argument("--train-csv", required=True)
    importer.add_argument("--output-dir", required=True)
    importer.add_argument("--version", type=int, required=True)
    labels = sub.add_parser("labels")
    folds = sub.add_parser("folds")
    for command in (labels, folds):
        command.add_argument("--train-csv", required=True)
        command.add_argument("--weak-labels", required=True)
        command.add_argument("--output", required=True)
    folds.add_argument("--provenance", required=True)
    folds.add_argument("--manifest-dir", required=True)
    folds.add_argument("--config", default="configs/baseline.json")
    folds.add_argument("--groups")
    args = parser.parse_args()
    try:
        if args.command == "convert-pilkwang":
            result = convert_pilkwang(args.source, args.output, args.audit_output)
        elif args.command == "import-vmohitrao":
            result = import_vmohitrao(
                args.source_dir,
                args.source_manifest,
                args.source_readme,
                args.train_csv,
                args.output_dir,
                args.version,
            )
        elif args.command == "labels":
            result = audit_labels(args.train_csv, args.weak_labels, args.output)
        else:
            result = audit_folds(
                args.train_csv,
                args.weak_labels,
                args.provenance,
                args.manifest_dir,
                args.output,
                args.config,
                args.groups,
            )
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")
    # Detailed distributions are in the private audit file; stdout remains compact.
    print(json.dumps({key: value for key, value in result.items() if key not in ("per_target", "per_fold")}, indent=2))
    if result.get("ready_for_training") is False:
        parser.exit(2, "Review fold observation warnings before training\n")


if __name__ == "__main__":
    main()
