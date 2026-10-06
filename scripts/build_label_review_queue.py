"""Create a private, training-only report evidence review queue; never change labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import tempfile
from collections import Counter, defaultdict
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

if __package__:
    from .audit_training_labels import audit_folds, expected_components, training_rows
else:
    from audit_training_labels import audit_folds, expected_components, training_rows


REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT_NAMES = ("studies.csv", "review.csv", "manifest.json", "fold-audit.json", "codebook.json")
REVIEW_FIELDS = (
    "evidence_text",
    "negation",
    "uncertainty",
    "current_finding",
    "historical_finding",
    "not_mentioned",
    "reviewed_state",
    "proposed_label",
    "reviewer",
    "reviewed_at",
    "notes",
)
STATE_EXPECTED = {"P": 1.0, "N": 0.0, "B": 0.0, "U": None, "M": None}


def private_output(path):
    """Require a fresh path under the repository's actual Git-ignored local outputs."""
    output = Path(path).resolve()
    if output.exists():
        raise ValueError("Use a new output directory; preserve previous review queues")
    if not any(output.is_relative_to(REPOSITORY / name) for name in ("artifacts", "data")):
        raise ValueError("Report review outputs must be inside this repository's Git-ignored artifacts/ or data/")
    for name in OUTPUT_NAMES:
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--", str(output / name)],
            cwd=REPOSITORY,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise ValueError("Every report review output must be Git-ignored and untracked")
    return output


def label_kind(raw):
    value = probability(raw, missing=True)
    if value is None:
        return "missing"
    if value == 0:
        return "negative"
    if value == 1:
        return "positive"
    return "soft"


def select_studies(rows, component_by_uid, states, size, seed):
    """Cycle rare target/state strata in seeded order, sampling at most one per component."""
    pools = defaultdict(list)
    for row in rows:
        for target in TARGETS:
            kind = label_kind(row[target])
            state = states[row[ID]][target] if states else ""
            pools[(target, kind, state)].append(row)
    strata = sorted(
        pools,
        key=lambda item: (
            len({component_by_uid[row[ID]] for row in pools[item]}),
            TARGETS.index(item[0]),
            item[1:],
        ),
    )
    for stratum, members in pools.items():
        members.sort(
            key=lambda row: hashlib.sha256(
                json.dumps([seed, *stratum, row[ID]], ensure_ascii=False).encode()
            ).hexdigest()
        )
    used, selected, offsets = set(), [], Counter()
    while len(selected) < size:
        added = False
        for stratum in strata:
            members = pools[stratum]
            while offsets[stratum] < len(members):
                row = members[offsets[stratum]]
                offsets[stratum] += 1
                component = component_by_uid[row[ID]]
                if component not in used:
                    used.add(component)
                    selected.append((row, stratum))
                    added = True
                    break
            if len(selected) == size:
                break
        if not added:
            break
    selected_ids = {row[ID] for row, _ in selected}
    coverage = [
        {
            "target": target,
            "label_kind": kind,
            "source_state": state,
            "available_studies": len(pools[(target, kind, state)]),
            "available_components": len({component_by_uid[row[ID]] for row in pools[(target, kind, state)]}),
            "selected_studies": sum(row[ID] in selected_ids for row in pools[(target, kind, state)]),
            "primary_review_studies": sum(stratum == (target, kind, state) for _, stratum in selected),
        }
        for target, kind, state in strata
    ]
    return selected, coverage


def build_queue(
    train_path,
    weak_path,
    provenance_path,
    manifest_dir,
    config_path,
    output_dir,
    *,
    groups_path=None,
    states_path=None,
    holdout_folds=(0,),
    size=150,
    seed=20261006,
    reason="Audit report evidence and missing/negative label semantics before a controlled teacher change",
):
    output = private_output(output_dir)
    if type(size) is not int or size < 1 or type(seed) is not int or not reason.strip():
        raise ValueError("Provide a positive queue size, integer seed and nonempty reason")
    root = Path(manifest_dir)
    inputs = {
        "train": Path(train_path),
        "weak": Path(weak_path),
        "provenance": Path(provenance_path),
        "config": Path(config_path),
        **{name: root / name for name in ("manifest.json", "weak.csv", "gold.csv")},
    }
    if groups_path:
        inputs["groups"] = Path(groups_path)
    if states_path:
        inputs["states"] = Path(states_path)
    hashes = {name: sha256(path) for name, path in inputs.items()}
    metadata = json.loads(inputs["manifest.json"].read_text(encoding="utf-8-sig"))
    folds = sorted(set(holdout_folds))
    if not folds or any(type(f) is not int or not 0 <= f < metadata["n_folds"] for f in folds):
        raise ValueError("Reserve at least one existing holdout fold")
    if len(folds) >= metadata["n_folds"]:
        raise ValueError("At least one training fold must remain")
    provenance = json.loads(inputs["provenance"].read_text(encoding="utf-8-sig"))
    if provenance.get("gold_used_for_tuning") is not False:
        raise ValueError("Training label review requires a reviewed source declaring no gold tuning")
    # Existing audit checks source hashes, label equality and transitive report/group leakage.
    with tempfile.TemporaryDirectory(prefix="rsna-review-audit-") as temporary:
        fold_audit = audit_folds(
            train_path,
            weak_path,
            provenance_path,
            manifest_dir,
            Path(temporary) / "audit.json",
            config_path,
            groups_path,
        )
    studies, official = training_rows(train_path)
    _, prepared_rows = read_csv(root / "weak.csv", (*SUBMISSION_COLUMNS, "group_id", "fold", "source"))
    prepared = index_studies(prepared_rows)
    components = expected_components(studies, groups_path)
    component_by_uid = {
        key: hashlib.sha256(min(component).encode()).hexdigest() for component in components for key in component
    }
    holdout = {key for key, row in prepared.items() if int(row["fold"]) in folds}
    blocked_components = {component_by_uid[key] for key in official | holdout}
    eligible = [
        row
        for row in prepared_rows
        if int(row["fold"]) not in folds and component_by_uid[row[ID]] not in blocked_components
    ]
    if not eligible:
        raise ValueError("No training studies remain after gold and holdout component exclusions")
    states = None
    if states_path:
        expected_hash = provenance.get("source_artifact_sha256", {}).get("label_states.csv")
        if not expected_hash or hashes["states"] != expected_hash:
            raise ValueError("Label states must match the label_states.csv hash in reviewed provenance")
        _, state_rows = read_csv(states_path, SUBMISSION_COLUMNS)
        states = index_studies(state_rows)
        _, source_rows = read_csv(weak_path, SUBMISSION_COLUMNS)
        source = index_studies(source_rows)
        if states.keys() != source.keys():
            raise ValueError("State rows must cover exactly the source weak labels")
        for key, row in source.items():
            for target in TARGETS:
                state = states[key][target]
                if state not in STATE_EXPECTED or STATE_EXPECTED[state] != probability(row[target], missing=True):
                    raise ValueError("Source state/value mismatch; preserve N/B and U/M distinctions")
    selected, coverage = select_studies(eligible, component_by_uid, states, size, seed)
    study_rows, review_rows = [], []
    for position, (row, stratum) in enumerate(selected, start=1):
        key = row[ID]
        study_rows.append(
            {
                "review_order": position,
                ID: key,
                "fold": row["fold"],
                "group_id": row["group_id"],
                "component_id": component_by_uid[key],
                "primary_target": stratum[0],
                "primary_label_kind": stratum[1],
                "primary_source_state": stratum[2],
                "report_language": "",
                "report_sha256": hashlib.sha256(studies[key]["Report"].encode()).hexdigest(),
                "Report": studies[key]["Report"],
            }
        )
        for target in TARGETS:
            review_rows.append(
                {
                    "review_order": position,
                    ID: key,
                    "target": target,
                    "primary_review": int(target == stratum[0]),
                    "existing_label": row[target],
                    "existing_label_kind": label_kind(row[target]),
                    "existing_source_state": states[key][target] if states else "",
                    **dict.fromkeys(REVIEW_FIELDS, ""),
                }
            )
    if {name: sha256(path) for name, path in inputs.items()} != hashes:
        raise ValueError("Inputs changed during review queue generation")
    output.mkdir(parents=True)
    write_csv(output / "studies.csv", study_rows[0].keys(), study_rows)
    write_csv(output / "review.csv", review_rows[0].keys(), review_rows)
    dump_json(output / "fold-audit.json", fold_audit)
    dump_json(
        output / "codebook.json",
        {
            "scope": "Manual report evidence review; not image truth, validation data or automatic label correction",
            "workflow": "Read studies.csv reports; review primary_review=1 first; fill remaining targets as time permits",
            "report_language": "Human-recorded language; empty is unreviewed, not automatically inferred",
            "evidence_text": "Copy the exact report evidence; leave empty if the target is not mentioned",
            "binary_fields": {
                key: "yes / no / unclear; empty means unreviewed"
                for key in ("negation", "uncertainty", "current_finding", "historical_finding", "not_mentioned")
            },
            "reviewed_state": "positive / explicit_negative / below_threshold / uncertain / not_mentioned / unresolved",
            "proposed_label": "Optional 0/1/soft proposal; blank is not negative. No tool applies this field to training",
            "source_state_contract": {
                "P": "positive=1",
                "N": "explicit absence=0",
                "B": "below threshold=0",
                "U": "uncertain=missing",
                "M": "unmentioned/withheld=missing",
            },
            "selection_warning": "Purposive rare-stratum sample; coverage is not a prevalence or accuracy estimate",
            "independence": "Keep every declared holdout fold out of extraction/prompt/rule development; freeze rules before evaluation",
            "privacy": "Keep all files local in Git-ignored outputs; reports and identifiers must not enter shared docs",
        },
    )
    result = {
        "schema_version": 1,
        "status": "prepared_for_manual_review_not_relabelled",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "selection_seed": seed,
        "split_seed": metadata["seed"],
        "holdout_folds": folds,
        "requested_studies": size,
        "selected_studies": len(selected),
        "review_cells": len(review_rows),
        "eligible_training_studies": len(eligible),
        "eligible_components": len({component_by_uid[row[ID]] for row in eligible}),
        "fewer_than_requested": len(selected) < size,
        "label_provenance": provenance,
        "inputs_sha256": hashes,
        "source_sha256": {
            str(path.relative_to(REPOSITORY)): sha256(path)
            for path in (
                Path(__file__).resolve(),
                Path(__file__).resolve().with_name("audit_training_labels.py"),
                REPOSITORY / "src/rsna_knee/contracts.py",
                REPOSITORY / "src/rsna_knee/prepare.py",
            )
        },
        "outputs_sha256": {name: sha256(output / name) for name in OUTPUT_NAMES if name != "manifest.json"},
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "sampling": "Round-robin target/state strata, rare component counts first, seeded SHA-256 row order, one study per transitive component",
        "stratum_coverage": coverage,
        "unavailable_binary_or_missing_strata": [
            {"target": target, "label_kind": kind}
            for target in TARGETS
            for kind in ("positive", "negative", "missing")
            if not any(row["target"] == target and row["label_kind"] == kind for row in coverage)
        ],
        "gold_studies_selected": 0,
        "holdout_studies_selected": 0,
        "labels_changed": False,
        "patient_independence_verified": False,
        "excluded_scope": "Only prepared weak training rows eligible; all-missing rows excluded by prepare remain excluded",
        "interpretation": "Training-side semantic audit only; not unbiased label accuracy, OOF, clinical truth or Public performance",
    }
    dump_json(output / "manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--weak-labels", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--groups")
    parser.add_argument("--label-states", help="Optional reviewed P/N/B/U/M source CSV; provenance hash required")
    parser.add_argument(
        "--holdout-fold",
        type=int,
        action="append",
        required=True,
        help="Repeat to protect multiple future evaluation folds",
    )
    parser.add_argument("--size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        result = build_queue(
            args.train_csv,
            args.weak_labels,
            args.provenance,
            args.manifest_dir,
            args.config,
            args.output_dir,
            groups_path=args.groups,
            states_path=args.label_states,
            holdout_folds=args.holdout_fold,
            size=args.size,
            seed=args.seed,
            reason=args.reason,
        )
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "selected_studies",
                    "review_cells",
                    "holdout_folds",
                    "fewer_than_requested",
                    "labels_changed",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
