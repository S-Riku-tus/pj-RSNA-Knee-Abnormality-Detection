"""Prepare a fixed64 whole-group unlabelled p011 profile; no remote execution."""

import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "artifacts/kaggle/native392-full-shards-v2-20261008/plan.json"
PLAN_SHA = "12bc17f371997b20d1b1f85de83b8792a2e058ee409e08b0bff70aa50d7ff425"
OUTPUT = ROOT / "artifacts/kaggle/p011-profile64-independent224-v3-20261009"
NOTEBOOK = ROOT / "notebooks/public/22a_profile_p011_independent224_64_v3.ipynb"
AUDIT = ROOT / "experiments/p011-profile64-preparation-v3-20261009.json"


def cohort64():
    assert hashlib.sha256(PLAN.read_bytes()).hexdigest() == PLAN_SHA
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for row in plan["studies"]:
        assert set(row) == {"StudyInstanceUID", "group_id", "fold", "source"}
        assert row["StudyInstanceUID"] not in plan["gold_ids"] and row["group_id"] not in plan["gold_groups"]
        groups[row["group_id"]].append(row)
    selected = []
    for group in sorted(groups, key=lambda g: hashlib.sha256(("p011-profile64-v1:" + g).encode()).hexdigest()):
        if len(selected) + len(groups[group]) <= 64:
            selected.extend(sorted(groups[group], key=lambda r: r["StudyInstanceUID"]))
        if len(selected) == 64:
            break
    assert len(selected) == 64
    uids = [row["StudyInstanceUID"] for row in selected]
    assert len(set(uids)) == 64
    return {
        "schema_version": "p011_fixed64_wholegroup_unlabelled_v1",
        "seed": "p011-profile64-v1",
        "uids": uids,
        "groups": {r["StudyInstanceUID"]: r["group_id"] for r in selected},
        "gold_UIDs": plan["gold_ids"],
        "gold_groups": plan["gold_groups"],
        "gold_UID_and_group_excluded": True,
        "whole_groups_preserved": True,
        "Report_or_targets_included": False,
        "structural_plan_sha256": PLAN_SHA,
        "ordered_uid_sha256": hashlib.sha256("\n".join(uids).encode()).hexdigest(),
    }


if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("_p011_builder", ROOT / "scripts/build_p011_independent224.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    result = builder.build(OUTPUT, NOTEBOOK, AUDIT, cohort=cohort64())
    print(
        json.dumps(
            {"studies": result["studies"], "notebook_sha256": result["notebook_sha256"], "candidate_gate_passed": False}
        )
    )
