"""Prepare only a p011 V3 diagnostic256; preserve every current64/model asset."""

import hashlib
import importlib.util
import json
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "artifacts/kaggle/p011-profile64-independent224-v3-20261009"
CURRENT_BOOK = ROOT / "notebooks/public/22a_profile_p011_independent224_64_v3.ipynb"
PLAN = ROOT / "artifacts/kaggle/native392-full-shards-v2-20261008/plan.json"
PLAN_SHA = "12bc17f371997b20d1b1f85de83b8792a2e058ee409e08b0bff70aa50d7ff425"
PARENT_PROOF = ROOT / "experiments/p011-artificial-preparation-validation-v3-20261009.json"
PARENT_PROOF_SHA = "c9549811f800c11ced15da8381a257879d59497d6f531ad7d83c09f8aa8f24b7"
CURRENT_BOOK_SHA = "62e6aaefda761077dafbbe56167732784f9c14ec11230df0e197481918974020"
CURRENT_PACKAGE_SHA = "18087a2a089be593fa327ddff74e0ffe166a4f886e922f664a3dfa5fbcf051a1"
CURRENT_COHORT_SHA = "342ac914493d88970ad531ecec67fce643f90177b37722f88b86662266dfa502"
OUTPUT = ROOT / "artifacts/kaggle/p011-profile256-independent224-v1-20261009"
NOTEBOOK = ROOT / "notebooks/public/22b_profile_p011_independent224_256_v1.ipynb"
AUDIT = ROOT / "experiments/p011-profile256-preparation-20261009.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_builder():
    spec = importlib.util.spec_from_file_location("_p011_256_parent", ROOT / "scripts/build_p011_independent224.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_assets():
    if sha(PARENT_PROOF) != PARENT_PROOF_SHA:
        raise ValueError("Frozen p011 V3 validation changed")
    proof = load_json(PARENT_PROOF)
    assets = {**proof["source_SHA"]}
    assets.update({r["notebook"]: r["sha256"] for r in proof["notebooks"]})
    assets.update({str(Path(p) / "package.json").replace("\\", "/"): h for p, h in proof["package_json_sha256"].items()})
    parent = load_json(CURRENT / "package.json")
    for name, value in proof["helper_sha256"].items():
        assets[str((CURRENT / name).relative_to(ROOT)).replace("\\", "/")] = value
    assets.update(
        {
            "scripts/p010_author224_runtime.py": proof["helper_sha256"]["p010_author224_metrics.py"],
            "scripts/p006_profile_runtime.py": proof["helper_sha256"]["p006_metrics.py"],
            "scripts/p007_profile_runtime.py": proof["helper_sha256"]["p007_metrics.py"],
            str((CURRENT / "p011_author224_stock.py").relative_to(ROOT)).replace("\\", "/"): parent[
                "stock_source_sha256"
            ],
            str((CURRENT / "cohort.json").relative_to(ROOT)).replace("\\", "/"): CURRENT_COHORT_SHA,
            str(PLAN.relative_to(ROOT)).replace("\\", "/"): PLAN_SHA,
        }
    )
    for path, expected in assets.items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Frozen asset changed: {path}")
    if sha(CURRENT_BOOK) != CURRENT_BOOK_SHA or sha(CURRENT / "package.json") != CURRENT_PACKAGE_SHA:
        raise ValueError("Current64 genealogy changed")
    return assets


def extend_cohort(plan, prefix, size=256):
    """Extend the sealed64 with complete supplied groups; structural fields only."""
    if size < len(prefix["uids"]) or isinstance(size, bool):
        raise ValueError("Extension must retain the complete current64 prefix")
    if plan.get("targets_used_for_selection") is not False:
        raise ValueError("Only target-free structural plans are allowed")
    groups, by_uid = defaultdict(list), {}
    gold_uids, gold_groups = set(plan["gold_ids"]), set(plan["gold_groups"])
    for row in plan["studies"]:
        if set(row) != {"StudyInstanceUID", "group_id", "fold", "source"}:
            raise ValueError("No target/Report fields are allowed")
        uid, group = row["StudyInstanceUID"], row["group_id"]
        if not isinstance(uid, str) or not uid or not isinstance(group, str) or not group or uid in by_uid:
            raise ValueError("Unique nonempty UID/group identities are required")
        if type(row["fold"]) is not int or not 0 <= row["fold"] < plan["n_folds"]:
            raise ValueError("Supplied fold contract changed")
        if row["source"] != "report_weak" or uid in gold_uids or group in gold_groups:
            raise ValueError("Gold UID/group or nonweak structural row encountered")
        groups[group].append(row)
        by_uid[uid] = row
    prefix_uids = prefix["uids"]
    if len(prefix_uids) != 64 or len(set(prefix_uids)) != 64:
        raise ValueError("Current64 must be unique and complete")
    if prefix["structural_plan_sha256"] != PLAN_SHA or prefix["groups"] != {
        uid: by_uid[uid]["group_id"] for uid in prefix_uids if uid in by_uid
    }:
        raise ValueError("Current64 no longer binds the supplied plan")
    selected = [by_uid[uid] for uid in prefix_uids]
    used_groups = {row["group_id"] for row in selected}
    selected_set = set(prefix_uids)
    if any({r["StudyInstanceUID"] for r in groups[g]} - selected_set for g in used_groups):
        raise ValueError("Current64 splits a supplied whole group")
    seed = prefix["seed"]
    for group in sorted(groups, key=lambda g: hashlib.sha256((seed + ":" + g).encode()).hexdigest()):
        if group in used_groups:
            continue
        if len(selected) + len(groups[group]) <= size:
            selected.extend(sorted(groups[group], key=lambda r: r["StudyInstanceUID"]))
            used_groups.add(group)
        if len(selected) == size:
            break
    uids = [row["StudyInstanceUID"] for row in selected]
    if len(uids) != size or len(set(uids)) != size or uids[:64] != prefix_uids:
        raise ValueError("Cannot extend the fixed prefix to the exact requested size")
    return {
        "schema_version": "p011_fixed256_wholegroup_unlabelled_v1",
        "seed": seed,
        "uids": uids,
        "groups": {r["StudyInstanceUID"]: r["group_id"] for r in selected},
        "folds": {r["StudyInstanceUID"]: r["fold"] for r in selected},
        "sources": {r["StudyInstanceUID"]: r["source"] for r in selected},
        "gold_UIDs": plan["gold_ids"],
        "gold_groups": plan["gold_groups"],
        "gold_UID_and_group_excluded": True,
        "whole_groups_preserved": True,
        "Report_or_targets_included": False,
        "structural_plan_sha256": PLAN_SHA,
        "ordered_uid_sha256": hashlib.sha256("\n".join(uids).encode()).hexdigest(),
        "first64_ordered_uid_sha256": prefix["ordered_uid_sha256"],
        "first64_cohort_sha256": CURRENT_COHORT_SHA,
        "current64_prefix_preserved": True,
        "extension_selected_from_teacher_or_scores": False,
    }


def cohort256():
    frozen_assets()
    return extend_cohort(load_json(PLAN), load_json(CURRENT / "cohort.json"))


def make_book(cohort):
    builder = load_builder()
    book, stock = builder.make_book(cohort=cohort)
    intro = book["cells"][0]["source"][0]
    if intro.count("64") != 1:
        raise ValueError("Original profile intro contract changed")
    book["cells"][0]["source"][0] = intro.replace("64", "256")
    parent = load_json(CURRENT_BOOK)
    # Only the diagnostic intro and the UID/Gold structural prelude may differ.
    differences = [i for i, (left, right) in enumerate(zip(parent["cells"], book["cells"], strict=True)) if left != right]
    if differences != [0, 12]:
        raise ValueError(f"Unexpected diagnostic256 cell delta: {differences}")
    expected_prelude = builder.code(builder.profile_prelude(cohort))
    if book["cells"][12] != expected_prelude:
        raise ValueError("Only the frozen original prelude may introduce the256 cohort")
    return book, stock


def build(output=OUTPUT, notebook=NOTEBOOK, audit=AUDIT):
    assets = frozen_assets()
    output, notebook, audit = map(Path, (output, notebook, audit))
    if any(path.exists() for path in (output, notebook, audit)):
        raise FileExistsError("Use fresh256 paths; retain existing source/evidence")
    cohort = cohort256()
    builder = load_builder()
    book, stock = make_book(cohort)
    output.mkdir(parents=True)
    builder.write(notebook, book)
    shutil.copyfile(notebook, output / notebook.name)
    for name, path in builder.HELPERS.items():
        shutil.copyfile(path, output / name)
    (output / "p011_author224_stock.py").write_bytes(stock.encode("utf-8"))
    for name in ("original-metadata.json", "inputs-manual.json", "public-pretrained-eligibility.json"):
        shutil.copyfile(CURRENT / name, output / name)
    builder.write(output / "cohort.json", cohort)
    parent = load_json(CURRENT / "package.json")
    checks = {
        **parent["checks"],
        "frozen_current64_serving_helpers_generators_unchanged": assets == frozen_assets(),
        "exact256_unique_UIDs_and_first64_order_prefix": len(cohort["uids"]) == 256
        and cohort["uids"][:64] == load_json(CURRENT / "cohort.json")["uids"],
        "supplied_wholegroups_folds_and_Gold_exclusion": True,
        "only_intro_and_structural_prelude_cells_changed": True,
        "all_six_embedded_source_literals_same_as_current64": True,
        "same_original14_models_decoder_coefficient_resource_and_math": True,
        "all_code_compile_from_frozen_original_make_book": True,
        "private256_launch_gate_closed_until_actual64_root_review": True,
    }
    receipt = {
        **parent,
        "schema_version": "p011_fixed256_local_preparation_v1",
        "checks": checks,
        "all_passed": all(checks.values()),
        "studies": 256,
        "group_count": len(set(cohort["groups"].values())),
        "notebook_sha256": sha(notebook),
        "ordered_uid_sha256": cohort["ordered_uid_sha256"],
        "cohort_json_sha256": sha(output / "cohort.json"),
        "generator_sha256": sha(__file__),
        "generation_reuses": "Frozen original make_book/profile_prelude/code/write; identical original build package layout",
        "parent64_notebook_sha256": CURRENT_BOOK_SHA,
        "parent64_package_sha256": CURRENT_PACKAGE_SHA,
        "parent64_cohort_sha256": CURRENT_COHORT_SHA,
        "parent_original_builder_sha256": sha(ROOT / "scripts/build_p011_independent224.py"),
        "parent_original_profile_builder_sha256": sha(ROOT / "scripts/build_p011_profile.py"),
        "frozen_asset_sha256": assets,
        "launch_gate_passed": False,
        "actual64_result_root_review_passed": False,
        "actual256_result_review_passed": False,
        "candidate_gate_passed": False,
        "launch_gate_requirement": "Root must separately verify the completed current private64 V3 result before any private256 Save/Run.",
        "runtime_documentation_note": "Frozen finish_profile qualification_remaining still mentions actual64; its schema and behavior are unchanged.",
        "API_calls": 0,
        "actual_MRI_model_GPU_runs": 0,
        "accuracy_evaluated": False,
        "decision": "Prepared locally; HOLD private256 launch pending actual64 root review",
    }
    if any(sha(output / name) != value for name, value in parent["helper_sha256"].items()):
        raise ValueError("Package helper byte drift")
    if sha(output / "p011_author224_stock.py") != parent["stock_source_sha256"]:
        raise ValueError("Stock LF byte drift")
    builder.write(output / "package.json", receipt)
    builder.write(audit, receipt)
    return receipt


if __name__ == "__main__":
    result = build()
    print(json.dumps({"studies": result["studies"], "notebook_sha256": result["notebook_sha256"], "launch_gate_passed": False}))
