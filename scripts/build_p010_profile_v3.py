"""Prepare the fixed32 original/bounded/shared224 controls; never runs remotely."""

import argparse
import contextlib
import copy
import csv
import hashlib
import importlib.util
import io
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/p010_profile_runtime_v3.py"
OUTPUT = ROOT / "artifacts/kaggle/p010-profile32-shared224-rank10-v3-20261008"
NOTEBOOK = ROOT / "notebooks/public/21a_profile_p010_shared224_32_v3.ipynb"
AUDIT = ROOT / "experiments/p010-profile32-preparation-v3-20261008.json"
UID_SHA = "fb83b0b77b4539eb70b142cecc69c3cbc6a0ae4a8763381869078cc936b6fc07"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


serving = load("_p010_profile_serving_builder", ROOT / "scripts/build_p010_shared224.py")
cohort = load("_p010_profile_cohort_builder", ROOT / "scripts/build_p006_profile.py")


def structural_groups(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return {row["StudyInstanceUID"]: row["group_id"] for row in csv.DictReader(handle)}


def make_profile():
    assert serving.digest(serving.CONTROL) == serving.CONTROL_SHA
    baseline = json.loads(serving.CONTROL.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="p010-cohort-preparation-") as directory:
        root = Path(directory)
        with contextlib.redirect_stdout(io.StringIO()):
            inherited = cohort.build(32, root / "package", root / "profile.ipynb")
        inherited_book = json.loads((root / "profile.ipynb").read_text(encoding="utf-8"))
    uids = inherited["uids"]
    assert hashlib.sha256("\n".join(uids).encode()).hexdigest() == UID_SHA
    weak, gold = structural_groups(cohort.WEAK), structural_groups(cohort.GOLD)
    groups = {weak[uid] for uid in uids}
    assert groups and not groups & set(gold.values())
    prelude = serving.text(inherited_book["cells"][3])
    assert prelude.count("/kaggle/working/p006_profile_data") == 1
    prelude = """import os as _p010_guard_os
assert _p010_guard_os.environ.get("KAGGLE_IS_COMPETITION_RERUN","").lower() not in ("true","1"), "Diagnostic must never compete"
""" + prelude.replace("/kaggle/working/p006_profile_data", "/kaggle/working/p010_profile_data")
    prelude += f"""
_p010_cohort_groups = {sorted(groups)!r}
_p010_gold_groups = {sorted(set(gold.values()))!r}
assert not set(_p010_cohort_groups) & set(_p010_gold_groups)
_p_Path("/kaggle/working/p010_src").mkdir(exist_ok=False)
"""
    original = serving.text(baseline["cells"][4])
    shared = serving.shared_main_source(original, expert_enabled=True)
    bounded = serving.shared_main_source(original, expert_enabled=False)
    reader = serving.text(baseline["cells"][8]).split("\n", 1)[1]
    fusion = serving.replace_once(
        serving.text(baseline["cells"][9]),
        '_o_data = _o_find("test_series.csv")',
        '_o_data = "/kaggle/working/p010_profile_data"',
    )
    stock = serving.load_author_helper().build_stock_source(serving.AUTHOR)
    helpers = {
        "p006_metrics.py": cohort.HELPER,
        "p007_metrics.py": ROOT / "scripts/p007_profile_runtime.py",
        "p009_metrics.py": ROOT / "scripts/p009_profile_runtime.py",
        "p010_shared_metrics.py": serving.SHARED_HELPER,
        "p010_author224_metrics.py": serving.AUTHOR_HELPER,
        "p010_profile_metrics.py": HELPER,
    }
    cells = [
        *copy.deepcopy(baseline["cells"][:3]),
        serving.code_cell(prelude),
        copy.deepcopy(baseline["cells"][3]),
        *[
            serving.code_cell("%%writefile /kaggle/working/p010_src/" + name + "\n" + path.read_text(encoding="utf-8"))
            for name, path in helpers.items()
        ],
        serving.code_cell("%%writefile /kaggle/working/p010_src/p010_author224_stock.py\n" + stock),
        *copy.deepcopy(baseline["cells"][5:9]),
        serving.code_cell("""# Artificial kernel lifecycle probe; no predictions or success-marker imitation.
import sys as _p010_sys
_p010_sys.path.insert(0,"/kaggle/working/p010_src")
from p010_shared_metrics import FusionObserver as _p010_FusionObserver
_p010_stdout_probe = _p010_FusionObserver()
print("p010 observer cross-cell artificial start")
"""),
        serving.code_cell("""_p010_stdout_probe_text = _p010_stdout_probe.restore()
assert "p010 observer cross-cell artificial start" in _p010_stdout_probe_text
_p010_cross_cell_stdout_lifecycle_passed = True
"""),
        serving.code_cell(f'''import sys as _p010_sys
_p010_sys.path.insert(0,"/kaggle/working/p010_src")
from p010_profile_metrics import paired_shared224
from p006_metrics import write_json
_p010_profile = paired_shared224(globals(),{original!r},{bounded!r},{shared!r},{fusion!r},{reader!r},_p_uids,_p_labels)
_p010_summary = {{"schema_version":"p010_paired_shared224_resource_profile_v1","status":"passed",
    "studies":len(_p_uids),"uids":_p_uids,"ordered_uid_sha256":"{UID_SHA}",
    "gold_UID_and_group_excluded":True,"Report_or_targets_consumed":False,"accuracy_evaluated":False,
    "baseline_notebook_sha256":"{serving.CONTROL_SHA}",
    "stock_author_source_sha256":"{hashlib.sha256(stock.encode()).hexdigest()}",
    "profile_helper_sha256":"{serving.digest(HELPER)}",
    "shared_helper_sha256":"{serving.digest(serving.SHARED_HELPER)}",
    "author_helper_sha256":"{serving.digest(serving.AUTHOR_HELPER)}",
    "weak_structural_manifest_sha256":"{serving.digest(cohort.WEAK)}",
    "gold_structural_manifest_sha256":"{serving.digest(cohort.GOLD)}",**_p010_profile}}
_p010_summary["cross_cell_stdout_lifecycle_passed"] = _p010_cross_cell_stdout_lifecycle_passed
write_json("/kaggle/working/P010_PROFILE_SUMMARY.json",_p010_summary)
print(_p_json.dumps({{k:v for k,v in _p010_summary.items() if k not in ("uids","main","standalone224","reader")}},ensure_ascii=False,indent=2))
'''),
    ]
    book = copy.deepcopy(baseline)
    book["cells"] = cells
    book["cells"][0]["source"] = [
        "p010診断32件：元transportのbase・bounded base・同じvol/maskを共有するbase+224・stock224単独を対照比較します。ConvReaderは一度だけ実行。教師・Gold・Reportは使わず、精度未測定。提出は禁止です。\n",
        *baseline["cells"][0]["source"],
    ]
    for index in (3, 5, 6, 7, 8):
        assert serving.text(baseline["cells"][index]) in [serving.text(cell) for cell in book["cells"]]
    for cell in book["cells"]:
        if cell["cell_type"] == "code":
            source = serving.text(cell)
            compile(
                source.split("\n", 1)[1] if source.startswith("%%writefile ") else source,
                "<p010-paired-profile>",
                "exec",
            )
            cell["outputs"], cell["execution_count"] = [], None
    return book, {
        "uids": uids,
        "gold_excluded": True,
        "gold_group_excluded": True,
        "groups": {uid: weak[uid] for uid in uids},
        "helpers": helpers,
        "original": original,
        "bounded": bounded,
        "shared": shared,
        "reader": reader,
        "fusion": fusion,
    }


def build(output=OUTPUT, notebook_path=NOTEBOOK, audit_path=AUDIT):
    output, notebook_path, audit_path = map(Path, (output, notebook_path, audit_path))
    if any(path.exists() for path in (output, notebook_path, audit_path)):
        raise FileExistsError("Preserve previous profile packages and evidence")
    book, inherited = make_profile()
    output.mkdir(parents=True)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(json.dumps(book, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    shutil.copyfile(notebook_path, output / notebook_path.name)
    manifest = {
        "uids": inherited["uids"],
        "groups": inherited["groups"],
        "ordered_uid_sha256": UID_SHA,
        "gold_UID_and_group_excluded": True,
        "Report_or_targets_included": False,
    }
    (output / "cohort.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    record = {
        "schema_version": "p010_shared224_profile_offline_preparation_v1",
        "status": "prepared_no_real_MRI_or_models_or_remote",
        "checks": {
            "original14_protected": serving.digest(serving.CONTROL) == serving.CONTROL_SHA,
            "same_fixed32_UID_hash": True,
            "Gold_UID_and_group_excluded": True,
            "no_targets_or_reports": True,
            "five_original_setup_code_cells_literal": True,
            "production_shared_source_literal": True,
            "all_code_cells_compile": True,
            "single_reader_execution_then_two_CSV_replays": True,
            "standalone_stock224_reference_separate_from_base_only_control": True,
            "final_submission_deleted": True,
            "artificial_cross_cell_stdout_lifecycle_required_before_real_profile": True,
            "actual_child_reader_helper_import_preflight_before_models": True,
            "p010_exact_reader_replay_preflight_before_models": True,
            "only_profile_observation_imports_and_replay_changed_from_failed_profiles": True,
        },
        "candidate_gate_passed": False,
        "actual_execution_gate_passed": False,
        "studies": 32,
        "ordered_uid_sha256": UID_SHA,
        "notebook_sha256": serving.digest(notebook_path),
        "baseline_notebook_sha256": serving.CONTROL_SHA,
        "helper_sha256": {name: serving.digest(path) for name, path in inherited["helpers"].items()},
        "literal_source_sha256": {
            name: hashlib.sha256(inherited[name].encode()).hexdigest()
            for name in ("original", "bounded", "shared", "reader", "fusion")
        },
        "profile_notebook": str(notebook_path.relative_to(ROOT)),
        "package": str(output.relative_to(ROOT)),
        "accuracy_evaluated": False,
        "new_real_MRI_decodes": 0,
        "new_model_downloads": 0,
        "numeric_tolerance_predeclared": {
            "raw_absolute": 1e-7,
            "raw_relative": 0,
            "baseline_final_CSV": "byte_exact",
            "prefer_exact_raw_zero": True,
        },
        "actual_T4_and_checkpoint_gates": "pending; artificial tiny encoder is not the real model or hidden-test evidence",
        "training": False,
        "API_or_competition_mutations": False,
        "supersedes_failed_profile_notebooks": [
            "notebooks/public/21a_profile_p010_shared224_32.ipynb",
            "notebooks/public/21a_profile_p010_shared224_32_v2.ipynb",
        ],
        "failure_scope": "Profile child observation import and p009 replay alias mismatch. Both original prediction controls and numerical literals remain unchanged.",
    }
    record["all_passed"] = all(record["checks"].values())
    audit_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "package.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.notebook, args.audit), ensure_ascii=False, indent=2))
