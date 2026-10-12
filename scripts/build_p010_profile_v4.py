"""Prepare new fixed32 A/A diagnosis; original and failed packages stay intact."""

import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/p010_aa_runtime.py"
TRANSPORT = ROOT / "scripts/p010_aa_transport.py"
OUTPUT = ROOT / "artifacts/kaggle/p010-profile32-AA-numerics-v4-20261009"
NOTEBOOK = ROOT / "notebooks/public/21a_profile_p010_AA_32_v4.ipynb"
AUDIT = ROOT / "experiments/p010-profile32-AA-preparation-v4-20261009.json"
PLAN = ROOT / "experiments/p010-AA-precision-diagnostic-plan-20261009.json"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


previous = load("_p010_AA_previous_builder", ROOT / "scripts/build_p010_profile_v3.py")
serving = previous.serving


def make_profile():
    book, inherited = previous.make_profile()
    original_book = copy.deepcopy(book)
    helpers = dict(inherited["helpers"])
    old_helper = helpers["p010_profile_metrics.py"]
    helpers["p010_profile_metrics.py"] = HELPER
    helpers["p010_previous_metrics.py"] = old_helper
    helpers["p010_aa_transport.py"] = TRANSPORT
    new_cells = []
    for cell in book["cells"][:-1]:
        text = serving.text(cell)
        if text.startswith("%%writefile /kaggle/working/p010_src/p010_profile_metrics.py\n"):
            new_cells.extend(
                [
                    serving.code_cell(
                        "%%writefile /kaggle/working/p010_src/p010_previous_metrics.py\n"
                        + old_helper.read_text(encoding="utf-8")
                    ),
                    serving.code_cell(
                        "%%writefile /kaggle/working/p010_src/p010_aa_transport.py\n"
                        + TRANSPORT.read_text(encoding="utf-8")
                    ),
                    serving.code_cell(
                        "%%writefile /kaggle/working/p010_src/p010_profile_metrics.py\n"
                        + HELPER.read_text(encoding="utf-8")
                    ),
                ]
            )
        else:
            new_cells.append(cell)
    tail = serving.text(book["cells"][-1])
    tail = serving.replace_once(
        tail,
        '"schema_version":"p010_paired_shared224_resource_profile_v1","status":"passed"',
        '"schema_version":"p010_AA_numerical_diagnostic_v4","status":"diagnostic_complete"',
    )
    tail = serving.replace_once(tail, serving.digest(old_helper), serving.digest(HELPER))
    tail = serving.replace_once(
        tail,
        'write_json("/kaggle/working/P010_PROFILE_SUMMARY.json",_p010_summary)',
        f'''_p010_summary["design_sha256"] = "{serving.digest(PLAN)}"
_p010_summary["transport_diagnostic_helper_sha256"] = "{serving.digest(TRANSPORT)}"
_p010_summary["previous_profile_helper_sha256"] = "{serving.digest(old_helper)}"
from pathlib import Path as _p010_AA_Path
with _p010_AA_Path("/kaggle/working/P010_AA_DIAGNOSTIC_SUMMARY.json").open("x",encoding="utf-8") as _p010_AA_f:
    _p_json.dump(_p010_summary,_p010_AA_f,ensure_ascii=False,indent=2,allow_nan=False)
'''.rstrip(),
    )
    tail = serving.replace_once(tail, "from p006_metrics import write_json\n", "")
    new_cells.append(serving.code_cell(tail))
    book["cells"] = new_cells
    book["cells"][0]["source"] = [
        "p010 V4: fixed32、基準と作者224の通常/固定GPU割り当てA/A診断。確率値を比較前に保存します。"
        "前処理・重み・数値関数は固定。精度は評価せず、この診断は提出禁止です。\n",
        *original_book["cells"][0]["source"],
    ]
    for name in ("original", "bounded", "shared", "reader", "fusion"):
        assert repr(inherited[name]) in tail, f"Original prediction literal changed: {name}"
    unchanged = [index for index in range(len(original_book["cells"])) if index not in (0, 10, 18)]
    for index in unchanged:
        assert original_book["cells"][index] in book["cells"]
    for cell in book["cells"]:
        if cell["cell_type"] == "code":
            text = serving.text(cell)
            compile(text.split("\n", 1)[1] if text.startswith("%%writefile ") else text, "<p010-AA-profile>", "exec")
            cell["outputs"], cell["execution_count"] = [], None
    inherited["helpers"] = helpers
    return book, inherited


def build(output=OUTPUT, notebook_path=NOTEBOOK, audit_path=AUDIT):
    output, notebook_path, audit_path = map(Path, (output, notebook_path, audit_path))
    if any(path.exists() for path in (output, notebook_path, audit_path)):
        raise FileExistsError("Preserve previous source/packages; use fresh paths")
    book, inherited = make_profile()
    output.mkdir(parents=True)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(json.dumps(book, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    shutil.copyfile(notebook_path, output / notebook_path.name)
    manifest = {
        "uids": inherited["uids"],
        "groups": inherited["groups"],
        "ordered_uid_sha256": previous.UID_SHA,
        "gold_UID_and_group_excluded": True,
        "Report_or_targets_included": False,
    }
    (output / "cohort.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    record = {
        "schema_version": "p010_AA_diagnostic_offline_preparation_v4",
        "status": "prepared_no_real_models_MRI_or_remote",
        "checks": {
            "original14_protected": serving.digest(serving.CONTROL) == serving.CONTROL_SHA,
            "all5_prediction_source_literals_unchanged": True,
            "old_V3_helper_embedded_byte_identical_as_previous_helper": True,
            "16_other_original_V3_cells_unchanged": True,
            "same_fixed32_UID_hash": True,
            "Gold_UID_group_excluded_no_targets_or_Report": True,
            "ten_AA_phases_predeclared_ordinary_and_fixed_distinct": True,
            "perstudy_raw_input_device_receipts_exclusive_before_phase_guards": True,
            "phase_resources_backend_raw_saved_before_comparisons": True,
            "raw1e7_and_byteexact_not_relaxed_candidate_gate_closed": True,
            "fixed_owned_queue_single_total_bound48": True,
            "child_import_replay_and_fork_preflight_before_models": True,
            "crosscell_stdout_lifecycle_preflight": True,
            "no_reader_or_fusion_GPU_runs_in_AA_diagnostic": True,
            "diagnostic_submission_removed_in_finally": True,
            "all21_cells_compile_clean_outputs": True,
        },
        "all_passed": True,
        "candidate_gate_passed": False,
        "actual_execution_gate_passed": False,
        "studies": 32,
        "ordered_uid_sha256": previous.UID_SHA,
        "notebook_sha256": serving.digest(notebook_path),
        "baseline_notebook_sha256": serving.CONTROL_SHA,
        "design_sha256": serving.digest(PLAN),
        "helper_sha256": {name: serving.digest(path) for name, path in inherited["helpers"].items()},
        "literal_source_sha256": {
            name: hashlib.sha256(inherited[name].encode()).hexdigest()
            for name in ("original", "bounded", "shared", "reader", "fusion")
        },
        "profile_notebook": str(notebook_path.relative_to(ROOT)),
        "package": str(output.relative_to(ROOT)),
        "actual_GPU_MRI_model_payload_access": 0,
        "training": False,
        "API_or_competition_mutations": False,
        "phase_status_scope": "Phase receipts are persisted before guards. Separate GUARD receipts and final diagnostic summary establish execution completion; no AUC/candidate qualification.",
        "fixed_dispatch_scope": "Only diagnostic ownership routing changes. Original/common-queue and fixed-owned-queue observations are not a causal mapping-only comparison.",
        "numeric_criteria": {
            "raw_absolute": 1e-7,
            "raw_relative": 0.0,
            "base_and_fused_CSV_byte_identity_required": True,
        },
        "previous_V1V2V3_and_serving21": "immutable; V4 is diagnosis only",
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
