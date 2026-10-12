"""Prepare a paired plain/mirror k16 profile on the fixed p006 cohort; never uploads."""

import argparse
import ast
import contextlib
import importlib.util
import io
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


profile = load("p006_profile_base", ROOT / "scripts/build_p006_profile.py")
serving = load("p007_mirror_serving", ROOT / "scripts/build_p007_convnext_mirror.py")
HELPER = ROOT / "scripts/p007_profile_runtime.py"
OUTPUT_DIR = ROOT / "artifacts/kaggle/p007-profile-32-plain-mirror-v2-20261008"
NOTEBOOK_PATH = ROOT / "notebooks/public/16a_profile_p007_plain_mirror_32.ipynb"
AUDIT_PATH = ROOT / "experiments/p007-profile32-preparation-v2-20261008.json"


def make_profile(count):
    if count not in (32, 256):
        raise ValueError("Expected the fixed nested cohort32 or256")
    if profile.digest(profile.CONTROL) != profile.CONTROL_SHA:
        raise ValueError("Fixed p005 source changed")
    candidate = profile.digest(serving.NOTEBOOK_PATH)
    candidate_audit = serving.control.read_json(serving.AUDIT_PATH)
    if candidate != candidate_audit["candidate_sha256"]:
        raise ValueError("The prepared p007 serving Notebook/audit differs")
    baseline = serving.control.read_json(profile.CONTROL)
    mirror_book = serving.control.read_json(serving.NOTEBOOK_PATH)
    serving.verify_mirror_change(baseline, mirror_book)
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        base_path = temporary / "profile.ipynb"
        with contextlib.redirect_stdout(io.StringIO()):
            inherited = profile.build(count, temporary / "package", base_path)
        book = serving.control.read_json(base_path)
    code = [profile.source(c) for c in book["cells"] if c["cell_type"] == "code"]
    assert all(serving.control.cell_source(baseline, i) in code for i in (3, 5, 6, 7, 8))
    old_tail = ast.parse(profile.source(book["cells"][-1]))
    paired_calls = [node for node in ast.walk(old_tail) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id == "paired_readers"]
    assert len(paired_calls) == 1
    fusion = ast.literal_eval(paired_calls[0].args[1])
    plain = ast.literal_eval(paired_calls[0].args[2])
    assert fusion == serving.control.cell_source(baseline, 9).replace(
        '_o_data = _o_find("test_series.csv")', '_o_data = "/kaggle/working/p006_profile_data"'
    )
    assert plain == serving.control.writefile_body(serving.control.cell_source(baseline, 8))
    mirrored = serving.control.writefile_body(serving.control.cell_source(mirror_book, 8))
    book["cells"][-1] = profile.code_cell(f'''from p007_metrics import paired_mirror_readers, write_json
_p007_readers, _p007_comparison = paired_mirror_readers(globals(), {fusion!r}, {plain!r}, {mirrored!r}, _p_uids, _p_labels)
_p007_summary = {{'schema_version':'p007_paired_mirror_resource_profile_v2','status':'passed',
  'studies':len(_p_uids),'uids':_p_uids,'gold_excluded':True,'Report_or_targets_consumed':False,
  'accuracy_evaluated':False,'cohort_source_weak_manifest_sha256':'{profile.digest(profile.WEAK)}',
  'baseline_notebook_sha256':'{profile.CONTROL_SHA}', 'candidate_notebook_sha256':'{candidate}',
  'p006_helper_sha256':'{profile.digest(profile.HELPER)}','p007_helper_sha256':'{profile.digest(HELPER)}',
  'main':_p_main_report,'readers':_p007_readers,'comparison':_p007_comparison}}
write_json('/kaggle/working/P007_PROFILE_SUMMARY.json', _p007_summary)
print(_p_json.dumps({{k:v for k,v in _p007_summary.items() if k not in ('uids','readers')}}, ensure_ascii=False, indent=2))
''')
    helper = profile.code_cell("%%writefile /kaggle/working/own_src/p007_metrics.py\n" + HELPER.read_text(encoding="utf-8"))
    # p006_metrics is written first; the new reader observer imports its unchanged helpers.
    book["cells"].insert(6, helper)
    book["cells"][0]["source"] = [
        f"p007診断{count}検査：p006同一UID、CoAt1回、ConvNeXt plain k16と左右鏡映2view k16のpaired資源診断。提出用ではありません。\n",
        *baseline["cells"][0]["source"],
    ]
    for cell in book["cells"]:
        if cell["cell_type"] == "code":
            source = profile.source(cell)
            if source.startswith("%%writefile "):
                source = source.split("\n", 1)[1]
            compile(source, "p007-resource-profile", "exec")
            cell["outputs"], cell["execution_count"] = [], None
    return book, inherited


def build(count=32, output_dir=OUTPUT_DIR, notebook_path=NOTEBOOK_PATH, audit_path=AUDIT_PATH):
    output_dir, notebook_path, audit_path = map(Path, (output_dir, notebook_path, audit_path))
    if any(path.exists() for path in (output_dir, notebook_path, audit_path)):
        raise FileExistsError("Use fresh package/Notebook/audit paths; existing artifacts are never overwritten")
    book, inherited = make_profile(count)
    uids = inherited["uids"]
    assert uids == profile.select_uids(count)
    output_dir.mkdir(parents=True)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    serving.control.write_json(notebook_path, book)
    shutil.copyfile(notebook_path, output_dir / notebook_path.name)
    audit = {
        "schema_version": "p007_profile_preparation_v2", "prepared_at_jst": "2026-10-08",
        "studies": count, "uids": uids, "same_p006_uid_selection": True,
        "ordered_uid_sha256": profile.hashlib.sha256("\n".join(uids).encode("utf-8")).hexdigest(),
        "gold_excluded": True, "Report_or_targets_used": False, "accuracy_evaluated": False,
        "baseline_sha256": profile.CONTROL_SHA,
        "candidate_sha256": profile.digest(serving.NOTEBOOK_PATH),
        "notebook_sha256": profile.digest(notebook_path),
        "generator_sha256": profile.digest(__file__),
        "p006_generator_sha256": profile.digest(ROOT / "scripts/build_p006_profile.py"),
        "p006_helper_sha256": profile.digest(profile.HELPER),
        "p007_helper_sha256": profile.digest(HELPER),
        "input_sha256": inherited["input_sha256"],
        "main_coat_executions": 1, "reader_executions": {"plain_k16": 1, "mirror2view_k16": 1},
        "profile_prediction_sources": {
            "plain_reader_body": "exact baseline14 body except entrypoint observation context",
            "mirror_reader_body": "exact candidate16 body except entrypoint observation context",
            "fusion": "exact baseline14 fusion except working training-alias data path",
            "model_decoder_inputs_coefficients": "unchanged pinned p0053Inputs and Originalcontainer",
        },
        "observational_changes": [
            "reuse p006 metadata alias/mask/pair/resource helpers and same fixed cohort",
            "separate returned-mask directories for plain and mirror",
            "top-level original model forward hooks count/shape/finite logits",
            "mirror helper pass-through observer records position/mask contract and actual TTA calls",
            "reset CoAt CSV before each reader; remove stale own CSV; reject fusion fallback",
            "fixed fusion formula, identical mask and complete forward-count gates",
            "AST original run no_grad decorator and actual load/forward grad-disabled gates",
            "delete final diagnostic submission.csv",
        ],
        "limitations": [
            "finite-logit hooks synchronizeCUDA and measurements include observers/pip/cache effects",
            "mask-valid is not proof every native DICOM decoded without skip",
            "sampled RSS may double-count shared pages and GPU memory is device-wide",
            "32 cases and unlabeled score differences do not prove hidden completion or AUC gain",
        ],
        "real_inference_remote_or_public_submission_performed_at_preparation": False,
        "supersedes_unexecuted_v1": {
            "notebook_sha256": "54de4d87061df65a501fe48ce6edbd281d3cd0962052810a27ae24f554846fc6",
            "package": "artifacts/kaggle/p007-profile-32-plain-mirror-v1-20261008",
            "reason": "The embedded p007 serving v1 moved run's no_grad decorator to its helper. V1 profile is invalidated before execution; V2 embeds the corrected serving source and rejects grad-enabled execution.",
        },
    }
    serving.control.write_json(audit_path, audit)
    serving.control.write_json(output_dir / "preparation.json", audit)
    for name in ("original-metadata.json", "inputs-manual.json", "public-pretrained-eligibility.json"):
        shutil.copyfile(serving.control.CONTROL_PACKAGE / name, output_dir / name)
    serving.control.write_json(output_dir / "package.json", {
        "schema_version": "p007_profile_package_v2",
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": profile.digest(p)}
                  for p in sorted(output_dir.iterdir()) if p.is_file()},
    })
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, choices=(32, 256), default=32)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK_PATH)
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    args = parser.parse_args()
    result = build(args.count, args.output_dir, args.notebook, args.audit)
    print(json.dumps({"studies": result["studies"], "notebook_sha256": result["notebook_sha256"]}))
