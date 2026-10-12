"""Prepare a fixed32 paired FOV140/160 profile without real data execution or uploads."""

import argparse
import contextlib
import copy
import hashlib
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


serving = load("p009_serving_builder", ROOT / "scripts/build_p009_coat_fov160.py")
profile = load("p009_cohort_builder", ROOT / "scripts/build_p006_profile.py")
HELPER = ROOT / "scripts/p009_profile_runtime.py"
READER_HELPER = ROOT / "scripts/p007_profile_runtime.py"
OUTPUT_DIR = ROOT / "artifacts/kaggle/p009-profile32-fov140-fov160-v2-20261008"
NOTEBOOK_PATH = ROOT / "notebooks/public/19a_profile_p009_fov140_fov160_32.ipynb"
AUDIT_PATH = ROOT / "experiments/p009-profile32-preparation-v2-20261008.json"
UID_HASH = "fb83b0b77b4539eb70b142cecc69c3cbc6a0ae4a8763381869078cc936b6fc07"


def make_profile():
    assert profile.digest(profile.CONTROL) == profile.CONTROL_SHA
    baseline = serving.control.read_json(profile.CONTROL)
    candidate = serving.control.read_json(serving.NOTEBOOK_PATH)
    serving.verify_change(baseline, candidate)
    production_audit = serving.control.read_json(serving.AUDIT_PATH)
    assert profile.digest(serving.NOTEBOOK_PATH) == production_audit["candidate_sha256"]
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        with contextlib.redirect_stdout(io.StringIO()):
            inherited = profile.build(32, directory / "package", directory / "profile.ipynb")
        base_profile = serving.control.read_json(directory / "profile.ipynb")
    assert hashlib.sha256("\n".join(inherited["uids"]).encode()).hexdigest() == UID_HASH
    prelude = profile.source(base_profile["cells"][3])
    assert prelude.count("/kaggle/working/p006_profile_data") == 1
    prelude = (
        "import os as _p_guard_os\n"
        "assert _p_guard_os.environ.get('KAGGLE_IS_COMPETITION_RERUN','').lower() not in ('1','true'), "
        "'Diagnostic profile must not be submitted to the competition'\n"
        + prelude.replace("/kaggle/working/p006_profile_data", "/kaggle/working/p009_profile_data")
    )
    main = serving.control.cell_source(baseline, 4)
    altered_main = serving.control.cell_source(candidate, 4)
    infer = serving.control.writefile_body(serving.control.cell_source(baseline, 8))
    fusion = serving.control.cell_source(baseline, 9)
    marker = '_o_data = _o_find("test_series.csv")'
    assert fusion.count(marker) == 1
    fusion = fusion.replace(marker, '_o_data = "/kaggle/working/p009_profile_data"', 1)
    tail = profile.code_cell(f'''import sys as _p_sys
_p_sys.path.insert(0, '/kaggle/working/own_src')
from p009_metrics import paired_fov
from p006_metrics import write_json
_p009_results = paired_fov(globals(), {main!r}, {altered_main!r}, {fusion!r}, {infer!r}, _p_uids, _p_labels)
_p009_summary = {{'schema_version':'p009_paired_FOV_resource_profile_v2','status':'passed',
  'studies':len(_p_uids),'uids':_p_uids,'gold_excluded':True,'Report_or_targets_consumed':False,
  'accuracy_evaluated':False,'ordered_uid_sha256':'{UID_HASH}',
  'cohort_source_weak_manifest_sha256':'{profile.digest(profile.WEAK)}',
  'baseline_notebook_sha256':'{profile.CONTROL_SHA}',
  'candidate_notebook_sha256':'{profile.digest(serving.NOTEBOOK_PATH)}',
  'p006_helper_sha256':'{profile.digest(profile.HELPER)}',
  'p007_helper_sha256':'{profile.digest(READER_HELPER)}',
  'p009_helper_sha256':'{profile.digest(HELPER)}', **_p009_results}}
write_json('/kaggle/working/P009_PROFILE_SUMMARY.json', _p009_summary)
print(_p_json.dumps({{k:v for k,v in _p009_summary.items() if k not in ('uids','main','reader')}}, ensure_ascii=False, indent=2))
''')
    book = copy.deepcopy(baseline)
    book["cells"] = [
        *book["cells"][:3], profile.code_cell(prelude), book["cells"][3],
        *[profile.code_cell("%%writefile /kaggle/working/own_src/" + name + "\n"
                            + path.read_text(encoding="utf-8"))
          for name, path in (("p006_metrics.py", profile.HELPER), ("p007_metrics.py", READER_HELPER),
                             ("p009_metrics.py", HELPER))],
        *book["cells"][5:9], tail,
    ]
    book["cells"][0]["source"] = [
        "p009診断32検査：同一UIDのCoAt FOV140/160を比較し、ConvNeXt k16は一度だけ実行して両融合で共用します。教師・Report・Goldは使いません。精度改善は未測定です。提出用ではありません。\n",
        *baseline["cells"][0]["source"],
    ]
    assert all(serving.control.cell_source(baseline, index) in
               [profile.source(cell) for cell in book["cells"] if cell["cell_type"] == "code"]
               for index in (3, 5, 6, 7, 8))
    for cell in book["cells"]:
        if cell["cell_type"] == "code":
            text = profile.source(cell)
            if text.startswith("%%writefile "):
                text = text.split("\n", 1)[1]
            compile(text, "p009-paired-profile", "exec")
            cell["outputs"], cell["execution_count"] = [], None
    return book, inherited


def build(output_dir=OUTPUT_DIR, notebook_path=NOTEBOOK_PATH, audit_path=AUDIT_PATH):
    output_dir, notebook_path, audit_path = map(Path, (output_dir, notebook_path, audit_path))
    if any(path.exists() for path in (output_dir, notebook_path, audit_path)):
        raise FileExistsError("Preserve earlier package/Notebook/audit evidence")
    book, inherited = make_profile()
    output_dir.mkdir(parents=True)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    serving.control.write_json(notebook_path, book)
    shutil.copyfile(notebook_path, output_dir / notebook_path.name)
    receipt = {
        "schema_version": "p009_profile_preparation_v2", "prepared_at_jst": "2026-10-08",
        "status": "prepared_no_real_inference_or_remote_execution",
        "studies": 32, "uids": inherited["uids"], "ordered_uid_sha256": UID_HASH,
        "same_p006_and_p007_32UID_cohort": True, "gold_excluded": True,
        "Report_or_targets_consumed": False, "accuracy_evaluated": False,
        "baseline_sha256": profile.CONTROL_SHA,
        "candidate_sha256": profile.digest(serving.NOTEBOOK_PATH),
        "notebook_sha256": profile.digest(notebook_path),
        "generator_sha256": profile.digest(__file__),
        "helper_sha256": {str(path.relative_to(ROOT)): profile.digest(path)
                          for path in (profile.HELPER, READER_HELPER, HELPER)},
        "input_sha256": inherited["input_sha256"],
        "main_coat_executions": 2, "reader_convnext_executions": 1, "fusion_executions": 2,
        "prediction_source_contract": {
            "FOV140": "Exact original14 main prediction source plus UID alias and pass-through mask/worker/forward observations; no p009 guards backported.",
            "FOV160": "Exact production19 prediction source plus the same observations.",
            "reader": "Exact original14 k16 body and its run no_grad decorator; original entrypoint enclosed in plain forward/coverage observer.",
            "fusion": "Exact original14 body except working UID data alias. Second execution intercepts only the exact reader subprocess and replays the hash-bound first reader CSV.",
            "weights_inputs_coefficients_container": "Unchanged pinned p0053Inputs, original checkpoints and OriginalT4offline environment.",
        },
        "gates": [
            "Artificial3task actualLinuxfork Pool transport preflight; each main _pre_one/_pre_init ForkingPickler loads identity; registered module removed on exit",
            "Actual every initialized worker builder CROP_MM140/160, IMG384 and source SHA",
            "32 valid masks,32 successful pairs,64 shape1x94x3x320x320 finite/no_grad forwards per main",
            "No main FALLBACK/FAILED or fusion EXCEPTION/FALLBACK; exactly one write/success marker per branch",
            "One reader load3models,24 k16 forward calls for8 B4 batches, all grad disabled and32 returned masks",
            "Same hash-bound reader CSV used in both fusions and independent standard-library fusion formula",
            "Specific diagnostic fastread.py and its exact cache path only; unrelated files preserved",
            "Stale main/reader CSV removal and no final submission.csv",
        ],
        "limitations": [
            "Local Windows cannot execute the Linuxfork probe; this is an explicit saved-Kaggle execution gate.",
            "Label-free changes and resource feasibility do not prove AUC improvement or hidden completion.",
            "Public fullfit weights do not provide independent OOF on this weak cohort.",
            "Pass-through finite hooks synchronizeCUDA; timings include observers, pip, decoding and cache effects.",
            "Valid masks do not prove every native slice decoded without skip.",
            "Sampled RSS can count shared pages repeatedly; GPU memory is device-wide.",
        ],
        "supersedes_failed_profile_v1": {
            "remote_ref": "rsraki/p009-profile-32-fov140-fov160", "saved_version": 1,
            "old_notebook_sha256": "5ee2308bdaa844f20ecdd187fdd2283609772da68df88f1dae6f04a98e83340f",
            "old_package": "artifacts/kaggle/p009-profile32-fov140-fov160-v1-20261008",
            "failure_evidence": "experiments/p009-profile32-first-failure-review-20261008.json",
            "fix_scope": "Profile exec module registration/task transport only; original140/main160/reader/fusion prediction strings and production19 are unchanged.",
        },
    }
    serving.control.write_json(audit_path, receipt)
    serving.control.write_json(output_dir / "preparation.json", receipt)
    for name in ("original-metadata.json", "inputs-manual.json", "public-pretrained-eligibility.json"):
        shutil.copyfile(serving.control.CONTROL_PACKAGE / name, output_dir / name)
    serving.control.write_json(output_dir / "package.json", {
        "schema_version": "p009_profile_package_v2",
        "files": {path.name: {"bytes": path.stat().st_size, "sha256": profile.digest(path)}
                  for path in sorted(output_dir.iterdir()) if path.is_file()},
    })
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK_PATH)
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    arguments = parser.parse_args()
    result = build(arguments.output_dir, arguments.notebook, arguments.audit)
    print(json.dumps({"studies": result["studies"], "notebook_sha256": result["notebook_sha256"]}))
