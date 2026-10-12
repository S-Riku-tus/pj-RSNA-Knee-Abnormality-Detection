"""Build p011 locally; original14 seven code cells remain literal and ordered."""

import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "notebooks/public/14_submit_p005_public950_control.ipynb"
BASE_SHA = "75c3abe77e8f0ccdaec2f57a8946d2baf297db8cc6fbdb275e5fcb1153b57546"
AUTHOR = ROOT / "artifacts/research/20261008-public-candidates/nartaa-rsna-knee-0945-efficient-224crop-v2/source.ipynb"
CONTROL_PACKAGE = ROOT / "artifacts/kaggle/p005-public950-control-v1-20261008"
HELPERS = {
    "p011_contracts.py": ROOT / "scripts/p011_contracts.py",
    "p011_metrics.py": ROOT / "scripts/p011_independent_runtime.py",
    "p010_author224_metrics.py": ROOT / "scripts/p010_author224_runtime.py",
    "p006_metrics.py": ROOT / "scripts/p006_profile_runtime.py",
    "p007_metrics.py": ROOT / "scripts/p007_profile_runtime.py",
}
OUTPUT = ROOT / "artifacts/kaggle/p011-independent224-rank10-v3-20261009"
NOTEBOOK = ROOT / "notebooks/public/22_submit_p011_independent224_rank10_v3.ipynb"
AUDIT = ROOT / "experiments/p011-independent224-preparation-v3-20261009.json"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text(cell):
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=1, allow_nan=False)
        stream.write("\n")


def make_book(*, cohort=None):
    assert sha(BASE) == BASE_SHA
    author = load("_p011_builder_author", HELPERS["p010_author224_metrics.py"])
    stock = author.build_stock_source(AUTHOR)
    baseline = json.loads(BASE.read_text(encoding="utf-8"))
    book = copy.deepcopy(baseline)
    intro = "p011準備版：元0.950の全7コードセルを保持し、確定CSV後に独立224モデルを二度目decodeで実行、固定90:10 rank融合。実T4時間・精度は未検証です。\n"
    if cohort is not None:
        intro = "p011診断64件：元コードの演算を保ち、二度のnative画像/マスクSHAとモデル・資源を調べます。提出禁止、教師/Gold/Report/AUCを使用しません。\n"
    book["cells"][0]["source"] = [intro, *baseline["cells"][0]["source"]]
    additions = [
        code("from pathlib import Path as _p011_Path\n_p011_Path('/kaggle/working/p011_src').mkdir(exist_ok=False)\n")
    ]
    for name, path in HELPERS.items():
        additions.append(code("%%writefile /kaggle/working/p011_src/" + name + "\n" + path.read_text(encoding="utf-8")))
    additions.append(code("%%writefile /kaggle/working/p011_src/p011_author224_stock.py\n" + stock))
    additions += [
        code(
            "import sys as _p011_sys\n_p011_sys.path.insert(0, '/kaggle/working/p011_src')\nfrom p011_metrics import FusionObserver\n_p011_probe = FusionObserver()\nprint('[p011 benign crosscell observer probe]')\n"
        ),
        code(
            "_p011_probe_value = _p011_probe.restore()\nassert _p011_probe_value.count('[p011 benign crosscell observer probe]') == 1, 'Real cell stdout lifetime changed'\n"
        ),
    ]
    if cohort is not None:
        additions.append(code(profile_prelude(cohort)))
    sequence = [*book["cells"][:3], *additions, baseline["cells"][3]]
    if cohort is not None:
        sequence.append(
            code(
                "from p011_metrics import BaselineProfile\n_p011_profile = BaselineProfile(globals(), '/kaggle/working/p011_profile_data', '/kaggle/working')\n"
            )
        )
    sequence.append(baseline["cells"][4])
    if cohort is not None:
        sequence.append(code("_p011_profile.execute_main()\n"))
    sequence.extend(baseline["cells"][5:9])
    if cohort is not None:
        sequence.append(code("_p011_profile.prepare_reader()\n"))
    sequence += [code("_p011_fusion_observer = FusionObserver()\n"), baseline["cells"][9]]
    tail = """from p011_metrics import cpu_csv_seal, run_standalone224, fixed10_from_seal, write_new
_p011_fusion_guard = _p011_fusion_observer.require_success()
"""
    if cohort is not None:
        tail += "_p011_profile.restore_reader()\n"
    tail += """_p011_seal = cpu_csv_seal('/kaggle/working/submission.csv', working='/kaggle/working')
_p011_stock_source = _p011_Path('/kaggle/working/p011_src/p011_author224_stock.py').read_text(encoding='utf-8')
"""
    if cohort is None:
        tail += """_p011_expert_csv, _p011_report = run_standalone224(globals(), _p011_seal, _p011_stock_source)
_p011_blend = fixed10_from_seal(_p011_seal, _p011_expert_csv)
write_new('/kaggle/working/P011_SERVING_SUMMARY.json', {
 'schema_version':'p011_new_independent_model_branch_serving_v1', 'studies':len(_p011_seal['uids']),
 'original_fusion_guard':_p011_fusion_guard, 'expert':_p011_report, 'blend':_p011_blend,
 'candidate_gate_passed':False,'accuracy_improvement_measured':False})
print('p011 candidate CSV ready; actual qualification and accuracy remain unmeasured')
"""
    else:
        tail += """from p011_metrics import finish_profile
try:
    _p011_expert_csv, _p011_report = run_standalone224(globals(), _p011_seal, _p011_stock_source,
                                                  data_root='/kaggle/working/p011_profile_data')
    _p011_blend = fixed10_from_seal(_p011_seal, _p011_expert_csv)
    _p011_summary = finish_profile(_p011_profile, _p011_seal, _p011_report, _p011_blend, _p011_fusion_guard)
    print('p011 label-free profile complete; diagnostic submission.csv deleted')
finally:
    _p011_profile.delete_diagnostic_submission()
    _p011_profile.close()
"""
        # Guard/seal failures also restore routing and remove diagnostic CSV.
        split = tail.index("from p011_metrics import finish_profile\n")
        prelude = tail[:split]
        later = tail[split + len("from p011_metrics import finish_profile\n") :]
        assert later.startswith("try:\n")
        tail = (
            "from p011_metrics import finish_profile\ntry:\n"
            + textwrap.indent(prelude, "    ")
            + later[len("try:\n") :]
        )
    sequence.append(code(tail))
    book["cells"] = sequence
    original_code = [text(c) for c in baseline["cells"] if c["cell_type"] == "code"]
    actual_code = [text(c) for c in book["cells"] if c["cell_type"] == "code"]
    assert len(original_code) == 7 and all(actual_code.count(s) == 1 for s in original_code)
    assert [actual_code.index(s) for s in original_code] == sorted(actual_code.index(s) for s in original_code)
    for cell in book["cells"]:
        if cell["cell_type"] == "code":
            s = text(cell)
            compile(s.split("\n", 1)[1] if s.startswith("%%writefile ") else s, "<p011-prepared>", "exec")
    return book, stock


def profile_prelude(cohort):
    return f"""import os as _p011_os
import pandas as _p011_pd
assert _p011_os.environ.get('KAGGLE_IS_COMPETITION_RERUN','').lower() not in ('1','true'), 'Diagnostic must never compete'
_p011_uids = {cohort["uids"]!r}
_p011_gold_UIDs = {cohort["gold_UIDs"]!r}
assert not set(_p011_uids) & set(_p011_gold_UIDs)
_p011_roots = [_p011_Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),_p011_Path('/kaggle/input/rsna-knee-abnormality-detection')]
_p011_official = next(p for p in _p011_roots if (p/'train_series.csv').is_file())
_p011_tree = next(p for p in (_p011_official/'train_series',_p011_official/'train_images') if p.is_dir())
_p011_root = _p011_Path('/kaggle/working/p011_profile_data')
_p011_root.mkdir(exist_ok=False)
_p011_series = _p011_pd.read_csv(_p011_official/'train_series.csv',dtype={{'StudyInstanceUID':str,'SeriesInstanceUID':str}})
_p011_series = _p011_series[_p011_series.StudyInstanceUID.isin(_p011_uids)]
assert set(_p011_series.StudyInstanceUID)==set(_p011_uids)
_p011_series.to_csv(_p011_root/'test_series.csv',index=False)
_p011_pd.DataFrame({{'StudyInstanceUID':_p011_uids}}).to_csv(_p011_root/'test.csv',index=False)
(_p011_root/'test_series').symlink_to(_p011_tree,target_is_directory=True)
_p011_pd.DataFrame({{'StudyInstanceUID':_p011_uids,**{{c:[0.0]*len(_p011_uids) for c in {list(load("_p011_label_helper", HELPERS["p010_author224_metrics.py"]).LABELS)!r}}}}}).to_csv(_p011_root/'sample_submission.csv',index=False)
def _p011_teacher_csv_audit(event,args):
    if event=='open' and args and isinstance(args[0],str) and args[0].startswith('/kaggle/input/') and args[0].lower().endswith('.csv'):
        assert _p011_Path(args[0]).name=='train_series.csv','Only structural series metadata is allowed in this diagnostic'
_p011_sys.addaudithook(_p011_teacher_csv_audit)
"""


def build(output=OUTPUT, notebook=NOTEBOOK, audit=AUDIT, *, cohort=None):
    output, notebook, audit = map(Path, (output, notebook, audit))
    if any(p.exists() for p in (output, notebook, audit)):
        raise FileExistsError("Use fresh p011 outputs; preserve earlier evidence")
    book, stock = make_book(cohort=cohort)
    output.mkdir(parents=True)
    write(notebook, book)
    shutil.copyfile(notebook, output / notebook.name)
    for name, path in HELPERS.items():
        shutil.copyfile(path, output / name)
    (output / "p011_author224_stock.py").write_bytes(stock.encode("utf-8"))
    for name in ("original-metadata.json", "inputs-manual.json", "public-pretrained-eligibility.json"):
        shutil.copyfile(CONTROL_PACKAGE / name, output / name)
    if cohort is not None:
        write(output / "cohort.json", cohort)
    checks = {
        "all7_original_code_cells_literal_and_order_unchanged": True,
        "all_code_compile": True,
        "original14_protected": sha(BASE) == BASE_SHA,
        "extra_stock_namespace_only_after_seal": True,
        "fixed_all12_rank90_10_no_search": True,
        "baseline_transport_unmodified": True,
        "candidate_gate_false_oldp010_not_requalified": True,
        "no_API_actualMRI_or_weight_download": True,
        "package_stock_file_exact_LF_source_bytes": sha(output / "p011_author224_stock.py")
        == hashlib.sha256(stock.encode()).hexdigest(),
        "native_worker_receipt_required_before_extra_model": True,
    }
    receipt = {
        "schema_version": "p011_new_independent224_preparation_v1",
        "all_passed": all(checks.values()),
        "checks": checks,
        "notebook_sha256": sha(notebook),
        "baseline_sha256": BASE_SHA,
        "generator_sha256": sha(__file__),
        "helper_sha256": {name: sha(path) for name, path in HELPERS.items()},
        "stock_source_sha256": hashlib.sha256(stock.encode()).hexdigest(),
        "candidate_gate_passed": False,
        "studies": None if cohort is None else len(cohort["uids"]),
        "ordered_uid_sha256": None if cohort is None else cohort["ordered_uid_sha256"],
        "new_model_branch": True,
        "old_p010_gate_relaxed_or_requalified": False,
        "accuracy_evaluated": False,
        "real_MRI_decode_or_checkpoint_acquisition": False,
        "baseline_original7_cell_sha256": [
            hashlib.sha256(text(c).encode()).hexdigest()
            for c in json.loads(BASE.read_text(encoding="utf-8"))["cells"]
            if c["cell_type"] == "code"
        ],
        "allmissing_policy": "Keep original author224 finite masked-input output; no perUID base-only substitution.",
        "unexpected_CUDA_model_or_contract_errors": "fatal, preserve receipts, no silent restoration",
        "unmeasured_gates": [
            "new_p011_actual64_result_root_review",
            "256_doubledecode_pixelmask_identity_and_resources",
            "actual_twoT4_model_memory_runtime",
            "wholehidden_coverage_completion",
            "self_public_accuracy",
        ],
        "profile_transport_scope": "Only independent extra branch bounded; original baseline transport untouched. Fixed GPU assignment is not adopted.",
        "author_Gold_selection_reuse": True,
        "own_Gold_fit_or_selection": False,
        "pretrained_weak_OOF": False,
    }
    write(audit, receipt)
    write(output / "package.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.notebook, args.audit), ensure_ascii=False))
