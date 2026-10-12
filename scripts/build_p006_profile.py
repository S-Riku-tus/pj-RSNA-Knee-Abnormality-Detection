"""Prepare label-free paired k16/k24 resource diagnostics; never uploads."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "notebooks/public/14_submit_p005_public950_control.ipynb"
CONTROL_SHA = "75c3abe77e8f0ccdaec2f57a8946d2baf297db8cc6fbdb275e5fcb1153b57546"
WEAK = ROOT / "data/manifests/v1-vmohitrao-research/weak.csv"
GOLD = ROOT / "data/manifests/v1-vmohitrao-research/gold.csv"
SERIES = ROOT / "data/exports/rsraki-rsna-knee-sv354838181/rsna-cache-v1/raw/train_series.csv"
HELPER = ROOT / "scripts/p006_profile_runtime.py"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(cell):
    return "".join(cell["source"])


def select_uids(count):
    with WEAK.open(encoding="utf-8-sig", newline="") as stream:
        weak = {row["StudyInstanceUID"] for row in csv.DictReader(stream)}
    with GOLD.open(encoding="utf-8-sig", newline="") as stream:
        gold = {row["StudyInstanceUID"] for row in csv.DictReader(stream)}
    with SERIES.open(encoding="utf-8-sig", newline="") as stream:
        available = {row["StudyInstanceUID"] for row in csv.DictReader(stream)}
    assert not weak & gold
    assert weak <= available
    assert count in (32, 256)
    ordered = sorted(weak, key=lambda uid: hashlib.sha256(("p006-profile-v1:" + uid).encode()).hexdigest())
    return ordered[:count]


def code_cell(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


def build(count, output, notebook_path):
    assert digest(CONTROL) == CONTROL_SHA, "Scored baseline changed"
    uids = select_uids(count)
    with GOLD.open(encoding="utf-8-sig", newline="") as stream:
        gold_uids = sorted({row["StudyInstanceUID"] for row in csv.DictReader(stream)})
    notebook = json.loads(CONTROL.read_text(encoding="utf-8"))
    original = copy.deepcopy(notebook)
    prelude = f'''# 診断専用。教師・Reportを読まず、提出用submission.csvは最後に削除する。
import json as _p_json
import hashlib as _p_hashlib
import os as _p_os
from pathlib import Path as _p_Path
import pandas as _p_pd
_p_uids = {uids!r}
_p_gold = {gold_uids!r}
assert not set(_p_uids) & set(_p_gold)
_p_roots = [_p_Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'), _p_Path('/kaggle/input/rsna-knee-abnormality-detection')]
_p_official = next((p for p in _p_roots if (p / 'train_series.csv').is_file()), None)
assert _p_official is not None, 'Competition training metadata is missing'
_p_train_tree = next((p for p in (_p_official/'train_series', _p_official/'train_images') if p.is_dir()), None)
assert _p_train_tree is not None
_p_series = _p_pd.read_csv(_p_official/'train_series.csv', dtype={{'StudyInstanceUID':str, 'SeriesInstanceUID':str}})
_p_series = _p_series[_p_series.StudyInstanceUID.isin(_p_uids)]
assert set(_p_series.StudyInstanceUID) == set(_p_uids)
_p_root = _p_Path('/kaggle/working/p006_profile_data')
_p_root.mkdir(exist_ok=False)
_p_series.to_csv(_p_root/'test_series.csv', index=False)
_p_pd.DataFrame({{'StudyInstanceUID':_p_uids}}).to_csv(_p_root/'test.csv', index=False)
(_p_root/'test_series').symlink_to(_p_train_tree, target_is_directory=True)
_p_labels = {['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']!r}
_p_sample = _p_pd.DataFrame({{'StudyInstanceUID':_p_uids, **{{c:[0.0]*len(_p_uids) for c in _p_labels}}}})
_p_sample.to_csv(_p_root/'sample_submission.csv', index=False)
_p_Path('/kaggle/working/own_src').mkdir(exist_ok=True)
'''
    helper_cell = code_cell("%%writefile /kaggle/working/own_src/p006_metrics.py\n" + HELPER.read_text(encoding="utf-8"))
    main = source(notebook["cells"][4])
    assert main.count("    ROOT = find_test_root()") == 1
    main = main.replace("    ROOT = find_test_root()", "    ROOT = '/kaggle/working/p006_profile_data'")
    marker = "            q.put((sid, vol, mask))"
    assert main.count(marker) == 1
    main = main.replace(marker, "            _p006_masks[sid] = int(mask.sum())\n" + marker)
    tail = 'if __name__ == "__main__":\n    main()'
    assert main.count(tail) == 1
    main = main.replace(tail, '''import sys as _p_sys
_p_sys.path.insert(0, '/kaggle/working/own_src')
from p006_metrics import Resources as _p_Resources
_p006_masks = {}
_p006_successful_pairs = []
_p006_original_pair = infer_pair_probs
def _p006_observe_pair(model, plain, mirrored, device):
    result = _p006_original_pair(model, plain, mirrored, device)
    _p006_successful_pairs.append({'device':str(device), 'finite':bool(np.isfinite(result).all())})
    return result
infer_pair_probs = _p006_observe_pair
with _p_Resources() as _p_main_resources:
    main()
_p_main_report = _p_main_resources.result()
_p_main_report['valid_slices_by_uid'] = _p006_masks
_p_main_report['successful_pair_calls'] = len(_p006_successful_pairs)
assert set(_p006_masks) == set(_p_uids)
assert all(v > 0 for v in _p006_masks.values())
assert len(_p006_successful_pairs) == len(_p_uids) * len(ARMS)
assert all(row['finite'] for row in _p006_successful_pairs)
''')
    notebook["cells"][4]["source"] = main.splitlines(keepends=True)
    fusion = source(notebook["cells"][9])
    assert fusion.count('_o_data = _o_find("test_series.csv")') == 1
    fusion = fusion.replace('_o_data = _o_find("test_series.csv")', '_o_data = "/kaggle/working/p006_profile_data"')
    infer_source = source(notebook["cells"][8]).split("\n", 1)[1]
    notebook["cells"][9] = code_cell(f'''from p006_metrics import paired_readers, write_json
_p_readers, _p_comparison = paired_readers(globals(), {fusion!r}, {infer_source!r}, _p_uids, _p_labels)
_p_summary = {{'schema_version':'p006_paired_resource_profile_v1','status':'passed','studies':len(_p_uids),
  'uids':_p_uids,'gold_excluded':True,'Report_or_targets_consumed':False,
  'accuracy_evaluated':False,'cohort_source_weak_manifest_sha256':'{digest(WEAK)}',
  'baseline_notebook_sha256':'{CONTROL_SHA}', 'main':_p_main_report,
  'readers':_p_readers,'comparison':_p_comparison}}
write_json('/kaggle/working/P006_PROFILE_SUMMARY.json', _p_summary)
print(_p_json.dumps({{k:v for k,v in _p_summary.items() if k not in ('uids','readers')}}, ensure_ascii=False, indent=2))
''')
    notebook["cells"][0]["source"] = [f"p006診断 {count}検査：教師を使わず、同じUIDのConvNeXt k16/k24とCoAtNetを計測。提出用ではありません。\n", *original["cells"][0]["source"]]
    # Setup before weight guard; helper after directory creation, before main execution.
    notebook["cells"].insert(3, code_cell(prelude))
    notebook["cells"].insert(5, helper_cell)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"], cell["execution_count"] = [], None
            text = source(cell)
            if text.startswith("%%writefile "):
                text = text.split("\n", 1)[1]
            compile(text, "p006-profile", "exec")
    output.mkdir(parents=True, exist_ok=False)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    assert not notebook_path.exists()
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (output / notebook_path.name).write_bytes(notebook_path.read_bytes())
    receipt = {
        "schema_version": "p006_profile_preparation_v1", "studies": count, "uids": uids,
        "gold_excluded": True, "accuracy_evaluated": False,
        "baseline_sha256": CONTROL_SHA, "notebook_sha256": digest(notebook_path),
        "helper_sha256": digest(HELPER), "generator_sha256": digest(__file__),
        "input_sha256": {str(p.relative_to(ROOT)): digest(p) for p in (WEAK, GOLD, SERIES)},
        "real_execution_at_preparation": False,
        "observational_changes": ["working train-to-test metadata/symlink for fixed gold-excluded UID cohort", "CoAt masks/pair success/time/RSS sampling", "paired unchanged fusion cell with reader default k16/k24 and returned-mask observer", "exact UID/columns/range/rank formula validation", "delete diagnostic submission.csv at completion"],
    }
    (output / "preparation.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"studies": count, "notebook": str(notebook_path), "sha256": digest(notebook_path)}))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, choices=(32, 256), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--notebook-path", type=Path, required=True)
    args = parser.parse_args()
    build(args.count, args.output_dir, args.notebook_path)
