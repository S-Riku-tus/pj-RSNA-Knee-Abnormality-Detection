"""Build p006 with only the Apex ConvNeXt inference default k16 -> k24 changed."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "notebooks/public/14_submit_p005_public950_control.ipynb"
BASELINE_SHA256 = "75c3abe77e8f0ccdaec2f57a8946d2baf297db8cc6fbdb275e5fcb1153b57546"
OUTPUT_DIR = ROOT / "artifacts/kaggle/p006-convnext-k24-v1-20261008"
NOTEBOOK_PATH = ROOT / "notebooks/public/15_submit_p006_convnext_k24.ipynb"
AUDIT_PATH = ROOT / "experiments/p006-convnext-density-implementation-20261008.json"
CONTROL_PACKAGE = ROOT / "artifacts/kaggle/p005-public950-control-v1-20261008"
INFER_CELL_INDEX = 8
FUSION_CELL_INDEX = 9
ORIGINAL_SIGNATURE = "def run(data_dir, split, ckpts, k=16, studies=None, workers=4, bs=4):"
CANDIDATE_SIGNATURE = "def run(data_dir, split, ckpts, k=24, studies=None, workers=4, bs=4):"
INFER_MAGIC = "%%writefile /kaggle/working/own_src/infer.py"
RUNTIME_COMMAND = (
    '_o_sp.run([_o_sys.executable, "/kaggle/working/own_src/infer.py", _o_data, "test", _o_own_csv] + _o_ck,'
)
PREFACE = """p006: 自己Public0.950のp005からConvNeXtの推論窓数だけ16→24に変更した候補。
この候補のPublic/隠し完走/時間は未測定。CoAtNetの重み・96slice/K94・native320・鏡映2view、
ConvNeXt3fold重み・reader・slot/series選択・batch4・workers4、12所見の融合係数・fallbackは同一です。
9番目のcellがown_src/infer.pyを生成し、10番目がその同じ絶対pathをsubprocessで実行します。
mainのrun呼出しはkを指定しないため、変更したdefault24がDatasetへ渡ります。
生成時に固定基準hash、限定1行diff、runtimeへの伝搬を検査し、ずれがあれば生成を拒否します。
このファイルは提出推論候補で、train診断用profileは別ファイル・別用途です。
元作者の帰属・公開checkpoint利用条件・gold選択履歴は以下に保持します。
以下のp005準備時点の未採点説明は履歴で、p005の自己Public0.950は10月8日に確認されています。

---

"""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_digest(source):
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def cell_source(book, index):
    return "".join(book["cells"][index]["source"])


def code_sources(book):
    return {i: cell_source(book, i) for i, cell in enumerate(book["cells"]) if cell["cell_type"] == "code"}


def writefile_body(source, expected=INFER_MAGIC):
    first, body = source.split("\n", 1)
    if first != expected:
        raise ValueError("Unexpected writefile destination; the runtime source binding differs")
    return body


def verify_runtime_binding(book, expected_k):
    """Prove the changed default is used by the exact original subprocess path."""
    if len(book["cells"]) != 10 or len(code_sources(book)) != 7:
        raise ValueError("Expected the fixed ten-cell/seven-code-cell Apex layout")
    source = cell_source(book, INFER_CELL_INDEX)
    tree = ast.parse(writefile_body(source))
    runs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run"]
    if len(runs) != 1:
        raise ValueError("Expected exactly one reader run function")
    run = runs[0]
    names = [arg.arg for arg in run.args.args]
    expected_names = ["data_dir", "split", "ckpts", "k", "studies", "workers", "bs"]
    defaults = dict(zip(names[-len(run.args.defaults):], run.args.defaults))
    if names != expected_names or ast.literal_eval(defaults["k"]) != expected_k:
        raise ValueError("Reader run signature/default does not match the density factor")
    main_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run"
    ]
    if len(main_calls) != 1 or main_calls[0].keywords or [ast.unparse(x) for x in main_calls[0].args] != [
        "data_dir", "split", "ckpts"
    ]:
        raise ValueError("Reader entrypoint overrides the default k or no longer invokes run")
    dataset_calls = [
        node for node in ast.walk(run)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "DicomStudyDataset"
    ]
    if len(dataset_calls) != 1 or [ast.unparse(x) for x in dataset_calls[0].args] != [
        "studies", "table", "root", "k"
    ]:
        raise ValueError("Reader density is no longer passed to the original Dataset")
    windows = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "window_centres"
    ]
    if len(windows) != 1 or [ast.unparse(x) for x in windows[0].args] != ["n", "self.k", "False"]:
        raise ValueError("Dataset no longer uses its runtime density for window selection")
    if cell_source(book, FUSION_CELL_INDEX).count(RUNTIME_COMMAND) != 1:
        raise ValueError("Fusion no longer executes the exact generated reader path")
    return {
        "writefile_cell_index": INFER_CELL_INDEX,
        "writefile_path": "/kaggle/working/own_src/infer.py",
        "subprocess_cell_index": FUSION_CELL_INDEX,
        "main_call_uses_default_k": True,
        "run_dataset_window_centres_k_propagation_verified": True,
        "effective_convnext_k_per_slot": expected_k,
    }


def prepare_notebook(baseline):
    verify_runtime_binding(baseline, 16)
    prepared = copy.deepcopy(baseline)
    source = cell_source(baseline, INFER_CELL_INDEX)
    if source.count(ORIGINAL_SIGNATURE) != 1 or CANDIDATE_SIGNATURE in source:
        raise ValueError("The exact k16 signature must occur once before limited replacement")
    replacement = source.replace(ORIGINAL_SIGNATURE, CANDIDATE_SIGNATURE, 1)
    prepared["cells"][INFER_CELL_INDEX]["source"] = replacement.splitlines(keepends=True)
    if prepared["cells"][0]["cell_type"] != "markdown":
        raise ValueError("Expected baseline attribution markdown")
    prepared["cells"][0]["source"] = (PREFACE + cell_source(baseline, 0)).splitlines(keepends=True)
    for cell in prepared["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"], cell["execution_count"] = [], None
    verify_runtime_binding(prepared, 24)
    verify_single_change(baseline, prepared)
    return prepared


def verify_single_change(baseline, prepared):
    before, after = code_sources(baseline), code_sources(prepared)
    changed = [i for i in before if before[i] != after.get(i)]
    if list(before) != list(after) or changed != [INFER_CELL_INDEX]:
        raise ValueError("Only the fixed reader default signature may change")
    if after[INFER_CELL_INDEX] != before[INFER_CELL_INDEX].replace(
        ORIGINAL_SIGNATURE, CANDIDATE_SIGNATURE, 1
    ):
        raise ValueError("The reader contains a change beyond k16 -> k24")
    if prepared["metadata"] != baseline["metadata"]:
        raise ValueError("Baseline Notebook metadata must remain unchanged")
    for index in (1, 2):
        if cell_source(baseline, index) != cell_source(prepared, index):
            raise ValueError("Original acknowledgement/reproduction markdown changed")
    return changed


def syntax_check(book):
    result = []
    for index, source in code_sources(book).items():
        if source.startswith("%%writefile "):
            first, body = source.split("\n", 1)
            if first not in {
                INFER_MAGIC,
                "%%writefile /kaggle/working/own_src/preprocess.py",
                "%%writefile /kaggle/working/own_src/knee.py",
            }:
                raise ValueError("Unexpected IPython writefile destination")
            compile(body, f"p006_cell_{index}_body", "exec")
        elif source.startswith("%%"):
            raise ValueError("Unexpected IPython magic")
        else:
            compile(source, f"p006_cell_{index}", "exec")
        result.append({"cell_index": index, "compiled_without_execution": True})
    return result


def prepare_package(
    baseline_path=BASELINE, output_dir=OUTPUT_DIR, notebook_path=NOTEBOOK_PATH,
    audit_path=AUDIT_PATH, control_package=CONTROL_PACKAGE,
):
    baseline_path, output_dir, notebook_path, audit_path, control_package = map(
        Path, (baseline_path, output_dir, notebook_path, audit_path, control_package)
    )
    if any(path.exists() for path in (output_dir, notebook_path, audit_path)):
        raise FileExistsError("Use fresh package, Notebook and audit paths; never overwrite a prior run")
    if digest(baseline_path) != BASELINE_SHA256:
        raise ValueError("Fixed p005 Notebook SHA256 mismatch")
    baseline = read_json(baseline_path)
    prepared = prepare_notebook(baseline)
    syntax = syntax_check(prepared)
    metadata = read_json(control_package / "original-metadata.json")
    inputs = read_json(control_package / "inputs-manual.json")
    if (
        metadata["enableInternet"] or not metadata["enableGpu"]
        or not metadata["dockerImage"].endswith(
            "@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"
        )
        or [x["sourceId"] for x in inputs["inputs"]] != [154281, 20325538, 20440357]
    ):
        raise ValueError("Fixed p005 original container/three Input snapshots differ")
    output_dir.mkdir(parents=True)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(notebook_path, prepared)
    shutil.copyfile(notebook_path, output_dir / notebook_path.name)
    shutil.copyfile(baseline_path, output_dir / "baseline-p005.ipynb")
    for name in ("original-metadata.json", "inputs-manual.json", "public-pretrained-eligibility.json"):
        shutil.copyfile(control_package / name, output_dir / name)
    audit = {
        "schema_version": "p006_convnext_k24_implementation_v1",
        "prepared_at_jst": "2026-10-08",
        "status": "prepared_single_factor_not_executed",
        "baseline": {"path": str(baseline_path), "sha256": BASELINE_SHA256, "own_public_lb": 0.950},
        "candidate": {"path": str(notebook_path), "sha256": digest(notebook_path), "own_public_lb": None},
        "runtime_binding": verify_runtime_binding(prepared, 24),
        "prediction_code_delta": {
            "changed_cell_indices": [8], "unchanged_other_code_cells": 6,
            "before": ORIGINAL_SIGNATURE, "after": CANDIDATE_SIGNATURE,
            "only_default_k_changed": True,
        },
        "code_cell_sha256": {
            str(i): {"baseline": source_digest(code_sources(baseline)[i]), "candidate": source_digest(source)}
            for i, source in code_sources(prepared).items()
        },
        "unchanged": [
            "CoAtNet checkpoint/96slice/K94/native320/anatomical mirror2views",
            "ConvNeXt3fold checkpoints/series/preprocessing/position slots",
            "reader batch4/workers4/AMP/fusion12weights/fallback policy",
            "original container digest/3 public Input versions/Internet OFF",
            "Notebook metadata and original acknowledgements",
        ],
        "syntax_checks": syntax,
        "limitations": [
            "Window density changes prediction math; improved AUC is unmeasured.",
            "ConvNeXt forward windows increase about1.5x; total time/VRAM is unmeasured.",
            "Public CoAt training exposure makes retrospective weak folds non-independent.",
            "Original decoder installation failures/masked slices and fallback are unchanged.",
            "Visible fusion success is required; fallback to pure CoAt is not this candidate success.",
        ],
        "new_training_or_gold_selection": False,
        "real_inference_upload_or_submission_in_builder": False,
        "api_publication_scope": "Separate SDK orchestration is responsible for execution and submission.",
    }
    write_json(audit_path, audit)
    write_json(output_dir / "provenance.json", audit)
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
        for path in sorted(output_dir.iterdir()) if path.is_file()
    }
    package = {"schema_version": "p006_convnext_k24_package_v1", "files": files}
    write_json(output_dir / "package.json", package)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK_PATH)
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    args = parser.parse_args()
    audit = prepare_package(args.baseline, args.output_dir, args.notebook, args.audit)
    print(json.dumps({"notebook_sha256": audit["candidate"]["sha256"], "status": audit["status"]}))


if __name__ == "__main__":
    main()
