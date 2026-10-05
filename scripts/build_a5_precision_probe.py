"""Build the offline A5 precision diagnostic; never run MRI/model inference locally."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "ba9491ac1e214ba55ca181de55118c9a793fe7f6163fc221fe1e68d596259dca"
DEFINITIONS = ROOT / "notebooks/public/a5_p002_definitions.py"
CONSTANTS = {
    "CROP_MM",
    "SIZE",
    "SLICE_BAND",
    "N_SLICE",
    "INTENSITY",
    "SLOTS",
    "N_SLOT",
    "LABELS",
    "N_SLOT_TYPES",
    "MASK_IDX",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "N_PLANE",
    "N_CONTRAST",
    "_PLANE_OF",
    "_CONTRAST_OF",
    "MICRO",
}


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def extract_definitions(notebook_path):
    """Copy exact source segments, excluding every top-level model/I/O/inference action."""
    payload = Path(notebook_path).read_bytes()
    if sha256(payload) != SOURCE_SHA:
        raise ValueError("Expected the fixed p002 V32 source notebook")
    original = json.loads(payload)
    source = "".join(original["cells"][5]["source"])
    lines = source.splitlines(keepends=True)
    keep = []
    provenance = []
    for node in ast.parse(source).body:
        allowed = isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))
        if isinstance(node, ast.Assign):
            names = {n.id for target in node.targets for n in ast.walk(target) if isinstance(n, ast.Name)}
            allowed = bool(names) and names <= CONSTANTS
        if not allowed:
            continue
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        segment = "".join(lines[start - 1 : node.end_lineno])
        keep.append(segment)
        provenance.append({"start": start, "end": node.end_lineno, "sha256": sha256(segment.encode())})
    header = (
        '# ruff: noqa\n# fmt: off\n"""Exact definitions from romantamrazov/rsna-knee-dinosaur-v4 Version 32, cell 5.\n'
        "Source notebook SHA-256: " + SOURCE_SHA + "\n"
        "Used by the A5 diagnostic only; top-level I/O, loading, inference and blending are removed.\n"
        "Public code attribution: https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v4\n"
        '"""\n\n'
    )
    result = header + "\n".join(keep)
    ast.parse(result)
    return result, {
        "notebook_sha256": SOURCE_SHA,
        "cell_index": 5,
        "cell_sha256": sha256(source.encode()),
        "segments": provenance,
        "definitions_sha256": sha256(result.encode()),
    }


def code_cell(source, name):
    ast.parse(source)
    return {
        "cell_type": "code",
        "id": name,
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "execution_count": None,
        "outputs": [],
    }


def probe_contract():
    audit = json.loads((ROOT / "docs/research/p002-effective-inputs-20261005.json").read_text(encoding="utf-8"))
    inputs = []
    for entry in audit["effective_inputs"]:
        if entry["key"] not in {"consolidated", "offline_timm", "competition"}:
            continue
        files = [
            {key: item[key] for key in ("path", "bytes", "sha256")}
            for item in entry["files"]
            if item["role"] in {"dinov3_fold_checkpoint", "offline_timm_wheel"}
        ]
        inputs.append(
            {
                "key": entry["key"],
                "ref": entry["ref"],
                "version": entry["version"],
                "roots": entry["root_candidates"],
                "files": files,
            }
        )
    provenance = json.loads(DEFINITIONS.with_suffix(".json").read_text(encoding="utf-8"))
    return {
        "schema": "p002_a5_precision_probe_v1",
        "inputs": inputs,
        "source": provenance,
        "checkpoint_count": 5,
        "modes": ["bf16", "fp16", "fp32"],
    }


def build(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new notebook path; existing notebooks are not overwritten")
    definitions = DEFINITIONS.read_text(encoding="utf-8")
    runtime = (ROOT / "scripts/a5_precision_probe_runtime.py").read_text(encoding="utf-8")
    contract = probe_contract()
    if sha256(definitions.encode()) != contract["source"]["definitions_sha256"]:
        raise ValueError("The copied p002 A5 definitions changed")
    intro = """# p002 A5の計算精度診断（提出用ではありません）

Public 0.937成功版のA5だけを、同じ5 checkpoint・前処理・MICRO=8・GPU 0で比較します。
**T4 x2・Internet OFF** にして、新しいprivate Notebookで実行してください。
Inputsは **大会データ**、**tonylica/rsna-knee-bend-dinov3-0917-repro-assets Version 5**、
**mattiaangeli/knee-mri-fold-weights Version 2** の3つです（既存p002の5 Inputsのままでも可）。

train_series.csvの4つのメタデータ列だけを読み、seedで64検査を固定します。
train.csv・Report・gold/weakラベルは読みません。これは速度・数値安定性の診断であり、AUC/CVではありません。
同一検査を一度だけ前処理し、保存cacheのSHAを固定してBF16/FP16/FP32で各3回比較します。
読み込み・前処理・warmupと、CUDA同期したforward所要時間を別々に保存します。
BF16が環境で使えない場合は失敗として記録し、別精度へ置き換えません。

実行後はOutputの`a5-precision-probe-.../report.json`、`comparisons.csv`、raw確率/logit NPYを確認します。
`adoption_decision`は常に未決定です。小さい差でも隠しtestで同じ順位・スコアになる保証はありません。
**submission.csvは生成せず、提出・アップロード・学習も行いません。** FP16を自動採用しません。
検査数・MICRO・checkpoint・画像条件は初回比較では変更しないでください。
実MRI・公開重みによるこの診断の完走はローカルでは未検証です。
"""
    bootstrap = (
        "import sys, types\n"
        "if 'rsna_a5_precision_probe' in sys.modules:\n"
        "    raise RuntimeError('Restart with a fresh Kaggle session')\n"
        "probe = types.ModuleType('rsna_a5_precision_probe')\n"
        "sys.modules[probe.__name__] = probe\n"
        f"exec(compile({runtime!r}, '<a5-probe-runtime>', 'exec'), probe.__dict__)\n"
        f"CONTRACT = {contract!r}\n"
        f"DEFINITION_SOURCE = {definitions!r}\n"
    )
    run = (
        "# 新しいOutputディレクトリへ保存。Kaggle上でだけ実行します。\n"
        "REPORT = probe.run_probe(CONTRACT, DEFINITION_SOURCE, studies=64, seed=20261005, repeats=3)\n"
        "print('診断結果:', REPORT['status'])\n"
        "print('保存先:', REPORT['run_directory'])\n"
        "print('採用判断:', REPORT['adoption_decision'])\n"
    )
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"isInternetEnabled": False, "isGpuEnabled": True, "dataSources": []},
        },
        "cells": [
            {
                "cell_type": "markdown",
                "id": "a5-probe-intro",
                "metadata": {},
                "source": intro.splitlines(keepends=True),
            },
            code_cell(bootstrap, "a5-probe-bootstrap"),
            code_cell(run, "a5-probe-run"),
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    return {"path": str(output), "sha256": sha256(output.read_bytes()), "status": "prepared_not_executed"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "notebooks/public/03_a5_precision_probe.ipynb")
    parser.add_argument("--extract-from", type=Path, help="One-time extraction from the hash-fixed V32 source")
    args = parser.parse_args()
    if args.extract_from:
        if DEFINITIONS.exists() or DEFINITIONS.with_suffix(".json").exists():
            raise FileExistsError("Existing definition/provenance files are never overwritten")
        definitions, provenance = extract_definitions(args.extract_from)
        DEFINITIONS.parent.mkdir(parents=True, exist_ok=True)
        DEFINITIONS.write_text(definitions, encoding="utf-8")
        DEFINITIONS.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(build(args.output), ensure_ascii=True, indent=2))
