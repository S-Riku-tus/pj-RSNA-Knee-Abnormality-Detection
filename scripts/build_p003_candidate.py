"""Build a self-contained, unexecuted, pinned P003 Kaggle notebook, offline."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks/public/vendor/haideptry-speedy-v2.ipynb"
INPUTS = ROOT / "docs/research/p003-input-contract-20261005.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def build(output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Use a new output directory; no existing experiment is overwritten")
    inputs = json.loads(INPUTS.read_text(encoding="utf-8"))
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    provenance = json.loads(SOURCE.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    if digest(SOURCE) != provenance["frozen_notebook_sha256"]:
        raise ValueError("Frozen V2 notebook hash mismatch")
    guard_path = ROOT / "scripts/p003_guard.py"
    guard = guard_path.read_text(encoding="utf-8")
    ast.parse(guard)
    source_cells = {str(i): "".join(c["source"]) for i, c in enumerate(source["cells"]) if c["cell_type"] == "code"}
    for value in source_cells.values():
        ast.parse(value)
    if {key: hashlib.sha256(value.encode()).hexdigest() for key, value in source_cells.items()} != provenance[
        "source_cells"
    ]:
        raise ValueError("Frozen inference code changed")
    dino = json.loads((ROOT / "docs/research/p002-effective-inputs-20261005.json").read_text(encoding="utf-8"))
    contract = {
        "candidate_id": inputs["configuration_id"],
        "inputs": inputs["inputs"],
        "source_notebook_sha256": provenance["original_notebook_sha256"],
        "source_cells": provenance["source_cells"],
        "cell_order": [int(i) for i in source_cells],
        "dino_member_ids": [member["id"] for member in dino["dino20_members"]],
    }
    input_lines = [f"- [{item['ref']}]({item['url']}) — Version {item['version']}" for item in inputs["inputs"]]
    intro = (
        """# p003: 0.937からの次の比較候補

haideptry **V2** の推論ソースを保持し、全4 CoAt系統・生確率・統合方式を追加検査する未実行候補です。
作者Public **0.943** は当方の実測値ではありません。0.945/0.955への到達は未確認です。
過去の採点時Input版は不明であり、今回固定した版・hashの組合せとして比較します。

新しいprivate NotebookにImportし、**GPU T4 x2 / Internet OFF** を選択します。
下記14 Inputsだけを指定版で追加し、Save Version → Save & Run All。元のp002は保持します。
すべてのセルを新しいセッションで順番に実行してください。
最後に **P003 READY FOR MANUAL SUBMISSION** と **P003_READY.json** を確認し、
**submission.csv** を手動提出します。停止した場合は提出せずログを確認してください。
表示用の少数test成功は、隠しtestの9時間完走や精度向上の証拠ではありません。

この候補はA5 BF16を原版のまま保持します。別添のA5精度診断による変更を混ぜません。
演算のFP32再試行は原版どおり記録し、途中失敗による定数埋め・モデル省略は拒否します。
ラベルによる学習・係数選択・自動アップロード・自動提出は行いません。

## Inputs

"""
        + "\n".join(input_lines)
        + "\n"
    )
    cells = [
        {"cell_type": "markdown", "id": "p003-instructions", "metadata": {}, "source": intro.splitlines(keepends=True)}
    ]
    bootstrap = (
        "import builtins as _p003_builtins\nimport types as _p003_types\nimport sys as _p003_sys\n"
        "if hasattr(_p003_builtins, '_rsna_p003_guard'):\n    raise RuntimeError('Restart with a fresh session')\n"
        "_p003_module = _p003_types.ModuleType('rsna_p003_guard')\n"
        "_p003_sys.modules['rsna_p003_guard'] = _p003_module\n"
        f"exec(compile({guard!r}, '<p003-guard>', 'exec'), _p003_module.__dict__)\n"
        f"_p003_guard = _p003_module.P003Guard({contract!r})\n"
        "_p003_builtins._rsna_p003_guard = _p003_guard\n_p003_guard.prepare()\n"
    )
    cells.append(code_cell(bootstrap, "p003-preflight"))
    for index, cell in enumerate(source["cells"]):
        if cell["cell_type"] == "code":
            wrapper = (
                "import builtins as _p003_builtins\n"
                f"_p003_builtins._rsna_p003_guard.run_cell({index}, {source_cells[str(index)]!r}, globals())\n"
            )
            cells.append(code_cell(wrapper, f"p003-source-{index}"))
        elif index != 0:
            cells.append({**cell, "id": f"p003-original-{index}"})
    cells.append(
        code_cell(
            "import builtins as _p003_builtins\n_p003_builtins._rsna_p003_guard.finish(globals())\n", "p003-finish"
        )
    )
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": cells,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
            "kaggle": {"isInternetEnabled": False, "isGpuEnabled": True, "dataSources": []},
        },
    }
    output_dir.mkdir(parents=True)
    path = output_dir / "02_submit_p003.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    manifest = {
        "candidate_id": contract["candidate_id"],
        "status": "prepared_not_executed",
        "notebook_sha256": digest(path),
        "source": provenance,
        "input_contract_sha256": digest(INPUTS),
        "guard_sha256": digest(guard_path),
        "generator_sha256": digest(Path(__file__)),
        "source_cells_unchanged": True,
        "own_public_score": None,
        "training_performed": False,
        "uploaded_or_submitted": False,
    }
    (output_dir / "handoff-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir), indent=2))
