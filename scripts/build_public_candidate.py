"""Rebuild the scored p002 source from repository files without executing public code.

The generated notebook is unexecuted; its cell sources match the 2026-10-05
scored handoff. Reproduction requires the separately pinned Kaggle Inputs.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        raise FileExistsError("Use a new handoff directory; existing assets are never overwritten")
    effective_path = ROOT / "docs/research/p002-effective-inputs-20261005.json"
    output_contract_path = ROOT / "docs/research/p002-output-contract-20261005.json"
    effective = json.loads(effective_path.read_text(encoding="utf-8"))
    output_contract = json.loads(output_contract_path.read_text(encoding="utf-8"))
    source_path = ROOT / output_contract["source"]["path"]
    if digest(source_path) != output_contract["source"]["sha256"]:
        raise ValueError("Original V32 source SHA mismatch")
    original = json.loads(source_path.read_text(encoding="utf-8"))
    guard_path = ROOT / "scripts/public_candidate_guard.py"
    guard_source = guard_path.read_text(encoding="utf-8")
    ast.parse(guard_source)
    source_cells = {
        str(i): hashlib.sha256("".join(c["source"]).encode()).hexdigest()
        for i, c in enumerate(original["cells"])
        if c["cell_type"] == "code"
    }
    inputs = []
    for item in effective["effective_inputs"]:
        files = (
            []
            if item["key"] == "competition"
            else [
                {key: file[key] for key in ("path", "bytes", "sha256", "role")}
                for file in item["files"]
                if file["required"]
            ]
        )
        if any(not file["sha256"] or file["bytes"] is None for file in files):
            raise ValueError("All public files must have expected hashes and sizes")
        inputs.append(
            {
                "key": item["key"],
                "ref": item["ref"],
                "version": item["version"],
                "url": item["url"],
                "mounts": [p.removeprefix("/kaggle/input/") for p in item["root_candidates"]],
                "files": files,
            }
        )
    contract = {
        "candidate_id": "p002-dinosaur-v32-current-pins-v1",
        "source_notebook_sha256": digest(source_path),
        "source_cells": source_cells,
        "inputs": inputs,
        "dino_member_ids": [m["id"] for m in effective["dino20_members"]],
        "timm_wheel": {"input": "offline_timm", "path": "timm-1.0.22-py3-none-any.whl", "version": "1.0.22"},
        "receipt_values": {name: value["required_values"] for name, value in output_contract["receipts"].items()},
        "receipt_shapes": {name: value.get("shape_values", {}) for name, value in output_contract["receipts"].items()},
        "check_receipt_inputs": True,
        "strict_input_mounts": True,
    }
    intro = """# p002 DINOsaur V32: manual Kaggle saved run

第1段階で準備した未実行候補です。GPUは **T4 x2**、Internetは **OFF**。
手順書の5 Inputsを指定版で追加してから、Save Version → Save & Run Allを行ってください。
旧p001/r001や旧p002のソース準備版を上書きせず、新しいprivate Notebookを使います。

元V32の10セルの推論ソース・前処理・係数は保持しています。各ソースを同じglobalsで順に実行し、
前後でInput hash、全member完走、fallback、最終CSVを追加検査します。
新しい版組合せであり作者0.937の再現済みモデルではありません。実MRIはここで初めて処理します。

最後の **P002 READY FOR MANUAL SUBMISSION** と **P002_READY.json** を確認し、
`submission.csv`だけを手動Submitしてください。エラー時は提出せずログを確認します。
再実行は新しいセッションで最初から行ってください。途中セルだけの再実行は拒否します。
"""
    cells = [{"cell_type": "markdown", "id": "p002-intro", "metadata": {}, "source": intro.splitlines(keepends=True)}]
    bootstrap = (
        "import builtins as _p002_builtins\nimport types as _p002_types\nimport sys as _p002_sys\n"
        "if hasattr(_p002_builtins, '_rsna_p002_guard'):\n    raise RuntimeError('Restart with a fresh session')\n"
        "_p002_module = _p002_types.ModuleType('rsna_p002_guard')\n"
        "_p002_sys.modules['rsna_p002_guard'] = _p002_module\n"
        f"exec(compile({guard_source!r}, '<p002-guard>', 'exec'), _p002_module.__dict__)\n"
        f"_p002_guard = _p002_module.CandidateGuard({contract!r})\n"
        "_p002_builtins._rsna_p002_guard = _p002_guard\n_p002_guard.prepare()\n"
    )
    cells.append(code_cell(bootstrap, "p002-preflight"))
    for i, cell in enumerate(original["cells"]):
        if cell["cell_type"] != "code":
            cell = dict(cell)
            cell["id"] = f"p002-attribution-{i}"
            cells.append(cell)
            continue
        source = "".join(cell["source"])
        wrapper = (
            f"import builtins as _p002_builtins\n_p002_builtins._rsna_p002_guard.run_cell({i}, {source!r}, globals())\n"
        )
        cells.append(code_cell(wrapper, f"p002-source-{i}"))
    cells.append(
        code_cell(
            "import builtins as _p002_builtins\n_p002_builtins._rsna_p002_guard.finish()\n", "p002-final-contract"
        )
    )
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"isInternetEnabled": False, "isGpuEnabled": True, "dataSources": []},
        },
    }
    output_dir.mkdir(parents=True)
    notebook_path = output_dir / "02_submit_p002.ipynb"
    # The scored handoff was generated on Windows. Keep its bytes on every OS.
    notebook_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\r\n"
    )
    (output_dir / "input-contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "candidate_id": contract["candidate_id"],
        "status": "prepared_for_manual_saved_run_not_executed",
        "historical_author_score_reproduction_claimed": False,
        "original_source": output_contract["source"],
        "notebook": {"path": str(notebook_path), "sha256": digest(notebook_path)},
        "generator_sha256": digest(Path(__file__)),
        "guard_sha256": digest(guard_path),
        "input_contract_sha256": digest(output_dir / "input-contract.json"),
        "effective_input_audit_sha256": digest(effective_path),
        "output_contract_audit_sha256": digest(output_contract_path),
        "original_code_cells": source_cells,
        "number_of_inputs": len(inputs),
        "public_files_with_hash_gates": sum(len(i["files"]) for i in inputs),
        "inference_source_modified": False,
        "added_checks": ["preflight", "per-cell completion observer", "final receipt/CSV check"],
        "real_data_or_model_downloaded": False,
        "real_decode_or_forward_run": False,
        "uploaded_or_submitted": False,
        "own_public_score": None,
    }
    (output_dir / "handoff-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir), ensure_ascii=True, indent=2))
