"""Build a label-free, unexecuted 64-study P003 Kaggle timing notebook, offline."""

import argparse
import ast
import hashlib
import json
import tempfile
from pathlib import Path

from build_p003_candidate import ROOT, code_cell, digest
from build_p003_candidate import build as build_candidate


def extracted_contract(bootstrap):
    for node in ast.walk(ast.parse(bootstrap)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "P003Guard":
            return ast.literal_eval(node.args[0])
    raise ValueError("Missing parent guard contract")


def profile_root_source(source):
    """Patch only the competition resolver: shadow must not compete with a flat mount."""
    parsed = ast.parse(source)
    node = next(item for item in parsed.body if isinstance(item, ast.FunctionDef) and item.name == "find_root")
    lines = source.splitlines(keepends=True)
    replacement = (
        "def find_root():\n"
        "    return _one_direct('profile competition', [os.environ['RSNA_COMP_ROOT']],\n"
        "                       ['test.csv', 'test_series.csv', 'test_series'])\n"
    )
    result = "".join(lines[: node.lineno - 1]) + replacement + "".join(lines[node.end_lineno :])
    ast.parse(result)
    return result


def build(output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Use a new output directory; no existing experiment is overwritten")
    with tempfile.TemporaryDirectory() as temporary:
        parent_dir = Path(temporary) / "parent"
        parent_manifest = build_candidate(parent_dir)
        notebook = json.loads((parent_dir / "02_submit_p003.ipynb").read_text(encoding="utf-8"))
    cells = notebook["cells"]
    bootstrap = "".join(cells[1]["source"])
    contract = extracted_contract(bootstrap)
    original_cells = dict(contract["source_cells"])
    contract["cell_order"] = [number for number in contract["cell_order"] if number != 29]
    contract["source_cells"] = {key: value for key, value in original_cells.items() if key != "29"}
    contract["profile"] = {"count": 64, "seed": 20261005, "original_source_cells": original_cells}
    runtime_path = ROOT / "scripts/p003_profile.py"
    runtime = runtime_path.read_text(encoding="utf-8")
    ast.parse(runtime)
    for cell in cells:
        if cell["id"] != "p003-source-6":
            continue
        wrapper = ast.parse("".join(cell["source"]))
        call = wrapper.body[-1].value
        source = profile_root_source(ast.literal_eval(call.args[1]))
        contract["source_cells"]["6"] = hashlib.sha256(source.encode()).hexdigest()
        cell["source"] = (
            f"import builtins as _p003_builtins\n_p003_builtins._rsna_p003_guard.run_cell(6, {source!r}, globals())\n"
        ).splitlines(keepends=True)
    # The parent's exact guard and pinned Input contract are embedded, not forked.
    bootstrap_nodes = ast.parse(bootstrap)
    guard = next(
        ast.literal_eval(node.args[0])
        for node in ast.walk(bootstrap_nodes)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "compile"
    )
    profile_bootstrap = (
        "import builtins as _p003_builtins\nimport types as _p003_types\nimport sys as _p003_sys\n"
        "if hasattr(_p003_builtins, '_rsna_p003_guard'):\n    raise RuntimeError('Restart with a fresh session')\n"
        "_p003_module = _p003_types.ModuleType('p003_guard')\n"
        "_p003_sys.modules['p003_guard'] = _p003_module\n"
        f"exec(compile({guard!r}, '<p003-guard>', 'exec'), _p003_module.__dict__)\n"
        "_p003_profile_module = _p003_types.ModuleType('p003_profile')\n"
        "_p003_sys.modules['p003_profile'] = _p003_profile_module\n"
        f"exec(compile({runtime!r}, '<p003-profile>', 'exec'), _p003_profile_module.__dict__)\n"
        f"_p003_guard = _p003_profile_module.P003ProfileGuard({contract!r})\n"
        "_p003_builtins._rsna_p003_guard = _p003_guard\n_p003_guard.prepare()\n"
    )
    cells[1] = code_cell(profile_bootstrap, "p003-profile-preflight")
    input_lines = [f"- [{item['ref']}]({item['url']}) — Version {item['version']}" for item in contract["inputs"]]
    introduction = (
        """# p003 全体時間診断：固定64検査・提出不可

**GPU T4 x2 / Internet OFF** の新しいprivate Notebookで実行してください。
下記の14 Inputsは提出候補p003と同一です。全セルを新しい保存セッションで順番に実行します。
train.csvのUIDだけ、train_series.csvのmetadataだけを使い、seed **20261005** のSHA-256順で
64検査を選択します。Report・所見ラベルを使わず、学習・AUC計算・係数選択を行いません。
公開重みの学習にこれらの画像が含まれる可能性があり、独立OOFではありません。

64件は元V2の48件超の**逐次CoAt分岐**を通ります。元の8時間予算と全モデル構成・数値設定を保ちます。
入力の参照先だけをtrain画像のsymlinkに変更し、最終提出セルを実行しません。
competition解決のfind_rootはshadow一箇所だけを参照する明示的な派生です。
その他の推論セルはp003の元ソースとhash一致を検査します。

完了時の **P003_PROFILE.json** に全体時間・セル別時間・UID/hash・入力監査hashを保存し、
予測は **profile_predictions.csv** に保存します。**submission.csvは生成しません。提出しないでください。**
元の中間診断・receiptも作業フォルダに残ります。停止時は不足モデルを省略せず失敗として扱います。
64件は隠しtest全体のサイズ・分布・メモリを再現しないため、9時間完走やスコア向上を保証しません。

## Inputs

"""
        + "\n".join(input_lines)
        + "\n"
    )
    cells[0] = {
        "cell_type": "markdown",
        "id": "p003-profile-instructions",
        "metadata": {},
        "source": introduction.splitlines(keepends=True),
    }
    notebook["cells"] = [cell for cell in cells if cell["id"] not in {"p003-source-29", "p003-original-28"}]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    output_dir.mkdir(parents=True)
    path = output_dir / "04_profile_p003_64.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    manifest = {
        "status": "prepared_not_executed",
        "candidate_id": contract["candidate_id"],
        "notebook_sha256": digest(path),
        "parent_candidate": parent_manifest,
        "profile_runtime_sha256": digest(runtime_path),
        "profile_generator_sha256": digest(Path(__file__)),
        "profile_config": contract["profile"],
        "executed_source_cells": contract["source_cells"],
        "omitted_source_cells": [29],
        "changed_source_cells": [6],
        "source_change": "find_root resolves only RSNA_COMP_ROOT profile shadow",
        "submission_publication_included": False,
        "labels_used": False,
        "training_performed": False,
        "uploaded_or_submitted": False,
    }
    (output_dir / "profile-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    print(json.dumps(build(arguments.output_dir), indent=2))
