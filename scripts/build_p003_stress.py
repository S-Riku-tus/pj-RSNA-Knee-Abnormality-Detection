"""Prepare a larger unchanged P003 schedule for a manual Kaggle diagnostic run."""

import argparse
import ast
import hashlib
import json
import tempfile
from pathlib import Path

from build_p003_candidate import ROOT, code_cell, digest
from build_p003_speed import build as build_speed
from build_p003_speed import embedded_module


def sized_profile_runtime(source, count):
    if source.count("PROFILE_COUNT = 64\n") != 1:
        raise ValueError("Frozen 64-study profile runtime changed")
    result = source.replace("PROFILE_COUNT = 64\n", f"PROFILE_COUNT = {count}\n")
    # These replacements affect messages/receipts only; source inference cells stay pinned.
    for before, after in (("64-study", f"{count}-study"), ("64 studies", f"{count} studies")):
        result = result.replace(before, after)
    ast.parse(result)
    return result


def build(output_dir, count=256):
    if not isinstance(count, int) or not 133 <= count <= 1300:
        raise ValueError("Use 133..1300 studies to cross the 1 GiB DINO cache boundary")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Use a new output directory; old experiments must remain immutable")
    with tempfile.TemporaryDirectory() as temporary:
        parent_dir = Path(temporary) / "parent"
        parent_manifest = build_speed(parent_dir, "profile")
        notebook = json.loads((parent_dir / "05_profile_p003_speed_64.ipynb").read_text(encoding="utf-8"))
    bootstrap = ast.parse("".join(notebook["cells"][1]["source"]))
    call = next(
        node
        for node in ast.walk(bootstrap)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "P003SpeedProfileGuard"
    )
    contract = ast.literal_eval(call.args[0])
    contract["candidate_id"] = f"p003-stress-{count}-v1-20261007"
    contract["profile"]["count"] = count
    contract["speed"].update(
        candidate_id=contract["candidate_id"],
        mode="stress_profile",
        reference=None,
        reference_comparison=None,
        factor="Larger label-free cohort; unchanged interleaved schedule",
    )
    contract["stress"] = {
        "count": count,
        "seed": 20261005,
        "reference_replay": False,
        "diagnostics": "2-second resource samples, child status, full-cache range and SHA256",
        "cache_zero_initialization_unchanged": True,
    }
    names = ("p003_guard", "p003_profile", "p003_speed_runtime", "p003_speed_guard", "p003_stress_runtime")
    sources = {name: (ROOT / "scripts" / (name + ".py")).read_text(encoding="utf-8") for name in names}
    sources["p003_profile"] = sized_profile_runtime(sources["p003_profile"], count)
    bootstrap_source = (
        "import builtins as _p003_builtins\nimport types as _types\nimport sys as _sys\n"
        "if hasattr(_p003_builtins, '_rsna_p003_guard'):\n    raise RuntimeError('Restart with a fresh session')\n"
        + "".join(embedded_module(name, sources[name]) for name in names)
        + "from p003_stress_runtime import P003StressProfileGuard\n"
        + f"_p003_guard = P003StressProfileGuard({contract!r})\n"
        + "_p003_builtins._rsna_p003_guard = _p003_guard\n_p003_guard.prepare()\n"
        + "from p003_speed_runtime import RaptorSchedule\n"
        + "_p003_speed_schedule = RaptorSchedule('/kaggle/working', profile=False)\n"
    )
    notebook["cells"][1] = code_cell(bootstrap_source, "p003-stress-preflight")
    old_intro = "".join(notebook["cells"][0]["source"])
    introduction = (
        f"# p003 規模診断：固定{count}件・提出不可\n\n"
        "T4×2 / Internet OFF の新しいprivate NotebookへImportし、下記の従来14 Inputsだけを指定版で追加します。"
        "Save Version → Save & Run All。今回失敗したf002 head Datasetは不要です。\n\n"
        f"既存05と同じ全モデル・前処理・AMP・microbatch・統合式・Raptor実行順を保ち、"
        f"seed20261005のSHA256順でラベル不要のtrain {count}検査を選びます。64件診断はDINO cacheの"
        "1GiB境界（133件）を通っていません。この版はmemmap経路を確認します。"
        "照合用reference再計算は行わず、64件で確認済みの数値処理を維持します。"
        "追加のfull-cache hash計算があるため、公平な速度ベンチマークではありません。\n\n"
        "完了時は P003_STRESS_SUMMARY.json の status=profile_complete_not_for_submission、"
        "studies、memmap到達、CPU/cgroupメモリ、disk初終・最小空き、各子処理の成否を確認します。"
        "停止時は P003_STRESS_FAILURE.json、P003_STRESS_EVENTS.jsonl、各CoAt logを確認します。"
        "cgroup v1/v2の観測が取得できない場合はmeasurement_status=incompleteとnull/availabilityを記録し、"
        "モデル完走とは別に扱います。観測なしをメモリ安全の証明にはしません。"
        "Report/教師ラベルを使わず、学習・AUC評価・係数変更を行いません。"
        "submission.csvは生成せず、profile_predictions.csvは提出しません。"
        "133件超が原因と確定したわけではなく、隠し完走や自己0.943を保証しません。\n\n"
        "08で使うdecoder Inputは追加しません。decoder導入と規模変更を同時に混ぜず、"
        "前処理エラーが起きた場合はその形式とUIDをログから切り分けます。\n\n## Inputs"
        + old_intro.split("## Inputs", 1)[1]
    )
    notebook["cells"][0]["source"] = introduction.splitlines(keepends=True)
    notebook["cells"][0]["id"] = "p003-stress-instructions"
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    output_dir.mkdir(parents=True)
    filename = f"11_profile_p003_stress_{count}.ipynb"
    path = output_dir / filename
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    manifest = {
        "status": "prepared_not_executed",
        "notebook": filename,
        "notebook_sha256": digest(path),
        "parent": parent_manifest,
        "profile_config": contract["profile"],
        "stress": contract["stress"],
        "executed_source_cells": contract["source_cells"],
        "all_inference_cells_equal_05_profile": True,
        "reference_replay_performed": False,
        "runtime_sha256": {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()},
        "generator_sha256": digest(__file__),
        "own_public_score": None,
        "training_performed": False,
        "uploaded_or_submitted": False,
    }
    (output_dir / "stress-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir, args.count), indent=2))
