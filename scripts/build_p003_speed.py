"""Build the isolated Raptor scheduling candidate and its label-free parity profile."""

import argparse
import ast
import difflib
import hashlib
import json
import tempfile
from pathlib import Path

from build_p003_candidate import ROOT, code_cell, digest
from build_p003_candidate import build as build_candidate
from build_p003_profile import build as build_profile


def scheduling_source(source):
    """Change GPU1 traversal only; tensor preparation and forward stay verbatim."""
    old = """    def gpu_one():
        with torch.cuda.device(1):
            run_single(1, torch.device('cuda:1'))
            run_single(3, torch.device('cuda:1'))
"""
    new = """    def gpu_one():
        with torch.cuda.device(1):
            _p003_speed_schedule.run_native_pair(
                globals(), arms, test_ids, series, series_root, outputs, torch.device('cuda:1'))
"""
    anchor = "rsna_phase('public_raptor', 'START')\n_ke_run_raptor_arms()"
    if source.count(old) != 1 or source.count(anchor) != 1:
        raise ValueError("Frozen Raptor scheduling anchors changed")
    result = source.replace(old, new).replace(anchor, "_p003_speed_schedule.install(globals())\n" + anchor)
    ast.parse(result)
    return result


def embedded_module(name, source):
    ast.parse(source)
    return (
        f"_mod = _types.ModuleType({name!r})\n_sys.modules[{name!r}] = _mod\n"
        f"exec(compile({source!r}, '<{name}>', 'exec'), _mod.__dict__)\n"
    )


def build(output_dir, mode="profile"):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Use a new output directory; previous runs are never overwritten")
    if mode not in ("profile", "submission"):
        raise ValueError("mode must be profile or submission")
    profile = mode == "profile"
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary) / "parent"
        parent_manifest = (build_profile if profile else build_candidate)(parent)
        parent_name = "04_profile_p003_64.ipynb" if profile else "02_submit_p003.ipynb"
        notebook = json.loads((parent / parent_name).read_text(encoding="utf-8"))
    cells = notebook["cells"]
    # Both bootstrap variants pass a literal pinned contract to their guard class.
    bootstrap = ast.parse("".join(cells[1]["source"]))
    calls = [
        node
        for node in ast.walk(bootstrap)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("P003Guard", "P003ProfileGuard")
    ]
    if len(calls) != 1:
        raise ValueError("Expected one pinned parent contract")
    contract = ast.literal_eval(calls[0].args[0])
    original = dict(contract.get("profile", {}).get("original_source_cells", contract["source_cells"]))
    target = next(cell for cell in cells if cell["id"] == "p003-source-27")
    call = ast.parse("".join(target["source"])).body[-1].value
    before = ast.literal_eval(call.args[1])
    after = scheduling_source(before)
    contract["source_cells"]["27"] = hashlib.sha256(after.encode()).hexdigest()
    contract["speed"] = {
        "candidate_id": "p003-native-raptor-interleave-v1-20261006",
        "mode": mode,
        "seed": 20261005 if profile else None,
        "factor": "GPU1 visits native384dense-v10 and native384-v8 per study, retaining both models",
        "weights_preprocessing_forward_and_final_blend_unchanged": True,
        "resident_native_models_before": 1,
        "resident_native_models_after": 2,
        "prefetch": "one current plus one future pair; one CPU preparer",
        "original_source_cells": original,
        "reference": "full64 original native arm order with uncached input reads" if profile else None,
        "reference_comparison": "exact input tensor SHA256, exact raw predictions, full64 per-arm ranks",
        "real_mri_parity_and_speedup_verified_at_build": False,
    }
    contract["candidate_id"] = contract["speed"]["candidate_id"]
    target["source"] = (
        f"import builtins as _p003_builtins\n_p003_builtins._rsna_p003_guard.run_cell(27, {after!r}, globals())\n"
    ).splitlines(keepends=True)
    modules = ("p003_guard", "p003_profile", "p003_speed_runtime", "p003_speed_guard")
    runtime_sources = {name: (ROOT / "scripts" / (name + ".py")).read_text(encoding="utf-8") for name in modules}
    bootstrap_source = (
        "import builtins as _p003_builtins\nimport types as _types\nimport sys as _sys\n"
        "if hasattr(_p003_builtins, '_rsna_p003_guard'):\n    raise RuntimeError('Restart with a fresh session')\n"
        + "".join(embedded_module(name, runtime_sources[name]) for name in modules)
        + "from p003_speed_guard import P003SpeedGuard, P003SpeedProfileGuard\n"
        + f"_p003_guard = {'P003SpeedProfileGuard' if profile else 'P003SpeedGuard'}({contract!r})\n"
        + "_p003_builtins._rsna_p003_guard = _p003_guard\n_p003_guard.prepare()\n"
        + "from p003_speed_runtime import RaptorSchedule\n"
        + f"_p003_speed_schedule = RaptorSchedule('/kaggle/working', profile={profile!r})\n"
    )
    cells[1] = code_cell(bootstrap_source, "p003-speed-preflight")
    instructions = (
        "# p003 Raptor実行順の改善候補："
        + ("64件の精度差・時間診断（提出不可）" if profile else "提出候補（未採点）")
        + "\n\n元02の予測式・全モデル・AMP・microbatchは維持し、GPU1のnative Raptor 2系統を"
        "検査ごとに交互実行してDICOM decode cacheを再利用します。モデルは2個同時に常駐するため"
        "GPUメモリ増加を記録します。作者0.943は当方の実測値ではありません。"
        "9時間以内の完走・高速化率・同等精度は未検証です。元の8時間内部予算も維持します。\n\n"
        + (
            "先にこの診断をT4 x2・Internet OFFで実行してください。元04と同じseed/64 UIDを使い、"
            "変更した2系統について元の検査順でも再計算し、入力tensor hash・生予測完全一致・順位差を"
            "検査します。追加のreference実行は時間測定に含まれるため、総時間を旧04と直接比較しないでください。"
            "referenceはprefetchなしの数値検証で、速度比較の基準ではありません。"
            "P003_SPEED_SUMMARY.json と P003_SPEED_RAPTOR.json の parity_passed=true を確認します。"
            "submission.csvは生成しません。\n\n"
            if profile
            else "先に05_profile_p003_speed_64.ipynbのparity_passed=true・メモリ・時間内訳を確認してください。"
            "診断結果が未確認のまま性能改善済みとして扱わないでください。"
            "元02とは別の新しいprivate NotebookでT4 x2・Internet OFFを指定して保存実行します。"
            "全セル完了と P003_READY.json / P003_SPEED_SUMMARY.json を確認してからsubmission.csvを手動提出します。\n\n"
        )
        + "P003_SPEED_PHASES.json はRaptorとCoAt各系統の開始・終了を逐次保存します。"
        "失敗時は P003_SPEED_FAILURE.json のtracebackと子処理logを確認します。"
        "一般的なUnhandled errorだけで9時間超過とは判断しません。モデル欠落・fallbackの拒否は維持します。"
        "\n\n## Inputs\n" + "".join(cells[0]["source"]).split("## Inputs", 1)[1]
    )
    cells[0]["source"] = instructions.splitlines(keepends=True)
    cells[0]["id"] = "p003-speed-instructions"
    output_dir.mkdir(parents=True)
    filename = "05_profile_p003_speed_64.ipynb" if profile else "05_submit_p003_speed.ipynb"
    path = output_dir / filename
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    diff = "".join(
        difflib.unified_diff(before.splitlines(True), after.splitlines(True), "frozen-cell27", "speed-cell27")
    )
    (output_dir / "cell27-scheduling.diff").write_text(diff, encoding="utf-8", newline="\n")
    manifest = {
        "status": "prepared_not_executed",
        "notebook": filename,
        "notebook_sha256": digest(path),
        "parent": parent_manifest,
        "speed": contract["speed"],
        "executed_source_cells": contract["source_cells"],
        "runtime_sha256": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in runtime_sources.items()},
        "generator_sha256": digest(__file__),
        "diff_sha256": digest(output_dir / "cell27-scheduling.diff"),
        "own_public_score": None,
        "training_performed": False,
        "uploaded_or_submitted": False,
    }
    (output_dir / "speed-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("profile", "submission"), default="profile")
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir, args.mode), indent=2))
