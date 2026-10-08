"""Prepare a label-free 1300-study memory profile with the original fixed head."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from rsna_knee.contracts import sha256, source_hashes

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_NAME = "13_profile_f002_memory_1300.ipynb"
BASE = ROOT / "notebooks/public/10_submit_f002_embedded_head.ipynb"
BASE_SHA256 = "8ca99ca50a5ddd60eef67ce7ec5a1c606d23194686b515e94f481b97680dc8ba"
BASELINE_RUNTIME_SHA256 = "a0d13ea7a93916dc6e7acd52ffe68a9a09e21bbe889d830746f7f9ed46d87f37"
PACKAGE = ROOT / "artifacts/kaggle/f002-memory-profile-v1-20261008"


def code_cell(source):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


def make_notebook():
    if sha256(BASE) != BASE_SHA256:
        raise ValueError("Original prepared embedded-head notebook changed")
    if sha256(ROOT / "scripts/frozen_streaming_runtime.py") != BASELINE_RUNTIME_SHA256:
        raise ValueError("Original streaming runtime changed")
    run = json.loads((ROOT / "artifacts/runs/f002-dinov2-attention-fold0/run.json").read_text(encoding="utf-8"))
    if source_hashes() != run["source_sha256"]:
        raise ValueError("Original sixteen core source files changed")
    book = copy.deepcopy(json.loads(BASE.read_text(encoding="utf-8")))
    book["cells"][0]["source"] = """# f002メモリ方式だけの1300検査診断：提出しない13

このNotebookは教師ラベルを使わず、既存のgold除外weak 4,207検査からseed20261007で1,300検査を選んで推論します。**profile_predictions.csvは提出しません。** 保存版の隠し再実行を許可する提出候補ではありません。

元10と同じCompetition・rsraki/rsna-frozen-feature-inputs-v1・rsraki/rsna-dicom-decoders-py313-v1の3 Inputsだけを接続し、新しいprivate NotebookでT4×2、Internet OFF、Save & Run Allを使います。旧head Datasetは追加しません。再学習・モデル再アップロードは不要です。

変更はnative MRI中間配列の保存場所だけです。全native sliceのglobal p1/p99、物理方向・PixelSpacing・整数化・letterbox・中心±1triplet・series選択・encoder・固定fold0 epoch11 headは維持します。native volumeとpercentile scratchを一時memmapにし、必要sliceだけ同じglobal percentileで正規化します。元16coreファイルとstreaming runtimeを変更せず、readerの一時bindingを終了時に戻します。

F002_STREAMING_SUMMARY.jsonのstatus=profile_complete_not_for_submission、phase=complete、processed=1300、receipt.valid=true、memory_reader.reader_binding_restored=true、temporary_files_remaining=0を確認します。memory-reader-events.jsonlには個別seriesの配列サイズ・disk・資源観測を記録します。

全4,207件や隠しtestの最大MRIサイズを網羅する診断ではありません。file-backed percentile partitionもRAMページを使うため、peak RSS一定や隠し完走を保証しません。一時disk/IOは増えます。不正geometryや壊れたseriesを捨てる処理・代替予測・精度改善は追加していません。
""".splitlines(keepends=True)
    reader = (ROOT / "scripts/bounded_feature_reader.py").read_text(encoding="utf-8")
    adapter = (ROOT / "scripts/bounded_streaming_adapter.py").read_text(encoding="utf-8")
    book["cells"][4:4] = [code_cell(reader), code_cell(adapter)]
    final = "".join(book["cells"][-1]["source"])
    if (
        final.count("MODE = 'test'") != 1
        or final.count("PROFILE_LIMIT = 256") != 1
        or final.count("result = run_streaming(") != 1
    ):
        raise ValueError("Unexpected original execution binding")
    final = final.replace("MODE = 'test'", "MODE = 'profile_train'").replace(
        "PROFILE_LIMIT = 256", "PROFILE_LIMIT = 1300"
    )
    final = final.replace(
        "    result = run_streaming(\n",
        "    result = run_memory_profile(\n        run_streaming, make_bounded_reader,\n",
    )
    final = final.replace(
        "        mode=MODE, weak_manifest=",
        f"        summary_path=SUMMARY_PATH, reader_sha256={sha256(ROOT / 'scripts/bounded_feature_reader.py')!r},\n"
        f"        baseline_runtime_sha256={BASELINE_RUNTIME_SHA256!r}, resource_fn=resource_snapshot,\n"
        "        mode=MODE, weak_manifest=",
    )
    final = final.replace(
        "    saved = DETAILS / 'F002_STREAMING_SUMMARY.json'\n",
        "    saved = SUMMARY_PATH if SUMMARY_PATH.exists() else DETAILS / 'F002_STREAMING_SUMMARY.json'\n",
    )
    book["cells"][-1] = code_cell(final)
    book["metadata"]["rsna_memory_profile"] = {
        "mode": "profile_train",
        "limit": 1300,
        "seed": 20261007,
        "labels_or_report_used": False,
        "for_submission": False,
        "reader_sha256": sha256(ROOT / "scripts/bounded_feature_reader.py"),
        "adapter_sha256": sha256(ROOT / "scripts/bounded_streaming_adapter.py"),
        "baseline_streaming_runtime_sha256": BASELINE_RUNTIME_SHA256,
        "base_notebook_sha256": BASE_SHA256,
        "change": "one factor: native intermediate pixel storage; original strict input contract retained",
    }
    for index, cell in enumerate(book["cells"]):
        if cell["cell_type"] == "code":
            cell["outputs"], cell["execution_count"] = [], None
            compile("".join(cell["source"]), f"{NOTEBOOK_NAME}:{index}", "exec")
    return book


def prepare():
    destination = ROOT / "notebooks/public" / NOTEBOOK_NAME
    if PACKAGE.exists() or destination.exists():
        raise ValueError("Preserve previous artifacts; use a new package/notebook name")
    book = make_notebook()
    serialized = json.dumps(book, ensure_ascii=False, indent=1) + "\n"
    if len(serialized.encode("utf-8")) >= 1_000_000:
        raise ValueError("Generated embedded notebook exceeds the conservative preparation limit")
    PACKAGE.mkdir()
    for path in (destination, PACKAGE / NOTEBOOK_NAME):
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
    package = {
        "created_at": "2026-10-08 JST",
        "status": "prepared_artificial_only_Kaggle_unexecuted",
        "notebook_name": NOTEBOOK_NAME,
        "notebook_sha256": sha256(destination),
        "notebook_bytes": destination.stat().st_size,
        "profile": book["metadata"]["rsna_memory_profile"],
        "required_manual_inputs": book["metadata"]["rsna_manual_inputs"],
        "fixed_delivery": book["metadata"]["rsna_delivery"],
        "core_source_sha256": source_hashes(),
        "builder_sha256": sha256(Path(__file__)),
        "real_MRI_decode_training_upload_submission_performed": False,
        "not_for_competition_submission": True,
        "profile_does_not_cover_all_weak_or_hidden_shapes": True,
    }
    with (PACKAGE / "package.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(package, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps({"prepared": str(destination), "bytes": destination.stat().st_size, "sha256": sha256(destination)})
    )


if __name__ == "__main__":
    prepare()
