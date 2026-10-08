"""Prepare an unchanged scored public control; never execute, download or publish it."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = (
    ROOT / "artifacts/research/20261008-public-candidates/goodpjw2008-rsna-knee-stack-2-5d-convnext-mil-lb-0-944-v2"
)
OUTPUT_DIR = ROOT / "artifacts/kaggle/p004-public944-control-v3-20261008"
NOTEBOOK_PATH = ROOT / "notebooks/public/12_submit_p004_public944_control.ipynb"
REF = "goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944"
SCRIPT_VERSION = 355300588
SUBMISSION_ID = 56837602
VERSION = 2
SOURCE_SHA256 = "582f4a1c768cb4b347483797cfff4dc2a3e7648f0bb603d8e878eeb9ccc2695f"
DOCKER_DIGEST = "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
CONTAINER_ID = 31430
ORIGINAL_URL = f"https://www.kaggle.com/code/{REF}?scriptVersionId={SCRIPT_VERSION}"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def code_sources(book):
    return [(i, "".join(cell["source"])) for i, cell in enumerate(book["cells"]) if cell["cell_type"] == "code"]


def syntax_check(book):
    """Compile Python and bodies of the original writefile magics, without running them."""
    results = []
    for index, source in code_sources(book):
        kind = "python"
        if source.startswith("%%writefile "):
            first, source = source.split("\n", 1)
            if first not in {
                "%%writefile /kaggle/working/own_src/preprocess.py",
                "%%writefile /kaggle/working/own_src/knee.py",
                "%%writefile /kaggle/working/own_src/infer.py",
            }:
                raise ValueError("Unexpected writefile magic destination")
            kind = "ipython_writefile_python_body"
        elif source.startswith("%%"):
            raise ValueError("Unreviewed IPython cell magic")
        compile(source, f"p004_original_cell_{index}", "exec")
        results.append({"cell": index, "syntax": kind, "compiled_without_execution": True})
    return results


def verified_evidence(source_dir=SOURCE_DIR):
    source_dir = Path(source_dir)
    metadata = read_json(source_dir / "metadata.json")
    review = read_json(source_dir / "review.json")
    view = read_json(source_dir / "view-model.json")
    if digest(source_dir / "source.ipynb") != SOURCE_SHA256 or review["source_sha256"] != SOURCE_SHA256:
        raise ValueError("Public scored source SHA256 mismatch")
    if metadata["ref"] != REF or metadata["currentVersionNumber"] != VERSION:
        raise ValueError("Wrong public reference/version")
    expected_submission = {"id": SUBMISSION_ID, "sourceScriptVersionId": SCRIPT_VERSION, "scoreFormatted": "0.944"}
    if review["submission"] != expected_submission or view["submission"] != expected_submission:
        raise ValueError("Submission is not the verified author0.944 scored V2")
    for record in (review["best_submission_score"], view["bestSubmissionScore"]):
        if record["scoreFormatted"] != "0.944" or record["kernelVersionNumber"] != VERSION:
            raise ValueError("Best score is attached to a different version")
    run = view["kernelRun"]
    if (review["script_version_id"], run["id"], run["kernelVersionNumber"]) != (
        SCRIPT_VERSION,
        SCRIPT_VERSION,
        VERSION,
    ):
        raise ValueError("Saved source/session is not the scored V2")
    if run["dockerImageVersionId"] != CONTAINER_ID or run["runInfo"]["dockerImageDigest"] != DOCKER_DIGEST:
        raise ValueError("Scored original container differs")
    if not metadata["dockerImage"].endswith("@sha256:" + DOCKER_DIGEST):
        raise ValueError("Metadata container digest differs")
    if (metadata["enableGpu"], metadata["enableInternet"], run["status"]) != (True, False, "COMPLETE"):
        raise ValueError("Scored GPU/Internet/run state differs")
    if len(review["inputs"]) != 19:
        raise ValueError("Expected exact scored19 Input graph")
    notebook = read_json(source_dir / "source.ipynb")
    if notebook["metadata"]["language_info"]["version"] != "3.12":
        raise ValueError("Expected scored Python3.12 Notebook metadata")
    return notebook, metadata, review, view


PREFACE = f"""このファイルは公開作者の採点済み Version 2 を固定したp004対照です。
作者goodpjw2008のPublic0.944は公式submission {SUBMISSION_ID}/script {SCRIPT_VERSION}で確認済みですが、私たちの再現値ではありません。
元作者の全29 code cellは変更せず、保存Outputと実行番号だけを消しています。以下の『ours』『we』は元作者を指します。

Kaggleでは下の採点済みURLのVersion2を開き、Copy & EditでOriginal Environmentと同版19 Inputsを保つ手順を使います。
このローカルファイルを最新環境へImportするだけではOriginal Containerは固定されません。
元Container31430/Python3.12、T4 GPU、Internet OFFを維持します。実行手順は配布packageのmanual-notes.txtを参照してください。
作者原版：{ORIGINAL_URL}
Copy & Edit後の保存版は、提出前に全29 code cell・Original環境・19 Input版がこのVersion2と一致するか照合します。
Version2ページからのコピーでも最新Version5を自動取得しないと保証はできないため、保存版URLを共有して確認してから提出してください。

自己の9時間以内の隠し完走・Public・追加精度改善は未測定。公開stackのgold使用履歴は独自学習・checkpoint選択と区別し、元の出典表記を維持します。

---

"""


def prepare_notebook(original):
    prepared = copy.deepcopy(original)
    if prepared["cells"][0]["cell_type"] != "markdown":
        raise ValueError("Expected original leading attribution markdown")
    prepared["cells"][0]["source"] = (PREFACE + "".join(original["cells"][0]["source"])).splitlines(keepends=True)
    for cell in prepared["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    if code_sources(prepared) != code_sources(original):
        raise AssertionError("Original code bytes/order changed")
    return prepared


def manual_inputs(metadata, review):
    entries = []
    for original in review["inputs"]:
        item = dict(original)
        mount = item["mountSlug"]
        if mount.startswith("datasets/"):
            item["kind"], item["ref"] = "dataset", mount[len("datasets/") :]
            item["version_identifier"] = f"dataset_version_id:{item['sourceId']}"
        elif mount.startswith("notebooks/"):
            item["kind"], item["ref"] = "notebook_output", mount[len("notebooks/") :]
            item["version_identifier"] = f"script_version_id:{item['sourceId']}"
        elif mount.startswith("models/"):
            item["kind"] = "model"
            refs = [ref for ref in metadata["modelDataSources"] if ref.lower() == mount[len("models/") :].lower()]
            if len(refs) != 1:
                raise ValueError("Model ref does not match scored graph")
            item["ref"] = refs[0]
            item["version_identifier"] = f"model_instance_version_id:{item['sourceId']}"
        elif mount.startswith("competitions/"):
            item["kind"], item["ref"] = "competition", mount[len("competitions/") :]
            item["version_identifier"] = f"databundle_version_id:{item['databundleVersionId']}"
        else:
            raise ValueError("Unsupported scored Input mount")
        entries.append(item)
    for kind, field in (
        ("dataset", "datasetDataSources"),
        ("notebook_output", "kernelDataSources"),
        ("model", "modelDataSources"),
        ("competition", "competitionDataSources"),
    ):
        if {item["ref"] for item in entries if item["kind"] == kind} != set(metadata[field]):
            raise ValueError("Scored metadata/Input references disagree")
    return entries


def manual_notes(inputs):
    lines = [
        "p004公開0.944採点済みVersion2の固定対照 — 手動操作 2026-10-08 JST",
        "",
        "この作業は元作者の採点済み予測パイプラインを、推論コードを変えずに自分のアカウントで実測する対照です。",
        "0.944は作者の実測Publicです。自分の採点成功・同じPublic・9時間以内の完走は未確認です。p002の自己0.937は保持します。",
        "最新Version5やタイトルだけで選ばず、次のVersion2/script355300588を開いてください。",
        ORIGINAL_URL,
        "",
        "1. 上の保存済みVersion2ページでVersion表示を確認し、Copy & Editを選びます。最新Versionへ移動しません。",
        "2. 新しい自分のNotebookをprivateにし、SettingsのEnvironmentでOriginal/元環境を保ちます。Latestへの変更はしません。",
        "   元containerID31430 / Python3.12 / digest " + DOCKER_DIGEST,
        "   Kaggle画面がOriginalを選べず、この環境を保持できない場合は、その画面/状態を記録して実行前に報告してください。",
        "   ローカル12.ipynbを新規NotebookへImportするだけでは、Input版やcontainerは固定されません。",
        "3. AcceleratorはGPU T4 x2、Internet OFF。元Inputs19個の同じ選択版を保ち、Update allやDataset最新化を押しません。",
        "   過去のf002 head Dataset・f002 decoder Datasetは、この公開対照に追加しません。",
        "   Copy & Editが最新V5のcodeを取得した可能性を消すには、そのコピー内へ本配布12.ipynbをImportして原版V2 codeを固定します。",
        "   ただしImport後にInput選択版/Environmentが保持されるGUI動作も保証できないため、必ず19 InputsとOriginal設定を再確認します。",
        "   確認できないままGPUを開始したりLatestへ切り替えたりせず、設定状態を記録して報告します。",
        "4. 推論code cell、係数、系列のfallback、モデル構成を編集せず、Save Version → Save & Run Allします。",
        "   本配布の12は原版全29 code cellのローカル照合用コピーです。Copy & Editが最新版を取得しないというGUI動作は独立未検証です。",
        "5. 保存版が正常完了してsubmission.csvが出たら、提出前にその保存版URL・Versionを共有して照合します。",
        "   全29 code cell・19 Input内部版ID・Original Container31430/digestが原作者採点Version2と一致することを確認します。",
        "   この照合後、その同じ保存Versionのsubmission.csvをコンペページから手動提出します。最新版V5の混入を見つけた場合は提出しません。",
        "   ログ末尾で『own reader: blended 3 checkpoints at weight 0.3』と『wrote /kaggle/working/_own.csv』を確認します。",
        "   『own reader FAILED, the stack's submission is kept』がある場合、readerが外れて元stackへ戻っているため、この0.944固定対照の成功とは記録せず提出を保留します。",
        "   これは追加の承認ではなく、提出する保存版が固定対照と一致するかという品質確認です。",
        "   提出Notebook識別子・保存Version・scriptVersionId・提出結果を記録して、この固定対照の成否を確認します。",
        "   私たちのローカル人工テストや作者の公開3件317.135秒は、隠し1300件の完走保証ではありません。",
        "",
        "元作者V2の19 Inputs（内部version IDはKaggle APIのIDで、画面に出るVersion番号とは別です）：",
    ]
    for i, item in enumerate(inputs, 1):
        lines.append(
            f"{i:02d}. {item['kind']} {item['ref']} | {item['version_identifier']} | /kaggle/input/{item['mountSlug']}"
        )
    lines.extend(
        [
            "",
            "原版InputsとOriginal Environmentを一緒に残すのが、この対照の条件です。Notebook JSONへ独自docker項目を追加していません。",
            "帰属：元Notebook/codeはgoodpjw2008と元Notebook記載の公開stack各作者。全ての原文出典表記を残します。",
            "reader DatasetはApache2.0と公式metadataで確認。Notebook全体の独立ライセンスは未確認で、readerのライセンスを全体へ転記していません。",
            "公開stackには既存のgold選択履歴があります。元作者readerのgold非学習は作者説明で、独立監査済みという意味ではありません。",
            "この公開固定対照のPublicと、goldを独自学習/選択に使わないf002/f003のweak検証は分けて記録します。",
            "外部アップロード・Input再配布・学習・提出を自動実行するコードは追加していません。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def audit_author_reader_log(source_dir):
    log = Path(source_dir) / "visible.log"
    entries = read_json(log)
    stdout = "".join(item.get("data", "") for item in entries if item.get("stream_name") == "stdout")
    evidence = {
        "log_sha256": digest(log),
        "reader_csv_written": "wrote /kaggle/working/_own.csv" in stdout,
        "reader_blended3_at_weight030": "own reader: blended 3 checkpoints at weight 0.3" in stdout,
        "reader_failed_fallback_message_absent": "own reader FAILED, the stack's submission is kept" not in stdout,
        "scope": "Original author visible3 test study execution; does not expose hidden runtime or our own reproduction.",
    }
    if not all(
        evidence[name]
        for name in ("reader_csv_written", "reader_blended3_at_weight030", "reader_failed_fallback_message_absent")
    ):
        raise ValueError("Author visible reader completion evidence differs")
    return evidence


def prepare_package(source_dir=SOURCE_DIR, output_dir=OUTPUT_DIR, notebook_path=NOTEBOOK_PATH):
    source_dir, output_dir, notebook_path = map(Path, (source_dir, output_dir, notebook_path))
    if output_dir.exists() or notebook_path.exists():
        raise FileExistsError("Use new package and Notebook paths; existing artifacts are never overwritten")
    original, metadata, review, view = verified_evidence(source_dir)
    prepared = prepare_notebook(original)
    compiled = syntax_check(prepared)
    inputs = manual_inputs(metadata, review)
    reader_log = audit_author_reader_log(source_dir)
    output_dir.mkdir(parents=True)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(notebook_path, prepared)
    copy_name = notebook_path.name
    shutil.copyfile(notebook_path, output_dir / copy_name)
    for name in ("source.ipynb", "metadata.json", "review.json", "visible.log"):
        shutil.copyfile(source_dir / name, output_dir / ("original-" + name))
    input_manifest = {
        "schema_version": "p004_manual_scored_input_graph_v1",
        "author_ref": REF,
        "scored_version": VERSION,
        "scored_script_version_id": SCRIPT_VERSION,
        "inputs": inputs,
        "automatic_upload_or_submission": False,
        "note": "Exact scored internal version IDs; retain via Copy & Edit of scored V2. These are not fabricated Kaggle SDK metadata fields.",
    }
    write_json(output_dir / "inputs-manual.json", input_manifest)
    provenance = {
        "schema_version": "p004_public_scored_control_v1",
        "prepared_at_jst": "2026-10-08",
        "source": {
            "ref": REF,
            "url": ORIGINAL_URL,
            "version": VERSION,
            "script_version_id": SCRIPT_VERSION,
            "sha256": SOURCE_SHA256,
        },
        "author_scored_submission": review["submission"],
        "own_public_lb": None,
        "own_visible_or_hidden_execution": False,
        "model_weights_downloaded": False,
        "real_mri_decoded": False,
        "training_performed": False,
        "upload_or_submission_performed": False,
        "source_best_score_binding": {
            "api_review_sha256": digest(source_dir / "review.json"),
            "api_view_sha256": digest(source_dir / "view-model.json"),
            "best_score": view["bestSubmissionScore"],
            "kernel_run_id": view["kernelRun"]["id"],
            "kernel_run_version": view["kernelRun"]["kernelVersionNumber"],
            "matching_code_cells": len(code_sources(prepared)),
        },
        "author_visible_reader_completion": reader_log,
        "environment_observed": {
            "original_docker_image_id": CONTAINER_ID,
            "docker_image_version_id": view["kernelRun"]["dockerImageVersionId"],
            "docker_image_digest": DOCKER_DIGEST,
            "metadata_docker_image": metadata["dockerImage"],
            "run_image_name": view["kernelRun"]["runInfo"]["dockerImageName"],
            "pinning_type": view["kernel"]["dockerImagePinningType"],
            "python_notebook_metadata": "3.12",
            "accelerator": metadata["machineShape"],
            "enable_internet": metadata["enableInternet"],
            "how_to_preserve": "Copy & Edit the scored V2, keep Original Environment and exact19 input snapshots; do not Import into Latest.",
        },
        "code_cell_sha256": {
            str(index): hashlib.sha256(source.encode("utf-8")).hexdigest() for index, source in code_sources(prepared)
        },
        "syntax_checks": compiled,
        "changes": [
            "Japanese provenance preface in original leading markdown",
            "clear cell outputs and execution_count",
        ],
        "unchanged": [
            "all29codecells byte-for-byte",
            "code cell order",
            "model composition",
            "blend coefficients",
            "prediction/series fallback",
            "original attribution markdown",
        ],
        "license_scope": {
            "reader_dataset": "Apache2.0 metadata verified",
            "whole_notebook": "independently unconfirmed; original attribution preserved; not relabeled Apache",
        },
        "gold_scope": "Existing public stack gold selection exposure remains separately audited; reader gold-not-fit statement is author claim, not independent proof. No independent training/head selection is performed.",
        "runtime_scope": {
            "author_visible_seconds": review["visible_runtime"],
            "hidden_seconds": None,
            "own_completion": False,
            "nine_hour_success_guarantee": False,
        },
    }
    write_json(output_dir / "provenance.json", provenance)
    with (output_dir / "manual-notes.txt").open("x", encoding="utf-8") as stream:
        stream.write(manual_notes(inputs))
    files = {
        path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    package = {
        "schema_version": "p004_public_control_package_v1",
        "files": files,
        "prepared_notebook_sha256": digest(notebook_path),
        "source_notebook_sha256": SOURCE_SHA256,
        "generator_sha256": digest(__file__),
        "code_cell_count": len(code_sources(prepared)),
        "input_count": len(inputs),
        "local_only": True,
    }
    write_json(output_dir / "package.json", package)
    return package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--notebook-path", type=Path, default=NOTEBOOK_PATH)
    args = parser.parse_args()
    result = prepare_package(output_dir=args.output_dir, notebook_path=args.notebook_path)
    print(
        json.dumps(
            {
                "code_cells": result["code_cell_count"],
                "inputs": result["input_count"],
                "sha256": result["prepared_notebook_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
