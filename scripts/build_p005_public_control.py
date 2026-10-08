"""Stage the scored public Apex control without running its inference or publishing it."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "artifacts/research/20261008-public-candidates"
SOURCE_DIR = RESEARCH / "sujanmajhisuzan-rsna-knee-apex-grandmaster-stack-v1"
VIEW_PATH = RESEARCH / "rsna-knee-apex-grandmaster-stack-GetKernelViewModel.json"
OUTPUT_DIR = ROOT / "artifacts/kaggle/p005-public950-control-v1-20261008"
NOTEBOOK_PATH = ROOT / "notebooks/public/14_submit_p005_public950_control.ipynb"
ELIGIBILITY_PATH = ROOT / "experiments/public-pretrained-eligibility-20261008.json"
REF = "sujanmajhisuzan/rsna-knee-apex-grandmaster-stack"
SOURCE_SHA256 = "c3b62e2f8be016abe541e9d5e537217fec73edce41f6fef15d95527c2dbfc75a"
SCRIPT_VERSION = 356192954
SUBMISSION_ID = 56930357
CONTAINER_ID = 31481
DOCKER_DIGEST = "2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"
ORIGINAL_URL = f"https://www.kaggle.com/code/{REF}?scriptVersionId={SCRIPT_VERSION}"
SPEC = importlib.util.spec_from_file_location("p004_control_tools", ROOT / "scripts/build_p004_public_control.py")
control_tools = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control_tools)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def verified_eligibility(path=ELIGIBILITY_PATH):
    path = Path(path)
    if not path.is_file():
        raise ValueError("The completed public-pretrained eligibility audit is required before handoff generation")
    eligibility = read_json(path)
    required = {
        "adoption_decision": "conditional_proceed_public_frozen_checkpoint_inference",
        "adopted_notebook_ref": REF,
        "adopted_notebook_version": 1,
        "adopted_script_version_id": SCRIPT_VERSION,
        "adopted_source_sha256": SOURCE_SHA256,
        "adopted_author_scored_submission_id": SUBMISSION_ID,
    }
    if not isinstance(eligibility, dict) or any(eligibility.get(key) != value for key, value in required.items()):
        raise ValueError("The completed eligibility decision does not adopt this fixed scored source")
    if not eligibility.get("adoption_conditions"):
        raise ValueError("Record the conditional adoption scope before handoff generation")
    return eligibility


def verified_evidence(source_dir=SOURCE_DIR, view_path=VIEW_PATH):
    source_dir = Path(source_dir)
    notebook = read_json(source_dir / "source.ipynb")
    metadata = read_json(source_dir / "metadata.json")
    view = read_json(view_path)
    if digest(source_dir / "source.ipynb") != SOURCE_SHA256:
        raise ValueError("Apex public scored source SHA256 mismatch")
    if metadata["ref"] != REF or metadata["currentVersionNumber"] != 1:
        raise ValueError("Wrong Apex public reference/version")
    if view["submission"] != {
        "id": SUBMISSION_ID,
        "sourceScriptVersionId": SCRIPT_VERSION,
        "scoreFormatted": "0.950",
    }:
        raise ValueError("Author0.950 submission is attached to another source/version")
    run = view["kernelRun"]
    if (run["id"], run["kernelVersionNumber"], run["status"]) != (SCRIPT_VERSION, 1, "COMPLETE"):
        raise ValueError("Apex scored saved run differs")
    best = view["bestSubmissionScore"]
    if (best["scoreFormatted"], best["kernelVersionNumber"]) != ("0.950", 1):
        raise ValueError("Apex best score/version differs")
    if (run["dockerImageVersionId"], run["runInfo"]["dockerImageDigest"]) != (CONTAINER_ID, DOCKER_DIGEST):
        raise ValueError("Apex original container differs")
    if not metadata["dockerImage"].endswith("@sha256:" + DOCKER_DIGEST):
        raise ValueError("Apex metadata container digest differs")
    if metadata["enableInternet"] or not metadata["enableGpu"]:
        raise ValueError("Apex scored Internet/GPU setting differs")
    if len(view["dataSources"]) != 3 or any(item.get("isPrivate") for item in view["renderableDataSources"]):
        raise ValueError("Apex public3Input graph differs or contains private inputs")
    expanded_metadata = {
        **metadata,
        "modelDataSources": metadata.get("modelDataSources", []),
        "kernelDataSources": metadata.get("kernelDataSources", []),
    }
    inputs = control_tools.manual_inputs(expanded_metadata, {"inputs": view["dataSources"]})
    if [item["sourceId"] for item in inputs] != [154281, 20325538, 20440357]:
        raise ValueError("Apex exact input snapshots differ")
    entries = read_json(source_dir / "visible.log")
    stdout = "".join(item.get("data", "") for item in entries if item.get("stream_name") == "stdout")
    all_streams = "".join(item.get("data", "") for item in entries)
    log_evidence = {
        "visible_log_sha256": digest(source_dir / "visible.log"),
        "python313_visible_log_path": "/python3.13/" in all_streams,
        "found3_reader_checkpoints": "[ConvNeXt Reader] Found 3 checkpoints" in stdout,
        "reader_csv_written": "wrote /kaggle/working/_own.csv" in stdout,
        "fusion_success": "[Apex Fusion SUCCESS] Successfully fused" in stdout,
        "fusion_exception_or_fallback_absent": "[Apex Fusion EXCEPTION]" not in stdout
        and "[Apex Fusion FALLBACK]" not in stdout,
        "libjpeg_wheel_install_failure_seen": "No matching distribution found for pylibjpeg-libjpeg" in all_streams,
        "decoder_scope": "CP313 cannot install the attached CP311/312 libjpeg wheel. The original check=False installation and series-level decode exception/mask behavior remain unchanged. Visible3 and author score do not quantify compression-specific skipped data or prove all compressed MRI was decoded.",
    }
    if not all(
        log_evidence[name]
        for name in (
            "python313_visible_log_path",
            "found3_reader_checkpoints",
            "reader_csv_written",
            "fusion_success",
            "fusion_exception_or_fallback_absent",
        )
    ):
        raise ValueError("Apex original visible completion evidence differs")
    return notebook, metadata, view, inputs, log_evidence


PREFACE = f"""p005公開0.950固定対照 — 公開学習済みcheckpointの凍結推論として条件付き採用。
利用根拠はexperiments/public-pretrained-eligibility-20261008.jsonです。公式Rulesの公開事前学習済みモデル条項とSWAcardのRSNAcompetition利用条件を区別して監査しました。
条件は公開weights/Input版の固定、帰属・原terms保持、追加gold調整なし、自分によるrestricted raw OAI取得・再学習なしです。Host個別承認や元作者のDUA履行を保証する判断ではありません。
作者sujanmajhisuzanのVersion1/script{SCRIPT_VERSION}、submission{SUBMISSION_ID}のPublic0.950を公式表示で確認しました。私たちの再現値ではありません。
元作者の全7 code cell・所見別係数・モデル構成・fallbackを変更せず、保存Outputと実行番号だけを消しています。以下の原文は作者の記述です。
採点済み原版：{ORIGINAL_URL}

Original Container31481/digest{DOCKER_DIGEST}、T4 GPU、Internet OFF、同版3 Inputsを記録しています。
可視ログはPython3.13で、libjpeg wheel導入失敗後も元コードのcheck=False/series maskで続行していました。decoder失敗を隠したり補修したりしていません。
最終Fusion成功とreader3 checkpointsの実行は作者ログで確認できますが、圧縮MRI全件の正常デコード・自分の隠し完走・同じ精度は保証しません。
Copy & Edit/Import後の保存版code・Input版・Original環境を独立照合してから、その同じ保存版を手動提出します。
OAI由来の公開学習済みcheckpointを使うため、controlled raw OAIへの新しいアクセス・元MRI取得はこの準備に含めません。gold選択履歴は独自学習と分けて記録します。

---

"""


def prepare_notebook(original):
    prepared = copy.deepcopy(original)
    if prepared["cells"][0]["cell_type"] != "markdown":
        raise ValueError("Expected original leading acknowledgement markdown")
    prepared["cells"][0]["source"] = (PREFACE + "".join(original["cells"][0]["source"])).splitlines(keepends=True)
    for cell in prepared["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"], cell["execution_count"] = [], None
    if control_tools.code_sources(prepared) != control_tools.code_sources(original):
        raise AssertionError("Apex original code bytes/order changed")
    return prepared


def manual_notes(inputs):
    lines = [
        "p005作者Public0.950採点Version1固定対照 — 公開checkpoint凍結推論として条件付き採用 2026-10-08 JST",
        "",
        "利用根拠：experiments/public-pretrained-eligibility-20261008.json。公開モデル条項とSWAcardのRSNAcompetition許可に基づく条件付き採用です。",
        "公開weights/Input版固定・帰属/terms保持・追加gold調整なし・自分によるOAI raw取得/再学習なしの範囲で進めます。Host個別承認や元作者DUA履行保証ではありません。",
        "既存12は別の公開固定対照として保持します。14の自分の採点が成功したという意味ではありません。",
        "自分のPublic/隠し完走は未測定です。作者の0.950は公式submission56930357/script356192954に結び付いた値です。",
        "",
        "1. 次の原作者採点済みVersion1を開き、Copy & Editでprivateコピーを作ります。",
        ORIGINAL_URL,
        "2. Original Environmentを保ち、Container31481/digest " + DOCKER_DIGEST + "を再確認します。",
        "   GPU T4x2・Internet OFF・元3 Inputsの選択版を維持します。Latestへの変更やUpdate allはしません。",
        "3. コピー元Version取り違え防止の候補として、そのコピー内へ14.ipynbをImportして全7codecellを固定できます。",
        "   ImportがInputs/envを保つGUI動作は未検証なので、必ず再確認します。Originalを保持できない場合はworker開始前に止めて状態を記録します。",
        "4. 推論/所見別係数/fallbackを編集せずSave Version→Save & Run Allします。",
        "   原作者visible.logでもCP313用libjpeg wheelが無くpip導入に失敗しています。原版挙動を保ち、別decoder追加で対照条件を変えません。",
        "5. saved run完了後、ログの『[ConvNeXt Reader] Found 3 checkpoints』『wrote /kaggle/working/_own.csv』『[Apex Fusion SUCCESS] Successfully fused』を確認します。",
        "   Apex Fusion EXCEPTION/FALLBACKがある場合は融合が外れて純0.949系へ戻っているため、この0.950固定対照の成功とは扱いません。",
        "6. 保存版URL/Versionを共有し、7codecell・3Input内部versionID・Originalcontainerの独立照合後に、その同保存版のsubmission.csvを手動提出します。",
        "   これは追加承認ではなく、実際の提出版が固定対照に一致するかの品質確認です。",
        "",
        "固定Input graph（内部versionIDは画面のVersion番号とは別）：",
    ]
    lines += [
        f"{i:02d}. {item['kind']} {item['ref']} | {item['version_identifier']} | /kaggle/input/{item['mountSlug']}"
        for i, item in enumerate(inputs, 1)
    ]
    lines += [
        "",
        "元sourceにはNartaa、DreadDevelopment、ConvNeXt reader作者の帰属およびOAI/NDA acknowledgementを残しています。",
        "公開sourceと学習済みweightsの利用条件確認を、restricted raw OAIへのアクセス許可と同一視しません。",
        "Gold58選択/開発は元公開モデルの履歴で、独立検証や自分のgold非利用実験とは区別します。",
        "Notebook全体の独立licenseは未確認。reader DatasetのApache2.0を全体へ転記していません。",
        "新checkpoint取得、実MRIdecode、学習、外部upload、submitはこの準備で実行していません。",
    ]
    return "\n".join(lines) + "\n"


def prepare_package(source_dir=SOURCE_DIR, view_path=VIEW_PATH, output_dir=OUTPUT_DIR, notebook_path=NOTEBOOK_PATH):
    source_dir, view_path, output_dir, notebook_path = map(Path, (source_dir, view_path, output_dir, notebook_path))
    if output_dir.exists() or notebook_path.exists():
        raise FileExistsError("Use fresh package and Notebook paths; existing outputs are never overwritten")
    eligibility = verified_eligibility(ELIGIBILITY_PATH)
    original, metadata, view, inputs, log_evidence = verified_evidence(source_dir, view_path)
    if eligibility.get("adopted_public_inputs") != view["dataSources"]:
        raise ValueError("The eligibility audit does not adopt the exact scored3Input snapshots")
    prepared = prepare_notebook(original)
    syntax = control_tools.syntax_check(prepared)
    output_dir.mkdir(parents=True)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(notebook_path, prepared)
    shutil.copyfile(notebook_path, output_dir / notebook_path.name)
    for name in ("source.ipynb", "metadata.json", "visible.log"):
        shutil.copyfile(source_dir / name, output_dir / ("original-" + name))
    shutil.copyfile(ELIGIBILITY_PATH, output_dir / "public-pretrained-eligibility.json")
    write_json(
        output_dir / "inputs-manual.json",
        {"schema_version": "p005_manual_scored_input_graph_v1", "inputs": inputs, "private_inputs": []},
    )
    provenance = {
        "schema_version": "p005_public950_control_v1",
        "prepared_at_jst": "2026-10-08",
        "eligibility_status": "conditionally_adopt_public_checkpoint_frozen_inference",
        "eligibility_audit": {"path": str(ELIGIBILITY_PATH.relative_to(ROOT)), "sha256": digest(ELIGIBILITY_PATH)},
        "adoption_conditions": eligibility["adoption_conditions"],
        "adoption_scope": "Fixed public checkpoints and original inference only; preserve attribution/terms, no new gold tuning, no raw controlled OAI access or retraining. Not individual Host approval or a warranty of origin DUA compliance.",
        "source": {
            "ref": REF,
            "url": ORIGINAL_URL,
            "version": 1,
            "script_version_id": SCRIPT_VERSION,
            "sha256": SOURCE_SHA256,
        },
        "author_scored_submission": view["submission"],
        "source_score_binding": {
            "view_sha256": digest(view_path),
            "kernel_run_id": view["kernelRun"]["id"],
            "kernel_version": view["kernelRun"]["kernelVersionNumber"],
            "best_submission": view["bestSubmissionScore"],
        },
        "own_public_lb": None,
        "own_visible_or_hidden_execution": False,
        "original_environment": {
            "docker_image_version_id": CONTAINER_ID,
            "docker_image_digest": DOCKER_DIGEST,
            "metadata_docker_image": metadata["dockerImage"],
            "gpu": metadata["machineShape"],
            "internet": metadata["enableInternet"],
            "python313_visible_log": log_evidence["python313_visible_log_path"],
        },
        "author_visible_log": log_evidence,
        "author_visible_runtime_seconds": view["kernelRun"]["runInfo"]["runTimeSeconds"],
        "own_hidden_runtime": None,
        "syntax_checks": syntax,
        "matching_original_code_cells": len(control_tools.code_sources(prepared)),
        "code_cell_sha256": {
            str(i): hashlib.sha256(source.encode()).hexdigest() for i, source in control_tools.code_sources(prepared)
        },
        "code_changes": [],
        "non_code_changes": [
            "Japanese audited adoption provenance preface in existing leading markdown",
            "clear saved outputs/execution_count",
        ],
        "gold_scope": "Gold58 reused for development/selection by upstream published model; not independent CV or our independent head training.",
        "external_pretraining_scope": "Public OAI-derived learned weights only; no raw controlled OAI access/data download. Conditional adopted scope is bound to the completed official-rule/public-model usage audit.",
        "whole_notebook_license": "independently unconfirmed; original credits and NDA acknowledgement preserved",
        "reader_dataset_license": "Apache2.0 metadata confirmed elsewhere; not assigned to whole Notebook",
        "model_or_real_mri_download": False,
        "real_mri_decode": False,
        "training_performed": False,
        "upload_or_submission_performed": False,
        "success_or_same_score_guarantee": False,
    }
    write_json(output_dir / "provenance.json", provenance)
    with (output_dir / "manual-notes.txt").open("x", encoding="utf-8") as stream:
        stream.write(manual_notes(inputs))
    package = {
        "schema_version": "p005_public_control_package_v1",
        "eligibility_status_at_preparation": "conditionally_adopt_public_checkpoint_frozen_inference",
        "files": {
            path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        },
        "prepared_notebook_sha256": digest(notebook_path),
        "source_notebook_sha256": SOURCE_SHA256,
        "generator_sha256": digest(__file__),
        "reused_offline_syntax_helpers_sha256": digest(ROOT / "scripts/build_p004_public_control.py"),
        "code_cell_count": len(control_tools.code_sources(prepared)),
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
    package = prepare_package(output_dir=args.output_dir, notebook_path=args.notebook_path)
    print(
        json.dumps(
            {
                "code_cells": package["code_cell_count"],
                "inputs": package["input_count"],
                "eligibility": package["eligibility_status_at_preparation"],
                "sha256": package["prepared_notebook_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
