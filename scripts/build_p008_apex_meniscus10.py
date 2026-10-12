"""Append one fixed Meniscus10 factor to the seven unchanged p005 code cells."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import io
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "notebooks/public/14_submit_p005_public950_control.ipynb"
BASELINE_SHA256 = "75c3abe77e8f0ccdaec2f57a8946d2baf297db8cc6fbdb275e5fcb1153b57546"
HELPERS = [
    ROOT / "scripts" / name
    for name in ("public0033_streaming_adapter.py", "public0033_profile_runtime.py", "public0033_apex_serving.py")
]
PROFILE_REVIEW = ROOT / "experiments/public0033-profile64-result-review-20261008.json"
CONTAINER = "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(cell):
    return "".join(cell["source"])


def code_cell(text):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def make_payload():
    buffer = io.BytesIO()
    members = {}
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(HELPERS):
            value = path.read_bytes()
            info = zipfile.ZipInfo(path.name, date_time=(2026, 10, 8, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, value)
            members[path.name] = {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    return buffer.getvalue(), members


def prepare_notebook(baseline):
    original_code = [source(cell) for cell in baseline["cells"] if cell["cell_type"] == "code"]
    if len(baseline["cells"]) != 10 or len(original_code) != 7:
        raise ValueError("expected frozen ten-cell/seven-code-cell Apex layout")
    body, members = make_payload()
    body_hash = hashlib.sha256(body).hexdigest()
    preface = """p008：自分のPublic0.950を確認したp005/Apexから、半月板専門枝だけを追加した新候補。
下の元7 code cell・CoAtNet・ConvNeXt k16・SWA/3fold重み・既存係数は変更していません。
p006/k24や鏡映TTAの変更は混ぜていません。
末尾の追加cellで別のDINOv2 Meniscus10を1回実行し、2列だけ固定90:10のglobal rank融合をします。
他10列のCSV値文字列とUID順を保持し、Apexの元CSVを別ファイルへ保存します。
live testが1件ならstockをskip。全slot欠損等の明示した入力不足だけはApex保持のfallbackです。
checkpoint不一致・CUDA OOM・想定外code errorはApexを保持して失敗を再送出し、成功扱いしません。
64件のKaggle実診断はT4x2/FP32/batch32で成功しましたが、今回の融合Public・隠し完走は未確認です。
診断の3557 selected framesはExplicitVRLittleEndianのみで、圧縮codec全般の確認ではありません。
全件弱教師fitのcheckpointであり、OOF評価や自分のgoldによる係数選択は行いません。
追加Inputはrenta0426/rsna-knee-public0033-meniscus-bag-v1 Version2。
既存reader Version1・SWA Version1・competitionを保持し、合計4 Inputsを接続します。
Original Container31481/digest2757e…、TeslaT4 x2、Internet OFF。
cacheは/kaggle/tempのscratch memmapへ検査ごとに作り、OutputへMRI cacheを残しません。
新しいP008_MENISCUS_SUMMARY.jsonのstatus=passed/specialist_used=trueを確認して保存版を照合します。
この準備はAPI実行・アップロード・競技提出を行いません。モデル・MRIのローカル取得も行いません。
追加checkpoint/runtime Apache2.0、DINOv2/competition条件は別に適用。既存Apexの帰属は以下に保持。
Dataset: https://www.kaggle.com/datasets/renta0426/rsna-knee-public0033-meniscus-bag-v1/versions/2
Cache source: https://www.kaggle.com/code/sujanmajhisuzan/rsna-knee-tri-specialist-superstack?scriptVersionId=355742320
以下のp005準備時点の未採点説明は履歴です。p005自身のPublic0.950は10月8日に確認済み。

---

"""
    extraction = f"""# Append only after the original Apex CSV has been generated.
import base64 as _p8_base64, hashlib as _p8_hashlib, io as _p8_io
import sys as _p8_sys, tempfile as _p8_tempfile, zipfile as _p8_zipfile
from pathlib import Path as _p8_Path
_p8_payload = _p8_base64.b64decode({base64.b64encode(body).decode()!r})
assert _p8_hashlib.sha256(_p8_payload).hexdigest() == {body_hash!r}
_p8_members = {members!r}
_p8_source = _p8_Path(_p8_tempfile.mkdtemp(prefix='p008_source_', dir='/kaggle/working'))
with _p8_zipfile.ZipFile(_p8_io.BytesIO(_p8_payload)) as _p8_zip:
    assert set(_p8_zip.namelist()) == set(_p8_members)
    for _p8_name in _p8_zip.namelist():
        assert _p8_Path(_p8_name).name == _p8_name and not _p8_Path(_p8_name).is_absolute()
        _p8_bytes = _p8_zip.read(_p8_name)
        assert len(_p8_bytes) == _p8_members[_p8_name]['bytes']
        assert _p8_hashlib.sha256(_p8_bytes).hexdigest() == _p8_members[_p8_name]['sha256']
        (_p8_source/_p8_name).write_bytes(_p8_bytes)
_p8_sys.path.insert(0, str(_p8_source))
from public0033_apex_serving import append_specialist as _p8_append
_p8_result = _p8_append(submission='/kaggle/working/submission.csv',
    input_root='/kaggle/input', output_directory='/kaggle/working/p008_meniscus10',
    scratch='/kaggle/temp')
assert _p8_result['status'] in ('passed','skipped_incumbent_retained','fallback_incumbent_retained')
assert _p8_Path('/kaggle/working/submission.csv').is_file()
"""
    prepared = copy.deepcopy(baseline)
    prepared["cells"][0]["source"] = (preface + source(prepared["cells"][0])).splitlines(keepends=True)
    prepared["cells"].append(code_cell(extraction))
    for cell in prepared["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"], cell["execution_count"] = [], None
            text = source(cell)
            if text.startswith("%%writefile "):
                text = text.split("\n", 1)[1]
            compile(text, "p008", "exec")
    if [source(cell) for cell in prepared["cells"][:-1] if cell["cell_type"] == "code"] != original_code:
        raise ValueError("original seven code cells changed")
    if prepared["metadata"] != baseline["metadata"]:
        raise ValueError("baseline metadata changed")
    return prepared, body, members


def build(output_directory, notebook_path, audit_path):
    output, notebook_path, audit_path = map(Path, (output_directory, notebook_path, audit_path))
    if any(path.exists() for path in (output, notebook_path, audit_path)):
        raise FileExistsError("immutable p008 preparation requires fresh paths")
    if sha256(BASELINE) != BASELINE_SHA256:
        raise ValueError("own scored p005 baseline changed")
    profile = json.loads(PROFILE_REVIEW.read_text(encoding="utf-8"))
    if (
        profile["status"] != "real64_single_branch_contract_passed_integration_preparation_allowed"
        or profile["checks_passed"] != 32
    ):
        raise ValueError("real profile64 independent integration gate has not passed")
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    notebook, body, members = prepare_notebook(baseline)
    settings = {
        "slug": "rsraki/p008-apex-meniscus10",
        "newTitle": "p008 Apex Meniscus10",
        "language": "python",
        "kernelType": "notebook",
        "datasetDataSources": [
            "goodpjw2008/rsna-knee-2-5d-convnext-reader/1",
            "nartaa/rsna-knee-publication-swa-weights-20261007/1",
            "renta0426/rsna-knee-public0033-meniscus-bag-v1/2",
        ],
        "competitionDataSources": ["rsna-knee-abnormality-detection"],
        "kernelDataSources": [],
        "modelDataSources": [],
        "categoryIds": ["gpu"],
        "isPrivate": True,
        "enableGpu": True,
        "enableInternet": False,
        "enableTpu": False,
        "dockerImagePinningType": "original",
        "dockerImage": CONTAINER,
        "kernelExecutionType": "SAVE_AND_RUN_ALL",
        "machineShape": "NvidiaTeslaT4",
    }
    output.mkdir(parents=True, exist_ok=False)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (output / notebook_path.name).write_bytes(notebook_path.read_bytes())
    (output / "p008-meniscus-code.zip").write_bytes(body)
    (output / "api-settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "id": settings["slug"],
        "title": settings["newTitle"],
        "code_file": notebook_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": settings["datasetDataSources"],
        "competition_sources": settings["competitionDataSources"],
        "kernel_sources": [],
    }
    (output / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    record = {
        "schema_version": "p008_apex_meniscus10_preparation_v1",
        "prepared_at": datetime.now(timezone(timedelta(hours=9))).isoformat(),
        "status": "offline_prepared_not_executed",
        "notebook": str(notebook_path.relative_to(ROOT)),
        "notebook_sha256": sha256(notebook_path),
        "baseline_notebook_sha256": BASELINE_SHA256,
        "generator_sha256": sha256(__file__),
        "helper_sha256": {path.name: sha256(path) for path in HELPERS},
        "payload_members": members,
        "payload_sha256": hashlib.sha256(body).hexdigest(),
        "original_code_cells": 7,
        "original_code_cells_unchanged": True,
        "original_code_cell_sha256": {
            str(index): hashlib.sha256(source(cell).encode()).hexdigest()
            for index, cell in enumerate(baseline["cells"])
            if cell["cell_type"] == "code"
        },
        "appended_code_cells": 1,
        "appended_code_cell_sha256": hashlib.sha256(source(notebook["cells"][-1]).encode()).hexdigest(),
        "total_cells": len(notebook["cells"]),
        "only_prediction_factor": "additional frozen DINOv2 Meniscus10 with prespecified90:10 global rank overlay on2Meniscus columns",
        "fixed_parent_weight": 0.90,
        "fixed_branch_weight": 0.10,
        "other10_value_strings_UID_order_preserved": True,
        "N_less_than2": "skip stock and retain incumbent bytes",
        "branch_error": "declared input unavailability retains incumbent with explicit fallback; checkpoint/CUDA/programming failures retain bytes, record failed and re-raise",
        "stale_branch_outputs": "refuse existing output directory and never reuse raw predictions",
        "cache": "sequential study decode into scratch disk memmap, cleaned on exit, never exported",
        "new_training_or_teacher_change": False,
        "own_gold_coefficient_selection": False,
        "branch_checkpoint_role": "weak fullfit notOOF",
        "local_real_MRI_or_model_payload_retrieval": False,
        "remote_upload_launch_submit": False,
        "p008_Public": None,
        "real_profile_gate": {
            "review": str(PROFILE_REVIEW.relative_to(ROOT)),
            "sha256": sha256(PROFILE_REVIEW),
            "checks": 32,
            "studies": 64,
            "note": "Resource/contract gate; no fusion accuracy or compressed-codec evidence.",
        },
        "Inputs": {
            "competition": {"ref": "rsna-knee-abnormality-detection", "expected_bundle_id": 20902466},
            "reader": {"ref": settings["datasetDataSources"][0], "expected_internal_version_id": 20325538},
            "SWA": {"ref": settings["datasetDataSources"][1], "expected_internal_version_id": 20440357},
            "Meniscus10": {"ref": settings["datasetDataSources"][2], "expected_internal_version_id": 19356288},
        },
        "original_container": CONTAINER,
        "api_settings_sha256": sha256(output / "api-settings.json"),
        "launch_submission_gate": [
            "Root authorizes/gates remote operations; generator does not perform them.",
            "Saved exact all11 cells, original7 unchanged, appended helper payload SHA exact.",
            "Private Original digest, T4x2, Internet OFF, only4 pinned Inputs.",
            "Visible P008 summary status passed and specialist_used true; N<2 skip/fallback is not demonstrated gain.",
            "Full UID12-column finite candidate, other10 strings, exact stock checkpoint/source, no stale outputs.",
            "Preserve own0.950 incumbent; score improvement and hidden full run remain unmeasured.",
        ],
    }
    (output / "preparation.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "notebook": str(notebook_path),
                "sha256": record["notebook_sha256"],
                "package": str(output),
                "original7_unchanged": True,
            }
        )
    )
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--notebook-path", required=True, type=Path)
    parser.add_argument("--audit-path", required=True, type=Path)
    args = parser.parse_args()
    build(args.output_dir.resolve(), args.notebook_path.resolve(), args.audit_path.resolve())
