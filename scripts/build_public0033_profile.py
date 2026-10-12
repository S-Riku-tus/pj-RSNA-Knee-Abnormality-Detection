"""Build a private, label-free public0033 batch32/T4x2 profile; never uploads."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEAK = ROOT / "data/manifests/v1-vmohitrao-research/weak.csv"
GOLD = ROOT / "data/manifests/v1-vmohitrao-research/gold.csv"
SERIES = ROOT / "data/exports/rsraki-rsna-knee-sv354838181/rsna-cache-v1/raw/train_series.csv"
EVIDENCE = ROOT / "artifacts/research/20261008-post950-public-research-v1"
MANIFEST = EVIDENCE / "selected-provenance-files/bundle_manifest.json"
API_FILES = EVIDENCE / "datasets/renta0426/rsna-knee-public0033-meniscus-bag-v1/files.json"
HELPERS = tuple(
    ROOT / "scripts" / name for name in ("public0033_streaming_adapter.py", "public0033_profile_runtime.py")
)
CONTAINER = "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"
COUNT = 64


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def uid_column(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return {row["StudyInstanceUID"] for row in csv.DictReader(stream)}


def select_cohort():
    weak, gold, available = (uid_column(path) for path in (WEAK, GOLD, SERIES))
    if weak & gold or not weak <= available:
        raise ValueError("weak/gold/series UID metadata contract failed")
    uids = sorted(weak)[:COUNT]
    if len(uids) != COUNT:
        raise ValueError("not enough weak studies for fixed64")
    return {
        "schema_version": "public0033_fixed_weak_uid_cohort64_v1",
        "uids": uids,
        "selection": "lexicographically_sorted_fixed_weak_manifest_UID_prefix_64",
        "study_uid_sha256": hashlib.sha256("".join(f"{uid}\n" for uid in uids).encode()).hexdigest(),
        "weak_manifest_sha256": sha256(WEAK),
        "gold_manifest_sha256": sha256(GOLD),
        "series_metadata_sha256": sha256(SERIES),
        "weak_UID_count": len(weak),
        "gold_UID_count": len(gold),
        "gold_overlap_count": len(set(uids) & gold),
        "gold_excluded": True,
        "Report_or_targets_consumed": False,
        "accuracy_evaluated": False,
    }


def verify_public_metadata():
    """Check small source evidence only; no MRI or checkpoint payload is retrieved."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    listing = json.loads(API_FILES.read_text(encoding="utf-8"))["datasetFiles"]
    declared = {name: entry["bytes"] for name, entry in manifest["files"].items() if name != "dataset-metadata.json"}
    declared["bundle_manifest.json"] = MANIFEST.stat().st_size
    actual = {entry["name"]: entry["totalBytes"] for entry in listing}
    if actual != declared or len(actual) != 9:
        raise ValueError("Dataset V2 exact nine-file listing differs from declared byte sizes")
    if manifest["batch_studies"] != 32 or manifest["required_gpu_count"] != 2 or manifest["precision"] != "fp32":
        raise ValueError("public stock runtime contract drift")
    for name in ("public0033_runtime.py", "run_contract.json", "task_complete.json", "fullfit0033_meniscus_bag.yaml"):
        path = MANIFEST.parent / name
        entry = manifest["files"][name]
        if sha256(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError("locally inspected public small-source provenance hash drift")
    return {
        "api_files": actual,
        "manifest_sha256": sha256(MANIFEST),
        "api_listing_sha256": sha256(API_FILES),
        "declared_payload": manifest["files"],
        "checkpoint_payload_downloaded_at_preparation": False,
        "stock_checkpoint_count": 1,
        "stock_checkpoint_replicas_at_inference": 2,
        "stock_manifest_historical_parent_route_not_applied": True,
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def payload(cohort):
    buffer = io.BytesIO()
    files = {path.name: path.read_bytes() for path in HELPERS}
    files["public0033_profile_cohort.json"] = (
        json.dumps(cohort, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 8, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, value)
    body = buffer.getvalue()
    return body, {
        name: {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()} for name, value in files.items()
    }


def build(output_directory, notebook_path):
    output = Path(output_directory)
    notebook_path = Path(notebook_path)
    if output.exists() or notebook_path.exists():
        raise FileExistsError("profile preparation requires fresh output paths")
    cohort = select_cohort()
    source = verify_public_metadata()
    body, members = payload(cohort)
    body_hash = hashlib.sha256(body).hexdigest()
    intro = """public0033 Meniscus10 単独64検査診断（提出用ではありません）。
自分のPublic0.950基盤を保持し、異なるDINOv2表現の半月板専門枝の実行可能性を測ります。精度は評価しません。
固定weak manifestのUID先頭64件だけを使い、gold58 UIDと重ならないことを準備時に確認しています。
Notebookはtrain.csv・Report・教師を読みません。実MRIデコードはKaggleに接続された64検査のみです。
competitionとrenta0426/rsna-knee-public0033-meniscus-bag-v1 Version2の計2 Inputsが必要です。
ApexのCoAtNet・ConvNeXt重みは、この単独診断には不要です。
Original Container31481/Python3.13、TeslaT4 x2、Internet OFF。
author stock runtimeはFP32・batch32・10重複窓・6slots・12slices・336pxのまま1回呼びます。
64件のuint8 cacheはscratch disk memmapへ1件ずつ書きます。全件匿名RAM配列は作りません。
実model strict load、12所見有限予測、UID順、slot masks、decoder、時間、RSS、両GPUのpeak VRAMを観測します。
PUBLIC0033_PROFILE_SUMMARY.jsonはすべての契約に成功した場合のみ保存します。失敗時はeventsとtracebackを確認します。
submission.csvは作りません。2列raw CSVと12列diagnostic CSVは採点に提出しないでください。
弱教師全件fitでありOOFモデルではありません。半月板の融合改善や隠しデータ完走は未確認です。
外部checkpoint/runtime Apache2.0。DINOv2とcompetition dataの条件は別に適用されます。
Source: https://www.kaggle.com/datasets/renta0426/rsna-knee-public0033-meniscus-bag-v1/versions/2
Cache reference: https://www.kaggle.com/code/sujanmajhisuzan/rsna-knee-tri-specialist-superstack?scriptVersionId=355742320
"""
    extraction = f"""# Code and fixed UID metadata only: no weights, MRI, Reports or teachers.
import base64, hashlib, io, json, sys, zipfile
from pathlib import Path
_profile_payload = base64.b64decode({base64.b64encode(body).decode()!r})
assert hashlib.sha256(_profile_payload).hexdigest() == {body_hash!r}
_profile_members = {members!r}
_profile_source = Path('/kaggle/working/public0033_profile_src')
_profile_source.mkdir(exist_ok=False)
with zipfile.ZipFile(io.BytesIO(_profile_payload)) as _profile_zip:
    assert set(_profile_zip.namelist()) == set(_profile_members)
    for _name in _profile_zip.namelist():
        assert Path(_name).name == _name and not Path(_name).is_absolute()
        _bytes = _profile_zip.read(_name)
        assert len(_bytes) == _profile_members[_name]['bytes']
        assert hashlib.sha256(_bytes).hexdigest() == _profile_members[_name]['sha256']
        (_profile_source / _name).write_bytes(_bytes)
sys.path.insert(0, str(_profile_source))
print('Prepared code-only profile helpers. No packages are installed or downloaded.')
"""
    invocation = """from public0033_profile_runtime import run_profile
_cohort = json.loads((_profile_source/'public0033_profile_cohort.json').read_text(encoding='utf-8'))
_summary = run_profile(_cohort, input_root='/kaggle/input',
    output_directory='/kaggle/working/public0033_profile64', scratch='/kaggle/temp')
assert _summary['status'] == 'passed'
assert not Path('/kaggle/working/submission.csv').exists()
"""
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "public0033_profile": {
                "studies": COUNT,
                "dataset_version": 2,
                "dataset_internal_version_id": 19356288,
                "docker_image": CONTAINER,
                "docker_image_pinning_type": "original",
                "internet": False,
                "accelerator": "Tesla T4 x2",
                "submitted": False,
                "real_execution_at_preparation": False,
            },
        },
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": intro.splitlines(keepends=True)},
            code_cell(extraction),
            code_cell(invocation),
        ],
    }
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "public0033-profile64", "exec")
    settings = {
        "slug": "rsraki/public0033-menis-profile64",
        "newTitle": "public0033 meniscus batch32 profile64",
        "language": "python",
        "kernelType": "notebook",
        "datasetDataSources": ["renta0426/rsna-knee-public0033-meniscus-bag-v1/2"],
        "kernelDataSources": [],
        "competitionDataSources": ["rsna-knee-abnormality-detection"],
        "categoryIds": ["gpu"],
        "isPrivate": True,
        "enableGpu": True,
        "enableInternet": False,
        "dockerImagePinningType": "original",
        "modelDataSources": [],
        "enableTpu": False,
        "dockerImage": CONTAINER,
        "kernelExecutionType": "SAVE_AND_RUN_ALL",
        "machineShape": "NvidiaTeslaT4",
    }
    kernel_metadata = {
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
    output.mkdir(parents=True, exist_ok=False)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (output / notebook_path.name).write_bytes(notebook_path.read_bytes())
    (output / "public0033-profile-code.zip").write_bytes(body)
    for name, value in (
        ("api-settings.json", settings),
        ("kernel-metadata.json", kernel_metadata),
        ("fixed-cohort.json", cohort),
    ):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    record = {
        "schema_version": "public0033_profile64_preparation_v1",
        "prepared_at": datetime.now(timezone(timedelta(hours=9))).isoformat(),
        "status": "prepared_not_executed",
        "notebook_path": str(notebook_path.relative_to(ROOT)),
        "notebook_sha256": sha256(notebook_path),
        "generator_sha256": sha256(__file__),
        "helper_sha256": {path.name: sha256(path) for path in HELPERS},
        "payload_sha256": body_hash,
        "payload_members": members,
        "studies": COUNT,
        "study_uid_sha256": cohort["study_uid_sha256"],
        "selection": cohort["selection"],
        "gold_excluded": True,
        "gold_overlap_count": 0,
        "Report_or_targets_consumed_at_Kaggle": False,
        "preparation_inputs_used_for_UID_exclusion_and_hashes_only": True,
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (WEAK, GOLD, SERIES)},
        "accuracy_evaluated": False,
        "real_MRI_decoded_at_preparation": False,
        "weights_retrieved_at_preparation": False,
        "remote_executed_at_preparation": False,
        "fusion_applied": False,
        "submission_created": False,
        "cache_data_bytes": COUNT * 6 * 12 * 336 * 336,
        "maximum_anonymous_cohort_studies_during_decode": 1,
        "actual_stock_max_batch_unverified": True,
        "stock_call_count_planned": 1,
        "stock_source_sha256": source["declared_payload"]["public0033_runtime.py"]["sha256"],
        "stock_dataset": {
            "ref": "renta0426/rsna-knee-public0033-meniscus-bag-v1",
            "version": 2,
            "internal_version_id": 19356288,
            "metadata_checked_at": "2026-10-08 JST",
            "source": source,
        },
        "original_container": CONTAINER,
        "api_launch_settings_sha256": sha256(output / "api-settings.json"),
        "launch_required_checks": [
            "exact saved 3-cell source",
            "private Original31481 digest; T4x2; Internet OFF",
            "competition20902466 and renta Dataset19356288 only; no Apex/head Inputs",
            "64 fixed weak UID hash and no Gold UID overlap",
            "summary generated only after both batch32 x10 finite FP32 passes",
            "no submission.csv or competition submission",
        ],
        "limits": [
            "Own Public0.950 incumbent remains unchanged.",
            "This is execution/resource profiling, not accuracy/OOF testing.",
            "Fullfit weak checkpoint does not support weak holdout coefficient selection.",
            "Larger hidden cohorts and actual gain remain unverified.",
        ],
    }
    (output / "preparation.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "notebook": str(notebook_path),
                "sha256": record["notebook_sha256"],
                "package": str(output),
                "cache_data_bytes": record["cache_data_bytes"],
            }
        )
    )
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--notebook-path", required=True, type=Path)
    args = parser.parse_args()
    build(args.output_dir.resolve(), args.notebook_path.resolve())
