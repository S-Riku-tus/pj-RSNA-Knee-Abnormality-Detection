"""Prepare native392 whole-group shards and CPU notebooks; never run or upload."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import io
import json
import re
import zipfile
from pathlib import Path

from rsna_knee.contracts import ID, dump_json, sha256
from rsna_knee.feature_contracts import json_hash, load_feature_config, validate_fold_audit
from rsna_knee.native_pixel_shards import EXECUTION_ENVIRONMENT, ORIGINAL_DOCKER, make_plan, validate_plan

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/v1-vmohitrao-research"
DEFAULT_PACKAGE = ROOT / "artifacts/kaggle/native392-full-shards-v1-20261008"


def original_builder():
    spec = importlib.util.spec_from_file_location(
        "immutable_pixel16_builder", ROOT / "scripts/build_native_triplet_pilot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reviewed_plan(manifest=MANIFEST, shard_count=24):
    manifest = Path(manifest)
    metadata = json.loads((manifest / "manifest.json").read_text(encoding="utf-8-sig"))
    validate_fold_audit(manifest, manifest / "fold-audit-v2.json", metadata)
    if metadata["seed"] != 20261002 or metadata["n_folds"] != 5 or metadata["weak_studies"] != 4207:
        raise ValueError("The original reviewed seed/folds/full4207 weak manifest is required")
    if metadata["label_provenance"].get("gold_used_for_tuning") is not False:
        raise ValueError("Reviewed manifest reports Gold use")
    builder = original_builder()
    rows = builder.structure_rows(manifest / "weak.csv", (ID, "group_id", "fold", "source"))
    for row in rows:
        row["fold"] = int(row["fold"])
    gold = builder.structure_rows(manifest / "gold.csv", (ID, "group_id"))
    if len(rows) != metadata["weak_studies"]:
        raise ValueError("Full weak structure row count differs")
    settings = load_feature_config(ROOT / "configs/experiments/f002-dinov2-attention.json")["preprocess"]
    plan = make_plan(
        rows,
        [row[ID] for row in gold],
        [row["group_id"] for row in gold],
        seed=metadata["seed"],
        n_folds=metadata["n_folds"],
        settings=settings,
        manifest_sha256={
            name: sha256(manifest / name) for name in ("manifest.json", "weak.csv", "gold.csv", "fold-audit-v2.json")
        },
        shard_count=shard_count,
    )
    return validate_plan(plan)


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(outputs=[], execution_count=None)
    return result


def notebook(plan, index, encoded, binding, decoder_manifest_sha256, phantom_manifest_sha256):
    setup = f"""from pathlib import Path
import base64, hashlib, io, json, os, sys, zipfile
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    raise RuntimeError('Pixel cache exports must never be submitted for scoring')
CODE = Path('/kaggle/working/native392-shard-code')
if CODE.exists():
    raise RuntimeError('Use a fresh immutable saved run')
payload = base64.b64decode({encoded!r}, validate=True)
if hashlib.sha256(payload).hexdigest() != {binding["code_zip_sha256"]!r}:
    raise ValueError('Embedded code ZIP hash differs')
expected = {binding["code_member_sha256"]!r}
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    if len(archive.infolist()) != len(expected) or set(archive.namelist()) != set(expected):
        raise ValueError('Code/structure-only member allowlist differs')
    validated = {{}}
    for name in expected:
        target = (CODE / name).resolve()
        if not target.is_relative_to(CODE.resolve()) or '\\\\' in name or ':' in name:
            raise ValueError('Unsafe code ZIP path')
        data = archive.read(name)
        if hashlib.sha256(data).hexdigest() != expected[name]:
            raise ValueError('Code member bytes differ')
        validated[name] = data
    CODE.mkdir()
    for name, data in validated.items():
        target = CODE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / 'src'))
CSV_OPENED = []
def prohibit_teacher_csv(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes)):
        path = os.fsdecode(args[0])
        if Path(path).name.lower().endswith('.csv') and '/kaggle/input/' in path.replace('\\\\', '/'):
            if Path(path).name.lower() != 'train_series.csv':
                raise RuntimeError('Teacher, Report, test and submission CSV reads are forbidden')
            CSV_OPENED.append(path)
sys.addaudithook(prohibit_teacher_csv)
print('Code/structural UID plan verified; no teacher or encoder loaded')
"""
    runtime = f"""from decoder_runtime import install_offline_decoders, verify_compressed_phantoms, decoder_inventory, resource_snapshot
from bounded_feature_reader import make_bounded_reader
from rsna_knee.feature_imaging import read_feature_windows
from rsna_knee.feature_contracts import json_hash
from rsna_knee.native_pixel_shards import export_pixel_shard
ROOT = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
if not ROOT.is_dir():
    ROOT = Path('/kaggle/input/rsna-knee-abnormality-detection')
DECODER = Path('/kaggle/input/datasets/rsraki/rsna-dicom-decoders-py313-v1')
if not DECODER.is_dir():
    DECODER = Path('/kaggle/input/rsna-dicom-decoders-py313-v1')
if not ROOT.is_dir() or not DECODER.is_dir():
    raise RuntimeError('Competition plus decoder Dataset Version2 are required')
decoder_install = install_offline_decoders(DECODER / 'wheels', {decoder_manifest_sha256!r})
decoder_install['synthetic_checks'] = verify_compressed_phantoms(DECODER / 'phantoms', {phantom_manifest_sha256!r})
if not all(item['available'] for item in decoder_inventory().values()):
    raise RuntimeError('Required lossless MRI decoders are unavailable')
PLAN = json.loads((CODE / 'native392-plan.json').read_text(encoding='utf-8'))
if json_hash(PLAN) != {json_hash(plan)!r}:
    raise ValueError('Full structural plan differs')
SHARD_INDEX = {index}
EXECUTION_BINDING = {binding!r}
WORKSPACE = Path('/kaggle/working/native392-native-tmp')
WORKSPACE.mkdir()
def native_budget_observer(event, data):
    if event == 'allocation_plan' and data['minimum_free_disk_bytes'] > PLAN['budgets']['native_temp_bytes']:
        raise ValueError('One native memmap would exceed the fixed temporary disk reserve')
reader = make_bounded_reader(WORKSPACE, observer=native_budget_observer)
def finalize_metadata():
    if list(WORKSPACE.iterdir()):
        raise RuntimeError('Native temporary files remain; shard cannot complete')
    if (Path('/kaggle/working') / 'submission.csv').exists():
        raise RuntimeError('Unexpected competition submission artifact')
    return {{'temporary_files_remaining': 0, 'input_csv_reads': sorted(set(CSV_OPENED)),
            'decoder_install': decoder_install, 'python': sys.version, 'accelerator': 'CPU',
            'storage_reader': 'one native memmap; exact shared392 triplets; native<=128MiB byteQC'}}
result = export_pixel_shard(ROOT, Path('/kaggle/working') / PLAN['shards'][SHARD_INDEX]['name'], PLAN, SHARD_INDEX,
                           reader=reader, qc_reader=read_feature_windows, execution_binding=EXECUTION_BINDING,
                           resource_fn=lambda: resource_snapshot('cpu'), finalize_metadata_fn=finalize_metadata)
print({{key: result[key] for key in ('status', 'processed', 'raw_pixel_upper_bytes', 'elapsed_seconds', 'preprocessing_byte_QC_count')}})
print(result['archive'])
"""
    result = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "rsna_native_pixel_shard": {
                "index": index,
                "study_count": plan["shards"][index]["study_count"],
                "full_plan_sha256": json_hash(plan),
                "code_zip_sha256": binding["code_zip_sha256"],
                "for_submission": False,
            },
        },
        "cells": [
            cell(
                "markdown",
                f"# native392 pixel cache {index + 1}/{plan['shard_count']}\n\n"
                "CPU・Internet OFF・private・Original Python3.13。Competition と decoder Dataset Version2 を接続します。"
                "教師ラベルと Report は読みません。元の392px・3×48 tripletの byte/QC/mask 契約を保持します。\n\n"
                "weak の whole group を固定割当し、Gold UID/group を除外します。この cache は学習用の画素であり、"
                "採点用 Notebook ではありません。完了しなかった shard は全件 index に組み込めません。",
            ),
            cell("code", setup),
            cell("code", runtime),
        ],
    }
    for value in result["cells"]:
        if value["cell_type"] == "code":
            compile("".join(value["source"]), "native392-shard", "exec")
    return result


def build(package=DEFAULT_PACKAGE, *, shard_count=24, notebook_dir=None, notebook_suffix=""):
    package = Path(package).resolve()
    notebook_dir = ROOT / "notebooks/public" if notebook_dir is None else Path(notebook_dir).resolve()
    plan = reviewed_plan(shard_count=shard_count)
    if notebook_suffix and not re.fullmatch(r"_[a-zA-Z0-9]+", notebook_suffix):
        raise ValueError("Only a safe alphanumeric generation suffix is allowed")
    names = [
        f"20a_export_native392_shard_{index + 1:02d}_of_{shard_count:02d}{notebook_suffix}.ipynb"
        for index in range(shard_count)
    ]
    if package.exists() or any((notebook_dir / name).exists() for name in names):
        raise ValueError("Use new immutable package and Notebook destinations")
    members = {
        f"src/rsna_knee/{name}": (ROOT / "src/rsna_knee" / name).read_bytes()
        for name in (
            "__init__.py",
            "contracts.py",
            "imaging.py",
            "feature_contracts.py",
            "feature_imaging.py",
            "triplet_cache.py",
            "native_pixel_shards.py",
        )
    }
    members["bounded_feature_reader.py"] = (ROOT / "scripts/bounded_feature_reader.py").read_bytes()
    members["decoder_runtime.py"] = original_builder().decoder_source().encode()
    members["native392-plan.json"] = (json.dumps(plan, sort_keys=True, ensure_ascii=False) + "\n").encode()
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(members.items()):
            item = zipfile.ZipInfo(name, date_time=(2026, 10, 8, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(item, data)
    binding = {
        "code_zip_sha256": hashlib.sha256(payload.getvalue()).hexdigest(),
        "code_member_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in members.items()},
        "environment": EXECUTION_ENVIRONMENT,
    }
    decoder = ROOT / "artifacts/kaggle/dicom-decoders-py313-v2-20261007"
    decoder_hash, phantom_hash = (
        sha256(decoder / "decoder-manifest.json"),
        sha256(decoder / "phantoms/phantom-manifest.json"),
    )
    package.mkdir(parents=True)
    notebook_dir.mkdir(parents=True, exist_ok=True)
    (package / "native392-code.zip").write_bytes(payload.getvalue())
    dump_json(package / "plan.json", plan)
    records = []
    for index, name in enumerate(names):
        book = notebook(plan, index, base64.b64encode(payload.getvalue()).decode(), binding, decoder_hash, phantom_hash)
        serialized = json.dumps(book, ensure_ascii=False, indent=1) + "\n"
        job = package / f"shard-{index + 1:02d}"
        job.mkdir()
        for path in (job / name, notebook_dir / name):
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(serialized)
        settings = {
            "kernel_ref": f"rsraki/native392-cache-shard-{index + 1:02d}-of-{shard_count:02d}",
            "title": f"native392 cache shard {index + 1:02d} of {shard_count:02d}",
            "notebook": name,
            "private": True,
            "accelerator": "CPU",
            "internet": False,
            "docker_image_pinning_type": "original",
            "docker_image": ORIGINAL_DOCKER,
            "competition_sources": ["rsna-knee-abnormality-detection"],
            "dataset_sources": ["rsraki/rsna-dicom-decoders-py313-v1/2"],
            "kernel_sources": [],
            "model_sources": [],
            "competition_submission_forbidden": True,
        }
        dump_json(job / "api-settings.json", settings)
        records.append(
            {"index": index, "directory": job.name, "notebook": name, "notebook_sha256": sha256(job / name), **settings}
        )
    counts = [value["study_count"] for value in plan["shards"]]
    pilot_summary = json.loads(
        (
            ROOT / "artifacts/research/20261008-native392-pixel16-results-v1/NATIVE392_PIXEL_PILOT_SUMMARY.json"
        ).read_text(encoding="utf-8")
    )
    mean_bytes, mean_seconds = pilot_summary["archive"]["bytes"] / 16, pilot_summary["elapsed_seconds"] / 16
    metadata = {
        "status": "prepared_no_execution_or_cache_acquisition",
        "plan_sha256": json_hash(plan),
        "plan_file_sha256": sha256(package / "plan.json"),
        "execution_binding": binding,
        "studies": plan["study_count"],
        "groups": plan["group_count"],
        "shard_count": shard_count,
        "shard_studies": counts,
        "max_shard_studies": max(counts),
        "raw_bytes_total": sum(value["raw_pixel_upper_bytes"] for value in plan["shards"]),
        "max_raw_shard_bytes": max(value["raw_pixel_upper_bytes"] for value in plan["shards"]),
        "decoder_manifest_sha256": decoder_hash,
        "phantom_manifest_sha256": phantom_hash,
        "jobs": records,
        "conditional_pilot_mean_plan": {
            "source_summary_sha256": sha256(
                ROOT / "artifacts/research/20261008-native392-pixel16-results-v1/NATIVE392_PIXEL_PILOT_SUMMARY.json"
            ),
            "full_archive_bytes": mean_bytes * plan["study_count"],
            "max_shard_archive_bytes": mean_bytes * max(counts),
            "max_shard_CPU_seconds": mean_seconds * max(counts),
            "all_CPU_job_seconds_sum": mean_seconds * plan["study_count"],
            "limits": "Linear means from16 selected studies, no population/resource/runtime guarantee. Concurrency/queue/network not estimated.",
        },
        "native_DICOM_downloaded_or_decoded_locally": False,
        "cache_downloaded": False,
        "fit_started": False,
        "remote_mutation": False,
        "for_submission": False,
    }
    dump_json(package / "package.json", metadata)
    commands = {
        "execution": "Root quality review must precede manual/API SAVE_AND_RUN_ALL; this builder contains no launch/upload/submit call.",
        "collection": "After all CPU jobs COMPLETE, collect only completed shard ZIP+summary into data/native392/shards. Not performed by this builder.",
        "build_full_index_after_all_shards": f".venv/Scripts/python.exe scripts/build_native392_index.py --package {package.relative_to(ROOT).as_posix()} --cache-root data/native392 --output data/native392/full-index.json --summary-glob shards/**/NATIVE392_PIXEL_SHARD_SUMMARY.json",
        "f005_preflight_after_index": ".venv/Scripts/python.exe scripts/train_native_dinov2.py preflight --manifest data/manifests/v1-vmohitrao-research --index data/native392/full-index.json --resource-gate artifacts/research/20261008-f005-native-training-preparation-v1/resource-gate-bound.json --config configs/experiments/f005-dinov2-native-last2.json --fold 0",
        "real_fit": "Separate root quality gate; use existing f005 handoff v2 commands with the validated full index and bound gate. No real fit is executed here.",
    }
    dump_json(package / "commands-not-executed.json", commands)
    print(
        json.dumps(
            {
                "package": str(package.relative_to(ROOT)),
                "studies": plan["study_count"],
                "shard_studies": counts,
                "max_raw_bytes": metadata["max_raw_shard_bytes"],
            }
        )
    )
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--shard-count", type=int, default=24)
    parser.add_argument("--notebook-dir", type=Path)
    parser.add_argument("--notebook-suffix", default="")
    args = parser.parse_args()
    build(
        args.package, shard_count=args.shard_count, notebook_dir=args.notebook_dir, notebook_suffix=args.notebook_suffix
    )
