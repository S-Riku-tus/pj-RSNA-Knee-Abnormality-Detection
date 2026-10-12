"""Build a fixed-group label-free pixel16 notebook, without downloads or execution."""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

from rsna_knee.contracts import ID, dump_json, sha256
from rsna_knee.feature_contracts import load_feature_config, validate_fold_audit
from rsna_knee.triplet_cache import raw_pixel_bytes, validate_cohort

ROOT = Path(__file__).resolve().parents[1]
NAME = "18a_profile_dinov2_native392_pixel16.ipynb"
MANIFEST = ROOT / "data/manifests/v1-vmohitrao-research"
PACKAGE = ROOT / "artifacts/kaggle/native392-pixel16-v1-20261008"


def structure_rows(path, fields):
    """Retain only structure columns from the already-reviewed local manifest."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        indices = [header.index(name) for name in fields]
        return [{name: row[index] for name, index in zip(fields, indices)} for row in reader]


def select_cohort(manifest=MANIFEST, count=16, held_out_fold=0):
    if count != 16:
        raise ValueError("This diagnostic is fixed to 16 studies")
    metadata = json.loads((manifest / "manifest.json").read_text(encoding="utf-8-sig"))
    validate_fold_audit(manifest, manifest / "fold-audit-v2.json", metadata)
    fields = (ID, "group_id", "fold", "source")
    weak = structure_rows(manifest / "weak.csv", fields)
    gold = structure_rows(manifest / "gold.csv", (ID, "group_id"))
    gold_ids, gold_groups = {row[ID] for row in gold}, {row["group_id"] for row in gold}
    groups = defaultdict(list)
    for row in weak:
        row["fold"] = int(row["fold"])
        if row[ID] in gold_ids or row["group_id"] in gold_groups or row["source"] != "report_weak":
            raise ValueError("Reviewed weak manifest contains a Gold/source conflict")
        groups[row["group_id"]].append(row)
    if any(len({row["fold"] for row in members}) != 1 for members in groups.values()):
        raise ValueError("The fixed group split is corrupt")
    eligible = {group: members for group, members in groups.items() if members[0]["fold"] != held_out_fold}
    ranked = sorted(eligible, key=lambda group: hashlib.sha256(f"{metadata['seed']}:{group}".encode()).hexdigest())
    selected = []
    for group in ranked:
        members = eligible[group]
        if len(members) <= count - len(selected):
            selected.extend(sorted(members, key=lambda row: row[ID]))
        if len(selected) == count:
            break
    cohort = {
        "schema_version": "fixed_weak_group_pixel_pilot_v1",
        "seed": metadata["seed"],
        "held_out_fold": held_out_fold,
        "n_folds": metadata["n_folds"],
        "studies": selected,
        "gold_ids": sorted(gold_ids),
        "gold_groups": sorted(gold_groups),
        "complete_selected_groups": {
            group: sorted(row[ID] for row in groups[group]) for group in sorted({row["group_id"] for row in selected})
        },
        "manifest_sha256": {
            name: sha256(manifest / name) for name in ("manifest.json", "weak.csv", "gold.csv", "fold-audit-v2.json")
        },
        "selection": "SHA256(seed:group_id) ascending; whole groups that fit the fixed 16-study budget",
        "targets_used_for_selection": False,
        "patient_independence_verified": False,
    }
    validate_cohort(cohort)
    return cohort


def decoder_source():
    path = ROOT / "scripts/frozen_streaming_runtime.py"
    text = path.read_text(encoding="utf-8")
    tree, lines = ast.parse(text), text.splitlines(keepends=True)
    keep = {"decoder_inventory", "install_offline_decoders", "verify_compressed_phantoms", "resource_snapshot"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in keep]
    if {node.name for node in nodes} != keep:
        raise ValueError("Audited offline decoder helpers missing")
    preamble = "import hashlib, importlib.metadata, json, shutil, subprocess, sys\nfrom pathlib import Path\n"
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "REQUIRED_SYNTAXES" for target in node.targets)
    )
    return (
        preamble
        + "".join(lines[assignment.lineno - 1 : assignment.end_lineno])
        + "\n\n"
        + "\n\n".join("".join(lines[node.lineno - 1 : node.end_lineno]) for node in nodes)
    )


def cell(kind, source):
    value = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        value.update(outputs=[], execution_count=None)
    return value


def build(package=PACKAGE, destination=None):
    package = Path(package).resolve()
    destination = ROOT / "notebooks/public" / NAME if destination is None else Path(destination).resolve()
    if package.exists() or destination.exists():
        raise ValueError("Use a new immutable package and notebook path")
    config = load_feature_config(ROOT / "configs/experiments/f002-dinov2-attention.json")
    cohort = select_cohort()
    members = {
        f"src/rsna_knee/{name}": (ROOT / "src/rsna_knee" / name).read_bytes()
        for name in (
            "__init__.py",
            "contracts.py",
            "imaging.py",
            "feature_contracts.py",
            "feature_imaging.py",
            "triplet_cache.py",
        )
    }
    members["bounded_feature_reader.py"] = (ROOT / "scripts/bounded_feature_reader.py").read_bytes()
    members["decoder_runtime.py"] = decoder_source().encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in sorted(members.items()):
            item = zipfile.ZipInfo(name, date_time=(2026, 10, 8, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(item, payload)
    encoded, digest = base64.b64encode(buf.getvalue()).decode(), hashlib.sha256(buf.getvalue()).hexdigest()
    member_hashes = {name: hashlib.sha256(payload).hexdigest() for name, payload in members.items()}
    decoder = ROOT / "artifacts/kaggle/dicom-decoders-py313-v2-20261007"
    setup = f"""from pathlib import Path
import base64, hashlib, io, json, os, sys, zipfile
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    raise RuntimeError('Pixel16 diagnostic must never be submitted to the competition')
CODE = Path('/kaggle/working/native392-code')
if CODE.exists():
    raise RuntimeError('Use a fresh saved run; preserve previous outputs')
payload = base64.b64decode({encoded!r}, validate=True)
if hashlib.sha256(payload).hexdigest() != {digest!r}:
    raise ValueError('Embedded code ZIP hash mismatch')
expected = {member_hashes!r}
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    if len(archive.infolist()) != len(expected) or set(archive.namelist()) != set(expected):
        raise ValueError('Code member allowlist mismatch')
    validated = {{}}
    for name in expected:
        target = (CODE / name).resolve()
        if not target.is_relative_to(CODE.resolve()) or '\\\\' in name or ':' in name:
            raise ValueError('Unsafe code ZIP path')
        data = archive.read(name)
        if hashlib.sha256(data).hexdigest() != expected[name]:
            raise ValueError('Embedded code member hash mismatch')
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
        name = Path(path).name.lower()
        if name.endswith('.csv') and '/kaggle/input/' in path.replace('\\\\', '/'):
            if name != 'train_series.csv':
                raise RuntimeError('Teacher, report, test, or submission CSV read is forbidden in this pixel diagnostic')
            CSV_OPENED.append(path)
sys.addaudithook(prohibit_teacher_csv)
print('Code-only package verified; no encoder or teacher CSV is loaded')
"""
    runtime = f"""from decoder_runtime import install_offline_decoders, verify_compressed_phantoms, decoder_inventory, resource_snapshot
from rsna_knee.contracts import dump_json
from rsna_knee.feature_imaging import read_feature_windows
from rsna_knee.triplet_cache import export_pixel_pilot
from bounded_feature_reader import make_bounded_reader
DECODER = Path('/kaggle/input/datasets/rsraki/rsna-dicom-decoders-py313-v1')
if not DECODER.is_dir():
    DECODER = Path('/kaggle/input/rsna-dicom-decoders-py313-v1')
ROOT = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
if not ROOT.is_dir():
    ROOT = Path('/kaggle/input/rsna-knee-abnormality-detection')
if not ROOT.is_dir() or not DECODER.is_dir():
    raise RuntimeError('Attach only Competition and decoder Dataset Version 2')
decoder_install = install_offline_decoders(DECODER / 'wheels', {sha256(decoder / "decoder-manifest.json")!r})
decoder_install['synthetic_checks'] = verify_compressed_phantoms(DECODER / 'phantoms', {sha256(decoder / "phantoms/phantom-manifest.json")!r})
if not all(value['available'] for value in decoder_inventory().values()):
    raise RuntimeError('Required lossless MRI decoders unavailable')
SETTINGS = {config["preprocess"]!r}
COHORT = {cohort!r}
WORKSPACE = Path('/kaggle/working/native392-native-tmp')
WORKSPACE.mkdir()
OUTPUT = Path('/kaggle/working/native392-pixel16')
reader = make_bounded_reader(WORKSPACE)
result = export_pixel_pilot(ROOT, OUTPUT, SETTINGS, COHORT, reader=reader,
                          qc_reader=read_feature_windows, resource_fn=lambda: resource_snapshot('cpu'))
if list(WORKSPACE.iterdir()):
    raise RuntimeError('Native temporary files were not cleaned')
if (Path('/kaggle/working') / 'submission.csv').exists():
    raise RuntimeError('Unexpected submission artifact')
result.update(decoder_install=decoder_install, temporary_files_remaining=0,
              input_csv_reads=sorted(set(CSV_OPENED)), accelerator='CPU',
              storage_reader='disk-backed native intermediate; exact shared preprocessing QC on native volumes <=128MiB')
dump_json(OUTPUT / 'NATIVE392_PIXEL_PILOT_SUMMARY.json', result)
dump_json('/kaggle/working/NATIVE392_PIXEL_PILOT_SUMMARY.json', result)
print({{key: result[key] for key in ('status', 'processed', 'raw_pixel_upper_bytes', 'elapsed_seconds', 'preprocessing_byte_QC_count')}})
print(result['archive'])
"""
    book = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "rsna_pixel_pilot": {
                "studies": 16,
                "for_submission": False,
                "teacher_csv_read": False,
                "code_zip_sha256": digest,
                "cohort_sha256": hashlib.sha256(json.dumps(cohort, sort_keys=True).encode()).hexdigest(),
            },
        },
        "cells": [
            cell(
                "markdown",
                "# native 392px・固定 weak/group 16 検査 pixelcache 診断\n\nCPU・Internet OFF・private 保存実行。Competition と decoder Dataset Version 2 の2 Inputsだけを接続します。教師 CSV と Report は読みません。Gold UID/group と fold0 検証群を除外し、選択 group 全体を保存します。\n\n元 f002 と同じ physical letterbox・3 series × 最大48 adjacent triplets・位置/mask/plane/flags を保持。DINO encoder・学習・予測・submission.csv はありません。約0.989 GiB の raw pixels 上限で、native 一時 memmap＋全体を展開しない圧縮 archive を使います。正解・Public 精度・全件時間はこの診断から判定できません。",
            ),
            cell("code", setup),
            cell("code", runtime),
            cell(
                "markdown",
                "完了条件: complete=true、processed=16、shared preprocessing byte QC >0、temporary_files_remaining=0、入力 CSV は train_series.csv のみ。native392-pixel16.zip と summary を次段の限定 profiler に使用します。Kaggle 提出には使いません。",
            ),
        ],
    }
    for index, item in enumerate(book["cells"]):
        if item["cell_type"] == "code":
            compile("".join(item["source"]), f"{NAME}:{index}", "exec")
    serialized = json.dumps(book, ensure_ascii=False, indent=1) + "\n"
    package.mkdir(parents=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for path in (destination, package / NAME):
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
    (package / "native392-code.zip").write_bytes(buf.getvalue())
    dump_json(package / "cohort.json", cohort)
    review = {
        "status": "prepared_artificial_checks_pending",
        "notebook": str(destination.relative_to(ROOT)),
        "notebook_sha256": sha256(destination),
        "notebook_bytes": destination.stat().st_size,
        "code_zip_sha256": digest,
        "code_member_sha256": member_hashes,
        "studies": 16,
        "raw_pixel_upper_bytes": raw_pixel_bytes(16, config["preprocess"]),
        "archive_budget_bytes": 1024**3,
        "required_inputs": ["rsna-knee-abnormality-detection", "rsraki/rsna-dicom-decoders-py313-v1/2"],
        "accelerator": "CPU",
        "internet": False,
        "private": True,
        "max_run_time_seconds": 9 * 3600,
        "decoder_manifest_sha256": sha256(decoder / "decoder-manifest.json"),
        "phantom_manifest_sha256": sha256(decoder / "phantoms/phantom-manifest.json"),
        "local_builder_structure_only": True,
        "runtime_teacher_CSV_read": False,
        "real_MRI_decode_performed_locally": False,
        "training_performed": False,
        "for_submission": False,
    }
    dump_json(package / "package.json", review)
    print(
        json.dumps(
            {
                key: review[key]
                for key in (
                    "notebook",
                    "notebook_sha256",
                    "notebook_bytes",
                    "raw_pixel_upper_bytes",
                    "required_inputs",
                    "accelerator",
                )
            }
        )
    )
    return review


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=PACKAGE)
    parser.add_argument("--notebook", type=Path)
    args = parser.parse_args()
    build(args.package, args.notebook)
