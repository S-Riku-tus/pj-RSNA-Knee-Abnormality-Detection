"""Generate an unexecuted Kaggle CPU cache template for local GPU training."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    result = {
        "cell_type": kind,
        "metadata": {},
        "source": (source.rstrip("\n") if kind == "code" else source).splitlines(keepends=True),
    }
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


cells = [
    cell(
        "markdown",
        """# RSNA Knee ローカル学習用キャッシュ作成

未実行テンプレートです。Kaggleの競技Inputを直接読み、自作KneeMIL用の画像キャッシュを作ります。元DICOMのダウンロード、学習、アップロード、競技提出は行いません。CPUで実行でき、GPU枠は不要です。

まずLIMIT=10で確認し、画像を目視してからLIMIT=Noneで全件を作ります。NotebookとOutputはprivateで扱ってください。出力には検査IDと読影レポートが含まれます。公開CoAtNet/DINOのキャッシュとは互換性がありません。
""",
    ),
    cell(
        "code",
        """import time
from pathlib import Path

started = time.perf_counter()
CODE_ZIP = Path("/kaggle/input/datasets/YOUR_USERNAME/rsna-knee-code/rsna-knee-code.zip")
WHEELS = None  # Optional attached Linux/Python-compatible decoder wheels.
LIMIT = 10  # Set to None only after inspecting the ten-study cache.
EXPORT_NAME = "rsna-cache-v1"  # Use a new name for changed code/config/inputs.
RESUME_CACHE = None  # Optional Path to the train-v1 directory of a prior private Output Input.
INPUT_VERSION = "TODO"  # Record the competition Input snapshot/date shown in your session.

WORK = Path("/kaggle/working")
assert Path(EXPORT_NAME).name == EXPORT_NAME and EXPORT_NAME not in ("", ".", "..")
EXPORT = WORK / EXPORT_NAME
candidates = [
    Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
    Path("/kaggle/input/rsna-knee-abnormality-detection"),
]
DATA_ROOT = next((p for p in candidates if (p / "train.csv").is_file()), None)
assert DATA_ROOT is not None, "Attach the competition Input after accepting its rules"
assert CODE_ZIP.is_file(), "Attach the repository code bundle and edit CODE_ZIP"
assert INPUT_VERSION.strip() and INPUT_VERSION != "TODO", "Record the Input snapshot/date"
""",
    ),
    cell(
        "code",
        """import importlib.util
import shutil
import subprocess
import sys
import zipfile

if WHEELS is not None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(WHEELS),
            "numpy",
            "pydicom",
            "Pillow",
            "pylibjpeg",
            "pylibjpeg-libjpeg",
            "pylibjpeg-openjpeg",
        ],
        check=True,
    )
for name in ["numpy", "pydicom", "PIL"]:
    assert importlib.util.find_spec(name), f"Missing {name}; prepare decoder dependencies"

# Preserve the exact preprocessing source alongside the cache for the Windows GPU terminal.
code_dir = EXPORT / "code"
with zipfile.ZipFile(CODE_ZIP) as archive:
    for entry in archive.namelist():
        (code_dir / entry).resolve().relative_to(code_dir.resolve())
    archive.extractall(code_dir)
sys.path.insert(0, str(code_dir / "src"))
""",
    ),
    cell(
        "code",
        """from rsna_knee.contracts import ID, dump_json, load_config, read_csv, sha256, source_hashes
from rsna_knee.imaging import build_cache, load_cached
from rsna_knee.prepare import audit

config = load_config(code_dir / "configs" / "baseline.json")
summary = audit(DATA_ROOT, EXPORT / "audit.json")
_, studies = read_csv(DATA_ROOT / "train.csv", (ID,))
requested = studies[:LIMIT] if LIMIT is not None else studies
p = config["preprocess"]
image_bytes = len(requested) * p["max_series"] * p["windows_per_series"] * 3 * p["image_size"] ** 2
assert image_bytes < 18_000_000_000, "Split the job or reduce input size before exceeding Output storage"
assert shutil.disk_usage(WORK).free > image_bytes + 1_000_000_000, "Insufficient working space"
cache_dir = EXPORT / "train-v1"
if RESUME_CACHE is not None and not cache_dir.exists():
    # Previous Output remains read-only; build_cache checks input hashes and preprocessing.
    shutil.copytree(RESUME_CACHE, cache_dir)

print({"csv_audit": summary, "uncompressed_image_bytes": image_bytes})
cache_result = build_cache(DATA_ROOT, "train", cache_dir, config, limit=LIMIT)
print(cache_result)
""",
    ),
    cell(
        "code",
        """import matplotlib.pyplot as plt

images, mask = load_cached(cache_dir, requested[0][ID], config)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for axis, window in zip(axes, images[mask][:3]):
    axis.imshow(window[1], cmap="gray")
    axis.axis("off")
plt.show()
# Review private coverage.json and compare with the corresponding original DICOM.
# This preview alone does not establish clinical information retention.
""",
    ),
    cell(
        "code",
        """import importlib.metadata

expected = {row[ID] for row in requested}
actual = {path.stem for path in cache_dir.glob("*.npz")}
assert expected <= actual, "Incomplete cache; review coverage.json before saving"
for key in sorted(expected):
    load_cached(cache_dir, key, config)
raw_dir = EXPORT / "raw"
raw_dir.mkdir(exist_ok=True)
for name in ["train.csv", "train_series.csv", "sample_submission.csv"]:
    shutil.copy2(DATA_ROOT / name, raw_dir / name)

# Avoid a second multi-GB archive inside /kaggle/working. Download the saved Output via the UI.
files = {
    path.relative_to(EXPORT).as_posix(): {"sha256": sha256(path), "bytes": path.stat().st_size}
    for path in sorted(EXPORT.rglob("*"))
    if path.is_file() and path.name != "export.json" and "__pycache__" not in path.parts
}
record = {
    "schema_version": 1,
    "complete": LIMIT is None and actual == {r[ID] for r in studies},
    "studies_requested": len(requested),
    "studies_total": len(studies),
    "debug_limit": LIMIT,
    "competition_input_version": INPUT_VERSION,
    "code_zip_sha256": sha256(CODE_ZIP),
    "source_sha256": source_hashes(),
    "config": config,
    "cache": cache_result,
    "runtime": {name: importlib.metadata.version(name) for name in ["numpy", "pydicom", "Pillow"]},
    "elapsed_seconds": time.perf_counter() - started,
    "output_bytes": sum(entry["bytes"] for entry in files.values()),
    "files": files,
}
dump_json(EXPORT / "export.json", record)
print({key: record[key] for key in ["complete", "studies_requested", "output_bytes", "elapsed_seconds"]})
assert record["output_bytes"] < 18_000_000_000, "Check the current Kaggle Output storage limit"
print("Save the private Notebook version, then download its Output. Record the saved version separately.")
""",
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for number, entry in enumerate(cells):
    entry["id"] = f"rsna-cache-{number}"
destination = ROOT / "notebooks" / "00_prepare_cache.ipynb"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
print(destination)
