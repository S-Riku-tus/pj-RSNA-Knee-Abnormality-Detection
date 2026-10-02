"""Generate the unexecuted Kaggle inference template; keep cell outputs empty."""

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
        """# RSNA Knee offline submission

自作KneeMIL checkpoint用。未実行テンプレートです。競技入力、コードzip、checkpointをAttachし、下のパスを編集してください。GPU有効・Internet無効で実行します。公開CoAtNet/DINO重みはこのNotebookに対応しません。
""",
    ),
    cell(
        "code",
        """import time
from pathlib import Path

notebook_started = time.perf_counter()

CODE_ZIP = Path("/kaggle/input/datasets/YOUR_USERNAME/rsna-knee-code/rsna-knee-code.zip")
CHECKPOINT = Path("/kaggle/input/datasets/YOUR_USERNAME/rsna-knee-checkpoint/best.pt")
WHEELS = None  # Optional Path to Linux/Python-compatible wheels attached as an Input.

candidates = [
    Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
    Path("/kaggle/input/rsna-knee-abnormality-detection"),
]
DATA_ROOT = next((p for p in candidates if (p / "test.csv").exists()), None)
assert DATA_ROOT is not None, "Attach the competition Input"
assert CODE_ZIP.is_file() and CHECKPOINT.is_file(), "Edit code/checkpoint Input paths"
WORK = Path("/kaggle/working")
""",
    ),
    cell(
        "code",
        """import importlib.util
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

for name in ["numpy", "pydicom", "PIL", "torch", "torchvision"]:
    assert importlib.util.find_spec(name), f"Missing {name}; attach compatible wheels"

code_dir = WORK / "rsna-code"
with zipfile.ZipFile(CODE_ZIP) as archive:
    for entry in archive.namelist():
        (code_dir / entry).resolve().relative_to(code_dir.resolve())
    archive.extractall(code_dir)
sys.path.insert(0, str(code_dir / "src"))
""",
    ),
    cell(
        "code",
        """import torch

from rsna_knee.contracts import SUBMISSION_COLUMNS, dump_json, read_csv
from rsna_knee.imaging import build_cache
from rsna_knee.runtime import predict

assert torch.cuda.is_available(), "Enable a CUDA GPU"
checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
config = checkpoint["config"]
header, _ = read_csv(DATA_ROOT / "sample_submission.csv", SUBMISSION_COLUMNS)
assert tuple(header) == SUBMISSION_COLUMNS, "Live competition target schema changed"
cache_dir = WORK / "test-cache"
coverage = build_cache(DATA_ROOT, "test", cache_dir, config)
del checkpoint
print(coverage)
""",
    ),
    cell(
        "code",
        """from rsna_knee.contracts import validate_submission

result = predict(DATA_ROOT / "test.csv", cache_dir, CHECKPOINT, WORK / "submission.csv", "cuda")
validate_submission(WORK / "submission.csv", DATA_ROOT / "test.csv", DATA_ROOT / "sample_submission.csv")
result["notebook_seconds"] = time.perf_counter() - notebook_started
dump_json(WORK / "submission.meta.json", result)
print(result)
assert result["notebook_seconds"] < 9 * 3600, "Notebook exceeds competition runtime limit"
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
    entry["id"] = f"rsna-{number}"
destination = ROOT / "notebooks" / "01_submit.ipynb"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
print(destination)
