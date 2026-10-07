"""Prepare profile/submission notebooks with the audited f002 weights; no uploads or execution."""

from __future__ import annotations

import json
from pathlib import Path

from rsna_knee.contracts import dump_json, sha256, source_hashes

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(outputs=[], execution_count=None)
    return result


def notebooks():
    original = json.loads(
        (ROOT / "artifacts/kaggle/f002-fold0-single-v1-20261006/07_predict_f002_fold0_single.ipynb").read_text(
            encoding="utf-8"
        )
    )
    codes = ["".join(item["source"]) for item in original["cells"] if item["cell_type"] == "code"]
    run_path = ROOT / "artifacts/runs/f002-dinov2-attention-fold0/run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run["source_sha256"] != source_hashes():
        raise ValueError("Baseline core implementation has changed")
    runtime = (ROOT / "scripts/frozen_streaming_runtime.py").read_text(encoding="utf-8")
    decoder_dir = ROOT / "artifacts/kaggle/dicom-decoders-py313-v2-20261007"
    decoder_hash = sha256(decoder_dir / "decoder-manifest.json")
    phantom_hash = sha256(decoder_dir / "phantoms/phantom-manifest.json")
    start = (
        """from pathlib import Path
import json, traceback, os
SUMMARY_PATH = Path('/kaggle/working/F002_STREAMING_SUMMARY.json')
startup = {'status': 'started', 'phase': 'load_code', 'public_lb': None, 'training_performed': False}
def save_startup():
    SUMMARY_PATH.write_text(json.dumps(startup, ensure_ascii=False, indent=2), encoding='utf-8')
save_startup()
try:
"""
        + "\n".join("    " + line for line in (codes[0] + "\n" + codes[1]).splitlines())
        + """
    startup.update(phase='code_ready', gpu=torch.cuda.get_device_name(0))
    save_startup()
except Exception as error:
    startup.update(status='failed', error=str(error), traceback=traceback.format_exc())
    save_startup()
    raise
"""
    )
    wheels = """DECODER_INPUT = Path('/kaggle/input/datasets/rsraki/rsna-dicom-decoders-py313-v1')
if not DECODER_INPUT.is_dir():
    DECODER_INPUT = Path('/kaggle/input/rsna-dicom-decoders-py313-v1')
try:
    startup['phase'] = 'offline_decoder_install'
    save_startup()
    if not DECODER_INPUT.is_dir():
        raise RuntimeError('Attach the new private rsna-dicom-decoders-py313-v1 Dataset; do not upload model weights again')
    decoder_install = install_offline_decoders(DECODER_INPUT / 'wheels', __MANIFEST_HASH__)
    startup['phase'] = 'synthetic_compressed_decoder_check'
    save_startup()
    decoder_install['synthetic_pixel_checks'] = verify_compressed_phantoms(DECODER_INPUT / 'phantoms', __PHANTOM_HASH__)
    startup.update(decoder_install=decoder_install, decoders=decoder_inventory(), phase='decoders_ready')
    save_startup()
except Exception as error:
    startup.update(status='failed', error=str(error), traceback=traceback.format_exc())
    save_startup()
    raise
""".replace("__MANIFEST_HASH__", repr(decoder_hash)).replace("__PHANTOM_HASH__", repr(phantom_hash))
    result = {}
    for name, mode in (
        ("08_profile_f002_serving_256.ipynb", "profile_train"),
        ("09_submit_f002_streaming.ipynb", "test"),
    ):
        title = "f002の256検査診断・提出不可" if mode == "profile_train" else "f002の検査ごと推論・診断付き提出候補"
        description = f"# {title}\n\nT4 GPU・Internet OFF・新しいprivate Notebook。既存Competition/encoder-code/headに、新しいdecoder Input一つを追加。前処理・encoder・fold0 epoch11の重みを保持。\n\n"
        description += (
            "学習は行わず、goldを含まない固定weakリストから256検査を診断します。`submission.csv`は作らず、Competitionへ提出しません。"
            if mode == "profile_train"
            else "公開test全件→隠し再実行時のtest全件を処理します。08の診断と09可視実行を確認してから、ユーザーが手動提出します。隠し例外の解消やPublicは未確認です。"
        )
        final = """HEAD_INPUT = Path('/kaggle/input/datasets/rsraki/rsna-f002-head-fold0-v1')
if not HEAD_INPUT.is_dir():
    HEAD_INPUT = Path('/kaggle/input/rsna-f002-head-fold0-v1')
MODE = __MODE__
PROFILE_LIMIT = 256
PROFILE_SEED = 20261007
DETAILS = Path('/kaggle/working/f002-streaming-diagnostics')
try:
    if MODE == 'profile_train' and os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        raise RuntimeError('Profile notebook is not for competition submission')
    result = run_streaming(
        ROOT, HEAD_INPUT / 'best.pt', provenance, ENCODER_REPO, ENCODER_WEIGHTS, DETAILS,
        mode=MODE, weak_manifest=INPUT / 'manifest/weak.csv' if MODE == 'profile_train' else None,
        profile_limit=PROFILE_LIMIT, selection_seed=PROFILE_SEED,
        expected_checkpoint_sha256=__HEAD_HASH__, expected_sources=__SOURCES__,
        expected_weak_sha256=__WEAK_HASH__, device='cuda')
    result['decoder_install'] = decoder_install
    SUMMARY_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
except Exception as error:
    saved = DETAILS / 'F002_STREAMING_SUMMARY.json'
    result = json.loads(saved.read_text(encoding='utf-8')) if saved.exists() else dict(startup)
    result.update(status='failed', error=str(error), traceback=traceback.format_exc(), decoder_install=decoder_install)
    SUMMARY_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    raise
"""
        for key, value in {
            "__MODE__": mode,
            "__HEAD_HASH__": run["selected"]["checkpoint_sha256"],
            "__SOURCES__": run["source_sha256"],
            "__WEAK_HASH__": run["manifest_sha256"]["weak.csv"],
        }.items():
            final = final.replace(key, repr(value))
        result[name] = {
            "cells": [
                cell("markdown", description),
                cell("code", start),
                cell("code", runtime),
                cell("code", wheels),
                cell("code", final),
            ],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
    return result


if __name__ == "__main__":
    destination = ROOT / "artifacts/kaggle/f002-streaming-v3-20261007"
    books = notebooks()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Use a new package directory; preserve the previous package")
    destination.mkdir(exist_ok=True)
    hashes = {}
    for name, book in books.items():
        for index, item in enumerate(book["cells"]):
            if item["cell_type"] == "code":
                compile("".join(item["source"]), f"{name}:{index}", "exec")
        serialized = json.dumps(book, ensure_ascii=False, indent=1) + "\n"
        for path in (ROOT / "notebooks/public" / name, destination / name):
            mode = "x"
            if path.parent == ROOT / "notebooks/public" and path.exists():
                previous = ROOT / "artifacts/kaggle/f002-streaming-v2-20261007" / name
                if path.read_bytes() != previous.read_bytes():
                    raise ValueError("Public notebook changed outside this preparation; preserve user edits")
                mode = "w"
            with path.open(mode, encoding="utf-8", newline="\n") as handle:
                handle.write(serialized)
        hashes[name] = sha256(destination / name)
    dump_json(
        destination / "package.json",
        {
            "status": "prepared_unexecuted",
            "notebooks_sha256": hashes,
            "runtime_sha256": sha256(ROOT / "scripts/frozen_streaming_runtime.py"),
            "core_sources_unchanged": source_hashes(),
            "training_started": False,
            "decoder_input": "rsraki/rsna-dicom-decoders-py313-v1",
            "hidden_failure_cause_confirmed": False,
        },
    )
    print(json.dumps({"prepared": list(hashes), "training_started": False}))
