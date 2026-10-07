"""Prepare an exact-head 09 derivative with three Inputs; no execution or upload."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from rsna_knee.contracts import sha256, source_hashes

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_NAME = "10_submit_f002_embedded_head.ipynb"
BASE_NOTEBOOK = ROOT / "artifacts/kaggle/f002-streaming-v3-20261007/09_submit_f002_streaming.ipynb"
BASE_NOTEBOOK_SHA256 = "26dd4be0b493899d946a5b4ee55202930015cd35f18ecf79188ea19d3000bd07"
HEAD_ZIP = ROOT / "artifacts/kaggle/f002-fold0-single-v1-20261006/rsna-f002-head-fold0-v1.zip"
HEAD_ZIP_SHA256 = "3cce03fda576e4af3d6e36b9dee99ab541afc6949dd807f81f7bdf5b44dad24a"
HEAD_SHA256 = "189ecab25e0d1c029e974ee23f3c46239e5eceef290b5dca33f6831dcb8e1b03"
RUNTIME_SHA256 = "a0d13ea7a93916dc6e7acd52ffe68a9a09e21bbe889d830746f7f9ed46d87f37"
PACKAGE = ROOT / "artifacts/kaggle/f002-embedded-head-v1-20261007"
REQUIRED_INPUTS = [
    {"kind": "competition", "reference": "rsna-knee-abnormality-detection"},
    {"kind": "dataset", "reference": "rsraki/rsna-frozen-feature-inputs-v1"},
    {"kind": "dataset", "reference": "rsraki/rsna-dicom-decoders-py313-v1"},
]


def code_cell(source):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


def make_notebook():
    """Check immutable inputs and return a notebook without modifying any source artifact."""
    if sha256(BASE_NOTEBOOK) != BASE_NOTEBOOK_SHA256 or sha256(HEAD_ZIP) != HEAD_ZIP_SHA256:
        raise ValueError("Immutable 09 notebook/head archive changed")
    run = json.loads((ROOT / "artifacts/runs/f002-dinov2-attention-fold0/run.json").read_text(encoding="utf-8"))
    if run["source_sha256"] != source_hashes():
        raise ValueError("Baseline core implementation changed")
    if run["selected"]["checkpoint_sha256"] != HEAD_SHA256:
        raise ValueError("Selected f002 fold0 head changed")
    if sha256(ROOT / "scripts/frozen_streaming_runtime.py") != RUNTIME_SHA256:
        raise ValueError("Audited streaming runtime changed")
    with zipfile.ZipFile(HEAD_ZIP) as archive:
        if set(archive.namelist()) != {"best.pt", "config.json", "checkpoint-provenance.json"}:
            raise ValueError("Unexpected head archive member")
        original_bytes = {name: archive.read(name) for name in archive.namelist()}
    if hashlib.sha256(original_bytes["best.pt"]).hexdigest() != HEAD_SHA256:
        raise ValueError("Archive head differs from selected checkpoint")
    if original_bytes["best.pt"] != (ROOT / "artifacts/runs/f002-dinov2-attention-fold0/best.pt").read_bytes():
        raise ValueError("Local checkpoint differs from the original delivered head")
    payload = {
        name: {
            "base64": base64.b64encode(raw).decode("ascii"),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        for name, raw in original_bytes.items()
    }
    book = copy.deepcopy(json.loads(BASE_NOTEBOOK.read_text(encoding="utf-8")))
    description = """# f002提出候補：元head Datasetの接続を外した10

新しいprivate NotebookへImportし、T4×2・Internet OFFにします。Inputsは次の3つだけです。

1. rsna-knee-abnormality-detection Competition
2. rsraki/rsna-frozen-feature-inputs-v1（従来のencoder/code）
3. rsraki/rsna-dicom-decoders-py313-v1（従来のdecoder）

**rsraki/rsna-f002-head-fold0-v1は接続しないでください。** 接続したままでは、Python開始前の同じmount失敗が残ります。旧09に上書きせず、新しいNotebookで3 Inputsを確認してください。

元head ZIPのbest.pt・config.json・checkpoint-provenance.jsonをこのNotebookに埋め込み、全件のサイズ/SHA256検証後にworkingへ復元します。fold0・弱ラベルBCE選択epoch11・前処理・encoder・09の推論コードは同一です。これはheadの配送方法だけの変更で、精度を改善する新モデルではありません。

256件診断は08 Version2で完了済みです。10をSave & Run Allし、F002_STREAMING_SUMMARY.jsonのstatus=passed、phase=complete、processed=studies、receipt.valid=trueを確認します。その保存版submission.csvをユーザーが手動提出します。08のprofile_predictions.csvは提出しません。隠し推論の成功やPublicは未確認です。残りInputsのKaggle側mount障害を防ぐものではありません。
"""
    book["cells"][0]["source"] = description.splitlines(keepends=True)
    helper = (ROOT / "scripts/embedded_head_payload.py").read_text(encoding="utf-8")
    embedded = helper + "\n\nEMBEDDED_HEAD_PAYLOAD = " + repr(payload) + "\n" + """try:
    startup['phase'] = 'materialize_embedded_head'
    save_startup()
    head_delivery = materialize_embedded_head(EMBEDDED_HEAD_PAYLOAD, Path('/kaggle/working/embedded-f002-head'))
    EMBEDDED_CHECKPOINT = Path(head_delivery['checkpoint_path'])
    startup.update(head_delivery=head_delivery, phase='embedded_head_ready')
    save_startup()
    print(json.dumps(head_delivery, ensure_ascii=False, indent=2))
except Exception as error:
    startup.update(status='failed', error=str(error), traceback=traceback.format_exc())
    save_startup()
    raise
"""
    book["cells"].insert(2, code_cell(embedded))
    final = "".join(book["cells"][-1]["source"])
    prefix = """HEAD_INPUT = Path('/kaggle/input/datasets/rsraki/rsna-f002-head-fold0-v1')
if not HEAD_INPUT.is_dir():
    HEAD_INPUT = Path('/kaggle/input/rsna-f002-head-fold0-v1')
"""
    if not final.startswith(prefix) or final.count("HEAD_INPUT / 'best.pt'") != 1:
        raise ValueError("Original 09 head binding differs from the audited source")
    final = final.removeprefix(prefix).replace("HEAD_INPUT / 'best.pt'", "EMBEDDED_CHECKPOINT")
    final = final.replace("    result['decoder_install'] = decoder_install\n", "    result['decoder_install'] = decoder_install\n    result['head_delivery'] = head_delivery\n")
    final = final.replace("    result.update(status='failed', error=", "    result['head_delivery'] = head_delivery\n    result.update(status='failed', error=")
    book["cells"][-1] = code_cell(final)
    book["metadata"]["rsna_manual_inputs"] = REQUIRED_INPUTS
    book["metadata"]["rsna_delivery"] = {
        "mode": "embedded_notebook_bytes",
        "head_dataset_required": False,
        "checkpoint_sha256": HEAD_SHA256,
        "base_09_notebook_sha256": BASE_NOTEBOOK_SHA256,
    }
    for index, item in enumerate(book["cells"]):
        if item["cell_type"] == "code":
            source = "".join(item["source"])
            compile(source, f"{NOTEBOOK_NAME}:{index}", "exec")
            if "rsna-f002-head-fold0-v1" in source or "HEAD_INPUT" in source:
                raise ValueError("External head Input binding remains in notebook code")
    return book, original_bytes, run


def prepare():
    public = ROOT / "notebooks/public" / NOTEBOOK_NAME
    if public.exists() or PACKAGE.exists():
        raise ValueError("Use a new package/notebook name; preserve existing artifacts")
    book, original_bytes, run = make_notebook()
    serialized = json.dumps(book, ensure_ascii=False, indent=1) + "\n"
    serialized_bytes = len(serialized.encode("utf-8"))
    if serialized_bytes >= 1_000_000:
        raise ValueError("Keep embedded notebook source below the conservative 1,000,000-byte preparation limit")
    PACKAGE.mkdir()
    for path in (public, PACKAGE / NOTEBOOK_NAME):
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
    for name in ("config.json", "checkpoint-provenance.json"):
        with (PACKAGE / name).open("xb") as handle:
            handle.write(original_bytes[name])
    input_manifest = {
        "required_inputs": REQUIRED_INPUTS,
        "excluded_input": "rsraki/rsna-f002-head-fold0-v1",
        "accelerator": "GPU T4 x2",
        "internet_enabled": False,
        "manual_attachment_required": True,
        "manual_submission_required": True,
        "head_dataset_required": False,
    }
    package = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "prepared_unexecuted",
        "change": "Head delivery from separate Dataset mount to byte-identical embedded notebook payload",
        "notebook_sha256": sha256(public),
        "notebook_bytes": public.stat().st_size,
        "conservative_notebook_size_limit_bytes": 1_000_000,
        "base_09_notebook_sha256": BASE_NOTEBOOK_SHA256,
        "original_head_zip_sha256": HEAD_ZIP_SHA256,
        "checkpoint_sha256": HEAD_SHA256,
        "embedded_files": {
            name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            for name, raw in original_bytes.items()
        },
        "runtime_sha256": RUNTIME_SHA256,
        "materializer_sha256": sha256(ROOT / "scripts/embedded_head_payload.py"),
        "builder_sha256": sha256(Path(__file__)),
        "core_sources_unchanged": source_hashes(),
        "feature_fingerprint": run["feature_fingerprint"],
        "fold": 0,
        "epoch": 11,
        "selection_data": "weak_validation_BCE",
        "required_inputs": REQUIRED_INPUTS,
        "head_dataset_required": False,
        "gold_used_for_training_or_selection": False,
        "report_required_for_inference": False,
        "new_download_or_real_mri_decode_or_training_or_upload_or_submission": False,
        "private_rerun_success_verified": False,
        "public_lb": None,
        "limitations": [
            "Other attached Inputs can still fail before Python starts",
            "Earlier hidden inference exception remains unconfirmed",
            "Input attachments require manual removal of the old head Dataset; metadata alone does not detach it",
        ],
    }
    for name, value in (("manual-input-manifest.json", input_manifest), ("package.json", package)):
        with (PACKAGE / name).open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    print(json.dumps({"prepared": str(public), "notebook_bytes": public.stat().st_size, "checkpoint_sha256": HEAD_SHA256}))


if __name__ == "__main__":
    prepare()
