"""Build CPU header-only same32 Notebook offline; no DICOM/Torch/model/API use."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "artifacts/kaggle/orthodiff-sag-mri32-pilot-20261017-v1"
RECEIPT = ROOT / "artifacts/qa/odm1/20261017T034429492163Z/files/ORTHODIFF_MRI_PILOT.json"
RUNTIME = ROOT / "scripts/orthodiffusion_header32_runtime.py"
OUTPUT = ROOT / "artifacts/kaggle/orthodiff-sag-header32-20261017-v1"
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"


def pin(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def prepare(output=OUTPUT):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Fresh header-only diagnostic package required")
    original = json.loads((PILOT / "config.json").read_bytes())
    receipt_raw = RECEIPT.read_bytes()
    receipt = json.loads(receipt_raw)
    if (hashlib.sha256(receipt_raw).hexdigest() != "f033452c022f1c292b810f639b3de59a9ef0455744047bc989ec88d92f102f75"
            or receipt["complete"] is not True or receipt["coverage"]["success"] != 24
            or receipt["requested_UIDs"] != original["uids"]):
        raise ValueError("Exact completed24/32 parent pilot UID binding required")
    config = {key: original[key] for key in ("uids", "UID_order_sha256", "competition_candidates", "series_metadata_pin")}
    config["parent_pilot_receipt_pin"] = pin(receipt_raw)
    raw = RUNTIME.read_bytes()
    tree = ast.parse(raw)
    compile(tree, "header32-runtime.py", "exec")
    imports = [ast.unparse(node) for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    if any("torch" in value or "numpy" in value for value in imports):
        raise ValueError("Header-only runtime cannot import scientific model libraries")
    cell = ("# CPU header-only diagnostic: fixed32, one header/Sagseries, no pixels/model/PD rule changes.\n"
            "import hashlib as _h\n"
            f"_raw={raw!r}\n"
            f"assert {{'bytes':len(_raw),'sha256':_h.sha256(_raw).hexdigest()}}=={pin(raw)!r}\n"
            "_ns={'__name__':'orthodiff_header32_candidate'}\nexec(compile(_raw,'<pinned-header-only-runtime>','exec'),_ns)\n"
            f"_result=_ns['run']({config!r})\n"
            "if not _result['complete']:\n    raise RuntimeError('Header-only diagnostic failed; inspect ORTHODIFF_HEADER32.json')\n")
    nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": "# Same32 sagittal header-only diagnostic\n\nCPU/private/offline, competition Input only. Reads one lexicographically first DICOM header per sagittal series with stop_before_pixels=True and an explicit tag allowlist. No PixelData, pixel decode, model, GPU, Gold/weak targets, Report, classifier changes, training or submission. TR/TE/sequence/vendor fields describe unknown-PD cases and known-PD controls; no new PD rule is applied.\n"},
                    {"cell_type": "code", "metadata": {}, "source": cell, "outputs": [], "execution_count": None}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "kaggle": {"isGpuEnabled": False, "isInternetEnabled": False}}, "nbformat": 4, "nbformat_minor": 5}
    source = (json.dumps(nb, indent=2) + "\n").encode()
    payload = {"slug": "rsraki/orthodiff-sag-header32-20261017-v1", "new_title": "orthodiff-sag-header32-20261017-v1", "text": source.decode(),
               "language": "python", "kernel_type": "notebook", "competition_data_sources": ["rsna-knee-abnormality-detection"],
               "dataset_data_sources": [], "kernel_data_sources": [], "is_private": True, "enable_gpu": False, "enable_internet": False,
               "docker_image": DOCKER, "docker_image_pinning_type": "original", "kernel_execution_type": "SAVE_AND_RUN_ALL"}
    manifest = {"schema_version": "orthodiff_same32_header_only_offline_v1", "source": pin(source), "runtime": pin(raw), "config": config,
                "AST_import_inventory": imports, "PDeligibility_changed": False, "scope": "header allowlist only, one header per official Sagittal series; no raw pixels or fullfile hashing",
                "local_activity": {"DICOM_model_Torch_GPU_network_API_Save_Submit": 0}}
    output.mkdir(parents=True)
    values = {"source.ipynb": source, "runtime.py": raw}
    values.update({name: (json.dumps(value, indent=2) + "\n").encode() for name, value in
                   (("save-payload.json", payload), ("manifest.json", manifest), ("config.json", config))})
    for name, content in values.items():
        (output / name).write_bytes(content)
    return {"output": str(output), "source": pin(source), "payload": pin(values["save-payload.json"]), "manifest": pin(values["manifest.json"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    print(json.dumps(prepare(parser.parse_args().output), indent=2))
