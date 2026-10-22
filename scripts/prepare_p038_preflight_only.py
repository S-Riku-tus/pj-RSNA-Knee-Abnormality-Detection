"""Offline CPU-only P038 mount/hash/decoder preflight pack; no science execution."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "artifacts/kaggle/p038-helios-v6-fixed-recipe-20261017-v3"
OUTPUT = ROOT / "artifacts/kaggle/p038-helios-preflight7-cpu-20261017-v1"
PARENT_SOURCE = "9e3669dd1542f0cde2457fcc9b54ccb4bac013832583c26f93b88711f1a87370"
PARENT_MANIFEST = "ae74c8a7a670ae8840cf269b315299657760bf8492275c2e52bd5e6fefa5886b"
PARENT_PAYLOAD = "63a55ad4de1fe5330445126d0edba159f158b0da6ea86b388bb2ba0a2e9550ac"
SLUG = "rsraki/p038c-preflight7-20261017-v1"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def pinned(path, expected):
    raw = Path(path).read_bytes()
    if digest(raw) != expected:
        raise ValueError("Fixed P038 parent artifact differs: " + str(path))
    return raw


def prepare(output=OUTPUT, *, parent=PARENT, slug=SLUG):
    output, parent = Path(output), Path(parent)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a fresh CPU preflight run directory")
    if not re.fullmatch(r"rsraki/[a-z0-9][a-z0-9-]{4,49}", slug):
        raise ValueError("Expected own private candidate slug")
    parent_source = pinned(parent / "source.ipynb", PARENT_SOURCE)
    manifest = json.loads(pinned(parent / "manifest.json", PARENT_MANIFEST))
    payload = json.loads(pinned(parent / "save-payload.json", PARENT_PAYLOAD))
    if payload["text"].encode() != parent_source or manifest["inputs"]["total"] != 7:
        raise ValueError("P038 source/payload/seven-Input binding differs")
    runtime = pinned(parent / "operational-runtime.py", manifest["operational_runtime"]["sha256"]).decode()
    config = manifest["guard_config"]
    text = ("import hashlib as _p038c_hash, json as _p038c_json, sys as _p038c_sys\n"
            "import tempfile as _p038c_temp, traceback as _p038c_trace\n"
            "from pathlib import Path as _p038c_Path\n"
            f"_p038c_runtime = {runtime!r}\n"
            f"if _p038c_hash.sha256(_p038c_runtime.encode()).hexdigest() != {manifest['operational_runtime']['sha256']!r}:\n"
            "    raise ValueError('Exact P038 operational runtime SHA differs')\n"
            "_p038c_runtime_ns = {'__name__': 'p038_operational_runtime'}\n"
            "exec(compile(_p038c_runtime, '<p038-preflight-runtime-only>', 'exec'), _p038c_runtime_ns)\n"
            f"_p038c_config = {config!r}\n"
            "_p038c_result = {'schema_version':'p038_CPU_only_actual_preflight_v1', 'complete':False,\n"
            " 'scope':'mounted public checkpoints plus existing own four synthetic decoder phantoms; no scientific inference',\n"
            " 'python':_p038c_sys.version, 'GPU_calls':0, 'model_deserialization_or_forward':0,\n"
            " 'competition_MRI_Report_Gold_reads':0, 'scientific_code_cells_run':0, 'Save_Submit_calls':0,\n"
            " 'actual_mounted_Input_version_independently_verified':False}\n"
            f"_p038c_result['P038_parent_source_sha256'] = {PARENT_SOURCE!r}\n"
            f"_p038c_result['operational_runtime_sha256'] = {manifest['operational_runtime']['sha256']!r}\n"
            "try:\n"
            "    with _p038c_temp.TemporaryDirectory(prefix='p038c-preflight-') as _p038c_work:\n"
            "        try:\n"
            "            _p038c_result['preflight'] = _p038c_runtime_ns['preflight'](_p038c_config, work=_p038c_work)\n"
            "        except Exception:\n"
            "            _p038c_partial = _p038c_Path(_p038c_work) / 'P038' / 'preflight.json'\n"
            "            if _p038c_partial.is_file():\n"
            "                _p038c_partial_bytes = _p038c_partial.read_bytes()\n"
            "                _p038c_result['partial_preflight'] = _p038c_json.loads(_p038c_partial_bytes)\n"
            "                _p038c_result['partial_preflight_sha256'] = _p038c_hash.sha256(_p038c_partial_bytes).hexdigest()\n"
            "            raise\n"
            "    _p038c_result['complete'] = True\n"
            "except Exception as _p038c_error:\n"
            "    _p038c_result.update(error_type=type(_p038c_error).__name__, error=str(_p038c_error)[:4000],\n"
            "                         traceback=_p038c_trace.format_exc()[-8000:],\n"
            "                         stderr=str(getattr(_p038c_error,'stderr',''))[-8000:])\n"
            "_p038c_output = _p038c_Path('/kaggle/working/P038_PREFLIGHT.json')\n"
            "with _p038c_output.open('x',encoding='utf-8') as _p038c_stream:\n"
            "    _p038c_json.dump(_p038c_result,_p038c_stream,indent=2,allow_nan=False)\n"
            "    _p038c_stream.write('\\n')\n"
            "print('P038_CPU_PREFLIGHT_RESULT',_p038c_json.dumps({'complete':_p038c_result['complete'],\n"
            "       'error_type':_p038c_result.get('error_type'),'scientific_code_cells_run':0}),flush=True)\n")
    ast.parse(text)
    book = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": (
        "# P038 CPU-only fixed Input and Reader decoder preflight\n\n"
        "Uses the exact P038 v3 operational guard and seven fixed Inputs. No author code cell is restored or run. "
        "Reads 12 mounted public checkpoint/manifest files only for SHA/size checks, installs offline decoder "
        "wheels into a private temporary directory and decodes only the existing own four pinned synthetic phantoms. "
        "Reader V1 has no author fixtures; this seventh Input supplies separate operational evidence. GPU OFF; "
        "no model deserialization/forward, competition MRI, Report, Gold, training, Save or Submit. "
        "One JSON output records success/failure, Python and positive fixture evidence; this is no accuracy qualification.\n")},
        {"cell_type": "code", "metadata": {}, "source": text, "execution_count": None, "outputs": []}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "kaggle": {"isGpuEnabled": False, "isInternetEnabled": False}}, "nbformat": 4, "nbformat_minor": 5}
    raw = (json.dumps(book, ensure_ascii=False, indent=1) + "\n").encode()
    new_payload = copy.deepcopy(payload)
    new_payload.update(slug=slug, new_title=slug.split("/")[1], text=raw.decode(), enable_gpu=False)
    new_payload.pop("machine_shape", None)
    evidence = {"schema_version": "p038_CPU_preflight_only_offline_pack_v1",
                "at_UTC": datetime.now(timezone.utc).isoformat(), "source_sha256": digest(raw), "source_bytes": len(raw),
                "P038_parent_source_sha256": PARENT_SOURCE, "P038_parent_manifest_sha256": PARENT_MANIFEST,
                "operational_runtime_sha256": manifest["operational_runtime"]["sha256"], "exact_guard_config": config,
                "inputs": manifest["inputs"], "requested_docker": manifest["requested_docker"],
                "GPU": False, "Internet": False, "private": True,
                "author_cells_included_or_called": False, "actual_CPU_preflight_executed": False,
                "planned_output": "P038_PREFLIGHT.json", "no_science_or_accuracy_qualification": True,
                "model_checkpoint_read_policy": "streamed SHA only, no torch/pickle/model deserialization",
                "fixture_policy": "same strict existing own four-phantom/three own CP313-plugin guard as P038 v2; temporary target prefix only differs",
                "partial_member_progress_on_failure_retained_before_tmp_cleanup": True,
                "runtime_function_globals_isolated": True,
                "builder_sha256": digest(Path(__file__).read_bytes()), "remote_operations": 0}
    output.mkdir(parents=True, exist_ok=False)
    (output / "source.ipynb").write_bytes(raw)
    for name, value in (("save-payload.json", new_payload), ("manifest.json", evidence)):
        (output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--slug", default=SLUG)
    args = parser.parse_args()
    value = prepare(args.output_dir, slug=args.slug)
    print(json.dumps({"source_sha256": value["source_sha256"], "source_bytes": value["source_bytes"], "GPU": False,
                      "Inputs": value["inputs"]["total"], "actual_CPU_preflight_executed": False}))


if __name__ == "__main__":
    main()
