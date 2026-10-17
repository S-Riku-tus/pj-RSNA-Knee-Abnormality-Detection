"""Build P032 locally, retaining P029M scientific cells and exact F006 numerics.

Only source/metadata bytes are parsed and compressed. No model loading, MRI,
training, network, upload, Kaggle Save or competition submission occurs here.
"""

from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import io
import json
import lzma
import platform
import struct
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECOVERY = ROOT / "artifacts/research/20261016-f006-execution-v1/recovery"
NATIVE_SHA = "ac4d5895248a4f46dca970b70371f711084830c98634a1ee1cfdd9f020c6b6f3"
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"
DATASETS = (
    "heliosli/rsna-orthofoundation-fold0-epoch7-20261004/1",
    "nartaa/rsna-knee-publication-swa-weights-20261007/1",
    "rsraki/n392-aucf0-816d6df4187b4878a3ccd7234ffa14dd/1",
    "rsraki/n392-aucf1-ddc10efce2d024a1d7a69884c47d50a7/1",
    "rsraki/n392rg-f0-8e978296019046e3a9fff8fcb6a4901f/1",
    "rsraki/n392rg-f1-82087594dae443ab82a701d9750b661a/1",
    "rsraki/rsna-dicom-decoders-py313-v1/1",
)
STAGE = "rsraki/p029-r6v1-stage-20261015-b2/1"


def digest(body):
    return hashlib.sha256(body).hexdigest()


def source(cell):
    value = cell["source"]
    return value if isinstance(value, str) else "".join(value)


def assignment(tree, name):
    values = [node.value for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
    if len(values) != 1:
        raise ValueError("One top-level assignment required: " + name)
    return values[0]


def inspect_native(path):
    body = Path(path).read_bytes()
    if digest(body) != NATIVE_SHA:
        raise ValueError("Exact recovered F006 native64 notebook SHA differs")
    book = json.loads(body)
    bootstrap = source(book["cells"][1])
    tree = ast.parse(bootstrap)
    call = assignment(tree, "_p18_joint")
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != "_p18_restore":
        raise ValueError("Expected pinned literal F006 restoration")
    encoded, length, joint_sha, compressed_sha = [ast.literal_eval(value) for value in call.args]
    compressed = base64.b85decode(encoded)
    if digest(compressed) != compressed_sha or length > 8 * 1024**2:
        raise ValueError("F006 compressed source SHA/size differs")
    joint = lzma.decompress(compressed)
    if len(joint) != length or digest(joint) != joint_sha:
        raise ValueError("F006 exact joint SHA differs")
    zip_length = struct.unpack("<Q", joint[:8])[0]
    archive_bytes, metadata_bytes = joint[8:8 + zip_length], joint[8 + zip_length:]
    inventory = ast.literal_eval(assignment(tree, "_p18_inventory"))
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        if len(archive.namelist()) != len(inventory) or set(archive.namelist()) != set(inventory):
            raise ValueError("Exact embedded F006 source inventory differs")
        for name, pin in inventory.items():
            raw = archive.read(name)
            if len(raw) != pin["bytes"] or digest(raw) != pin["sha256"]:
                raise ValueError("Embedded F006 member SHA differs: " + name)
            if name.endswith(".py"):
                compile(raw, "F006/" + name, "exec")
    payload = json.loads(metadata_bytes)
    for name, entry in payload["metadata"].items():
        if digest(entry["UTF8"].encode()) != entry["sha256"]:
            raise ValueError("Embedded original metadata SHA differs: " + name)
    context = payload["context"]
    cpu = json.loads(payload["metadata"][context["metadata"]["CPU_smoke"]["path"]]["UTF8"])
    if {key: value["dataset_ref"] for key, value in context["transport_jobs"].items()} != {
        "CLS:0": DATASETS[2], "CLS:1": DATASETS[3], "REG:0": DATASETS[4], "REG:1": DATASETS[5]
    }:
        raise ValueError("F006 four-model Inputs differ")
    compile(bootstrap, "F006-bootstrap", "exec")
    return bootstrap, {"source_notebook_sha256": digest(body), "source_member_count": len(inventory),
                       "source_members": inventory, "transport_jobs": context["transport_jobs"],
                       "model_fingerprints": cpu["model_fingerprints"],
                       "historical_metadata_not_new_execution_proof": True}


def inspect_parent(path):
    body = Path(path).read_bytes()
    book = json.loads(body)
    if len(book["cells"]) != 11 or len([c for c in book["cells"] if c["cell_type"] == "code"]) != 8:
        raise ValueError("Expected recovered P029M eight-code-cell structure")
    history = json.loads((ROOT / "experiments/p030-author-r7-switch-static-comparison-20261015.json").read_text(encoding="utf-8-sig"))
    expected = [row["R6_sha256"] for row in history["code_cell_comparison"]]
    scientific = [source(book["cells"][index]) for index in range(4, 11)]
    branch = "    cands += [INPUT / 'notebooks' / 'rsraki' / 'p029-r6v1-stage-20261015-b2' / slug]\n"
    if scientific[4].count(branch) != 1:
        raise ValueError("Exact b2 mount branch required")
    original = list(scientific)
    original[4] = original[4].replace(branch, "")
    if [digest(value.encode()) for value in original] != expected:
        raise ValueError("Recovered P029M scientific cells differ from archived R6 hashes")
    if digest(source(book["cells"][1]).encode()) != "411e9598b806757de14adf8de6e586eae3209c0460797f8e24eeb65fdf960536":
        raise ValueError("Exact P029M nine-member b2 guard required")
    return book, {"recovered_notebook_bytes": len(body), "recovered_notebook_sha256": digest(body),
                  "original_scientific_cell_sha256": expected,
                  "b2_scientific_cell_sha256": [digest(value.encode()) for value in scientific],
                  "all_seven_match_archived_original_after_single_mount_branch_normalization": True,
                  "serialized_notebook_bytes_differ_from_old_receipt": len(body) != 620588,
                  "cell_sources_pinned_independently_of_notebook_serialization": True}


def zip_sources(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, body in sorted(files.items()):
            entry = zipfile.ZipInfo(name, (2026, 10, 16, 0, 0, 0))
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, body)
    return buffer.getvalue()


def cell(kind, text):
    value = {"cell_type": kind, "metadata": {}, "source": text}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def prepare(output, *, recovery=RECOVERY, mode="serving", diagnostic_count=192):
    output, recovery = Path(output), Path(recovery)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a fresh P032 run directory")
    if mode not in ("serving", "diagnostic") or not 1 <= diagnostic_count <= 256:
        raise ValueError("Invalid P032 mode/count")
    parent, parent_review = inspect_parent(recovery / "P029M-source.ipynb")
    bootstrap, native_review = inspect_native(recovery / "F006-native64-source.ipynb")
    child = "BOOTSTRAP = " + repr(bootstrap) + "\n" + (ROOT / "scripts/p032_native_child.py").read_text(encoding="utf-8-sig")
    # Keep future import in the legal first statement position.
    child = child.replace("from __future__ import annotations\n", "", 1)
    compile(child, "p032_native_child.py", "exec")
    extension = (ROOT / "scripts/p032_parent_extension.py").read_text(encoding="utf-8-sig")
    files = {"p032_native_child.py": child.encode(),
             "p032_parent_extension.py": extension.encode(),
             "p032blend/__init__.py": b"",
             "p032blend/contracts.py": (ROOT / "src/rsna_knee/contracts.py").read_bytes(),
             "p032blend/fixed20_blend.py": (ROOT / "src/rsna_knee/fixed20_blend.py").read_bytes()}
    for name, raw in files.items():
        compile(raw, name, "exec")
    raw_bundle = zip_sources(files)
    packed = base64.b85encode(lzma.compress(raw_bundle, preset=9)).decode("ascii")
    leading = cell("code", "import time as _p032_clock\n_p032_started = _p032_clock.monotonic()\n")
    final_source = ("# New isolated native stage; original P029M cells above are unchanged.\n"
                    "exec(compile(" + repr(extension) + ", 'p032_parent_extension.py', 'exec'), globals())\n"
                    "_p032_runtime = restore_runtime(" + repr(packed) + ", " + repr(digest(raw_bundle)) + ")\n"
                    "_p032_result = execute(globals(), runtime=_p032_runtime, started=_p032_started, "
                    "mode=" + repr(mode) + ", diagnostic_count=" + repr(diagnostic_count) + ")\n")
    compile(final_source, "P032-final-cell", "exec")
    notice = cell("markdown", "# P032: existing P029M + existing F006 pair, fixed 20%\n\n"
                  "Private, Internet OFF, T4 x2. No training or API submission. "
                  "Parent scientific cells are preserved. Native errors/timeouts retain the parent CSV "
                  "and are explicitly not an accuracy comparison. MCL/Synovitis retain parent values. "
                  "The previous hidden failure cause is unknown.\n")
    book = copy.deepcopy(parent)
    if mode == "serving":
        for item in book["cells"]:
            item["metadata"] = {}
            if item["cell_type"] == "code":
                item.update(outputs=[], execution_count=None)
        book["cells"] = [notice, leading, *book["cells"], cell("code", final_source)]
        retained = [source(item) for item in book["cells"][2:-1]]
        if retained != [source(item) for item in parent["cells"]]:
            raise ValueError("Parent source changed during packaging")
    else:
        book["cells"] = [notice, leading, cell("code", final_source)]
    book["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                        "kaggle": {"isGpuEnabled": True, "isInternetEnabled": False}}
    serialized = (json.dumps(book, ensure_ascii=False, indent=1) + "\n").encode()
    inline_bytes = len(serialized)
    compressed_parent = mode == "serving" and inline_bytes > 950000
    if compressed_parent:
        # The API has a conservative source-size limit. Keep the exact recovered
        # parent notebook as a hashed runtime member; execute each unchanged
        # code-cell string with IPython so %%writefile keeps its real semantics.
        parent_bytes = (recovery / "P029M-source.ipynb").read_bytes()
        files["P029M-source.ipynb"] = parent_bytes
        raw_bundle = zip_sources(files)
        packed = base64.b85encode(lzma.compress(raw_bundle, preset=9)).decode("ascii")
        setup = ("exec(compile(" + repr(extension) + ", 'p032_parent_extension.py', 'exec'), globals())\n"
                 "_p032_runtime = restore_runtime(" + repr(packed) + ", " + repr(digest(raw_bundle)) + ")\n")
        parent_runner = (
            "_p032_parent_bytes = (_p032_runtime / 'P029M-source.ipynb').read_bytes()\n"
            "if hashlib.sha256(_p032_parent_bytes).hexdigest() != " + repr(digest(parent_bytes)) + ":\n"
            "    raise ValueError('Completed parent source pin differs')\n"
            "_p032_book = json.loads(_p032_parent_bytes)\n"
            "_p032_code = [c['source'] if isinstance(c['source'], str) else ''.join(c['source']) "
            "for c in _p032_book['cells'] if c['cell_type'] == 'code']\n"
            "if [hashlib.sha256(s.encode()).hexdigest() for s in _p032_code] != "
            + repr([digest(source(c).encode()) for c in parent["cells"] if c["cell_type"] == "code"]) + ":\n"
            "    raise ValueError('Parent cell source pins differ')\n"
            "_p032_parent_completed = 0\n"
            "for _p032_index, _p032_source in enumerate(_p032_code):\n"
            "    _p032_execution = get_ipython().run_cell(_p032_source, store_history=False)\n"
            "    _p032_error = _p032_execution.error_before_exec or _p032_execution.error_in_exec\n"
            "    if _p032_error is not None:\n"
            "        raise _p032_error\n"
            "    _p032_parent_completed += 1\n"
            "if _p032_parent_completed != 8:\n"
            "    raise RuntimeError('All eight unchanged P029M code cells must complete')\n"
        )
        extension_runner = (
            "_p032_result = execute(globals(), runtime=_p032_runtime, started=_p032_started, "
            "mode='serving', diagnostic_count=" + repr(diagnostic_count) + ")\n"
        )
        for name, value in (("compressed-setup", setup), ("parent-runner", parent_runner),
                            ("extension-runner", extension_runner)):
            compile(value, name, "exec")
        book["cells"] = [notice, *[copy.deepcopy(c) for c in parent["cells"] if c["cell_type"] == "markdown"],
                         leading, cell("code", setup), cell("code", parent_runner), cell("code", extension_runner)]
        serialized = (json.dumps(book, ensure_ascii=False, indent=1) + "\n").encode()
    if len(serialized) > 950000:
        raise ValueError(f"P032 Notebook exceeds conservative API source capacity: {len(serialized)}")
    manifest = {
        "schema_version": "p032_offline_preparation_v1", "mode": mode, "diagnostic_count": diagnostic_count,
        "source_sha256": digest(serialized), "source_bytes": len(serialized),
        "parent_execution_transport": "hashed_exact_cells_IPython_run_cell" if compressed_parent else "inline_unchanged_cells",
        "inline_notebook_source_bytes_before_capacity_fallback": inline_bytes,
        "parent": parent_review, "native": native_review,
        "runtime_bundle_sha256": digest(raw_bundle), "runtime_bundle_bytes": len(raw_bundle),
        "runtime_files": {name: {"sha256": digest(raw), "bytes": len(raw)} for name, raw in files.items()},
        "inputs": {"datasets": list(DATASETS), "kernels": [STAGE],
                   "competitions": ["rsna-knee-abnormality-detection"], "total": 9},
        "platform": {"docker": DOCKER, "Internet": False, "GPU": "Tesla T4 x2"},
        "integration": {"native_weight": 0.2, "parent_reranked": False, "final_reranked": False,
                        "FP32_separate_multiply_add": True,
                        "structural_failure_disables_entire_native_column": True,
                        "native_child_timeout_or_exception_retains_parent_CSV_and_records_no_comparison": True,
                        "whole_notebook_budget_seconds": 30600, "final_reserve_seconds": 300,
                        "diagnostic_native_budget_limit_seconds": 3600,
                        "serving_native_budget_limit_seconds": 28800,
                        "old_S6_parent_gate_or_two_controller_cleanup_not_reused": True},
        "observability": {"child_faulthandler_repeat_seconds": 120,
                          "child_stage_begin_end_events": True,
                          "parent_stage_begin_end_events": ["execute", "import_torch", "CUDA_available",
                                                            "CUDA_device_count", "GPU_cleanup", "native_child"],
                          "parent_heartbeat_seconds": 30,
                          "maximum_new_child_log_bytes_per_relay": 2000,
                          "already_relayed_child_log_bytes_repeated": False},
        "scope": {"source_and_metadata_only": True, "models_MRI_training_network_Save_Submit": 0,
                  "new_actual_GPU_forward": False, "new_native_diagnostic": False,
                  "new_visible_complete": False, "new_hidden_complete": False, "Public_gain_verified": False},
        "environment": {"python": platform.python_version()},
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "source.ipynb").write_bytes(serialized)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, default=RECOVERY)
    parser.add_argument("--mode", choices=("serving", "diagnostic"), default="serving")
    parser.add_argument("--diagnostic-count", type=int, default=192)
    args = parser.parse_args(argv)
    result = prepare(args.output, recovery=args.recovery, mode=args.mode, diagnostic_count=args.diagnostic_count)
    print(json.dumps({key: result[key] for key in ("mode", "source_bytes", "source_sha256")}))


if __name__ == "__main__":
    main()
