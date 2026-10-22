"""P038 V6 operational guards and exact-cell transport; stdlib until decoder subprocess.

Importing this file does not read Inputs, run scientific cells, load models, decode
MRI, call CUDA or access the network. Only a future explicit Kaggle run calls it.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import json
import lzma
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())
    temporary.replace(path)


def environment_snapshot():
    """Distribution metadata only: never import Torch or initialize CUDA."""
    result = {"python": sys.version, "platform": sys.platform, "packages": {}}
    for name in ("torch", "numpy", "pandas", "timm", "pydicom", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    return result


def dataset_root(input_root, owner, slug, required):
    base = Path(input_root)
    candidates = [base / slug, base / "datasets" / owner / slug]
    candidates += list((base / "datasets" / owner / slug).glob("*"))
    found = {path.resolve() for path in candidates if path.is_dir() and (path / required).is_file()}
    require(len(found) == 1, "Exactly one fixed dataset mount required: " + slug)
    return found.pop()


def verify_members(root, inventory, progress=None):
    root, result = Path(root), {}
    require(root.is_dir() and not root.is_symlink(), "Regular fixed bundle directory required")
    for name, expected in inventory.items():
        require(Path(name).name == name and name not in {"", ".", ".."}, "Unsafe member name")
        path = root / name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == expected["bytes"],
                "Fixed member missing/size differs: " + name)
        observed = sha(path)
        require(observed == expected["sha256"], "Fixed member SHA differs: " + name)
        result[name] = {"bytes": path.stat().st_size, "sha256": observed}
        if progress is not None:
            progress(name, result[name])
    return result


DECODER_CHECK = '''import hashlib,importlib.metadata,json,os,sys
import numpy as np
import pydicom
A=sys.argv[1]
manifest=json.load(open(A+'/phantom-manifest.json'))
ref=np.load(A+'/'+manifest['expected_filename'],allow_pickle=False)
rows=[]
for entry in manifest['files']:
    f=A+'/'+entry['filename']
    ds=pydicom.dcmread(f)
    syntax=str(ds.file_meta.TransferSyntaxUID)
    if syntax!=entry['syntax']: raise ValueError('Synthetic transfer syntax differs')
    value=ds.pixel_array
    if value.shape!=ref.shape or value.dtype!=ref.dtype or not np.array_equal(value,ref):
        raise ValueError('Decoder fixture mismatch: '+os.path.basename(f))
    rows.append({'file':os.path.basename(f),'sha256':hashlib.sha256(open(f,'rb').read()).hexdigest(),'equal':True,
      'syntax':syntax,'shape':list(value.shape),'dtype':str(value.dtype),'shape_dtype_values_equal':True})
print(json.dumps({'complete':True,'fixtures':rows,'reference_sha256':hashlib.sha256(open(A+'/'+manifest['expected_filename'],'rb').read()).hexdigest(),
 'python':sys.version,'pydicom_path':pydicom.__file__,'pydicom_version':pydicom.__version__,
 'numpy_version':np.__version__,'reference_shape':list(ref.shape),'reference_dtype':str(ref.dtype),
 'packages':{p:importlib.metadata.version(p) for p in ['pylibjpeg','pylibjpeg-libjpeg','pylibjpeg-openjpeg']}}))
'''


def validate_decoder_result(value, expected_names):
    rows = value.get("fixtures", [])
    require(value.get("complete") is True and bool(expected_names), "Positive decoder fixture result required")
    require(len(rows) == len(expected_names) and {row["file"] for row in rows} == set(expected_names)
            and all(row.get("equal") is True for row in rows), "Every available public decoder fixture must pass")
    return value


def strict_decoder_preflight(wheel_root, work, fixture_root, config):
    """Fail before scientific inference on incompatible wheels or silent fixture loss."""
    wheel_root, work, fixture_root = Path(wheel_root), Path(work), Path(fixture_root)
    # These are existing OWN synthetic fixtures, not nonexistent author fixtures.
    fixture_members = verify_members(fixture_root, config["fixture_members"])
    phantom = json.loads((fixture_root / "phantom-manifest.json").read_bytes())
    fixtures = sorted(row["filename"] for row in phantom["files"])
    require(len(fixtures) == len(set(fixtures)) == 4 and set(fixtures) == set(config["fixture_syntaxes"]),
            "Exactly four fixed synthetic fixtures required")
    require(phantom["expected_filename"] == "expected.npy"
            and phantom["expected_sha256"] == config["fixture_members"]["expected.npy"]["sha256"],
            "Synthetic reference SHA binding differs")
    require({row["filename"]: row["syntax"] for row in phantom["files"]} == config["fixture_syntaxes"],
            "Fixed four synthetic transfer syntaxes differ")
    require({path.name for path in fixture_root.glob("*.dcm")} == set(fixtures), "Synthetic fixture inventory differs")
    abi = "cp%d%d" % sys.version_info[:2]
    require(sys.platform == "linux" and abi == config["decoder_ABI"] == "cp313", "Pinned author Docker needs LinuxCP313")
    verify_members(wheel_root, config["decoder_wheel_members"])
    wheels = [wheel_root / name for name in config["decoder_wheel_members"]
              if "-py3-none-any" in name or ("-" + abi + "-" + abi + "-") in name]
    require(len(wheels) == len(config["decoder_wheel_members"]) == 3, "Three pinned own decoder wheels for CP313 required")
    target = work / "_own_env"  # Exactly the path the unmodified author Reader will use.
    command = [sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--target", str(target),
               *map(str, wheels)]
    install = subprocess.run(command, check=True, capture_output=True, text=True, timeout=180)
    (work / "P038" / "decoder-install.log").write_text(install.stdout + install.stderr, encoding="utf-8")
    environment = {**os.environ, "PYTHONPATH": str(target) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                   "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}
    checked = subprocess.run([sys.executable, "-c", DECODER_CHECK, str(fixture_root)], env=environment,
                             check=True, capture_output=True, text=True, timeout=180)
    value = validate_decoder_result(json.loads(checked.stdout), fixtures)
    for row in value["fixtures"]:
        require(row.get("syntax") == config["fixture_syntaxes"][row["file"]]
                and row.get("sha256") == config["fixture_members"][row["file"]]["sha256"]
                and row.get("shape_dtype_values_equal") is True
                and row.get("shape") == value["reference_shape"] and row.get("dtype") == value["reference_dtype"],
                "Positive fixed synthetic fixture evidence differs")
    require(value["reference_sha256"] == config["fixture_members"]["expected.npy"]["sha256"], "Decoded reference SHA differs")
    value.update(target=str(target), synthetic_fixture_members=fixture_members, fixture_root=str(fixture_root),
                 fixture_source="existing own synthetic four-phantom Dataset/V1; not author Reader fixtures",
                 wheel_files=[{"name": path.name, "sha256": sha(path), "bytes": path.stat().st_size} for path in wheels],
                 no_index=True, no_deps=True, install_returncode=install.returncode,
                 fixture_process_returncode=checked.returncode, actual_MRI_read=False, CUDA_calls=0)
    return value


def preflight(config, *, input_root="/kaggle/input", work="/kaggle/working"):
    root, work = Path(input_root), Path(work)
    output = work / "P038"
    output.mkdir(exist_ok=False)
    r6 = root / "notebooks" / "rsraki" / config["stage_slug"] / config["R6_folder"]
    reader = dataset_root(root, "goodpjw2008", config["Reader_slug"], "knee.py")
    # Prove the author's broad _o_find(knee.py) will select this guarded Reader.
    candidates = []
    for pattern in ("*/", "*/*/", "*/*/*/"):
        candidates += [path for path in sorted(root.glob(pattern)) if (path / "knee.py").is_file()]
    require(candidates and candidates[0].resolve() == reader, "Author Reader discovery would select another Input")
    state = {"schema_version": "p038_fixed_public_recipe_preflight_v1", "complete": False,
             "python": sys.version, "fixed_members": {}, "actual_mounted_Input_version_proven": False,
             "scientific_model_forward_calls": 0, "competition_MRI_Report_Gold_reads": 0}
    state["schema_version"] = "p038_helios_scoredV6_preflight_v1"
    state["producer_descriptor"] = config["producer_descriptor"]
    state["environment_before_preflight"] = environment_snapshot()
    state["phase"] = "fixed_member_guards"
    write_json(output / "preflight.json", state)
    try:
        for name, directory in (("R6", r6), ("Reader", reader)):
            state["fixed_members"][name] = {"root": str(directory), "members": {}}
            def progress(member, receipt, family=name):
                state["fixed_members"][family]["members"][member] = receipt
                state["phase"] = "verified_" + family + "/" + member
                write_json(output / "preflight.json", state)
            verify_members(directory, config[name], progress)
        state["phase"] = "synthetic_fixture_and_Reader_decoder_preflight"
        write_json(output / "preflight.json", state)
        decoder_root = dataset_root(root, "rsraki", config["fixture_input_slug"], "decoder-manifest.json")
        state["decoder_manifest"] = verify_members(decoder_root, config["decoder_manifest_members"])
        fixture_root = decoder_root / "phantoms"
        state["decoder"] = strict_decoder_preflight(decoder_root / "wheels", work, fixture_root, config)
        state["environment_after_preflight"] = environment_snapshot()
        state["complete"], state["phase"] = True, "all_required_guards_and_four_synthetic_decoders_passed"
        write_json(output / "preflight.json", state)
    except Exception as error:
        state.update(error_type=type(error).__name__, error=str(error)[:4000])
        write_json(output / "preflight.json", state)
        raise
    print("P038_PREFLIGHT_PASS", json.dumps({"members": sum(len(row["members"]) for row in state["fixed_members"].values()),
                                            "fixtures": len(state["decoder"]["fixtures"]),
                                            "python": sys.version.split()[0]}), flush=True)
    return state


def restore_cells(encoded, expected_compressed, expected_raw, inventory):
    compressed = base64.b85decode(encoded)
    require(hashlib.sha256(compressed).hexdigest() == expected_compressed and len(compressed) < 900000,
            "P038 compressed recipe SHA/size differs")
    raw = lzma.decompress(compressed, memlimit=128 * 1024**2)
    require(len(raw) < 2 * 1024**2 and hashlib.sha256(raw).hexdigest() == expected_raw,
            "P038 restored recipe SHA/size differs")
    cells = json.loads(raw)
    require(type(cells) is list and len(cells) == len(inventory), "P038 cell count differs")
    for text, expected in zip(cells, inventory):
        require(type(text) is str and len(text.encode()) == expected["bytes"]
                and hashlib.sha256(text.encode()).hexdigest() == expected["sha256"], "P038 exact cell bytes differ")
    return cells


OBSERVED = ("raptor_raw.csv", "raptor_sub.csv", "raptor224_raw.csv", "raptor224_sub.csv", "ours_sub.csv",
            "submission.csv", "_public_stack.csv", "_own.csv", "blend_receipt.json", "submission_receipt.json")


def observe(work, cell_index):
    """Copy only already-created predictions/receipts, without reranking or patching."""
    work = Path(work)
    destination = work / "P038" / "audit" / ("cell-%02d" % cell_index)
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    for name in OBSERVED:
        path = work / name
        if path.is_file():
            require(not path.is_symlink(), "Regular scientific output required")
            shutil.copyfile(path, destination / name)
            rows.append({"file": name, "bytes": path.stat().st_size, "sha256": sha(destination / name)})
    return rows


def execute_cells(cells, indices, shell, *, work="/kaggle/working"):
    require(len(cells) == len(indices) == 8 and list(indices) == list(range(2, 10)),
            "All eight fixed author V6 code cells in original index order required")
    output = Path(work) / "P038"
    receipt = {"schema_version": "p038_exact_author_cells_execution_v1", "complete": False,
               "completed": [], "own_Public_verified": False, "hidden_coverage_verified": False,
               "single_factor_scientific_effect_claimed": False, "Save_Submit_calls": 0}
    for index, text in zip(indices, cells):
        started = time.monotonic()
        result = shell.run_cell(text, store_history=False)
        error = result.error_before_exec or result.error_in_exec
        if error is not None:
            receipt.update(failed_cell=index, error_type=type(error).__name__)
            write_json(output / "execution.json", receipt)
            raise error
        row = {"original_cell_index": index, "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
               "elapsed_seconds": time.monotonic() - started, "environment_after_cell": environment_snapshot()}
        if index in (3, 5, 7, 8, 9):
            row["already_created_artifacts"] = observe(work, index)
        receipt["completed"].append(row)
        write_json(output / "execution.json", receipt)
    receipt["complete"] = True
    write_json(output / "execution.json", receipt)
    print("P038_ALL_AUTHOR_CELLS_COMPLETE", json.dumps({"cells": 8, "own_Public_verified": False}), flush=True)
    return receipt
