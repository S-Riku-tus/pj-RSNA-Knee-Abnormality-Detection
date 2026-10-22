"""Prepare an inert F009/P033 Notebook locally; never upload or call Kaggle.

No-argument/default invocation reads no input bodies. An output request builds
source only unless --execute and a future qualified root release are supplied.
Historical bootstrap text is parsed as literals, never executed.
"""
from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import importlib
import importlib.util
import io
import json
import lzma
import os
import re
import stat
import struct
import sys
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "artifacts/kaggle/p039-p033-native20-20261017-v2"
CONTROL_SHA = "6453b3d09787abab07205d832c0176f0e78d050127e98d83c9f835bc5f799452"
CONTROL_MANIFEST_SHA = "a5b6109cdfba4c198550daa53efcfc029325f94ebe97de721278b8f31d18043e"
PARENT_SHA = "57284b71d271c3d4bf09595a6acade490ea911c8f7438a595724f260f3f28ccb"
PLAN = ROOT / "experiments/f009-P039-parent-fixed20-CLS-pair-conditional-20261017.json"
PLAN_SHA = "37bb0c60826da6deddda0e3929fd8aeb4a2f56ad91371af9dab365835a69bba2"
SUPPORT = "_f009_kaggle_support"
FIXED_SUPPORT = {
    "f009_mixed_runtime.py": "73695814776300ed7c2edd8f2ba535f99b2af9afe4c335346ff06ac1530f6a90",
    "f009_transport.py": "8c613c2c2d24fc6b1d1c6222482c581b03ac5027c15b10f5e7116e7f7056af47",
    "f009_source_namespace.py": "cd2b5fd71355801b67f4f4596487d42728e4974f84b10eac15155ef35c6e5526",
    "isolated_uid_executor.py": "685ec00ccc595ccb274817f960c28bdd3bc5926657f52e222c86863e5806d8f8",
    "operational_uid_recovery.py": "40337f6bc2eaa5d75ab1157bcc2a9dec76bd0022445df2c012b6d89c1d45a814",
    "contracts.py": "aa246fd1a49ecbe88f1795f49904ef9675f119c935e99b9ffd5eee8220cbc71b",
}
PARENT_DATASETS = (
    "heliosli/rsna-orthofoundation-fold0-epoch7-20261004/1",
    "nartaa/rsna-knee-publication-swa-weights-20261007/1",
)
DECODER = "rsraki/rsna-dicom-decoders-py313-v1/1"
STAGE = "rsraki/p029-r6v1-stage-20261015-b2/1"
COMPETITION = "rsna-knee-abnormality-detection"
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"
MAX_SOURCE = 16 * 1024**2
MAX_RUNTIME = 96 * 1024**2  # Source<=16 MiB + root metadata<=64 MiB + ZIP/sidecar headroom.


def require(value, message):
    if not value:
        raise ValueError(message)


def raw_pin(body):
    return {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def read(path):
    return json.loads(Path(path).read_bytes())


def source(cell):
    value = cell["source"]
    return "".join(value) if isinstance(value, list) else value


def code_cells(book):
    return [source(cell) for cell in book["cells"] if cell["cell_type"] == "code"]


def literal_assignment(tree, name):
    values = [node.value for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)]
    require(len(values) == 1, "One literal boundary required: " + name)
    return values[0]


def safe_name(name):
    require(isinstance(name, str) and "\\" not in name and ":" not in name, "Unsafe source member")
    path = PurePosixPath(name)
    require(path.parts and not path.is_absolute() and path.as_posix() == name
            and all(p not in ("", ".", "..") and not p.endswith((".", " ")) for p in path.parts),
            "Noncanonical source member")
    require(all(not re.fullmatch(r"(?i:con|prn|aux|nul|com[1-9]|lpt[1-9])", p.split(".", 1)[0])
                for p in path.parts), "Reserved source member")
    return name


def checked_zip(raw, inventory):
    require(0 < len(raw) <= MAX_SOURCE and 0 < len(inventory) <= 512, "Bounded source archive required")
    result = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        require(len(entries) == len(inventory) and {e.filename for e in entries} == set(inventory),
                "Source inventory extras/missing/duplicate")
        require(len({e.filename.casefold() for e in entries}) == len(entries), "Case-colliding source members")
        require(sum(e.file_size for e in entries) <= MAX_SOURCE, "Expanded source cap exceeded")
        for entry in entries:
            safe_name(entry.filename)
            require(not entry.is_dir() and stat.S_IFMT(entry.external_attr >> 16) in (0, stat.S_IFREG)
                    and entry.flag_bits & 1 == 0 and entry.compress_type == zipfile.ZIP_STORED
                    and entry.file_size == inventory[entry.filename]["bytes"], "Unsafe source ZIP entry")
            body = archive.read(entry)
            require(raw_pin(body) == inventory[entry.filename], "Exact source member SHA/size differs")
            if entry.filename.endswith(".py"):
                compile(body, entry.filename, "exec")
            result[entry.filename] = body
    return result


def zip_sources(files):
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, body in sorted(files.items()):
            safe_name(name)
            item = zipfile.ZipInfo(name, (2026, 10, 17, 0, 0, 0))
            item.external_attr = (stat.S_IFREG | 0o600) << 16
            archive.writestr(item, body)
    return target.getvalue()


def inspect_control(control=CONTROL, plan_path=PLAN):
    """Unpack verified literals, without importing/executing any old science."""
    control = Path(control)
    raw, manifest_raw = ((control / name).read_bytes() for name in ("source.ipynb", "manifest.json"))
    require(raw_pin(raw)["sha256"] == CONTROL_SHA and raw_pin(manifest_raw)["sha256"] == CONTROL_MANIFEST_SHA,
            "Frozen P039 source/manifest differs")
    book, manifest = json.loads(raw), json.loads(manifest_raw)
    codes = code_cells(book)
    require(len(codes) == 4, "Frozen four-cell P039 carrier required")
    calls = [node.value for node in ast.parse(codes[1]).body if isinstance(node, ast.Assign)
             and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
             and node.value.func.id == "restore_runtime"]
    require(len(calls) == 1, "Single P039 compressed carrier boundary required")
    encoded, expected = [ast.literal_eval(n) for n in calls[0].args]
    archive = lzma.decompress(base64.b85decode(encoded), memlimit=256 * 1024**2)
    require(raw_pin(archive)["sha256"] == expected == manifest["runtime_bundle"]["sha256"],
            "P039 exact carrier digest differs")
    outer_files = checked_zip(archive, manifest["runtime_files"])
    parent_raw = outer_files["P033-source.ipynb"]
    require(raw_pin(parent_raw)["sha256"] == PARENT_SHA, "Fixed P033 parent body differs")
    plan_raw = Path(plan_path).read_bytes()
    require(raw_pin(plan_raw)["sha256"] == PLAN_SHA, "Fixed conditional plan differs")
    plan = json.loads(plan_raw)
    parent_codes = code_cells(json.loads(parent_raw))
    require(len(parent_codes) == 8 and [raw_pin(s.encode())["sha256"] for s in parent_codes]
            == plan["frozen_control"]["parent_eight_code_cell_sha256"], "Parent eight science cells differ")
    child_tree = ast.parse(outer_files["p032_native_child.py"].decode())
    bootstrap = ast.literal_eval(literal_assignment(child_tree, "BOOTSTRAP"))
    bootstrap_tree = ast.parse(bootstrap)
    joint_call = literal_assignment(bootstrap_tree, "_p18_joint")
    require(isinstance(joint_call, ast.Call) and isinstance(joint_call.func, ast.Name)
            and joint_call.func.id == "_p18_restore", "Pinned old restore literal required")
    packed, length, joint_sha, compressed_sha = [ast.literal_eval(n) for n in joint_call.args]
    compressed = base64.b85decode(packed)
    require(raw_pin(compressed)["sha256"] == compressed_sha, "Old source compressed bytes differ")
    joint = lzma.decompress(compressed, memlimit=256 * 1024**2)
    require(len(joint) == length <= MAX_SOURCE and raw_pin(joint)["sha256"] == joint_sha,
            "Old source joint SHA/size differs")
    archive_length = struct.unpack("<Q", joint[:8])[0]
    require(0 < archive_length < len(joint) - 8, "Exact source/metadata boundary required")
    original_inventory = ast.literal_eval(literal_assignment(bootstrap_tree, "_p18_inventory"))
    components = checked_zip(joint[8:8 + archive_length], original_inventory)
    historical = json.loads(joint[8 + archive_length:])
    for item in historical["metadata"].values():
        require(raw_pin(item["UTF8"].encode())["sha256"] == item["sha256"], "Historical text SHA differs")
    # Only old REG grants are runtime inputs. Old CLS/authority/CPU phases stay unused.
    reg_metadata = {}
    for fold in (0, 1):
        job = plan["intended_intervention"]["old_REG_transport_refs"]["REG:" + str(fold)]
        for role, key in (("permission", "root_permission"), ("peer", "peer_review")):
            value = historical["metadata"][job[key]]["UTF8"].encode()
            expected_sha = job["root_permission_sha" if role == "permission" else "peer_sha"]
            require(raw_pin(value)["sha256"] == expected_sha, "Exact historical REG grant differs")
            reg_metadata["metadata/REG/fold" + str(fold) + "-" + role + ".json"] = value
    return {"book": book, "manifest": manifest, "outer_files": outer_files,
            "components": components, "component_inventory": original_inventory,
            "reg_metadata": reg_metadata, "plan": plan, "plan_raw": plan_raw,
            "parent_raw": parent_raw, "parent_code_sha256": [raw_pin(s.encode())["sha256"] for s in parent_codes]}


# This stdlib restorer is also exercised directly over tiny artificial sources.
RESTORER = '''import base64, hashlib, io, json, lzma, os, re, stat, zipfile
from pathlib import Path, PurePosixPath

def f009_restore_sources(encoded, compressed_pin, archive_pin, inventory, destination):
    def require(value, message):
        if not value:
            raise ValueError(message)
    def pin(body):
        return {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
    def name_ok(name):
        require(isinstance(name, str) and "\\\\" not in name and ":" not in name, "Unsafe member")
        p = PurePosixPath(name)
        require(p.parts and not p.is_absolute() and p.as_posix() == name
                and all(x not in ("", ".", "..") and not x.endswith((".", " ")) for x in p.parts), "Unsafe member")
        require(all(not re.fullmatch(r"(?i:con|prn|aux|nul|com[1-9]|lpt[1-9])", x.split(".", 1)[0]) for x in p.parts), "Reserved member")
    destination = Path(destination).absolute()
    parent = destination.parent.resolve(strict=True)
    staging = destination.with_name(destination.name + ".partial")
    require(destination.resolve() == parent / destination.name and staging.resolve() == parent / staging.name
            and not destination.exists() and not destination.is_symlink()
            and not staging.exists() and not staging.is_symlink(), "Fresh same-parent source outputs required")
    compressed = base64.b85decode(encoded)
    require(pin(compressed) == compressed_pin and len(compressed) <= 4 * 1024**2, "Compressed source pin differs")
    raw = lzma.decompress(compressed, memlimit=256 * 1024**2)
    require(pin(raw) == archive_pin and len(raw) <= 96 * 1024**2, "Source ZIP pin differs")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        require(len(entries) == len(inventory) and {e.filename for e in entries} == set(inventory)
                and len({e.filename.casefold() for e in entries}) == len(entries), "Exact positive source inventory required")
        require(sum(e.file_size for e in entries) <= 96 * 1024**2, "Expanded source cap exceeded")
        for e in entries:
            name_ok(e.filename)
            require(not e.is_dir() and stat.S_IFMT(e.external_attr >> 16) in (0, stat.S_IFREG)
                    and e.compress_type == zipfile.ZIP_STORED and not e.flag_bits & 1
                    and e.file_size == inventory[e.filename]["bytes"], "Unsafe source ZIP entry")
            require(pin(archive.read(e)) == inventory[e.filename], "Source SHA mismatch before publication")
        staging.mkdir(exist_ok=False)
        try:
            for e in entries:
                target = staging / e.filename
                require(target.resolve().is_relative_to(staging.resolve()), "Source path escape")
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(archive.read(e))
            require(all(pin((staging / name).read_bytes()) == expected for name, expected in inventory.items()),
                    "Written source member differs")
            require(staging.resolve() == parent / staging.name and destination.resolve() == parent / destination.name
                    and not staging.is_symlink() and not destination.exists() and not destination.is_symlink(),
                    "Checked fresh source publication required")
            os.rename(staging, destination)
        except BaseException as error:
            with (staging / "RESTORE_FAILED.json").open("x", encoding="utf-8") as stream:
                json.dump({"complete": False, "error_type": type(error).__name__}, stream)
            raise
    return destination
'''


def replace_once(value, before, after):
    require(value.count(before) == 1, "Single exact extension boundary required: " + before[:70])
    return value.replace(before, after, 1)


@contextmanager
def new_support_modules():
    """Import only new stdlib helpers outside the canonical old REG package."""
    namespace = "_f009_notebook_builder_support"
    require(not any(n == namespace or n.startswith(namespace + ".") for n in sys.modules),
            "Fresh new source-preparation package required")
    package = ModuleType(namespace)
    package.__path__ = [str(ROOT / "src/rsna_knee")]
    package.__package__ = namespace
    package.__spec__ = importlib.util.spec_from_loader(namespace, loader=None, is_package=True)
    sys.modules[namespace] = package
    try:
        driver = importlib.import_module(namespace + ".f009_kaggle_driver")
        context = importlib.import_module(namespace + ".f009_kaggle_context")
        yield driver, context
    finally:
        for name in list(sys.modules):
            if name == namespace or name.startswith(namespace + "."):
                del sys.modules[name]


def native_shim():
    return '''"""Fixed CLI adapter; no old model bootstrap or import."""
import argparse
from pathlib import Path
import sys

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=("serving",))
    p.add_argument("--output", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--budget-seconds", required=True, type=float)
    p.add_argument("--diagnostic-count", required=True, type=int)
    p.add_argument("--native-executor", required=True, choices=("isolated-uid",))
    a = p.parse_args(argv)
    if not (a.diagnostic_count == 192 and 120 <= a.budget_seconds <= 28800):
        raise ValueError("Fixed serving/UID/deadline contract required")
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    from _f009_kaggle_support import f009_kaggle_driver as driver
    context = root / "f009-context.json"
    context_sha = (root / "f009-context.sha256").read_text(encoding="ascii").strip()
    driver.main(["--context", str(context), "--context-sha", context_sha,
        "--output", a.output, "--baseline", a.baseline,
        "--budget-seconds", str(a.budget_seconds), "--execute"])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
'''


def new_extension(old, *, adapter_schema, result_schema):
    value = replace_once(old, '"p039_p033_f006_fixed20_execution_v1"',
                         '"f009_p033_CLS_pair_fixed20_execution_v1"')
    value = replace_once(value, 'argv = [sys.executable, "-B", "-u", str(package / "p032_native_child.py"),',
                         'argv = [sys.executable, "-B", "-u", str(package / "f009_native_child.py"),')
    value = replace_once(value, 'child_script=str(package / "p032_native_child.py")',
                         'child_script=str(package / "f009_native_child.py")')
    value = replace_once(value, '"p035_f006_isolated_uid_adapter_v1"', repr(adapter_schema))
    value = replace_once(value, '"native_isolated_uid_operational_recovery_v1"', repr(result_schema))
    value = value.replace('"F006_native_rank"', '"F009_CLS_old_REG_native_rank"')
    # Operational names only; mathematical calls and UID/failure handling stay exact.
    value = value.replace('"P032"', '"F009_P033_NATIVE20"').replace('"P032_final.tmp"', '"F009_P033_final.tmp"')
    value = value.replace('"P032_PARENT_PROGRESS"', '"F009_P033_PARENT_PROGRESS"')
    value = value.replace('"P032_COMPLETE"', '"F009_P033_COMPLETE"')
    compile(value, "f009_parent_extension.py", "exec")
    return value


def cell(text):
    return {"cell_type": "code", "metadata": {}, "source": text, "outputs": [], "execution_count": None}


def runtime_paths():
    return {"source_roots": {"common": "components/common", "regional": "components/regional",
            "mixed": "components/mixed", "serving": "components/serving",
            "REG_transport": "components/REG_transport", "support": SUPPORT},
        "router": "new-serving/serving_router.py", "parent": "P033-source.ipynb",
        "child": "f009_native_child.py", "extension": "f009_parent_extension.py",
        "REG_metadata": {"REG:" + str(n): {
            "permission": "metadata/REG/fold" + str(n) + "-permission.json",
            "peer": "metadata/REG/fold" + str(n) + "-peer.json"} for n in (0, 1)}}


def assemble_sources(control, driver):
    files = {"components/" + name: body for name, body in control["components"].items()}
    original = control["outer_files"]
    for name, body in original.items():
        if name.startswith("p032blend/"):
            files[name] = body
    files["P033-source.ipynb"] = control["parent_raw"]
    files[SUPPORT + "/__init__.py"] = b'"""New F009 conditional runtime support; no canonical REG imports."""\n'
    for name, digest in FIXED_SUPPORT.items():
        body = (ROOT / "src/rsna_knee" / name).read_bytes()
        require(raw_pin(body)["sha256"] == digest, "Fixed new support source differs: " + name)
        files[SUPPORT + "/" + name] = body
    for name in ("f009_kaggle_driver.py", "f009_kaggle_context.py"):
        files[SUPPORT + "/" + name] = (ROOT / "src/rsna_knee" / name).read_bytes()
    require(driver.ADAPTER_SCHEMA == "f009_P033_isolated_uid_native_adapter_v1"
            and driver.RESULT_SCHEMA == "f009_amended_CLS_P039_REG_owned_UID_native_result_v1",
            "Exact new adapter/result schemas required")
    files["new-serving/serving_router.py"] = driver.derive_router_source(control["components"]["serving/serving_router.py"])
    require(type(files["new-serving/serving_router.py"]) is bytes, "New named router must be exact UTF8 bytes")
    files["f009_native_child.py"] = native_shim().encode()
    files["f009_parent_extension.py"] = new_extension(original["p032_parent_extension.py"].decode(),
        adapter_schema=driver.ADAPTER_SCHEMA, result_schema=driver.RESULT_SCHEMA).encode()
    for name, body in files.items():
        safe_name(name)
        if name.endswith(".py"):
            compile(body, name, "exec")
    require(all(files["components/" + name] == raw for name, raw in control["components"].items()),
            "Old component tree byte identity required")
    return files


def make_notebook(files, control, context, *, entry_enabled):
    require(sum(len(body) for body in files.values()) <= MAX_RUNTIME, "Whole runtime source/metadata cap exceeded")
    raw_archive = zip_sources(files)
    packed = lzma.compress(raw_archive, preset=9)
    all_pins = {name: raw_pin(body) for name, body in files.items()}
    first = ("# Source-only packages must stop before parent/model/MRI work.\n"
             f"_f009_actual_entry_enabled = {entry_enabled!r}\n"
             "if _f009_actual_entry_enabled is not True:\n"
             "    raise RuntimeError('F009 source-only package: actual joint/CPU/transport release absent')\n"
             "import time as _f009_clock\n_f009_started = _f009_clock.monotonic()\n")
    setup = ("_f009_restore_namespace = {'__name__': '_f009_source_restore'}\n"
             "exec(compile(" + repr(RESTORER) + ", 'f009-source-restorer.py', 'exec'), _f009_restore_namespace)\n"
             "_f009_runtime = _f009_restore_namespace['f009_restore_sources'](" + repr(base64.b85encode(packed).decode())
             + ", " + repr(raw_pin(packed)) + ", " + repr(raw_pin(raw_archive)) + ", " + repr(all_pins)
             + ", '/kaggle/working/F009_NATIVE_RUNTIME')\n")
    parent_runner = ("import hashlib as _f009_hashlib\nimport json as _f009_json\n"
        "_f009_parent_body = (_f009_runtime / 'P033-source.ipynb').read_bytes()\n"
        f"if _f009_hashlib.sha256(_f009_parent_body).hexdigest() != {PARENT_SHA!r}:\n"
        "    raise ValueError('Exact confirmed P033 parent body differs')\n"
        "_f009_parent_book = _f009_json.loads(_f009_parent_body)\n"
        "_f009_parent_code = [c['source'] if isinstance(c['source'],str) else ''.join(c['source']) "
        "for c in _f009_parent_book['cells'] if c['cell_type']=='code']\n"
        f"if [_f009_hashlib.sha256(s.encode()).hexdigest() for s in _f009_parent_code] != {control['parent_code_sha256']!r}:\n"
        "    raise ValueError('Exact eight P033 code cells differ')\n"
        "_f009_parent_completed = 0\n"
        "for _f009_cell in _f009_parent_code:\n"
        "    _f009_run = get_ipython().run_cell(_f009_cell,store_history=False)\n"
        "    _f009_error = _f009_run.error_before_exec or _f009_run.error_in_exec\n"
        "    if _f009_error is not None:\n        raise _f009_error\n"
        "    _f009_parent_completed += 1\n"
        "if _f009_parent_completed != 8:\n    raise RuntimeError('All eight P033 cells must complete')\n")
    last = ("_f009_extension_namespace = {'__name__':'_f009_parent_extension'}\n"
        "exec(compile((_f009_runtime/'f009_parent_extension.py').read_bytes(),"
        "'f009_parent_extension.py','exec'),_f009_extension_namespace)\n"
        "_f009_result = _f009_extension_namespace['execute'](globals(),runtime=_f009_runtime,"
        "started=_f009_started,mode='serving',diagnostic_count=192,blend_policy='success-subset',native_executor='isolated-uid')\n"
        "if (_f009_result.get('complete') is not True or _f009_result.get('native_complete') is not True "
        "or _f009_result.get('native_contribution_used') is not True or _f009_result.get('all_native_disabled') is not False "
        "or _f009_result.get('operational_recovery_contract_passed') is not True):\n"
        "    raise RuntimeError('F009 extension failed; restored parent is an artifact, never candidate success')\n")
    notice = {"cell_type": "markdown", "metadata": {}, "source":
        "# Conditional F009 CLS pair + fixed P033 parent\n\nSource preparation only unless the separately qualified "
        "actual release is enabled. P033 eight scientific cells are exact. Only the two CLS roles change; old REG "
        "refs, UID180, four full forwards, failure union and success-subset20 math remain fixed. "
        "Old bootstrap/model preflight is never executed. Actual inference and Public improvement are unqualified.\n"}
    book = {"cells": [notice, *[cell(text) for text in (first, setup, parent_runner, last)]],
            "metadata": copy.deepcopy(control["book"]["metadata"]), "nbformat": 4, "nbformat_minor": 5}
    for text in code_cells(book):
        compile(text, "F009-carrier-cell", "exec")
    raw = (json.dumps(book, ensure_ascii=False, indent=1) + "\n").encode()
    require(len(raw) <= 950000, "Notebook source capacity exceeded")
    require(context["actual_entry_enabled"] is entry_enabled, "Context/first-cell actual gate differs")
    return raw, raw_archive, all_pins


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--slug", default="rsraki/f009-p033-native20-source-20261017-v1")
    parser.add_argument("--execute", action="store_true", help="Future qualified metadata release only; no API calls")
    parser.add_argument("--release", type=Path)
    parser.add_argument("--expected-release-sha256")
    args = parser.parse_args(argv)
    if args.output is None:
        require(not args.execute and args.release is None and args.expected_release_sha256 is None,
                "Explicit output and actual release required")
        print(json.dumps({"source_preparation_only": True, "input_bodies_read": 0,
                          "actual_pack": False, "API_Save_Submit": 0}))
        return 0
    result = prepare(args.output, slug=args.slug, execute=args.execute, release=args.release,
                     expected_release_sha256=args.expected_release_sha256)
    print(json.dumps({"source": result["source"], "actual_entry_enabled": result["actual_entry_enabled"],
                      "Save_Submit": 0}))
    return 0


def prepare(output, *, slug="rsraki/f009-p033-native20-source-20261017-v1", execute=False,
            release=None, expected_release_sha256=None):
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink(), "Fresh F009 Notebook preparation directory required")
    require(re.fullmatch(r"rsraki/f009-p033-native20-[a-z0-9-]+", slug), "Fixed own private slug namespace required")
    require(type(execute) is bool and (execute or (release is None and expected_release_sha256 is None)),
            "Actual release arguments require explicit --execute")
    if execute:
        require(release is not None and re.fullmatch("[0-9a-f]{64}", expected_release_sha256 or ""),
                "Explicit qualified root release path/SHA required")
    control = inspect_control()
    with new_support_modules() as (driver, validator):
        files = assemble_sources(control, driver)
        source_inventory = {name: raw_pin(body) for name, body in files.items()}
        require(sum(row["bytes"] for row in source_inventory.values()) <= MAX_SOURCE, "Source-only body cap exceeded")
        if execute:
            context, metadata = validator.prepare_context(release, expected_release_sha256,
                source_inventory=source_inventory, runtime_paths=runtime_paths())
        else:
            context, metadata = validator.context_template(source_inventory=source_inventory,
                                                           runtime_paths=runtime_paths())
        require(isinstance(context, dict) and context["actual_entry_enabled"] is execute,
                "Validator must distinguish actual from source-only scope")
        require(type(metadata) is dict and "source-inventory.json" in metadata,
                "Validator-generated exact source inventory required")
        for name, body in metadata.items():
            safe_name(name)
            require((name.startswith("metadata/") or name == "source-inventory.json") and type(body) is bytes
                    and name.endswith((".json", ".py")) and len(body) <= 16 * 1024**2,
                    "Bounded nondata metadata bytes only")
        require(json.loads(metadata["source-inventory.json"]) == source_inventory,
                "Returned source inventory cannot change member pins")
        files.update(control["reg_metadata"])
        for name, body in metadata.items():
            require(name not in files or files[name] == body, "Context cannot replace pinned source/REG metadata")
            files[name] = body
        plan_name = "metadata/conditional-plan.json"
        require(plan_name not in files or files[plan_name] == control["plan_raw"], "Exact conditional plan collision")
        files[plan_name] = control["plan_raw"]
        files["f009-context.json"] = json_bytes(context)
        files["f009-context.sha256"] = (raw_pin(files["f009-context.json"])["sha256"] + "\n").encode("ascii")
        raw, archive, all_pins = make_notebook(files, control, context, entry_enabled=execute)
        payload = manual_payload(context, raw.decode(), slug) if execute else None
    manifest = {"schema_version": "f009_P033_native20_Notebook_source_preparation_v1",
        "source": raw_pin(raw), "runtime_archive": raw_pin(archive), "runtime_files": all_pins,
        "source_inventory": raw_pin(files["source-inventory.json"]),
        "context": raw_pin(files["f009-context.json"]), "final_context_root_review_required_before_Save": True,
        "source_member_count": len(source_inventory), "old_component_source_count": len(control["components"]),
        "parent_source_sha256": PARENT_SHA, "parent_code_sha256": control["parent_code_sha256"],
        "conditional_plan_sha256": PLAN_SHA, "actual_entry_enabled": execute,
        "actual_metadata_release_verified": execute, "actual_Runtime_GPU_or_Public_qualified": False,
        "old_bootstrap_executed": False, "old_CLS_Input_or_model_preflight_used": False,
        "old_source89_qualification_inherited": False, "native_decoder_UID_failure_numeric_algorithms_changed": False,
        "metadata_bodies_not_checkpoint_model_MRI": True, "manual_save_payload_present": payload is not None,
        "API_upload_Save_Submit": 0, "builder": raw_pin(Path(__file__).read_bytes())}
    manifest["entry_scope"] = {"serving_source_only": True, "same192_diagnostic_entrypoint_implemented": False,
        "diagnostic_count_192_argument": "serving CLI compatibility validation only; not diagnostic execution",
        "conditional_plan_same192_then_visible_qualification_required_and_outstanding": True}
    staging = output.with_name(output.name + ".partial")
    parent = output.parent.resolve()
    require(output.resolve() == parent / output.name and staging.resolve() == parent / staging.name
            and not staging.exists() and not staging.is_symlink(), "Fresh owned same-parent preparation stage required")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        artifacts = {"source.ipynb": raw, "runtime-bundle.zip": archive, "manifest.json": json_bytes(manifest),
            "context.json": files["f009-context.json"], "source-inventory.json": files["source-inventory.json"]}
        if payload is not None:
            artifacts["save-payload.json"] = json_bytes(payload)
        for name, body in artifacts.items():
            (staging / name).write_bytes(body)
        require(not output.exists() and not output.is_symlink() and staging.resolve() == parent / staging.name,
                "Checked fresh preparation publication required")
        os.rename(staging, output)
    except BaseException as error:
        (staging / "PREPARATION_FAILED.json").write_bytes(json_bytes({"complete": False, "error_type": type(error).__name__}))
        raise
    return manifest


def manual_payload(context, notebook_text, slug):
    """Construct data only; this module has no API client or uploader."""
    jobs, inputs = context["transport_jobs"], context["inputs"]
    require(context["actual_entry_enabled"] is True and context["artificial_fixture_only"] is False,
            "Actual root context required for a manual payload")
    datasets = [*PARENT_DATASETS, jobs["CLS:0"]["dataset_ref"], jobs["CLS:1"]["dataset_ref"],
                jobs["REG:0"]["dataset_ref"], jobs["REG:1"]["dataset_ref"], DECODER]
    require(inputs == {"datasets": datasets, "kernels": [STAGE], "competitions": [COMPETITION], "total": 9},
            "Only new CLS2 Inputs may replace old P039 Inputs")
    require(all(re.fullmatch(r"rsraki/[a-z0-9-]+/1", jobs["CLS:"+str(f)]["dataset_ref"])
                for f in (0, 1)) and jobs["CLS:0"]["dataset_ref"] != jobs["CLS:1"]["dataset_ref"],
            "Two distinct explicitly frozen own private CLS Inputs V1 required")
    return {"slug": slug, "new_title": slug.split("/")[1], "text": notebook_text,
        "language": "python", "kernel_type": "notebook", "dataset_data_sources": datasets,
        "kernel_data_sources": [STAGE], "competition_data_sources": [COMPETITION], "is_private": True,
        "enable_gpu": True, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "docker_image": DOCKER, "docker_image_pinning_type": "original",
        "kernel_execution_type": "SAVE_AND_RUN_ALL"}


if __name__ == "__main__":
    raise SystemExit(main())
