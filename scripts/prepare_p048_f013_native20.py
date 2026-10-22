"""Build a fresh, inert P048 carrier. No network, uploads, submission or Torch."""
from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import json
import lzma
import re
from pathlib import Path

import prepare_f009_p033_native20 as B
from f013_cuda_evidence import validate_prior_cuda
from f013_p039_bridge import NATIVE50_POLICY_SHA, NATIVE50_SHA, POLICY_SHA

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "artifacts/research/20261020-P048-F013-P039-bridge-prospective-v1/policy.json"
NATIVE50 = ROOT / "artifacts/research/20261020-P048-native50-paired-weak-prospective-v1/result.json"


def replace_function(source, name, replacement):
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    B.require(len(nodes) == 1, "One named child integration boundary")
    node = nodes[0]
    lines = source.splitlines(keepends=True)
    return "".join(lines[:node.lineno - 1]) + replacement.rstrip() + "\n" + "".join(lines[node.end_lineno:])


def child_source(old):
    source = old.decode()
    replacement = '''def execute_native_router(args, active, plan, router, *, task_source_sha256, started):
    import f013_p039_bridge
    return f013_p039_bridge.execute(args, active, plan, router,
        task_source_sha256=task_source_sha256, started=started, runtime=Path(__file__).parent)
'''
    source = replace_function(source, "execute_native_router", replacement)
    source = B.replace_once(source, 'schema_version="p035_f006_isolated_uid_adapter_v1"', 'schema_version="p048_f013_isolated_uid_adapter_v1"')
    source = B.replace_once(source, '"model_fingerprints": result["reconstruction_component"]["model_fingerprints"],',
                '"model_fingerprints": result["reconstruction_component"]["model_fingerprints"],\n'
                '        "F013_current_constructor_qualification": result["F013_current_constructor_qualification"],\n'
                '        "F013_bridge_policy_SHA": result["F013_bridge_policy_SHA"],\n'
                '        "old_F009_HOLD_unchanged": result["old_F009_HOLD_unchanged"],')
    compile(source, "p048_native_child.py", "exec")
    return source.encode()


def extension_source(old):
    source = old.decode()
    for before, after in (
        ('"p039_p033_f006_fixed20_execution_v1"', '"p048_F013_CLS_P039_REG_fixed20_execution_v1"'),
        ('str(package / "p032_native_child.py")', 'str(package / "p048_native_child.py")'),
        ('"p035_f006_isolated_uid_adapter_v1"', '"p048_f013_isolated_uid_adapter_v1"'),
    ):
        count = source.count(before)
        B.require(count in (1, 2), "Exact parent adapter boundary")
        source = source.replace(before, after)
    source = source.replace('"F006_native_rank"', '"F013_CLS_original_REG_native_rank"')
    source = B.replace_once(source, 'or adapter.get("native_executor") != "isolated-uid"',
                'or adapter.get("native_executor") != "isolated-uid"\n'
                '                    or adapter.get("F013_current_constructor_qualification") is not True\n'
                f'                    or adapter.get("F013_bridge_policy_SHA") != {POLICY_SHA!r}\n'
                '                    or adapter.get("old_F009_HOLD_unchanged") is not True')
    compile(source, "p048_parent_extension.py", "exec")
    return source.encode()


def qualified_release(evidence):
    weak_raw = NATIVE50.read_bytes()
    B.require(hashlib.sha256(weak_raw).hexdigest() == NATIVE50_SHA, "Frozen fixed native50 weak result")
    weak = json.loads(weak_raw)
    B.require(weak["complete"] is True and weak["bothfold_fixed10_positive"] is True
              and weak["policy_SHA"] == NATIVE50_POLICY_SHA
              and all(weak["folds"][str(f)]["fixed10_delta"] > 0 for f in (0, 1)), "Both fixed weak family deltas positive")
    if evidence is None:
        return {"actual_entry_enabled": False, "both_fixed_folds_artificial_CUDA_pass": False,
                "native50_check_SHA": NATIVE50_SHA, "old_F009_HOLD_unchanged": True}, None
    path = Path(evidence).resolve(strict=True)
    B.require(path.is_relative_to((ROOT / "artifacts/research").resolve()) and path.suffix == ".json", "Local root qualification evidence")
    raw = path.read_bytes()
    value = validate_prior_cuda(raw)
    return {"actual_entry_enabled": True, "both_fixed_folds_artificial_CUDA_pass": True,
            "CUDA_evidence_SHA": hashlib.sha256(raw).hexdigest(), "old_F009_HOLD_unchanged": True,
            "native50_check_SHA": NATIVE50_SHA,
            "prior_CUDA_hardware": value["GPU"], "current_twoT4_GPU_gate_still_required": True,
            "scope": "Own new F013 execution eligibility; not old F009 release, MRI, accuracy or formal Submit approval"}, raw


def prepare(output, slug, *, cuda_evidence=None):
    output = Path(output)
    B.require(not output.exists() and not output.is_symlink(), "Fresh output")
    B.require(re.fullmatch(r"rsraki/p048-[a-z0-9-]{5,43}", slug), "Own distinct P048 slug")
    B.require(B.raw_pin(POLICY.read_bytes())["sha256"] == POLICY_SHA, "Frozen prospective bridge policy")
    policy = json.loads(POLICY.read_bytes())
    for name, pin in policy["pins"].items():
        B.require(B.raw_pin((ROOT / name).read_bytes()) == pin, "Frozen original proof/source differs")
    release, proof = qualified_release(cuda_evidence)
    control = B.inspect_control()
    files = copy.deepcopy(control["outer_files"])
    original_child = files.pop("p032_native_child.py")
    original_extension = files.pop("p032_parent_extension.py")
    files["p048_native_child.py"] = child_source(original_child)
    files["p048_parent_extension.py"] = extension_source(original_extension)
    for name in ("f013_p039_bridge.py", "f013_cpu_qualifier.py", "f013_cuda_evidence.py"):
        files[name] = (ROOT / "scripts" / name).read_text(encoding="utf8").encode()
    files["P048-bridge-policy.json"] = POLICY.read_bytes()
    files["P048-release.json"] = B.json_bytes(release)
    files["P048-native50-weak-check.json"] = NATIVE50.read_bytes()
    if proof is not None:
        files["P048-prior-CUDA-evidence.json"] = proof
    raw_archive = B.zip_sources(files)
    packed = lzma.compress(raw_archive, preset=6)
    inventory = {name: B.raw_pin(body) for name, body in files.items()}
    first = (f"_p048_enabled={release['actual_entry_enabled']!r}\n"
             "if _p048_enabled is not True:\n    raise RuntimeError('P048 source only: actual prior CUDA proof absent')\n"
             "import time\n_p048_started=time.monotonic()\n")
    setup = ("_p048_restore={'__name__':'_p048_source_restore'}\n"
             f"exec(compile({B.RESTORER!r},'p048-source-restore','exec'),_p048_restore)\n"
             "_p048_runtime=_p048_restore['f009_restore_sources']("
             f"{base64.b85encode(packed).decode()!r},{B.raw_pin(packed)!r},{B.raw_pin(raw_archive)!r},{inventory!r},'/kaggle/working/P048_NATIVE_RUNTIME')\n")
    # The complete P033 notebook is a byte-pinned member; execute all eight
    # original cells without editing any scientific cell or its failure policy.
    parent = ("import json,hashlib\n"
              "_p048_parent=(_p048_runtime/'P033-source.ipynb').read_bytes()\n"
              f"assert hashlib.sha256(_p048_parent).hexdigest()=={B.PARENT_SHA!r}\n"
              "_p048_cells=[c['source'] if isinstance(c['source'],str) else ''.join(c['source']) for c in json.loads(_p048_parent)['cells'] if c['cell_type']=='code']\n"
              f"assert [hashlib.sha256(c.encode()).hexdigest() for c in _p048_cells]=={control['parent_code_sha256']!r}\n"
              "for _p048_cell in _p048_cells:\n"
              "    _p048_run=get_ipython().run_cell(_p048_cell,store_history=False)\n"
              "    _p048_error=_p048_run.error_before_exec or _p048_run.error_in_exec\n"
              "    if _p048_error is not None:\n        raise _p048_error\n")
    last = ("_p048_extension={'__name__':'_p048_parent_extension'}\n"
            "exec(compile((_p048_runtime/'p048_parent_extension.py').read_bytes(),'p048_parent_extension.py','exec'),_p048_extension)\n"
            "_p048_result=_p048_extension['execute'](globals(),runtime=_p048_runtime,started=_p048_started,mode='serving',diagnostic_count=192,blend_policy='success-subset',native_executor='isolated-uid')\n"
            "assert _p048_result['complete'] and _p048_result['native_complete'] and _p048_result['native_contribution_used'] and not _p048_result['all_native_disabled']\n")
    book = {"cells": [{"cell_type": "markdown", "metadata": {}, "source":
              "# P048 F013 CLS + original P039 REG, fixed native20\n\nSeparate exploratory gamma-trained CLS pair. Original F009 HOLD retained. No actual entry without prior artificial CUDA proof; current eight GPU models must pass artificial checks before MRI. No Public gain claimed.\n"},
              *[B.cell(s) for s in (first, setup, parent, last)]],
              "metadata": copy.deepcopy(control["book"]["metadata"]), "nbformat": 4, "nbformat_minor": 5}
    for source in B.code_cells(book):
        compile(source, "P048-cell", "exec")
    raw = (json.dumps(book, indent=1) + "\n").encode()
    B.require(len(raw) < 950000, "Kaggle source size cap")
    payload = json.loads((B.CONTROL / "save-payload.json").read_bytes())
    payload.update(slug=slug, new_title=slug.split("/")[1], text=raw.decode())
    payload["dataset_data_sources"].append("rsraki/f013-gamma-e12-qualification-20261020-v1/1")
    manifest = {"schema_version": "P048_F013_CLS_P039_REG_source_preparation_v1", "policy_SHA": POLICY_SHA,
                "source_SHA": B.raw_pin(raw)["sha256"], "original_P039_source_SHA": B.CONTROL_SHA,
                "P033_exact_eight_cell_SHA": control["parent_code_sha256"], "runtime_files": inventory,
                "runtime_archive": B.raw_pin(raw_archive), "actual_entry_enabled": release["actual_entry_enabled"],
                "actual_GPU_or_MRI_executed": False, "formal_Submit": 0, "API_calls": 0}
    output.mkdir(parents=True, exist_ok=False)
    for name, body in (("source.ipynb", raw), ("runtime-sources.bin", raw_archive),
                       ("save-payload.json", B.json_bytes(payload)), ("manifest.json", B.json_bytes(manifest))):
        (output / name).write_bytes(body)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--cuda-evidence", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.slug, cuda_evidence=args.cuda_evidence), indent=2))
