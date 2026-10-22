"""Prepare inert five-private-Input CPU proof, without API, model or MRI work."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import lzma
import re
from pathlib import Path

import prepare_f009_p033_native20 as B
from f013_p039_cpu_qualifier import POLICY_SHA

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "artifacts/research/20261020-F013-P039-mixed-CPU-prospective-v1/policy.json"


def prepare(output, slug):
    output = Path(output)
    B.require(not output.exists() and not output.is_symlink(), "Fresh output")
    B.require(re.fullmatch(r"rsraki/f013-mixed-cpu-[a-z0-9-]{5,30}", slug), "Own unique CPU qualifier slug")
    B.require(B.raw_pin(POLICY.read_bytes())["sha256"] == POLICY_SHA, "Frozen mixed CPU policy")
    policy = json.loads(POLICY.read_bytes())
    for name, pin in policy["pins"].items():
        B.require(B.raw_pin((ROOT / name).read_bytes()) == pin, "Frozen predecessor/source differs")
    prior = json.loads((ROOT / "experiments/F013-fixed-E12-CPU-qualified-not-release-20261020.json").read_bytes())
    B.require(prior["all_pass"] is True and all(prior["checks"].values()), "Actual prior standalone CPU proof")
    control = B.inspect_control()
    files = dict(control["outer_files"])
    for name in ("f013_cpu_qualifier.py", "f013_p039_bridge.py", "f013_p039_cpu_qualifier.py"):
        files[name] = (ROOT / "scripts" / name).read_text(encoding="utf8").encode()
    files["mixed-CPU-policy.json"] = POLICY.read_bytes()
    inventory = {name: B.raw_pin(body) for name, body in files.items()}
    raw = B.zip_sources(files)
    packed = lzma.compress(raw, preset=6)
    loader = ("import os,sys\nos.environ['CUDA_VISIBLE_DEVICES']=''\nos.environ['XFORMERS_DISABLED']='1'\n"
              "_f013_restorer={'__name__':'_f013_mixed_CPU_restore'}\n"
              f"exec(compile({B.RESTORER!r},'F013-CPU-source-restore','exec'),_f013_restorer)\n"
              "_f013_runtime=_f013_restorer['f009_restore_sources']("
              f"{base64.b85encode(packed).decode()!r},{B.raw_pin(packed)!r},{B.raw_pin(raw)!r},{inventory!r},'/kaggle/working/F013_P039_CPU_RUNTIME')\n"
              "sys.path.insert(0,str(_f013_runtime))\nimport f013_p039_cpu_qualifier\n"
              "_f013_result=f013_p039_cpu_qualifier.run(_f013_runtime,'/kaggle/working/F013-P039-current-CPU')\n")
    compile(loader, "F013-CPU-loader", "exec")
    book = {"cells": [{"cell_type": "markdown", "metadata": {}, "source":
              "# F013/P039 actual CPU constructor qualification\n\nFive own private model Inputs only. No competition, MRI, Report, labels, Gold, CUDA or training. Original P039 source/preflight/CPU model guards run before the exact new F013 CLS constructor. Old F009 HOLD stays unchanged. Never submit this notebook. CPU evidence cannot enable serving or replace GPU/MRI qualification.\n"}, B.cell(loader)],
              "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
    source = (json.dumps(book, indent=1) + "\n").encode()
    B.require(len(source) < 950000, "Notebook source size cap")
    old_payload = json.loads((B.CONTROL / "save-payload.json").read_bytes())
    inputs = [name for name in old_payload["dataset_data_sources"] if name.startswith("rsraki/n392")]
    B.require(len(inputs) == 4, "Exactly the unchanged four own model Inputs")
    inputs.append("rsraki/f013-gamma-e12-qualification-20261020-v1/1")
    payload = json.loads((ROOT / "artifacts/kaggle/f013-fixed-e12-cpu-qualification-20261020-v1/save-payload.json").read_bytes())
    payload.update(slug=slug, new_title=slug.split("/")[1], text=source.decode(), dataset_data_sources=inputs,
                   session_timeout_seconds=1800)
    B.require(payload["enable_gpu"] is False and not payload["kernel_data_sources"] and not payload["competition_data_sources"], "CPU only, no competition or notebook Input")
    manifest = {"schema_version": "F013_P039_current_mixed_CPU_source_preparation_v1", "policy_SHA": POLICY_SHA,
                "source_SHA": hashlib.sha256(source).hexdigest(), "runtime_files": inventory, "runtime_archive": B.raw_pin(raw),
                "private_five_Inputs": inputs, "CPU_only_no_MRI": True, "actual_execution_or_API_Save_Submit": 0}
    output.mkdir(parents=True, exist_ok=False)
    for name, body in (("source.ipynb", source), ("runtime-sources.bin", raw), ("manifest.json", B.json_bytes(manifest)),
                       ("save-payload.json", B.json_bytes(payload))):
        (output / name).write_bytes(body)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slug", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.slug), indent=2))
