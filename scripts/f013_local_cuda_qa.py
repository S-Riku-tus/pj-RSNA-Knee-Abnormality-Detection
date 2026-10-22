"""Exact F013 artificial CUDA checks on a naturally free local RTX 4090.

Only the old helper's explicit device-name constant is adapted. No network,
uploads, MRI, labels, model fitting, or legacy qualification edits occur.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import f013_cpu_qualifier as C

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "artifacts/research/20261021-F013-local-RTX4090-CUDA-prospective-v1/policy.json"
POLICY_SHA = "5da3be045253c1d52582d42da669fa24b31767924c78b19b8581e33334fd57f4"
ORIGINAL = ROOT / "scripts/f013_cuda_qa.py"
ORIGINAL_SHA = "d55b4cf4bd3aec386d10251add5be29693b3be3eb0e7bf8a79f663dfcd6b3d75"
GENERIC = ROOT / "artifacts/research/20261012-f006-fixed10-secondary-seal-source-v1/candidateAUC12-fold0"
STAGE = ROOT / "artifacts/staging/f013-fixed-E12-qualification-20261020-v2"
GPU = "NVIDIA GeForce RTX 4090"


def derive_child(raw):
    C.require(hashlib.sha256(raw).hexdigest() == ORIGINAL_SHA, "Exact original CUDA helper")
    original = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == "child")
    candidate, changed = copy.deepcopy(original), []

    class Hardware(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == "Tesla T4":
                changed.append(True)
                return ast.copy_location(ast.Constant(value=GPU), node)
            return node

    candidate = Hardware().visit(candidate)
    C.require(len(changed) == 1, "Only one explicit device-name constant")

    class Undo(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == GPU:
                return ast.copy_location(ast.Constant(value="Tesla T4"), node)
            return node

    C.require(ast.dump(Undo().visit(copy.deepcopy(candidate))) == ast.dump(original), "All nonhardware checks exactly preserved")
    return ast.fix_missing_locations(ast.Module(body=[candidate], type_ignores=[]))


def verify_pins():
    C.require(C.sha(POLICY) == POLICY_SHA, "Frozen local GPU policy")
    policy = json.loads(POLICY.read_bytes())
    for ref in policy["pins"].values():
        path = ROOT / ref["path"]
        C.require(path.stat().st_size == ref["bytes"] and C.sha(path) == ref["sha256"], "Fixed prior source/proof/assets")
    return policy


def resource_snapshot():
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.free,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=20, check=True)
    lines = result.stdout.strip().splitlines()
    C.require(len(lines) == 1, "One local GPU")
    name, free, used, util = [part.strip() for part in lines[0].split(",")]
    C.require(name == GPU and float(free) >= 16000, "Naturally free exact RTX4090; never stop other work")
    return {"GPU": name, "free_MiB": float(free), "used_MiB": float(used), "utilization_percent": float(util)}


def stage(output, policy):
    manifest = json.loads((STAGE / "package-manifest.json").read_bytes())
    archive = STAGE / "f013-models.bin"
    C.require(C.sha(archive) == manifest["archive"]["sha256"] and archive.stat().st_size == manifest["archive"]["bytes"], "Fixed local own checkpoint archive")
    package = output / "package"
    C.extract(archive, manifest["members"], package, exact=True)
    generic = json.loads((ROOT / policy["pins"]["generic_manifest"]["path"]).read_bytes())
    inventory = {name: ref for name, ref in generic["members"].items() if name.startswith("generic/")}
    C.require(len(inventory) == 162, "Only 162 generic members, no previous trained checkpoints")
    for name, ref in inventory.items():
        source, target = GENERIC / name, package / name
        C.require(source.is_file() and not source.is_symlink() and source.stat().st_size == ref["bytes"]
                  and C.sha(source) == ref["sha256"], "Original generic bytes")
        target.parent.mkdir(parents=True, exist_ok=True)
        C.require(not target.exists(), "Fresh independent generic copy")
        shutil.copyfile(source, target)
        C.require(C.sha(target) == ref["sha256"], "Exact copied generic bytes")
    return package, {**manifest["members"], **inventory}


def run(output):
    output = Path(output).resolve()
    C.require(output.is_relative_to((ROOT / "artifacts").resolve()) and not output.exists(), "Fresh workspace run")
    policy = verify_pins()
    derive_child(ORIGINAL.read_bytes())
    C.require("torch" not in sys.modules, "No parent Torch/model work")
    resource_snapshot()
    output.mkdir(parents=True, exist_ok=False)
    (output / "policy.json").write_bytes(POLICY.read_bytes())
    package, inventory = stage(output, policy)
    receipts, resources = [], []
    for fold in (0, 1):
        resources.append(resource_snapshot())
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", CUBLAS_WORKSPACE_CONFIG=":4096:8", XFORMERS_DISABLED="1",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        with (output / ("fold%d.stdout.txt" % fold)).open("x") as out, (output / ("fold%d.stderr.txt" % fold)).open("x") as err:
            completed = subprocess.run([sys.executable, "-u", __file__, "--child", str(fold), "--package", str(package),
                        "--output", str(output / ("fold%d.json" % fold))], env=env, stdout=out, stderr=err, timeout=600, check=False)
        C.require(completed.returncode == 0, "Local actual CUDA child failed fold%d; logs retained" % fold)
        item = json.loads((output / ("fold%d.json" % fold)).read_bytes())
        C.require(item["complete"] and item["fold"] == fold and item["GPU"] == GPU, "Both actual local fold proofs")
        receipts.append(item)
    for name, ref in inventory.items():
        C.require(C.sha(package / name) == ref["sha256"], "All staged source/checkpoint/generic bytes preserved")
    verify_pins()
    result = {"schema_version": "F013_local_RTX4090_actual_artificial_CUDA_v1", "complete": True,
              "policy_SHA": POLICY_SHA, "source_SHA": C.sha(__file__), "original_GPU_helper_SHA": ORIGINAL_SHA,
              "only_one_hardware_guard_constant_changed_inverse_AST_exact": True, "GPU": GPU, "devices": 1,
              "resources_before_each_fold": resources, "folds": receipts,
              "all_181_staged_members_preserved": len(inventory) == 181, "original_F009_HOLD_unchanged": True,
              "MRI_Report_Gold_training_API_Save_Submit_upload": 0,
              "Kaggle_T4_realMRI_fusion_or_Public_qualified": False,
              "future_actual_current_8_models_twoT4_GPU_QA_still_required_before_nativeMRI": True}
    (output / "completion.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps({"complete": True, "output": output.as_posix(), "GPU": GPU,
                      "fold_probability_maxabs": [r["CPUmanual_FP32_reference_probability_maxabs"] for r in receipts]}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", type=int, choices=(0, 1))
    parser.add_argument("--package", type=Path)
    args = parser.parse_args()
    if args.child is None:
        run(args.output)
    else:
        verify_pins()
        namespace = {"__name__": "_f013_local_hardware_adapter", "C": C, "os": os, "sys": sys, "json": json, "Path": Path}
        exec(compile(derive_child(ORIGINAL.read_bytes()), "F013-original-CUDA-child-local-hardware", "exec"), namespace)
        namespace["child"](args.package, args.child, args.output)
