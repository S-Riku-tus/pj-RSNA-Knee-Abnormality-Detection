"""Build a fixed staged-input artificial GPU qualification without importing it.

Only stdlib AST/text/hash operations run locally. No Torch/model/source import,
MRI, network, GPU, training, upload, API mutation, Save or Submit is performed.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OBSERVATION = ROOT / "artifacts/qa/od1/20261017T032200330465Z/files/orthodiffusion-sagittal-stage"
RUNTIME = ROOT / "scripts/orthodiffusion_artificial_gpu_runtime.py"
OUTPUT = ROOT / "artifacts/kaggle/orthodiff-sag-artificial-gpu-20261017-v1"
STAGE = "rsraki/orthodiff-sag-assets-20261017-v3-enabled/1"
CODE_COMMIT = "99d15186bb8face1728263023d8414d9405c8f7e"
HF_COMMIT = "d6920d49e7d1b3dd0774eed6b8e47e386a54dd81"
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"
SOURCE_BLOBS = {
    "LICENSE": (1065, "2c85895159d871d56e2ddd1237828aeb93ffc76c"),
    "requirements.txt": (1381, "3986e1d9de896356b210d9f0ee1359c593a37014"),
    "configs/config_linear.yaml": (1018, "a4b2e8ce7deea4978e79f3b1b3878181a49f6c6b"),
    "diffusion_model/unet.py": (14680, "b5df551bee5050bcc73435e290c1c93918010cba"),
    "diffusion_model/modules.py": (16751, "d739fbdef7f724c1ca64e45da1d2173f8feb9a74"),
    "diffusion_model/fp16_util.py": (8009, "b0ac5b288c73961540916f0091d02e74e443ceb4"),
    "diffusion_model/logger.py": (14046, "b3aa971d5a1b2b73554beca2945132f4bef1e89d"),
    "diffusion_model/trainer.py": (22894, "668480ba5ae99901dabffeddff9b11a521aa867f"),
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def pin(raw):
    return {"bytes": len(raw), "sha256": digest(raw)}


def encode_json(value):
    return (json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n").encode()


def fragment(text, node):
    return "".join(text.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])


def source_package(observation=OBSERVATION):
    """Verify all available original text against the immutable commit tree."""
    observation = Path(observation)
    manifest_raw = (observation / "asset-manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    rows = {row["path"]: row for row in manifest["assets"]}
    expected_paths = {"upstream/" + key for key in SOURCE_BLOBS} | {"weights/sagittal_model.pt"}
    if len(rows) != 9 or set(rows) != expected_paths:
        raise ValueError("Original fixed nine-asset package required")
    texts = {}
    for relative, (size, blob) in SOURCE_BLOBS.items():
        raw = (observation / "upstream" / relative).read_bytes()
        actual_blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        row = rows["upstream/" + relative]
        if (len(raw), actual_blob, row["bytes"], row["git_blob_sha1"], digest(raw)) != (size, blob, size, blob, row["sha256"]):
            raise ValueError("Actual source bytes or immutable original blob differ: " + relative)
        texts[relative] = raw.decode("utf-8")
    weight = rows["weights/sagittal_model.pt"]
    if (weight["bytes"], weight["sha256"]) != (552919566, "3afc323c58b6446c1ea6dc4f09133d744e5a1609dba7662ac9a15ca688e34951"):
        raise ValueError("Fixed positive LFS checkpoint pin required")
    # Checkpoint remains remote only: never open/download/deserialize it locally.
    return manifest_raw, manifest, texts


def exact_selected(text, *, functions=(), methods=()):
    tree = ast.parse(text)
    selected = [next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name) for name in functions]
    if methods:
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GaussianDiffusion")
        selected.extend(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name) for name in methods)
    return selected


def vendor_package(texts):
    """Preserve original UNet/modules bytes; exact AST fragments only for trims."""
    selected_fp = exact_selected(texts["diffusion_model/fp16_util.py"], functions=("convert_module_to_f16", "convert_module_to_f32"))
    fp16 = "# Exact two upstream conversion functions; original LICENSE applies.\nimport torch.nn as nn\n\n" + "\n\n".join(fragment(texts["diffusion_model/fp16_util.py"], node) for node in selected_fp) + "\n"
    selected_diff = exact_selected(texts["diffusion_model/trainer.py"],
                                   functions=("exists", "default", "extract", "cosine_beta_schedule"),
                                   methods=("__init__", "q_sample", "get_feature"))
    features = ("# Exact upstream feature helpers/methods; original LICENSE applies.\n"
                "from functools import partial\nfrom inspect import isfunction\nimport numpy as np\nimport torch\n"
                "from torch import nn\nfrom torch.cuda.amp import autocast\n\n"
                + "\n\n".join(fragment(texts["diffusion_model/trainer.py"], node) for node in selected_diff[:4])
                + "\n\nclass GaussianDiffusion(nn.Module):\n"
                + "\n\n".join(fragment(texts["diffusion_model/trainer.py"], node) for node in selected_diff[4:]) + "\n")
    pieces = {"__init__.py": "", "unet.py": texts["diffusion_model/unet.py"],
              "modules.py": texts["diffusion_model/modules.py"], "fp16_util.py": fp16,
              "features.py": features, "LICENSE": texts["LICENSE"]}
    original = [ast.dump(node, include_attributes=False) for node in selected_fp + selected_diff]
    new_fp = [node for node in ast.parse(fp16).body if isinstance(node, ast.FunctionDef)]
    new_diff = ast.parse(features)
    new_helpers = [node for node in new_diff.body if isinstance(node, ast.FunctionDef)]
    new_cls = next(node for node in new_diff.body if isinstance(node, ast.ClassDef))
    if original != [ast.dump(node, include_attributes=False) for node in new_fp + new_helpers + new_cls.body]:
        raise ValueError("Extracted scientific functions/method AST changed")
    imports = {}
    for name, text in pieces.items():
        if name.endswith(".py"):
            tree = ast.parse(text)
            imports[name] = sorted({ast.unparse(node) for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))})
            compile(tree, name, "exec")  # Compile only; no upstream code executed/imported.
    audit = {"original_commit": CODE_COMMIT, "unet_and_modules_original_bytes": True,
             "trimmed_function_AST_exact": True, "imports_by_AST_only": imports,
             "closure_external": ["stdlib abc/math/functools/inspect", "numpy", "torch/nn/F/cuda.amp"],
             "not_imported": ["original trainer/Trainer", "original fp16_util/logger", "APEX", "NIfTI", "TensorBoard"],
             "logger_note": "logger optional TensorFlow imports are lazy; trimming avoids an unneeded logger dependency, not a claim that import necessarily loads TensorFlow",
             "upstream_attention_checkpoint_true_preserved": True,
             "source_imported_or_GPU_qualified_locally": False}
    return {name: {"text": text, **pin(text.encode())} for name, text in pieces.items()}, audit


def protocol():
    return {"version": "orthodiff_sagittal_artificial_v1", "direction": "sagittal", "input_shape": [1, 1, 16, 256, 256],
            "fixture": "torch.linspace(-1,1,16*256*256).reshape(1,1,16,256,256), FP32", "precision": "float32",
            "timestep": 100, "feature": "mid_2", "expected_feature_shape": [1, 256, 1, 16, 16],
            "pool": "feature.float().mean(dim=(2,3,4))", "expected_pool_shape": [1, 256], "AMP": False,
            "passes": 3, "seed_protocol": "UID/series/clip/baseSeed SHA256; first two use20261017, third20261018; selected CUDA default generator only in fork_rng",
            "strict_load": "checkpoint['ema']; only leading denoise_fn.module. rename; all full-model keys/shapes/dtypes/finite and thirteen exact schedule buffers",
            "minimum_free_cuda_bytes": 3 * 1024**3,
            "resource_limits": {"cuda_reserved_bytes": 14 * 1024**3, "process_RSS_bytes": 8 * 1024**3,
                                "one_pass_seconds": 120, "total_seconds": 600},
            "resource_limits_are_predeclared_artificial_qualification_only": True,
            "existing_sampled16_cache_is_faithful_continuous16": False,
            "new_32case_Kaggle_continuous16_MRI_pilot_required": True,
            "accuracy_or_Patient_independence_or_Gold_clean_claim": False}


def prepare(output=OUTPUT, slug="rsraki/orthodiff-sag-artificial-20261017-v1"):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a fresh artificial qualification run")
    manifest_raw, upstream, texts = source_package()
    vendor, audit = vendor_package(texts)
    runtime = RUNTIME.read_bytes()
    compile(ast.parse(runtime), "orthodiffusion_artificial_gpu_runtime.py", "exec")
    assets = [{key: row[key] for key in ("path", "bytes", "sha256", "git_blob_sha1")} for row in upstream["assets"]]
    config = {"input_kernel": {"ref": STAGE, "kernel_id": 139145710, "version": 1, "source_version_id": 359210870},
              "stage_manifest_pin": pin(manifest_raw), "assets": assets,
              "stage_candidates": ["/kaggle/input/notebooks/" + STAGE.rsplit("/", 1)[0] + "/orthodiffusion-sagittal-stage",
                                   "/kaggle/input/" + STAGE.split("/")[1] + "/orthodiffusion-sagittal-stage"],
              "code_commit": CODE_COMMIT, "HF_revision": HF_COMMIT, "protocol": protocol()}
    bundle_raw = encode_json({"runtime": runtime.decode(), "runtime_pin": pin(runtime), "config": config, "vendor": vendor})
    encoded = base64.b85encode(zlib.compress(bundle_raw, 9)).decode()
    cell = ("# Artificial tensors only. CUDA required. No competition/Dataset Input or network.\n"
            "import base64 as _od_b64, hashlib as _od_hash, json as _od_json, zlib as _od_zlib\n"
            f"_od_raw = _od_zlib.decompress(_od_b64.b85decode({encoded!r}))\n"
            f"assert _od_hash.sha256(_od_raw).hexdigest() == {digest(bundle_raw)!r}\n"
            "_od_bundle = _od_json.loads(_od_raw)\n"
            "_od_runtime = _od_bundle['runtime'].encode('utf-8')\n"
            "assert {'bytes':len(_od_runtime), 'sha256':_od_hash.sha256(_od_runtime).hexdigest()} == _od_bundle['runtime_pin']\n"
            "_od_ns = {'__name__':'orthodiffusion_artificial_candidate'}\n"
            "exec(compile(_od_runtime, '<pinned-artificial-runtime>', 'exec'), _od_ns)\n"
            "_od_result = _od_ns['run'](_od_bundle['config'], _od_bundle['vendor'])\n"
            "if not _od_result['complete']:\n    raise RuntimeError('Artificial GPU qualification failed; inspect ORTHODIFF_ARTIFICIAL_GPU.json')\n")
    notebook = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": "# OrthoDiffusion frozen sagittal artificial GPU qualification\n\nInput is the fixed private generic asset stage V1 only. No MRI, Report, labels, Gold, training, internet or submission. Full EMA strict load precedes three feature-only FP32 passes of the same synthetic tensor: a same-seed repeat and a different-seed check. Fixed t100/mid_2/global-mean pooling and resource gates are predeclared. This does not establish MRI fidelity, clinical performance, leaderboard improvement or hidden-test capacity. The existing sampled16 cache is not a continuous16 reproduction. New32case Kaggle MRI pilot is a separate later candidate. Original MIT LICENSE is preserved.\n"},
                            {"cell_type": "code", "metadata": {}, "source": cell, "execution_count": None, "outputs": []}],
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "kaggle": {"isGpuEnabled": True, "isInternetEnabled": False}}, "nbformat": 4, "nbformat_minor": 5}
    source = encode_json(notebook)
    if len(source) >= 950000:
        raise ValueError("Notebook capacity exceeded")
    payload = {"slug": slug, "new_title": slug.split("/")[1], "text": source.decode(), "language": "python",
               "kernel_type": "notebook", "dataset_data_sources": [], "competition_data_sources": [],
               "kernel_data_sources": [STAGE], "is_private": True, "enable_gpu": True, "enable_internet": False,
               "machine_shape": "NvidiaTeslaT4", "docker_image": DOCKER,
               "docker_image_pinning_type": "original", "kernel_execution_type": "SAVE_AND_RUN_ALL"}
    summary = {"schema_version": "orthodiffusion_artificial_gpu_offline_package_v1", "source": pin(source),
               "runtime": pin(runtime), "bundle": pin(bundle_raw), "config": config, "source_import_audit": audit,
               "vendor": {name: {key: row[key] for key in ("bytes", "sha256")} for name, row in vendor.items()},
               "only_kernel_Input": STAGE, "GPU_required": True, "private": True, "Internet": False,
               "local_activity": {"Torch_model_source_import": 0, "checkpoint_open_or_download": 0, "MRI": 0,
                                  "GPU": 0, "training": 0, "Gold": 0, "Save_Submit": 0},
               "actual_GPU_qualification": "not executed; root manual Save required"}
    output.mkdir(parents=True)
    for name, raw in {"source.ipynb": source, "save-payload.json": encode_json(payload), "manifest.json": encode_json(summary),
                      "config.json": encode_json(config), "vendor.json": encode_json(vendor), "runtime.py": runtime,
                      "source-import-audit.json": encode_json(audit)}.items():
        (output / name).write_bytes(raw)
    (output / "vendor").mkdir()
    for name, row in vendor.items():
        (output / "vendor" / name).write_bytes(row["text"].encode())
    return {"output": str(output), "source": pin(source), "payload": pin((output / "save-payload.json").read_bytes()),
            "manifest": pin((output / "manifest.json").read_bytes())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--slug", default="rsraki/orthodiff-sag-artificial-20261017-v1")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.slug), indent=2))


if __name__ == "__main__":
    main()
