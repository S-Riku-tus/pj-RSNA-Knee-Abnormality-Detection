"""Prepare a disabled-by-default, public generic OrthoDiffusion CPU stage offline.

No model/source download, scientific import, torch, MRI, GPU, training, Gold,
authentication, Kaggle API, upload, Save or Submit is performed by this builder.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts/research/20261017-independent-model-research-v1"
OUTPUT = ROOT / "artifacts/kaggle/orthodiff-sagittal-public-assets-20261017-v3"
RUNTIME = ROOT / "scripts/orthodiffusion_public_stage_runtime.py"
CODE_COMMIT = "99d15186bb8face1728263023d8414d9405c8f7e"
HF_COMMIT = "d6920d49e7d1b3dd0774eed6b8e47e386a54dd81"
MODEL_SHA = "3afc323c58b6446c1ea6dc4f09133d744e5a1609dba7662ac9a15ca688e34951"
MODEL_BYTES = 552919566
SLUG = "rsraki/orthodiff-sag-assets-20261017-v3"
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"
BLOBS = {
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


def git_blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def saved(path):
    raw = path.read_bytes()
    receipt = json.loads(path.with_name(path.name + ".receipt.json").read_bytes())
    if digest(raw) != receipt["sha256"] or len(raw) != receipt["bytes"]:
        raise ValueError("Saved primary evidence SHA/size differs: " + str(path))
    return raw


def source_assets():
    tree = json.loads(saved(EVIDENCE / "orthodiff-tree.json"))
    if tree["sha"] != CODE_COMMIT or tree["truncated"] is not False:
        raise ValueError("Exact complete upstream source tree required")
    entries = {row["path"]: row for row in tree["tree"]}
    assets = []
    for relative, (size, blob) in BLOBS.items():
        if (entries[relative]["size"], entries[relative]["sha"], entries[relative]["type"]) != (size, blob, "blob"):
            raise ValueError("Fixed original source blob differs: " + relative)
        path = EVIDENCE / "orthodiff" / relative
        sha = None
        if path.is_file():
            raw = saved(path)
            if len(raw) != size or git_blob(raw) != blob:
                raise ValueError("Saved original source does not match fixed Git blob")
            sha = digest(raw)
        assets.append({"kind": "source", "path": "upstream/" + relative, "bytes": size,
                       "git_blob_sha1": blob, "sha256": sha,
                       "url": "https://raw.githubusercontent.com/lt-0123/OrthoDiffusion/" + CODE_COMMIT + "/" + relative})
    model = json.loads(saved(EVIDENCE / "orthodiff-hf-model.json"))
    if (model["sha"] != HF_COMMIT or model["id"] != "lanstat0123/orthodiffusion"
            or any(model[key] is not False for key in ("private", "gated", "disabled"))):
        raise ValueError("Fixed public nongated HF metadata required")
    weight = next(row for row in model["siblings"] if row["rfilename"] == "sagittal_model.pt")
    if (weight["size"], weight["lfs"]["size"], weight["lfs"]["sha256"]) != (MODEL_BYTES, MODEL_BYTES, MODEL_SHA):
        raise ValueError("Positive sagittal LFS SHA/size differs")
    assets.append({"kind": "checkpoint", "path": "weights/sagittal_model.pt", "bytes": MODEL_BYTES,
                   "sha256": MODEL_SHA, "git_blob_sha1": None,
                   "url": "https://huggingface.co/lanstat0123/orthodiffusion/resolve/" + HF_COMMIT + "/sagittal_model.pt"})
    return assets


def fragment(text, node):
    return "".join(text.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])


def vendor_proposal():
    """Text/AST extraction only. Never imports the extracted scientific code."""
    raw = saved(EVIDENCE / "orthodiff/diffusion_model/trainer.py")
    text = raw.decode()
    tree = ast.parse(text)
    wanted = ("exists", "default", "extract", "cosine_beta_schedule")
    helpers = [next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name) for name in wanted]
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GaussianDiffusion")
    methods = [next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
               for name in ("__init__", "q_sample", "get_feature")]
    proposed = ("# Proposal only: exact selected OrthoDiffusion MIT source; not imported or qualified.\n"
                "# Original trainer.py commit " + CODE_COMMIT + "; preserve upstream LICENSE.\n"
                "from functools import partial\nfrom inspect import isfunction\n"
                "import numpy as np\nimport torch\nfrom torch import nn\nfrom torch.cuda.amp import autocast\n\n"
                + "\n\n".join(fragment(text, node) for node in helpers)
                + "\n\nclass GaussianDiffusion(nn.Module):\n" + "\n\n".join(fragment(text, node) for node in methods) + "\n")
    parsed = ast.parse(proposed)
    proposed_cls = next(node for node in parsed.body if isinstance(node, ast.ClassDef))
    if [ast.dump(node, include_attributes=False) for node in proposed_cls.body] != [ast.dump(node, include_attributes=False) for node in methods]:
        raise ValueError("Proposed feature-only diffusion methods changed")
    spec = {"proposal_only_not_imported_or_runtime_qualified": True, "original_trainer_sha256": digest(raw),
            "proposed_module_sha256": digest(proposed.encode()),
            "exact_helper_fragments": {node.name: {"line": node.lineno, "sha256": digest(fragment(text, node).encode())} for node in helpers},
            "exact_GaussianDiffusion_methods": {node.name: {"line": node.lineno, "sha256": digest(fragment(text, node).encode())} for node in methods},
            "UNet_path": "original unet.py/create_model plus original modules.py; complete architecture required for strict state coverage",
            "fp16_trim_proposal": "after pinned fp16_util bytes are staged, preserve only exact convert_module_to_f16/f32 functions and their verified imports; do not import training/logger unless closure requires it",
            "source_bytes_not_yet_observed": [name for name in BLOBS if not (EVIDENCE / "orthodiff" / name).is_file()],
            "remaining_import_audit": "AST inspect staged modules/fp16/logger first; no claim that unseen import dependencies already close",
            "core_feature_dependencies": ["torch==2.6.0+cu124 (author)", "numpy==2.1.2 (author)", "stdlib functools/inspect"],
            "whole_author_requirements_are_not_stage_installs": True,
            "excluded_heavy_paths": ["trainer.Trainer", "APEX", "tensorboard", "nibabel/NIfTI dataset loader", "PIL", "distributed training", "raw image concat/pickle", "author 8-label classifier"],
            "later_checkpoint_gate": ["verify staged LFS SHA/bytes before any deserialization", "weights_only=True/CPU first; fail if unsupported, never unsafe fallback",
                "require explicit checkpoint['ema'] mapping", "only normalize denoise_fn.module. to denoise_fn.; reject duplicate resulting keys",
                "require exact full denoiser parameter/buffer names, shapes and finite values; no strict=False or silently random missing keys",
                "verify diffusion schedule buffers against retained exact constructor; no unrecorded dtype or architecture substitutions"],
            "later_artificial_GPU_gate": ["GPU CUDA CLI check; one logical GPU and B=1,C=1,D=16,H=W=256", "eval/inference_mode, params frozen, t tensor100/name mid_2, one denoiser call and no decoder blocks",
                "fixed deterministic UID/series/clip/seed noise with fork_rng; document as an adapter choice", "finite Bx256 global D/H/W mean, measured actual spatial feature shape",
                "record checkpoint/source/env hashes, wall time, peak allocated/reserved/RSS, input/output bytes and failures; resource fit unverified"],
            "later_32case_pilot": "new Kaggle-only continuous16 extraction, metadata-fixed 32 studies without Gold labels/Report; streaming features; no existing sampled16 equivalence assertion"}
    return proposed, spec


def feature_contract():
    return {"direction": "sagittal_only", "checkpoint": "sagittal_model.pt", "code_commit": CODE_COMMIT, "HF_revision": HF_COMMIT,
            "author_input": {"sequence": "PDW MRI; exact RSNA metadata selection/orientation rule must be frozen before new pilot",
                "shape": ["B", 1, 16, 256, 256], "depth": "central contiguous16 actual ordered slices; d<16 excluded, never sampled/duplicate/padded",
                "central_indices": "start=d//2-8,end=d//2+8 when d>16; all slices when d==16",
                "spatial_resize": "cv2.INTER_LINEAR per slice to256x256", "normalization": "entire retained volume minmax/(max-min+1e-8), then2*x-1",
                "axis": "NIfTI H,W,D toC,D,H,W; RSNA DICOM orientation adapter separately audited",
                "source_refs": ["dataset.py:176-179,210-240", "linear_classifier.py:307-312"]},
            "model": {"input_size": 256, "depth_size": 16, "channels": 1, "base_channels": 64, "num_res_blocks": 1,
                "channel_mult": [1, 1, 2, 2, 4], "timesteps_schedule": 1000, "feature_timestep": 100,
                "feature": "mid_2", "feature_only_single_denoiser_pass": True, "decoder_forward": False,
                "feature_spatial_shape_not_yet_verified": True},
            "fixed_pooling": {"operation": "feature.float().mean(dim=(2,3,4))", "expected_channels": 256,
                "trainable_pool_parameters": 0, "selection": "predeclared global mean, not author trained SAP/local-pool replication"},
            "noise": "UID+SeriesUID+centralclip+fixedseed deterministic hash, scoped Torch fork_rng; no batch/fold/order dependence",
            "existing_sampled16_cache_is_faithful_contiguous16": False, "new_32case_Kaggle_pilot_required": True,
            "Gold_fit_selection_or_patient_independence_claimed": False, "Report_required_for_inference": False,
            "head": "new masked12-label weak-label head on existing group folds; later task, never author unmasked8-label BCE",
            "resource_or_Public_improvement_verified": False}


def prepare(output=OUTPUT, *, slug=SLUG, authorize_download=False, known_own_score=None):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a fresh public asset stage run")
    if not re.fullmatch(r"rsraki/[a-z0-9][a-z0-9-]{4,49}", slug):
        raise ValueError("Own private stage slug required")
    if known_own_score is not None and (not math.isfinite(known_own_score) or not 0 <= known_own_score <= 1):
        raise ValueError("Finite observed own official score required")
    assets = source_assets()
    proposed, vendor = vendor_proposal()
    contract = feature_contract()
    decision = {"authorize_download": bool(authorize_download), "known_own_official_score": known_own_score,
                "threshold_stop": .960, "score_API_called": False,
                "scope": "existing user authorization; operator selects stage execution and checks latest own score before manual Save",
                "pending_experiment_score_return_required": False,
                "limitation": "no live score polling; pending P037/other experiments do not block independent preparation or authorized staging"}
    runtime = RUNTIME.read_text(encoding="utf-8")
    ast.parse(runtime)
    code = ("import hashlib as _od_hash\n" + f"_od_runtime = {runtime!r}\n"
            f"if _od_hash.sha256(_od_runtime.encode()).hexdigest() != {digest(runtime.encode())!r}:\n"
            "    raise ValueError('Public stage runtime SHA differs')\n"
            "_od_namespace = {'__name__': 'orthodiffusion_public_stage_only'}\n"
            "exec(compile(_od_runtime, '<orthodiff-public-CPU-stage>', 'exec'), _od_namespace)\n"
            f"_od_assets = {assets!r}\n_od_contract = {contract!r}\n_od_decision = {decision!r}\n"
            "_od_result = _od_namespace['stage'](_od_assets, _od_contract, _od_decision)\n")
    ast.parse(code)
    book = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": (
        "# Fixed public OrthoDiffusion sagittal assets: CPU stage only\n\n"
        "Private, Internet ON, CPU, zero Inputs. One552,919,566-byte public HF checkpoint at a fixed revision "
        "plus eight fixed upstream text/source files. Stream to owned scratch, verify LFS SHA256 and every source "
        "Git blob SHA1/size (plus SHA256 where already observed), then atomically publish the complete package. "
        "No torch, checkpoint deserialization, model forward, scientific source import, GPU, MRI, Report, Gold, "
        "training, upload or Submit. Default download flag is disabled for offline preparation; root may enable "
        "it under the existing user authorization without waiting for pending experiment scores. Root checks "
        "the latest own official score before manual Save; a supplied known score>=.960 always stops before "
        "the first request. No score API is called. MIT upstream license retained. "
        "This is generic asset staging, not an accuracy result. Existing sampled16 cache is not contiguous16; "
        "strict checkpoint coverage, artificial GPU resource checks and a new32case Kaggle pilot remain required.\n")},
        {"cell_type": "code", "metadata": {}, "source": code, "execution_count": None, "outputs": []}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "kaggle": {"isGpuEnabled": False, "isInternetEnabled": True}}, "nbformat": 4, "nbformat_minor": 5}
    raw = (json.dumps(book, ensure_ascii=False, indent=1) + "\n").encode()
    payload = {"slug": slug, "new_title": slug.split("/")[1], "text": raw.decode(), "language": "python", "kernel_type": "notebook",
        "dataset_data_sources": [], "kernel_data_sources": [], "competition_data_sources": [], "is_private": True,
        "enable_gpu": False, "enable_internet": True, "docker_image": DOCKER,
        "docker_image_pinning_type": "original", "kernel_execution_type": "SAVE_AND_RUN_ALL"}
    manifest = {"schema_version": "orthodiffusion_sagittal_public_CPU_stage_offline_preparation_v2",
        "prepared_at_UTC": datetime.now(timezone.utc).isoformat(), "source_sha256": digest(raw), "source_bytes": len(raw),
        "runtime_sha256": digest(runtime.encode()), "builder_sha256": digest(Path(__file__).read_bytes()),
        "assets": assets, "total_public_download_bytes_if_later_authorized": sum(row["bytes"] for row in assets),
        "decision": decision, "Input_count": 0, "GPU": False, "Internet": True, "private": True, "requested_docker": DOCKER,
        "feature_contract": contract, "vendor_proposal_sha256": digest(proposed.encode()),
        "actual_asset_download_or_execution_performed": False, "asset_content_not_yet_verified": True,
        "checkpoint_strict_coverage_or_GPU_resource_qualified": False, "model_MRI_GPU_training_Gold_credentials_Save_Submit_calls": 0}
    output.mkdir(parents=True, exist_ok=False)
    (output / "source.ipynb").write_bytes(raw)
    (output / "stage-runtime.py").write_bytes(runtime.encode("utf-8"))
    (output / "feature-vendor-proposal.py").write_bytes(proposed.encode("utf-8"))
    for name, value in (("manifest.json", manifest), ("save-payload.json", payload), ("feature-vendor-spec.json", vendor)):
        (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--slug", default=SLUG)
    parser.add_argument("--authorize-download", action="store_true")
    parser.add_argument("--known-own-score", type=float)
    args = parser.parse_args()
    value = prepare(args.output_dir, slug=args.slug, authorize_download=args.authorize_download,
                    known_own_score=args.known_own_score)
    print(json.dumps({"source_sha256": value["source_sha256"], "source_bytes": value["source_bytes"],
                      "assets": len(value["assets"]), "decision": value["decision"], "download_performed": False}))


if __name__ == "__main__":
    main()
