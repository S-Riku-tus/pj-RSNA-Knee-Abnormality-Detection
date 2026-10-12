"""CUDA-only, artificial native392 freeze/last2 profiler. No real labels or fitting."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import platform
import stat
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from rsna_knee.contracts import TARGETS, dump_json, sha256
from rsna_knee.feature_contracts import (
    load_feature_config,
    new_directory,
    source_tree_hash,
    validate_encoder_provenance,
)
from rsna_knee.feature_imaging import encode_windows, load_frozen_encoder
from rsna_knee.finetune_model import NativeTripletMIL
from rsna_knee.runtime import ensure_cuda, masked_bce

ROOT = Path(__file__).resolve().parents[1]
GENERIC_ZIP = ROOT / "artifacts/kaggle/f001-inputs-v1/rsna-frozen-feature-inputs-v1.zip"
WEIGHTS_MEMBER = "encoder/dinov2_vits14_pretrain.pth"
PROVENANCE_MEMBER = "encoder/encoder-provenance.json"
SOURCE_PREFIX = "encoder/dinov2-source/"


def allowed_member(name):
    if name in (WEIGHTS_MEMBER, PROVENANCE_MEMBER):
        return True
    if not name.startswith(SOURCE_PREFIX):
        return False
    relative = name[len(SOURCE_PREFIX) :]
    path = PurePosixPath(relative)
    return relative in {"hubconf.py", "LICENSE", "MODEL_CARD.md", "README.md"} or (
        relative.startswith("dinov2/") and (relative.endswith(".py") or path.name.startswith("LICENSE"))
    )


def extract_generic_assets(archive_path, output):
    """Extract only an explicit generic encoder whitelist, with no traversal/overwrite."""
    output = Path(output)
    if output.exists():
        raise ValueError("Generic extraction requires a new immutable asset directory")
    receipts, members, folded = {}, [], set()
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            name = PurePosixPath(info.filename)
            if (
                name.is_absolute()
                or ".." in name.parts
                or "\\" in info.filename
                or ":" in info.filename
                or info.orig_filename != info.filename
                or stat.S_ISLNK(info.external_attr >> 16)
            ):
                raise ValueError("Unsafe ZIP path or symlink")
            if not allowed_member(info.filename) or info.is_dir():
                continue
            if info.filename.casefold() in folded or info.file_size > 100 * 1024**2:
                raise ValueError("Duplicate/oversized whitelisted ZIP member")
            folded.add(info.filename.casefold())
            members.append(info)
        if len(members) > 512 or sum(info.file_size for info in members) > 110 * 1024**2:
            raise ValueError("Whitelisted assets exceed the extraction budget")
        required = {WEIGHTS_MEMBER, PROVENANCE_MEMBER, SOURCE_PREFIX + "hubconf.py", SOURCE_PREFIX + "LICENSE"}
        if not {name.casefold() for name in required}.issubset(folded):
            raise ValueError("Required generic weights/source/provenance missing")
        provenance = json.loads(archive.read(PROVENANCE_MEMBER))
        validate_encoder_provenance(provenance)
        output.mkdir(parents=True)
        for info in members:
            target = (output / info.filename).resolve()
            if not target.is_relative_to(output.resolve()):
                raise ValueError("Resolved member escapes the new asset directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with archive.open(info) as source, target.open("xb") as dest:
                while block := source.read(1024**2):
                    digest.update(block)
                    dest.write(block)
            receipts[info.filename] = {"bytes": target.stat().st_size, "sha256": digest.hexdigest()}
    weights = output / WEIGHTS_MEMBER
    repo = output / SOURCE_PREFIX
    if sha256(weights) != provenance["weights_sha256"] or source_tree_hash(repo) != provenance["source_tree_sha256"]:
        raise ValueError("Extracted generic assets do not match the reviewed provenance")
    receipt = {
        "archive": str(archive_path),
        "archive_sha256": sha256(archive_path),
        "members": receipts,
        "provenance": provenance,
        "weights_source_hashes_valid": True,
        "labels_or_reports_extracted": False,
        "competition_checkpoints_extracted": False,
    }
    dump_json(output / "extraction-receipt.json", receipt)
    return repo, weights, provenance, receipt


def synthetic_study(valid_triplets, device, seed):
    if not 1 <= valid_triplets <= 144:
        raise ValueError("Artificial triplet budget must be 1..144 for one study")
    rng = np.random.default_rng(seed)
    images = np.zeros((1, 3, 48, 3, 392, 392), np.uint8)
    mask = np.zeros((1, 3, 48), bool)
    positions = np.zeros((1, 3, 48), np.float32)
    # Populate every series when possible, retaining the real 3x48 tensor contract.
    for index in range(valid_triplets):
        slot, window = index % 3, index // 3
        images[0, slot, window] = rng.integers(0, 256, (3, 392, 392), dtype=np.uint8)
        mask[0, slot, window] = True
        positions[0, slot, window] = window / 47
    arrays = {
        "images": images,
        "mask": mask,
        "positions": positions,
        "planes": np.array([[0, 1, 2]], np.int64),
        "flags": np.array([[[1, 1], [1, 0], [0, 0]]], np.float32),
    }
    return {name: torch.from_numpy(value).to(device) for name, value in arrays.items()}, arrays


def check(name, passed, details=None):
    if not passed:
        raise AssertionError(f"Artificial contract failed: {name}: {details}")
    return {"name": name, "passed": True, "details": details}


def profile(args):
    # Fail before output creation, source extraction, checkpoint loading or GPU work.
    ensure_cuda()
    if not 1 <= args.valid_triplets <= 144 or args.window_chunk < 1 or args.repeats < 1:
        raise ValueError("Invalid artificial microbatch/resource budget")
    output = new_directory(args.output)
    config = load_feature_config(args.config)
    if config != load_feature_config(ROOT / "configs/experiments/f002-dinov2-attention.json"):
        raise ValueError("Profiler control must use the fixed original f002 config")
    repo, weights, provenance, extraction = extract_generic_assets(args.assets_zip, output / "generic-assets")
    device = torch.device("cuda")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    inputs, numpy_inputs = synthetic_study(args.valid_triplets, device, config["seed"])
    encoder = load_frozen_encoder(repo, weights, provenance, device)
    # Load exactly once in this fresh process; both arms share the same untouched encoder bytes.
    original_head, control_logits = None, None
    records, checks = [], []
    initial_encoder_digest = hashlib.sha256(
        b"".join(value.detach().cpu().numpy().tobytes() for value in encoder.state_dict().values())
    ).hexdigest()
    for arm, blocks in (("freeze", 0), ("last2", 2)):
        encoder.zero_grad(set_to_none=True)
        torch.manual_seed(config["seed"])
        model = NativeTripletMIL(
            encoder,
            config,
            trainable_blocks=blocks,
            window_chunk=args.window_chunk,
            checkpoint_last2=args.checkpoint_last2,
        ).to(device)
        if original_head is None:
            original_head = copy.deepcopy(model.head.state_dict())
        model.head.load_state_dict(original_head, strict=True)
        model.eval()
        with torch.no_grad():
            online_logits = model(**inputs)
            mask = numpy_inputs["mask"][0]
            valid_pixels = numpy_inputs["images"][0][mask]
            reference_config = copy.deepcopy(config)
            reference_config["encoder"]["batch_size"] = args.window_chunk
            reference_features = encode_windows(encoder, valid_pixels, reference_config, device)
            padded = np.zeros((1, 3, 48, 384), np.float16)
            padded[0][mask] = reference_features
            reference_logits = model.head(
                torch.from_numpy(padded).to(device),
                inputs["mask"],
                inputs["positions"],
                inputs["planes"],
                inputs["flags"],
            )
        delta = float((online_logits - reference_logits).abs().max())
        checks.append(
            check(f"{arm}: shared CLS/head forward equivalence", delta <= 1e-5, {"max_logit_abs_diff": delta})
        )
        checks.append(
            check(
                f"{arm}: finite fixed target contract",
                online_logits.shape == (1, len(TARGETS)) and bool(torch.isfinite(online_logits).all()),
                list(TARGETS),
            )
        )
        if control_logits is None:
            control_logits = online_logits.detach().clone()
        else:
            arm_delta = float((online_logits - control_logits).abs().max())
            checks.append(
                check(
                    "Freeze / last2 forward values unchanged before fitting",
                    arm_delta <= 1e-5,
                    {"max_logit_abs_diff": arm_delta},
                )
            )
        contract = model.parameter_contract()
        checks.append(
            check(
                f"{arm}: final norm/prefix frozen and encoder eval",
                contract["final_norm_frozen"]
                and contract["encoder_eval"]
                and all(name.startswith(("blocks.10.", "blocks.11.")) for name in contract["trainable_encoder_names"]),
            )
        )
        invalid = dict(inputs)
        invalid["mask"] = torch.zeros_like(inputs["mask"])
        try:
            model(**invalid)
        except ValueError as error:
            rejected = "valid slice" in str(error)
        else:
            rejected = False
        checks.append(check(f"{arm}: all-masked study fails clearly", rejected))
        model.train()
        durations = []
        torch.cuda.reset_peak_memory_stats()
        for repeat in range(args.repeats):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            began = time.perf_counter()
            logits = model(**inputs)
            # Generated targets are artificial: diagnostic backward only, no optimizer step or fitting.
            targets = torch.tensor([[0, 1] * 6], dtype=torch.float32, device=device)
            observed = torch.ones_like(targets, dtype=torch.bool)
            observed[:, 3] = False
            loss = masked_bce(logits, targets, observed)
            loss.backward()
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - began)
        selected = {name: value for name, value in model.encoder.named_parameters() if value.requires_grad}
        nonzero = [
            name for name, value in selected.items() if value.grad is not None and bool(value.grad.abs().sum() > 0)
        ]
        finite = all(value.grad is not None and bool(torch.isfinite(value.grad).all()) for value in selected.values())
        forbidden = [
            name
            for name, value in model.encoder.named_parameters()
            if not value.requires_grad and value.grad is not None
        ]
        checks.append(
            check(
                f"{arm}: selected encoder gradients finite and nonzero",
                finite and (blocks == 0 or len(nonzero) == len(selected)),
                {"selected": len(selected), "nonzero": len(nonzero)},
            )
        )
        checks.append(check(f"{arm}: frozen encoder receives no gradient", not forbidden, forbidden))
        checks.append(
            check(
                f"{arm}: head receives finite gradients",
                all(
                    value.grad is not None and bool(torch.isfinite(value.grad).all())
                    for value in model.head.parameters()
                ),
            )
        )
        # Missing targets must not change either loss or output-logit gradient.
        isolated = logits.detach().clone().requires_grad_(True)
        missing_changed = targets.clone()
        missing_changed[:, 3] = 173.0
        a = masked_bce(isolated, targets, observed)
        b = masked_bce(isolated, missing_changed, observed)
        (ga,) = torch.autograd.grad(a, isolated, retain_graph=True)
        (gb,) = torch.autograd.grad(b, isolated)
        checks.append(
            check(
                f"{arm}: missing label is not a negative",
                bool(torch.equal(a, b) and torch.equal(ga, gb) and (ga[:, 3] == 0).all()),
            )
        )
        records.append(
            {
                "arm": arm,
                "parameters": contract,
                "valid_triplets": args.valid_triplets,
                "forward_backward_seconds": durations,
                "loss": float(loss.detach()),
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "gradient_names": nonzero,
                "optimizer_steps": 0,
            }
        )
        del model, logits, loss
        gc.collect()
        torch.cuda.empty_cache()
    final_encoder_digest = hashlib.sha256(
        b"".join(value.detach().cpu().numpy().tobytes() for value in encoder.state_dict().values())
    ).hexdigest()
    checks.append(
        check("Generic encoder values unchanged; no optimizer fitting", initial_encoder_digest == final_encoder_digest)
    )
    checkpoint_review = None
    if args.checkpoint_equivalence:
        qc = NativeTripletMIL(encoder, config, trainable_blocks=2, window_chunk=args.window_chunk).to(device).eval()
        qc.head.load_state_dict(original_head, strict=True)
        trials = []
        for enabled in (False, True):
            qc.zero_grad(set_to_none=True)
            qc.checkpoint_last2 = enabled
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            began = time.perf_counter()
            logits = qc(**inputs)
            target = torch.tensor([[0, 1] * 6], device=device, dtype=torch.float32)
            observed = torch.ones_like(target, dtype=torch.bool)
            observed[:, 3] = False
            objective = masked_bce(logits, target, observed)
            objective.backward()
            torch.cuda.synchronize()
            trial = {
                "checkpoint_last2": enabled,
                "seconds": time.perf_counter() - began,
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "logits": logits.detach().cpu(),
                "gradients": {
                    name: parameter.grad.detach().cpu().clone()
                    for name, parameter in qc.encoder.named_parameters()
                    if parameter.requires_grad
                },
            }
            trials.append(trial)
            del logits, objective
        max_logit_diff = float((trials[0]["logits"] - trials[1]["logits"]).abs().max())
        max_gradient_diff = max(
            float((trials[0]["gradients"][name] - value).abs().max()) for name, value in trials[1]["gradients"].items()
        )
        for name, value in trials[1]["gradients"].items():
            torch.testing.assert_close(value, trials[0]["gradients"][name], rtol=1e-3, atol=2e-6)
        checks.append(
            check(
                "Checkpoint OFF/ON same logits and last2 gradients",
                max_logit_diff <= 1e-5,
                {
                    "max_logit_abs_diff": max_logit_diff,
                    "max_gradient_abs_diff": max_gradient_diff,
                    "gradient_tolerance": {"rtol": 1e-3, "atol": 2e-6},
                },
            )
        )
        checkpoint_review = {
            "valid_triplets": args.valid_triplets,
            "all144_valid_mask": bool(inputs["mask"].all()),
            "max_logit_abs_diff": max_logit_diff,
            "max_gradient_abs_diff": max_gradient_diff,
            "gradient_tolerance": {"rtol": 1e-3, "atol": 2e-6},
            "trials": [
                {key: value for key, value in trial.items() if key not in ("logits", "gradients")} for trial in trials
            ],
            "optimizer_steps": 0,
        }
    review = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "artificial_cuda_profile_complete",
        "real_MRI_used": False,
        "real_labels_used": False,
        "training_fit_performed": False,
        "optimizer_steps": 0,
        "config": config,
        "input_sha256": {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in numpy_inputs.items()},
        "fold": None,
        "config_sha256": sha256(args.config),
        "seed": config["seed"],
        "targets": list(TARGETS),
        "cohort": "one artificial study, shared padded 3x48 shape; budgeted valid image count",
        "checks": checks,
        "checks_passed": len(checks),
        "arms": records,
        "all144_valid_mask": bool(inputs["mask"].all()),
        "checkpoint_equivalence": checkpoint_review,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "xformers_disabled": os.getenv("XFORMERS_DISABLED"),
            "attention": "official MemEffAttention falls back to PyTorch SDPA for ordinary tensors; actual CUDA backend not observed",
            "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
        },
        "extraction_receipt": extraction,
        "source_sha256": {
            "finetune_model.py": sha256(ROOT / "src/rsna_knee/finetune_model.py"),
            "profile_dinov2_last2.py": sha256(Path(__file__)),
            "feature_model.py": sha256(ROOT / "src/rsna_knee/feature_model.py"),
        },
        "limits": [
            "Artificial valid-window count only; not 16 simultaneous studies or full real training",
            "Current window chunk/checkpoint are engineering factors recorded separately; no accuracy measured",
            "Higher-resolution pixel pilot and real weak held-out fitting remain separate gates",
        ],
    }
    dump_json(output / "DINO_LAST2_PROFILE_SUMMARY.json", review)
    print(
        json.dumps(
            {
                "status": review["status"],
                "checks_passed": len(checks),
                "arms": [
                    {
                        "arm": x["arm"],
                        "seconds": x["forward_backward_seconds"],
                        "peak_MiB": x["gpu_peak_reserved_bytes"] / 1024**2,
                    }
                    for x in records
                ],
            }
        )
    )
    return review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-zip", type=Path, default=GENERIC_ZIP)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/f002-dinov2-attention.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--valid-triplets", type=int, default=6)
    parser.add_argument("--window-chunk", type=int, default=2)
    parser.add_argument("--checkpoint-last2", action="store_true")
    parser.add_argument("--checkpoint-equivalence", action="store_true")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    profile(args)


if __name__ == "__main__":
    main()
