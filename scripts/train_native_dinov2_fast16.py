"""Prepare/preflight/train native392 paired controls, or bounded label-free resource QA.

There is no automatic upload, Notebook execution, competition submission or raw
MRI download. Real training requires a completed pixel resource gate/full index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from rsna_knee.contracts import dump_json, sha256
from rsna_knee.feature_contracts import json_hash, load_feature_config, new_directory
from rsna_knee.feature_imaging import load_frozen_encoder
from rsna_knee.finetune_model import NativeTripletMIL
from rsna_knee.native_training_fast16 import (
    ENGINEERING_LINEAGE,
    apply_native_checkpoint,
    backward_observed_micro,
    load_native_checkpoint,
    make_native_checkpoint,
    native_source_hashes,
    predict_native,
    prefix_digest,
    preflight_native_training,
    tensor_study,
    train_native,
    validate_native_config,
)
from rsna_knee.runtime import ensure_cuda
from rsna_knee.triplet_cache import load_archive_study

ROOT = Path(__file__).resolve().parents[1]


def add_assets(parser):
    for name in ("repo", "weights", "provenance"):
        parser.add_argument(f"--{name}", type=Path, required=True)


def qa_arm_config(config, directory, blocks):
    """Use the requested arm itself, keeping the existing opposite control."""
    if blocks == config["online"]["trainable_blocks"]:
        return config
    name = "f005-dinov2-native-frozen-control-fast16.json" if blocks == 0 else "f005-dinov2-native-last2-fast16.json"
    return validate_native_config(load_feature_config(Path(directory) / name))


def qa(args):
    ensure_cuda()
    config = validate_native_config(load_feature_config(args.config))
    torch.cuda.set_per_process_memory_fraction(
        21_000_000_000 / torch.cuda.get_device_properties(0).total_memory, device=0
    )
    out = new_directory(args.output)
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    pixel_info = None
    if args.pilot_archive:
        if args.pilot_summary is None or args.independent_pixel_review is None:
            raise ValueError("Processed pilot QA requires its complete summary and independent16 review")
        summary = json.loads(args.pilot_summary.read_text(encoding="utf-8"))
        review = json.loads(args.independent_pixel_review.read_text(encoding="utf-8"))
        if (
            review.get("all_passed") is not True
            or review.get("studies") != 16
            or not all(review.get("checks", {}).values())
        ):
            raise ValueError("Independent full16 cache checks have not passed")
        if (
            summary.get("complete") is not True
            or summary["processed"] != 16
            or summary.get("teacher_CSV_read") is not False
        ):
            raise ValueError("Processed pilot summary has not passed the16-study teacher-free gate")
        if (
            args.pilot_archive.stat().st_size != summary["archive"]["bytes"]
            or sha256(args.pilot_archive) != summary["archive"]["sha256"]
        ):
            raise ValueError("Processed pilot archive SHA256 differs")
        if review.get("archive_sha256") != summary["archive"]["sha256"]:
            raise ValueError("Independent full16 review belongs to another archive")
        key = max(summary["coverage"], key=lambda study: summary["coverage"][study]["valid_triplets"])
        began = time.perf_counter()
        arrays = load_archive_study(args.pilot_archive, key, summary)
        read_seconds = time.perf_counter() - began
        valid = int(arrays["mask"].sum())
        pixel_info = {
            "UID": key,
            "valid_triplets": valid,
            "max_valid_UID_selected": True,
            "read_decompress_hash_QC_seconds": read_seconds,
            "archive_sha256": summary["archive"]["sha256"],
            "archive_bytes": summary["archive"]["bytes"],
            "summary_sha256": sha256(args.pilot_summary),
            "independent_full16_review_sha256": sha256(args.independent_pixel_review),
            "independent_review_referenced": bool(review),
            "all16_QC_repeated": False,
            "whole_archive_sha_check_outside_read_timer": True,
        }
    else:
        # Import the previously verified artificial generator, without its profiler execution.
        from profile_dinov2_last2 import synthetic_study

        _, arrays_batched = synthetic_study(args.artificial_valid_triplets, "cpu", config["seed"])
        arrays = {name: value[0] for name, value in arrays_batched.items()}
        valid = int(arrays["mask"].sum())
    torch.cuda.synchronize()
    began = time.perf_counter()
    inputs = tensor_study(arrays)
    torch.cuda.synchronize()
    transfer_seconds = time.perf_counter() - began
    encoder = load_frozen_encoder(args.repo, args.weights, provenance, "cuda")
    records, checks = [], []
    arm_config_hashes = {}
    for blocks in (0, 2):
        arm_config = qa_arm_config(config, args.config.parent, blocks)
        arm_config_hashes[str(blocks)] = json_hash(arm_config)
        torch.manual_seed(config["seed"])
        encoder.zero_grad(set_to_none=True)
        model = (
            NativeTripletMIL(encoder, arm_config, trainable_blocks=blocks, window_chunk=arm_config["online"]["window_chunk"], checkpoint_last2=True)
            .cuda()
            .eval()
        )
        prefix_sha = prefix_digest(model)
        with torch.no_grad():
            before = model(**inputs)
        # Serialize an explicitly artificial checkpoint. It cannot enter real inference.
        path = out / f"artificial-{'freeze' if blocks == 0 else 'last2'}.pt"
        checkpoint = make_native_checkpoint(
            model,
            provenance,
            fold=0,
            epoch=0,
            weak_bce=0.0,
            prefix_sha256=prefix_sha,
            audit={"artificial": True},
            artificial=True,
        )
        torch.save(checkpoint, path)
        loaded = load_native_checkpoint(path, allow_artificial=True)
        apply_native_checkpoint(model, loaded, provenance)
        with torch.no_grad():
            after = model(**inputs)
        difference = float((before - after).abs().max())
        assert difference == 0 and bool(torch.isfinite(after).all())
        checks.append(
            {
                "name": f"{'freeze' if blocks == 0 else 'last2'} checkpoint strict roundtrip",
                "passed": True,
                "max_logit_abs_diff": difference,
                "artificial_checkpoint_real_prediction_forbidden": True,
            }
        )
        model.train()
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        began = time.perf_counter()
        logits = model(**inputs)
        torch.cuda.synchronize()
        forward_seconds = time.perf_counter() - began
        targets = torch.tensor([[0, 1] * 6], device="cuda", dtype=torch.float32)
        observed = torch.ones_like(targets, dtype=torch.bool)
        observed[:, 3] = False
        began = time.perf_counter()
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        denominator = int(observed.sum()) * arm_config["online"]["effective_batch_studies"]
        numerator = backward_observed_micro(logits, targets, observed, denominator, scaler=scaler)
        torch.cuda.synchronize()
        backward_seconds = time.perf_counter() - began
        selected = [(name, value) for name, value in model.encoder.named_parameters() if value.requires_grad]
        assert all(value.grad is not None and bool(torch.isfinite(value.grad).all()) for _, value in selected)
        assert all(value.grad is None for value in model.encoder.parameters() if not value.requires_grad)
        assert all(value.grad is not None and bool(torch.isfinite(value.grad).all()) for value in model.head.parameters())
        assert prefix_digest(model) == prefix_sha
        assert max(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()) <= 21_000_000_000
        checks.append(
            {
                "name": f"{'freeze' if blocks == 0 else 'last2'} selected suffix finite/frozen prefix no gradients",
                "passed": True,
            }
        )
        try:
            load_native_checkpoint(path)
        except ValueError as error:
            assert "Artificial checkpoints" in str(error)
        else:
            raise AssertionError("Artificial checkpoint was accepted by real inference")
        checks.append(
            {
                "name": f"{'freeze' if blocks == 0 else 'last2'} fast16 actual chunk/prefix/head finite and artificial checkpoint real-load rejection",
                "passed": model.window_chunk == arm_config["online"]["window_chunk"] == 16,
                "frozen_prefix_sha256_before_after": prefix_sha,
                "AMP_scaled_gradient_finite": True,
            }
        )
        records.append(
            {
                "arm": "freeze" if blocks == 0 else "last2",
                "valid_triplets": valid,
                "parameters": model.parameter_contract(),
                "forward_seconds": forward_seconds,
                "backward_seconds": backward_seconds,
                "transfer_seconds": transfer_seconds,
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "checkpoint_sha256": sha256(path),
                "checkpoint_bytes": path.stat().st_size,
                "optimizer_steps": 0,
                "effective_batch_studies": 16,
                "observed_label_count": int(observed.sum()),
                "effective_observed_denominator": denominator,
                "loss_numerator": numerator,
                "grad_scaler_scale": scaler.get_scale(),
                "scaler_step_or_update_calls": 0,
            }
        )
        del model, logits, scaler
    result = {
        "status": "native392_label_free_QA_fast16_complete",
        "engineering_lineage": ENGINEERING_LINEAGE,
        "input_array_sha256": {
            name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in arrays.items()
        },
        "real_processed_MRI_pixels_used": bool(args.pilot_archive),
        "raw_DICOM_decoded_or_downloaded": False,
        "real_teacher_labels_read": False,
        "real_fit": False,
        "optimizer_steps": 0,
        "studies_simultaneously_on_GPU": 1,
        "effective_batch16_capacity_proven": False,
        "pilot": pixel_info,
        "arms": records,
        "checks": checks,
        "checks_passed": len(checks),
        "config_sha256": sha256(args.config),
        "config_semantic_sha256": arm_config_hashes,
        "source_sha256": native_source_hashes(),
        "CLI_sha256": sha256(Path(__file__)),
        "runtime": {"torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0)},
        "limits": [
            "Single warmed resource observation; image-content gradients are not accuracy evidence",
            "Read timer includes decompression/member SHA/array QC; no full archive unpacking",
            "Backward uses artificial labels only; no optimizer is constructed or stepped",
        ],
    }
    dump_json(out / "NATIVE392_QA_SUMMARY.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "processed_pixels": bool(args.pilot_archive),
                "valid_triplets": valid,
                "checks": len(checks),
                "arms": [
                    {k: r[k] for k in ("arm", "forward_seconds", "backward_seconds", "gpu_peak_reserved_bytes")}
                    for r in records
                ],
            }
        )
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    train = sub.add_parser("train")
    for mode in (preflight, train):
        for name in ("manifest", "index", "config", "resource-gate"):
            mode.add_argument(f"--{name}", type=Path, required=True)
        mode.add_argument("--fold", choices=(0, 1), type=int, required=True)
    train.add_argument("--output", type=Path, required=True)
    add_assets(train)
    predict = sub.add_parser("predict")
    for name in ("root", "checkpoint", "output", "workspace"):
        predict.add_argument(f"--{name}", type=Path, required=True)
    add_assets(predict)
    quality = sub.add_parser("qa")
    for name in ("config", "output"):
        quality.add_argument(f"--{name}", type=Path, required=True)
    add_assets(quality)
    quality.add_argument("--artificial-valid-triplets", type=int, choices=range(1, 145), default=144)
    quality.add_argument("--pilot-archive", type=Path)
    quality.add_argument("--pilot-summary", type=Path)
    quality.add_argument("--independent-pixel-review", type=Path)
    args = parser.parse_args()
    ensure_cuda()  # CLI config/provenance/cache/label reads all follow the CUDA gate.
    if args.command == "qa":
        return qa(args)
    if args.command in ("preflight", "train"):
        config = validate_native_config(load_feature_config(args.config))
        if args.command == "preflight":
            training, validation, store, _ = preflight_native_training(
                args.manifest, args.index, config, args.fold, args.resource_gate
            )
            print(
                json.dumps(
                    {
                        "ready": True,
                        "train_studies": len(training),
                        "valid_studies": len(validation),
                        "cache_studies": len(store.studies),
                        "fitting_performed": False,
                    }
                )
            )
            return
        provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
        return train_native(
            args.manifest,
            args.index,
            args.output,
            config,
            args.fold,
            args.repo,
            args.weights,
            provenance,
            args.resource_gate,
        )
    if args.command == "predict":
        from bounded_feature_reader import make_bounded_reader

        args.workspace.mkdir(parents=True, exist_ok=False)
        reader = make_bounded_reader(args.workspace)
        provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
        return predict_native(
            args.root, args.checkpoint, args.output, args.repo, args.weights, provenance, reader=reader
        )


if __name__ == "__main__":
    main()
