"""Optional CUDA check: existing generic encoder, two synthetic windows, no optimizer.

This does not prepare a trainer, data export, learned checkpoint or submission.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from rsna_knee.contracts import TARGETS, sha256
from rsna_knee.feature_imaging import load_frozen_encoder
from rsna_knee.native_training_fast16 import prefix_digest
from rsna_knee.regional_triplet_model import RegionalTripletMIL
from rsna_knee.runtime import ensure_cuda

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    ensure_cuda()  # Config/assets/output access follows the unchanged CUDA guard.
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts/qa") or output.exists():
        raise ValueError("A fresh project artifacts/qa directory is required")
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    if provenance["weights_sha256"] != "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9":
        raise ValueError("Use only the already-audited unmodified generic DINOv2 ViT-S/14")
    names = {
        0: "configs/experiments/f005-dinov2-native-frozen-control-fast16.json",
        2: "configs/experiments/f005-dinov2-native-last2-fast16.json",
    }
    configs = {arm: json.loads((ROOT / path).read_text(encoding="utf-8")) for arm, path in names.items()}
    encoder = load_frozen_encoder(args.repo, args.weights, provenance, "cuda")
    if encoder.patch_size != 14:
        raise ValueError("The patch lattice requires the exact ViT-S/14 backbone")
    torch.manual_seed(20261002)
    images = torch.zeros(1, 3, 48, 3, 392, 392, dtype=torch.uint8, device="cuda")
    images[0, 0, 0] = torch.randint(0, 256, (3, 392, 392), dtype=torch.uint8, device="cuda")
    mask = torch.zeros(1, 3, 48, dtype=torch.bool, device="cuda")
    mask[0, 0, 0] = True
    inputs = {
        "images": images,
        "mask": mask,
        "positions": torch.zeros(1, 3, 48, device="cuda"),
        "planes": torch.tensor([[0, 3, 3]], device="cuda"),
        "flags": torch.zeros(1, 3, 2, device="cuda"),
    }
    rows = []
    for arm in (0, 2):
        torch.manual_seed(20261002)
        encoder.zero_grad(set_to_none=True)
        model = RegionalTripletMIL(
            encoder, configs[arm], trainable_blocks=arm, window_chunk=16, checkpoint_last2=True
        ).cuda()
        model.train(arm == 2)
        prefix_before = prefix_digest(model)
        norm_events = []
        handle = encoder.norm.register_forward_hook(
            lambda _, args, value: norm_events.append(
                {"shape": list(value.shape), "grad_enabled": torch.is_grad_enabled()}
            )
        )
        torch.cuda.reset_peak_memory_stats()
        try:
            if arm == 0:
                with torch.no_grad():
                    logits = model(**inputs)
            else:
                logits = model(**inputs)
                logits.square().sum().backward()  # Explicit artificial objective; no teacher or optimizer.
            torch.cuda.synchronize()
            assert logits.shape == (1, 12) and bool(torch.isfinite(logits).all())
            assert len(norm_events) == 1 and norm_events[0]["shape"] == [1, 785, 384]
            for name, parameter in encoder.named_parameters():
                if parameter.requires_grad:
                    assert arm == 2 and name.startswith(("blocks.10.", "blocks.11."))
                    assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                else:
                    assert parameter.grad is None
            if arm == 2:
                assert all(
                    value.grad is not None and bool(torch.isfinite(value.grad).all())
                    for value in model.head.parameters()
                )
            assert prefix_digest(model) == prefix_before
            rows.append(
                {
                    "trainable_blocks": arm,
                    "valid_windows": 1,
                    "norm_events": norm_events,
                    "logits_shape": list(logits.shape),
                    "logits_finite": True,
                    "prefix_SHA_unchanged": True,
                    "prefix_sha256": prefix_before,
                    "suffix_gradients_finite": arm == 2,
                    "frozen_encoder_and_norm_gradients_absent": True,
                    "head_gradients_finite": arm == 2,
                    "parameters": model.parameter_contract(),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "optimizer_constructed_or_steps": 0,
                }
            )
        finally:
            handle.remove()
        model.zero_grad(set_to_none=True)
        del model, logits
        torch.cuda.empty_cache()
    result = {
        "schema_version": "regional_triplet_artificial_two_windows_v1",
        "all_passed": True,
        "backbone": "existing generic DINOv2 ViT-S/14",
        "backbone_weights_sha256": provenance["weights_sha256"],
        "logical_encoder_windows_total": 2,
        "maximum_valid_windows_per_forward": 1,
        "backward_calls": 1,
        "suffix_checkpoint_recomputation": True,
        "optimizer_steps": 0,
        "real_MRI_or_teacher_read": False,
        "new_model_acquisition": False,
        "state_or_head_parameterization_changed": False,
        "targets": list(TARGETS),
        "arms": rows,
        "source_sha256": sha256(ROOT / "src/rsna_knee/regional_triplet_model.py"),
        "CLI_sha256": sha256(Path(__file__)),
        "input_image_sha256": hashlib.sha256(images.cpu().numpy().tobytes()).hexdigest(),
        "runtime": {"torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0)},
        "candidate_trainer_checkpoint_or_serving_qualified": False,
        "limits": ["Only two synthetic logical windows, no full144 capacity or real training accuracy qualification."],
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "checks.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {"all_passed": True, "windows": 2, "optimizer_steps": 0, "output_sha256": sha256(output / "checks.json")}
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "weights", "provenance", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
