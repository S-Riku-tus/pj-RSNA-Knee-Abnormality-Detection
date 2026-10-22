"""Local F015 artificial qualification or explicit one-fold CUDA training.

No weight download, model upload, Kaggle action or automatic fold launch exists.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = str(ROOT / "artifacts/cache/f015-hf")
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from rsna_knee import convnext_small_model as M  # noqa: E402
from rsna_knee import convnext_small_training as T  # noqa: E402
from rsna_knee.contracts import SUBMISSION_COLUMNS, TARGETS  # noqa: E402
from rsna_knee.feature_contracts import json_hash  # noqa: E402
from rsna_knee.runtime import ensure_cuda  # noqa: E402


def fixture(valid):
    images = torch.randint(0, 256, (1, 3, 48, 3, 392, 392), device="cuda", dtype=torch.uint8)
    mask = torch.zeros((1, 3, 48), device="cuda", dtype=torch.bool)
    if valid == 144:
        mask[:] = True
    else:
        for series, window in ((0, 0), (0, 20), (1, 13), (2, 25)):
            mask[0, series, window] = True
    return {"images": images, "mask": mask,
            "positions": torch.linspace(0, 1, 48, device="cuda")[None, None, :].expand(1, 3, 48).clone(),
            "planes": torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.long),
            "flags": torch.tensor([[[1., 0.], [0., 1.], [1., 1.]]], device="cuda")}


def qualify(output, config, provenance, weights):
    ensure_cuda()
    T.validate_native_config(config)
    output = Path(output).resolve()
    M.require(output.is_relative_to((ROOT / "artifacts").resolve()) and not output.exists(), "Fresh workspace QA")
    M.require(torch.cuda.device_count() == 1 and torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 4090", "Actual local RTX4090")
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    model = M.create_model(weights, config, provenance)
    prefix = M.prefix_digest(model)
    checks = {}
    model.eval()
    im = (torch.arange(3 * 392 * 392, device="cuda").reshape(1, 3, 392, 392) % 251).float() / 255
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        reference = model.encoder.forward_head(model.encoder.forward_features(im), pre_logits=True).to(torch.float16).float()
        actual = model.encode_normalized(im)
    checks["manual_frozen_prefix_suffix_equals_original_prelogits"] = torch.equal(actual, reference) and actual.shape == (1, 768)
    small = fixture(4)
    CPU_rng, GPU_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    with torch.inference_mode():
        logits = model(**small).float()
        checks["finite_ordered12_GPU"] = logits.shape == (1, 12) and bool(torch.isfinite(logits).all())
        poisoned = dict(small, images=small["images"].clone())
        poisoned["images"][~small["mask"]] = 255
        checks["masked_padding_exact"] = torch.equal(logits, model(**poisoned).float())
        calls = []
        hook = model.encoder.stem.register_forward_hook(lambda *args: calls.append(True))
        refused = False
        try:
            model(**dict(small, mask=torch.zeros_like(small["mask"])))
        except ValueError:
            refused = True
        hook.remove()
        checks["allmissing_refused_before_encoder"] = refused and not calls
    checks["eval_CPU_CUDA_RNG_unchanged"] = torch.equal(CPU_rng, torch.get_rng_state()) and torch.equal(GPU_rng, torch.cuda.get_rng_state())
    checks["same_ordered12_submission_contract"] = len(TARGETS) == 12 and tuple(SUBMISSION_COLUMNS) == ("StudyInstanceUID", *TARGETS)
    del poisoned, small, logits, im, reference, actual
    torch.cuda.empty_cache()
    full = fixture(144)
    model.train()
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    logits = model(**full).float()
    targets = (torch.arange(12, device="cuda") % 2).float()[None, :]
    observed = torch.ones_like(targets, dtype=torch.bool)
    loss = T.observed_bce_sum(logits, targets, observed) / 12
    loss.backward()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_reserved()
    checks["full144_forward_backward_finite"] = bool(torch.isfinite(logits).all()) and bool(torch.isfinite(loss)) and all(
        p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.requires_grad)
    checks["only_exact_18_suffix_tensors_trainable"] = {n for n, p in model.encoder.named_parameters() if p.requires_grad} == M.suffix_keys(model) and len(M.suffix_keys(model)) == 18
    checks["frozen_prefix_no_grad_and_bitunchanged"] = M.prefix_digest(model) == prefix and all(
        p.grad is None for n, p in model.encoder.named_parameters() if n not in M.suffix_keys(model))
    checks["full144_peak_under16GiB"] = peak < 16 * 1024**3
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = M.make_checkpoint(model, provenance, fold=0, epoch=0, weak_bce=0., prefix_sha256=prefix, audit={}, artificial=True)
    torch.save(checkpoint, output / "artificial.pt")
    refused = False
    try:
        M.load_checkpoint(output / "artificial.pt")
    except ValueError:
        refused = True
    checks["real_consumer_rejects_artificial"] = refused
    restored = M.load_checkpoint(output / "artificial.pt", allow_artificial=True)
    clone = copy.deepcopy(model).cpu()
    with torch.no_grad():
        for name, value in clone.encoder.named_parameters():
            if name in M.suffix_keys(clone):
                value.add_(.125)
        for value in clone.head.parameters():
            value.add_(.125)
    M.apply_checkpoint(clone, restored, provenance, allow_artificial=True)
    checks["strict_roundtrip_restores_all_selected_tensors"] = M.prefix_digest(clone) == prefix and all(
        torch.equal(value, restored["encoder_suffix"][name]) for name, value in clone.encoder.state_dict().items() if name in M.suffix_keys(clone)) and all(
        torch.equal(value, restored["head"][name]) for name, value in clone.head.state_dict().items())
    M.require(all(checks.values()), "CNN artificial contract failed: " + str([k for k, v in checks.items() if not v]))
    result = {"schema_version": "F015_native392_CNN_resource_gate_v1", "complete": True, "all_pass": True,
              "checks": checks, "config_hash": json_hash(config), "source_sha256": M.source_hashes(),
              "generic_weights_sha256": M.WEIGHTS_SHA, "policy_SHA": T.POLICY_SHA,
              "GPU": torch.cuda.get_device_name(0), "Torch": torch.__version__, "full144_peak_reserved_bytes": peak,
              "full144_forward_backward_seconds": seconds, "frozen_prefix_sha256": prefix,
              "parameter_contract": model.parameter_contract(), "optimizer_updates": 0, "MRI_Report_Gold_labels_reads": 0,
              "QA_script_SHA": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "library_versions": {"timm": "1.0.22", "safetensors": "0.6.2"}, "Kaggle_or_accuracy_qualified": False}
    (output / "resource-gate.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps({"QA_pass": True, "checks": len(checks), "peak_MiB": peak / 1024**2, "full144_seconds": seconds}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("qa", "train"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, default=ROOT / "artifacts/tools/f014-convnext-site-v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--fold", type=int, choices=(0, 1))
    args = parser.parse_args()
    ensure_cuda()  # Preserve CUDA-first behavior before config/labels/cache/model reads.
    config, provenance = [json.loads(p.read_bytes()) for p in (args.config, args.provenance)]
    M.use_vendor(args.vendor)
    if args.mode == "qa":
        qualify(args.output, config, provenance, args.weights)
    else:
        M.require(args.fold is not None and args.manifest and args.index and args.gate, "Explicit one-fold genuine training inputs")
        try:
            T.train_convnext_small(args.manifest, args.index, args.output, config, args.fold,
                             args.weights.parent, args.weights, provenance, args.gate)
        except T.TrainingStopRequested:
            print("F015_STOPPED_BY_ROOT_REQUEST", flush=True)


if __name__ == "__main__":
    main()
