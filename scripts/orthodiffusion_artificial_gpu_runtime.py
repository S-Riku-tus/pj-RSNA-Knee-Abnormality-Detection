"""Offline artificial-only qualification; Torch is imported only inside run().

No network, dataset, MRI, Gold, training, Kaggle API, upload or submission path.
The CLI requires CUDA; failure never changes resolution/precision/checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping
from pathlib import Path

BUFFER_NAMES = (
    "betas", "alphas_cumprod", "alphas_cumprod_prev", "sqrt_alphas_cumprod",
    "sqrt_one_minus_alphas_cumprod", "log_one_minus_alphas_cumprod",
    "sqrt_recip_alphas_cumprod", "sqrt_recipm1_alphas_cumprod",
    "posterior_variance", "posterior_log_variance_clipped",
    "posterior_mean_coef1", "posterior_mean_coef2", "posterior_mean_coef3",
)
PACKAGE = "od_qual_pinned"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalize_ema(ema):
    require(isinstance(ema, Mapping) and bool(ema), "Explicit nonempty checkpoint['ema'] mapping required")
    normalized = {}
    changed = []
    for key, value in ema.items():
        require(isinstance(key, str), "EMA key must be text")
        new = "denoise_fn." + key[len("denoise_fn.module."):] if key.startswith("denoise_fn.module.") else key
        require(new not in normalized, "Duplicate normalized EMA key: " + new)
        normalized[new] = value
        if new != key:
            changed.append([key, new])
    return normalized, changed


def validate_metadata(actual, expected):
    """Pure metadata check, also tested without importing Torch."""
    require(set(actual) == set(expected), "EMA key coverage differs: missing=" + str(sorted(set(expected) - set(actual)))
            + "; unexpected=" + str(sorted(set(actual) - set(expected))))
    for key in sorted(expected):
        require(actual[key] == expected[key], "EMA shape/dtype differs: " + key)


def scoped_seed(uid, series, clip, base_seed):
    require(isinstance(base_seed, int) and not isinstance(base_seed, bool), "Integer base seed required")
    raw = json.dumps(["orthodiff-artificial-noise-v1", uid, series, clip, base_seed],
                     ensure_ascii=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**63 - 1)


def file_pin(path):
    h = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            total += len(chunk)
            h.update(chunk)
    return {"bytes": total, "sha256": h.hexdigest()}


def resolve_stage(candidates):
    present = [Path(path) for path in candidates if Path(path).is_dir()]
    require(len(present) == 1, "Exactly one declared immutable stage mount required: " + str(present))
    return present[0]


def guard_stage(stage, config):
    manifest_path = stage / "asset-manifest.json"
    require(file_pin(manifest_path) == config["stage_manifest_pin"], "Staged manifest SHA/size differs")
    manifest = json.loads(manifest_path.read_bytes())
    actual_rows = {row["path"]: row for row in manifest["assets"]}
    require(len(actual_rows) == 9 and set(actual_rows) == {row["path"] for row in config["assets"]},
            "All eight source/text assets and one checkpoint required")
    verified = []
    for row in config["assets"]:
        rel = Path(row["path"])
        require(not rel.is_absolute() and ".." not in rel.parts, "Asset path escapes stage")
        expected = {key: row[key] for key in ("bytes", "sha256")}
        require({key: actual_rows[row["path"]][key] for key in expected} == expected, "Staged declared pin differs")
        member = stage / rel
        require(not member.is_symlink() and member.is_file(), "Regular staged member required")
        require(file_pin(member) == expected, "Staged member bytes/SHA differs: " + row["path"])
        if row.get("git_blob_sha1"):
            raw = member.read_bytes()
            blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            require(blob == row["git_blob_sha1"], "Original Git blob differs")
        verified.append({"path": row["path"], **expected, "verified": True})
    return verified


def install_vendor(folder, vendor):
    package = folder / PACKAGE
    package.mkdir()
    for relative, row in vendor.items():
        require(relative in {"__init__.py", "unet.py", "modules.py", "fp16_util.py", "features.py", "LICENSE"},
                "Unexpected vendored member")
        raw = row["text"].encode("utf-8")
        require({"bytes": len(raw), "sha256": sha(raw)} == {key: row[key] for key in ("bytes", "sha256")},
                "Vendored text pin differs")
        (package / relative).write_bytes(raw)
    require(not any(key == PACKAGE or key.startswith(PACKAGE + ".") for key in sys.modules),
            "Vendored package already imported")
    return package


def tensor_info(value):
    raw = value.detach().cpu().contiguous().numpy().tobytes()
    return {"shape": list(value.shape), "dtype": str(value.dtype), "bytes": len(raw), "sha256": sha(raw)}


def write_receipt(path, state):
    raw = (json.dumps(state, ensure_ascii=True, indent=2, allow_nan=False) + "\n").encode()
    temp = path.with_name(path.name + ".partial")
    temp.write_bytes(raw)
    os.replace(temp, path)


def run(config, vendor, output=Path("/kaggle/working/ORTHODIFF_ARTIFICIAL_GPU.json")):
    """One artificial fixture: same-seed repeat and a different-seed pass, CUDA0."""
    output = Path(output)
    require(not output.exists(), "New qualification output required")
    started = time.monotonic()
    state = {"schema_version": "orthodiffusion_artificial_gpu_v1", "complete": False,
             "phase": "start", "protocol": config["protocol"], "input_kernel": config["input_kernel"],
             "scope": {"artificial_only": True, "MRI": 0, "Gold": 0, "Report": 0, "training": 0,
                       "network": 0, "Save_Submit": 0, "Public_improvement_verified": False,
                       "continuous16_MRI_pilot_qualified": False}, "phases_seconds": {}}

    def progress(phase):
        state["phase"] = phase
        state["elapsed_seconds"] = time.monotonic() - started
        write_receipt(output, state)

    try:
        require(config["protocol"]["input_shape"] == [1, 1, 16, 256, 256], "Fixed artificial shape required")
        require(config["protocol"]["feature"] == "mid_2" and config["protocol"]["timestep"] == 100,
                "Fixed feature contract required")
        require(config["protocol"]["precision"] == "float32" and config["protocol"]["passes"] == 3,
                "Fixed precision/repeat contract required")
        stage = resolve_stage(config["stage_candidates"])
        state["stage_path"] = str(stage)
        state["verified_assets"] = guard_stage(stage, config)
        state["stage_manifest_pin"] = config["stage_manifest_pin"]
        progress("all_nine_assets_guarded_before_scientific_import")

        # Set before CUDA context initialization; no runtime installation or downloads.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import resource

        import numpy as np
        import torch

        require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "CUDA GPU required; no CPU fallback")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        state["environment"] = {"python": platform.python_version(), "torch": torch.__version__,
                                "numpy": np.__version__, "cuda": torch.version.cuda,
                                "cudnn": torch.backends.cudnn.version(), "visible_cuda": torch.cuda.device_count(),
                                "selected_device": str(device), "device_name": torch.cuda.get_device_name(device),
                                "cublas_workspace": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
        free, total = torch.cuda.mem_get_info(device)
        state["cuda_before"] = {"free_bytes": free, "total_bytes": total}
        require(free >= config["protocol"]["minimum_free_cuda_bytes"], "Insufficient initial CUDA memory")

        with tempfile.TemporaryDirectory(prefix="od-artificial-") as temp:
            package = install_vendor(Path(temp), vendor)
            sys.path.insert(0, temp)
            try:
                unet = importlib.import_module(PACKAGE + ".unet")
                features = importlib.import_module(PACKAGE + ".features")
                imported = {}
                for name in ("__init__", "unet", "modules", "fp16_util", "features"):
                    module = sys.modules[PACKAGE if name == "__init__" else PACKAGE + "." + name]
                    path = Path(module.__file__).resolve()
                    require(path == (package / (name + ".py")).resolve(), "Vendor import origin differs")
                    pin = file_pin(path)
                    require(pin == {key: vendor[name + ".py"][key] for key in ("bytes", "sha256")}, "Imported member pin differs")
                    imported[name] = {"path": str(path), **pin}
                state["imported_vendor"] = imported
                progress("CUDA_and_import_closure_checked")
                begin = time.monotonic()
                # Model initialization uses CPU RNG only and restores it; noise has a separate scoped seed.
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(1729)
                    denoiser = unet.create_model(image_size=256, num_channels=64, num_res_blocks=1,
                                                channel_mult="", in_channels=1, out_channels=1,
                                                use_checkpoint=False, use_fp16=False)
                    model = features.GaussianDiffusion(denoiser, image_size=256, depth_size=16,
                                                       channels=1, timesteps=1000, loss_type="l2")
                expected = model.state_dict()
                require(set(dict(model.named_buffers())) == set(BUFFER_NAMES), "Exact thirteen schedule buffers required")
                expected_buffers = {key: expected[key].clone() for key in BUFFER_NAMES}
                checkpoint = torch.load(stage / "weights/sagittal_model.pt", map_location="cpu", weights_only=True)
                require(isinstance(checkpoint, Mapping) and "ema" in checkpoint, "Explicit EMA checkpoint required")
                ema, changes = normalize_ema(checkpoint["ema"])
                require(all(isinstance(value, torch.Tensor) for value in ema.values()), "Tensor-only EMA mapping required")
                def metadata(rows):
                    return {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in rows.items()}
                validate_metadata(metadata(ema), metadata(expected))
                require(all(torch.isfinite(value).all().item() for value in ema.values()), "Nonfinite checkpoint tensor")
                schedule_checks = {key: bool(torch.equal(ema[key], expected_buffers[key])) for key in BUFFER_NAMES}
                require(all(schedule_checks.values()), "EMA diffusion schedule differs from exact retained constructor")
                incompat = model.load_state_dict(ema, strict=True)
                require(not incompat.missing_keys and not incompat.unexpected_keys, "Strict load not complete")
                require(all(torch.equal(model.state_dict()[key], value) for key, value in ema.items()), "Loaded EMA bytes differ")
                state["checkpoint"] = {"weights_only": True, "strict": True, "arm": "ema", "normalized_prefix_count": len(changes),
                                       "normalized_prefix_changes": changes, "keys": len(ema),
                                       "parameter_count": sum(value.numel() for value in model.parameters()),
                                       "full_key_shapes_dtypes": metadata(ema), "schedule_buffers_exact": schedule_checks,
                                       "buffer_hashes": {key: tensor_info(ema[key]) for key in BUFFER_NAMES},
                                       "missing_keys": [], "unexpected_keys": [], "all_loaded_tensors_equal": True}
                # Drop duplicate CPU training/model arms before moving the complete frozen architecture.
                del checkpoint, ema, expected, expected_buffers
                model.eval().requires_grad_(False).to(device)
                require(not model.training and all(not value.requires_grad for value in model.parameters()), "Frozen eval model required")
                state["phases_seconds"]["construct_strict_load_to_gpu"] = time.monotonic() - begin
                progress("full_EMA_strict_loaded_and_frozen")

                torch.cuda.reset_peak_memory_stats(device)
                x = torch.linspace(-1.0, 1.0, 16 * 256 * 256, dtype=torch.float32).reshape(1, 1, 16, 256, 256).to(device)
                t = torch.tensor([100], dtype=torch.long, device=device)
                seed = scoped_seed("ARTIFICIAL-UID-000", "ARTIFICIAL-SAGITTAL-000", [0, 16], 20261017)
                state["fixture"] = {"kind": "synthetic_linspace_minus1_plus1", "input": tensor_info(x),
                                    "timestep": tensor_info(t), "noise_seed": seed,
                                    "seed_fields": ["ARTIFICIAL-UID-000", "ARTIFICIAL-SAGITTAL-000", [0, 16], 20261017]}
                counts = {"denoiser": 0, "decoder_blocks": 0, "final_out": 0}
                noisy = []

                def count_denoiser(_module, args):
                    counts["denoiser"] += 1
                    noisy.append(tensor_info(args[0]))

                def count_decoder(_module, _args, _result):
                    counts["decoder_blocks"] += 1

                def count_out(_module, _args, _result):
                    counts["final_out"] += 1

                handles = [denoiser.register_forward_pre_hook(count_denoiser), denoiser.out.register_forward_hook(count_out)]
                handles.extend(block.register_forward_hook(count_decoder) for block in denoiser.output_blocks)
                passes = []
                reference = None
                cpu_rng = torch.random.get_rng_state().clone()
                cuda_rng = [value.clone() for value in torch.cuda.get_rng_state_all()]
                rng_before = {"CPU": tensor_info(cpu_rng), "all_visible_CUDA": [tensor_info(value) for value in cuda_rng]}
                try:
                    for index in range(3):
                        start_counts = dict(counts)
                        torch.cuda.synchronize(device)
                        begin = time.monotonic()
                        pass_seed = seed if index < 2 else scoped_seed("ARTIFICIAL-UID-000", "ARTIFICIAL-SAGITTAL-000", [0, 16], 20261018)
                        with torch.random.fork_rng(devices=[0]), torch.inference_mode():
                            # Avoid torch.manual_seed(), which seeds every visible CUDA device.
                            torch.cuda.default_generators[0].manual_seed(pass_seed)
                            feature = model.get_feature(x, t, name="mid_2", ret_in=False, ret_mid=True,
                                                        ret_out=False, use_amp=False)
                            require(list(feature.shape) == [1, 256, 1, 16, 16] and feature.dtype == torch.float32,
                                    "Measured mid_2 shape/dtype differs")
                            require(torch.isfinite(feature).all().item(), "Nonfinite feature")
                            pooled = feature.float().mean(dim=(2, 3, 4))
                            require(list(pooled.shape) == [1, 256] and torch.isfinite(pooled).all().item(), "Invalid global pooled feature")
                            feature_cpu = feature.detach().cpu()
                            pooled_cpu = pooled.detach().cpu()
                        torch.cuda.synchronize(device)
                        elapsed = time.monotonic() - begin
                        require(torch.equal(cpu_rng, torch.random.get_rng_state())
                                and all(torch.equal(before, after) for before, after in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True)),
                                "Scoped CPU/all-visible CUDA RNG was not restored")
                        rng_after = {"CPU": tensor_info(torch.random.get_rng_state()),
                                     "all_visible_CUDA": [tensor_info(value) for value in torch.cuda.get_rng_state_all()]}
                        delta = {key: counts[key] - start_counts[key] for key in counts}
                        require(delta == {"denoiser": 1, "decoder_blocks": 0, "final_out": 0}, "Feature forward path differs")
                        if reference is None:
                            reference = (feature_cpu.clone(), pooled_cpu.clone())
                        elif index == 1:
                            require(torch.equal(reference[0], feature_cpu) and torch.equal(reference[1], pooled_cpu),
                                    "Same scoped fixture is not bitwise repeatable")
                        else:
                            require(not torch.equal(reference[0], feature_cpu) and not torch.equal(reference[1], pooled_cpu),
                                    "Different scoped noise seed did not vary features and pooled output")
                        passes.append({"index": index, "noise_seed": pass_seed, "seconds": elapsed, "forward_counts": delta,
                                       "noisy_input": noisy[-1], "mid_2": tensor_info(feature_cpu),
                                       "pooled": tensor_info(pooled_cpu), "pooled_float32_values": pooled_cpu[0].tolist(),
                                       "scoped_RNG_restored": True, "RNG_before": rng_before, "RNG_after": rng_after})
                        state["passes"] = passes
                        progress("artificial_pass_" + str(index + 1) + "_complete")
                        del feature, pooled, feature_cpu, pooled_cpu
                finally:
                    for handle in handles:
                        handle.remove()
                require(passes[0]["noisy_input"] == passes[1]["noisy_input"], "Repeated noise differs")
                require(passes[0]["noisy_input"]["sha256"] != passes[2]["noisy_input"]["sha256"], "Different seed noise did not change")
                state["resources"] = {"peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                                      "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
                                      "peak_process_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                      "total_seconds": time.monotonic() - started}
                gate = config["protocol"]["resource_limits"]
                require(state["resources"]["peak_cuda_reserved_bytes"] <= gate["cuda_reserved_bytes"], "CUDA budget exceeded")
                require(state["resources"]["peak_process_RSS_bytes"] <= gate["process_RSS_bytes"], "CPU RSS budget exceeded")
                require(all(row["seconds"] <= gate["one_pass_seconds"] for row in passes), "Artificial pass time budget exceeded")
                require(state["resources"]["total_seconds"] <= gate["total_seconds"], "Total qualification time budget exceeded")
                state["repeat_bitwise_equal"] = True
                state["different_seed_noise_feature_and_pool_differ"] = True
                state["forward_counts"] = counts
                state["complete"] = True
                progress("artificial_GPU_qualification_complete")
            finally:
                sys.path.remove(temp)
                for key in list(sys.modules):
                    if key == PACKAGE or key.startswith(PACKAGE + "."):
                        del sys.modules[key]
        state["owned_vendor_temp_removed"] = not package.exists()
        write_receipt(output, state)
    except Exception as error:
        state["complete"] = False
        state["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        state["elapsed_seconds"] = time.monotonic() - started
        write_receipt(output, state)
    print("ORTHODIFF_ARTIFICIAL_GPU " + json.dumps({"complete": state["complete"], "phase": state["phase"],
                                                  "receipt": str(output), "sha256": file_pin(output)["sha256"]}, sort_keys=True))
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/ORTHODIFF_ARTIFICIAL_GPU.json"))
    args = parser.parse_args()
    state = run(json.loads(args.config.read_bytes()), json.loads(args.vendor.read_bytes()), args.output)
    if not state["complete"]:
        raise SystemExit("Artificial GPU qualification failed; see receipt")


if __name__ == "__main__":
    main()
