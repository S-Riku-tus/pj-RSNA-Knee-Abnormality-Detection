"""CUDA-only artificial repeatability checks for dedicated dense regional training.

Only existing, explicitly supplied generic encoder assets are loaded. No MRI,
teacher CSV, training manifest, external API, or real fitting is used. All
artificial outputs go to a new directory underneath this workspace's artifacts/qa.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path

# Set this before importing torch or any CUDA-dependent producer.
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8"):
    raise ValueError("Dedicated regional execution requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch

from rsna_knee.contracts import TARGETS, dump_json, sha256
from rsna_knee.feature_contracts import json_hash, load_feature_config, new_directory
from rsna_knee.finetune_model import NativeTripletMIL
from rsna_knee.native_training_fast16 import load_native_checkpoint, make_native_checkpoint
from rsna_knee.regional_training_fast16 import (
    EXECUTION,
    REPRESENTATION,
    RESOURCE_GATE_SCHEMA,
    apply_regional_checkpoint,
    cli_hashes,
    configure_regional_determinism,
    dense_regional_backward,
    finish_regional_group,
    load_regional_checkpoint,
    make_regional_checkpoint,
    make_regional_optimizer,
    new_regional_model,
    prefix_digest,
    regional_source_hashes,
    validate_regional_config,
)
from rsna_knee.runtime import ensure_cuda

ROOT = Path(__file__).resolve().parents[1]
GROUP_SIZES = [16, 10, 4]
MEMORY_LIMIT_BYTES = 21_000_000_000
# Fixed before execution; do not adjust these using QA results or real labels.
TOLERANCES = {
    "logits": {"atol": 2e-5, "rtol": 2e-5},
    "bce": {"atol": 5e-6, "rtol": 5e-6},
    "unscaled_gradients": {"atol": 2e-6, "rtol": 2e-3},
    "parameters": {"atol": 2e-6, "rtol": 2e-5},
    "exp_avg": {"atol": 2e-8, "rtol": 2e-3},
    "exp_avg_sq": {"atol": 2e-12, "rtol": 3e-3},
}


def require(result, name, passed):
    result["checks"][name] = bool(passed)
    if not passed:
        raise AssertionError(f"Regional artificial contract failed: {name}")


def rng_state():
    return {"cpu": torch.get_rng_state().clone(), "cuda": torch.cuda.get_rng_state().clone()}


def restore_rng(state):
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state(state["cuda"])


def same_rng(left, right):
    return all(torch.equal(left[key], right[key]) for key in ("cpu", "cuda"))


def synthetic_inputs(*, full, seed):
    generator = np.random.default_rng(seed)
    arrays = {
        "images": np.zeros((1, 3, 48, 3, 392, 392), dtype=np.uint8),
        "mask": np.zeros((1, 3, 48), dtype=bool),
        "positions": np.full((1, 3, 48), np.nan, dtype=np.float32),
        "planes": np.array([[0, 1, 2]], dtype=np.int64),
        "flags": np.array([[[1, 1], [1, 0], [0, 0]]], dtype=np.float32),
    }
    selected = [(series, window) for series in range(3) for window in range(48)] if full else [(0, 47), (1, 17), (2, 3)]
    for series, window in selected:
        arrays["images"][0, series, window] = generator.integers(0, 256, (3, 392, 392), dtype=np.uint8)
        arrays["mask"][0, series, window] = True
        arrays["positions"][0, series, window] = window / 47
    inputs = {name: torch.from_numpy(value).cuda() for name, value in arrays.items()}
    hashes = {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in arrays.items()}
    return inputs, hashes


def synthetic_targets(index):
    values = [[float((target + index) % 2) for target in range(len(TARGETS))]]
    targets = torch.tensor(values, dtype=torch.float32, device="cuda")
    observed = torch.ones_like(targets, dtype=torch.bool)
    observed[:, index % len(TARGETS)] = False
    observed[:, (index * 5 + 3) % len(TARGETS)] = False
    targets[~observed] = float("nan")
    return targets, observed


def trainable_state(model, *, gradients=False, scale=1.0):
    selected = {name: value for name, value in model.named_parameters() if value.requires_grad}
    result = {}
    for name, parameter in selected.items():
        value = parameter.grad if gradients else parameter
        if value is None or not bool(torch.isfinite(value).all()):
            raise AssertionError(f"Missing/nonfinite trainable {'gradient' if gradients else 'parameter'}: {name}")
        result[name] = value.detach().float().cpu().clone().div_(scale)
    if not result:
        raise AssertionError("No trainable regional parameters")
    return result


def restore_trainable(model, state):
    selected = {name: value for name, value in model.named_parameters() if value.requires_grad}
    if set(selected) != set(state):
        raise AssertionError("Trainable parameter inventory changed")
    with torch.no_grad():
        for name, value in selected.items():
            value.copy_(state[name].to(value.device, value.dtype))
    model.zero_grad(set_to_none=True)


def compare_tensors(left, right, tolerance):
    if set(left) != set(right):
        raise AssertionError("Dense main/replay tensor inventories differ")
    records, all_passed = {}, True
    for name in sorted(left):
        first, second = left[name], right[name]
        if first.shape != second.shape:
            raise AssertionError(f"Dense main/replay shapes differ: {name}")
        difference = (first.double() - second.double()).abs()
        passed = bool(torch.isfinite(first).all() and torch.isfinite(second).all()) and torch.allclose(
            first, second, **tolerance
        )
        records[name] = {
            "passed": passed,
            "elements": first.numel(),
            "max_abs_diff": float(difference.max()),
            "main_max_abs": float(first.abs().max()),
            "replay_max_abs": float(second.abs().max()),
        }
        all_passed = all_passed and passed
    return all_passed, records


def optimizer_snapshot(optimizer):
    saved = optimizer.state_dict()
    return {
        "param_groups": copy.deepcopy(saved["param_groups"]),
        "state": {
            key: {
                name: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
                for name, value in item.items()
            }
            for key, item in saved["state"].items()
        },
    }


def compare_optimizers(left, right):
    if left["param_groups"] != right["param_groups"] or set(left["state"]) != set(right["state"]):
        raise AssertionError("AdamW groups/state inventory differ")
    details, passed = {}, True
    for key in left["state"]:
        first, second = left["state"][key], right["state"][key]
        if set(first) != set(second):
            raise AssertionError("AdamW per-parameter state inventory differs")
        for name in first:
            label = f"{key}:{name}"
            if name in ("exp_avg", "exp_avg_sq"):
                accepted, diff = compare_tensors({label: first[name]}, {label: second[name]}, TOLERANCES[name])
                details.update(diff)
            else:
                accepted = (
                    torch.equal(first[name], second[name])
                    if isinstance(first[name], torch.Tensor)
                    else first[name] == second[name]
                )
                details[label] = {"passed": bool(accepted)}
            passed = passed and bool(accepted)
    return passed, details


def backward_micro(model, inputs, targets, observed, denominator, scaler):
    head_calls = []
    hook = model.head.register_forward_hook(lambda _module, _args, _output: head_calls.append(True))
    try:
        value = dense_regional_backward(model, inputs, targets, observed, denominator, scaler=scaler)
    finally:
        hook.remove()
    if len(head_calls) != 1:
        raise AssertionError("Each dense artificial microstudy must call the global head exactly once")
    logits = value["logits"]
    if (
        logits.requires_grad
        or value["observed"] != int(observed.sum())
        or value["valid_windows"] != int(inputs["mask"].sum())
    ):
        raise AssertionError("Dense backward receipt differs from artificial input/mask")
    if logits.shape != (1, len(TARGETS)) or not bool(torch.isfinite(logits).all()):
        raise AssertionError("Malformed/nonfinite regional logits")
    return logits.detach().cpu().clone(), float(value["observed_bce_sum"])


def update_diagnostics(initial, main, replay):
    """Persist offending coordinates before assertions, without changing tolerance."""
    result = {}
    groups = main["optimizer"]["param_groups"]
    keys = [key for group in groups for key in group["params"]]
    # Optimizer order is head first, then encoder; model named_parameters is encoder first.
    names = [name for name in initial if name.startswith("head.")] + [
        name for name in initial if name.startswith("encoder.")
    ]
    key_by_name = dict(zip(names, keys, strict=True))
    for name in initial:
        left, right = main["parameters"][name], replay["parameters"][name]
        threshold = TOLERANCES["parameters"]["atol"] + TOLERANCES["parameters"]["rtol"] * right.abs()
        failed = (left - right).abs() > threshold
        if not failed.any():
            continue
        flat = ((left - right).abs() / threshold).flatten().argmax().item()
        record = {
            "failed_elements": int(failed.sum()),
            "worst_flat_index": flat,
            "initial_parameter": float(initial[name].flatten()[flat]),
            "main_parameter": float(left.flatten()[flat]),
            "replay_parameter": float(right.flatten()[flat]),
            "main_unscaled_gradient": float(main["gradients"][name].flatten()[flat]),
            "replay_unscaled_gradient": float(replay["gradients"][name].flatten()[flat]),
        }
        key = key_by_name[name]
        for arm, values in (("main", main), ("replay", replay)):
            for moment in ("exp_avg", "exp_avg_sq"):
                record[f"{arm}_{moment}"] = float(values["optimizer"]["state"][key][moment].flatten()[flat])
        result[name] = record
    return result


def resource_record(start):
    torch.cuda.synchronize()
    return {
        "seconds": time.perf_counter() - start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def start_resource():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    return time.perf_counter()


def expect_rejection(result, name, action):
    try:
        action()
    except ValueError as error:
        result["details"][name] = {"rejection": str(error)}
        require(result, name, True)
    else:
        require(result, name, False)


def identity_checks(model, provenance, inputs, out, result):
    model.eval()
    prefix = prefix_digest(model)
    before_state = trainable_state(model)
    with torch.inference_mode():
        before_logits = model(**inputs).cpu()
    checkpoint = make_regional_checkpoint(
        model,
        provenance,
        fold=0,
        epoch=0,
        weak_bce=0.0,
        prefix_sha256=prefix,
        audit={"artificial": True},
        artificial=True,
    )
    regional_path = out / "artificial-regional.pt"
    torch.save(checkpoint, regional_path)
    loaded = load_regional_checkpoint(regional_path, allow_artificial=True)
    expect_rejection(
        result,
        "artificial_checkpoint_default_apply_rejected",
        lambda: apply_regional_checkpoint(model, loaded, provenance),
    )
    # Make roundtrip restoration observable instead of reapplying unchanged weights.
    with torch.no_grad():
        model.head.output_bias.add_(0.125)
    apply_regional_checkpoint(model, loaded, provenance, allow_artificial=True)
    with torch.inference_mode():
        after_logits = model(**inputs).cpu()
    roundtrip = all(torch.equal(before_state[name], value) for name, value in trainable_state(model).items())
    require(
        result,
        "dedicated_artificial_roundtrip_exact",
        roundtrip and torch.equal(before_logits, after_logits) and prefix_digest(model) == prefix,
    )
    expect_rejection(
        result, "artificial_checkpoint_real_load_rejected", lambda: load_regional_checkpoint(regional_path)
    )
    expect_rejection(
        result,
        "regional_checkpoint_old_CLS_load_rejected",
        lambda: load_native_checkpoint(regional_path, allow_artificial=True),
    )

    native_config = load_feature_config(ROOT / "configs/experiments/f005-dinov2-native-last2-fast16.json")
    native_model = NativeTripletMIL(
        model.encoder, native_config, trainable_blocks=2, window_chunk=16, checkpoint_last2=True
    )
    native_checkpoint = make_native_checkpoint(
        native_model,
        provenance,
        fold=0,
        epoch=0,
        weak_bce=0.0,
        prefix_sha256=prefix_digest(native_model),
        audit={"artificial": True},
        artificial=True,
    )
    native_path = out / "artificial-CLS.pt"
    torch.save(native_checkpoint, native_path)
    expect_rejection(
        result,
        "CLS_checkpoint_regional_load_rejected",
        lambda: load_regional_checkpoint(native_path, allow_artificial=True),
    )
    expect_rejection(
        result,
        "CLS_checkpoint_regional_apply_rejected",
        lambda: apply_regional_checkpoint(model, native_checkpoint, provenance, allow_artificial=True),
    )
    require(
        result,
        "CLS_rejected_before_copy",
        all(torch.equal(before_state[name], value) for name, value in trainable_state(model).items()),
    )

    tampered = copy.deepcopy(checkpoint)
    tampered["representation"] = {"name": "CLS_only_same_shape"}
    # Deliberately different tensors make a late semantic rejection observable.
    tampered["head"]["output_bias"].add_(1.0)
    tampered_path = out / "artificial-tampered-representation.pt"
    torch.save(tampered, tampered_path)
    expect_rejection(
        result,
        "tampered_representation_load_rejected",
        lambda: load_regional_checkpoint(tampered_path, allow_artificial=True),
    )
    expect_rejection(
        result,
        "tampered_representation_apply_rejected",
        lambda: apply_regional_checkpoint(model, tampered, provenance, allow_artificial=True),
    )
    require(
        result,
        "tampered_representation_rejected_before_copy",
        all(torch.equal(before_state[name], value) for name, value in trainable_state(model).items()),
    )
    result["details"]["artificial_checkpoint_sha256"] = {
        path.name: sha256(path) for path in (regional_path, native_path, tampered_path)
    }


def run(args):
    determinism = configure_regional_determinism()
    ensure_cuda()  # Guard before config/asset reads, output mutation, or any CUDA work.
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "artifacts/qa").resolve()) or output == (ROOT / "artifacts/qa").resolve():
        raise ValueError("Artificial QA outputs require a new directory beneath artifacts/qa")
    config = validate_regional_config(load_feature_config(args.config))
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    out = new_directory(output)
    result = {
        "schema_version": RESOURCE_GATE_SCHEMA,
        "all_passed": False,
        "checks": {},
        "details": {},
        "representation": REPRESENTATION,
        "representation_sha256": json_hash(REPRESENTATION),
        "execution": EXECUTION,
        "source_sha256": regional_source_hashes(),
        "CLI_sha256": cli_hashes(),
        "config_sha256": sha256(args.config),
        "config_semantic_sha256": json_hash(config),
        "encoder_provenance": provenance,
        "environment": {
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "determinism": determinism,
        },
        "full144_valid_windows": 0,
        "group_sizes": [],
        "optimizer_steps_artificial": 0,
        "real_teacher_labels_read": False,
        "real_MRI_pixels_read": False,
        "raw_DICOM_decoded_or_downloaded": False,
        "real_fit": False,
        "tolerances_prespecified": TOLERANCES,
        "memory_limit_bytes": MEMORY_LIMIT_BYTES,
    }
    try:
        require(result, "shared_deterministic_execution_enabled", determinism == EXECUTION["determinism"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.cuda.set_per_process_memory_fraction(
            min(0.98, MEMORY_LIMIT_BYTES / torch.cuda.get_device_properties(0).total_memory)
        )
        model = new_regional_model(args.repo, args.weights, provenance, config).train()
        prefix = prefix_digest(model)
        initial = trainable_state(model)
        result["trainable_parameters_compared"] = sum(value.numel() for value in initial.values())
        require(
            result,
            "encoder_eval_final_norm_frozen",
            not model.encoder.training and not any(value.requires_grad for value in model.encoder.norm.parameters()),
        )
        require(
            result,
            "head_FP32_and_fixed_chunk16",
            model.window_chunk == 16 and all(value.dtype == torch.float32 for value in model.head.parameters()),
        )
        full, full_hashes = synthetic_inputs(full=True, seed=config["seed"])
        targets, observed = synthetic_targets(0)
        denominator = int(observed.sum()) * 16
        full_rng = rng_state()
        full_records = {}
        for path in ("main", "replay"):
            restore_trainable(model, initial)
            model.train()
            restore_rng(full_rng)
            scaler = torch.amp.GradScaler("cuda", enabled=True)
            start = start_resource()
            logits, bce = backward_micro(model, full, targets, observed, denominator, scaler)
            resource = resource_record(start)
            full_records[path] = {
                "logits": logits,
                "bce": bce,
                "gradients": trainable_state(model, gradients=True, scale=scaler.get_scale()),
                "rng": rng_state(),
                "resource": resource,
                "scale": scaler.get_scale(),
            }
            require(
                result,
                f"full144_{path}_frozen_prefix_no_grad",
                prefix_digest(model) == prefix
                and all(value.grad is None for value in model.parameters() if not value.requires_grad),
            )
            require(
                result,
                f"full144_{path}_memory_within_limit",
                max(resource["peak_allocated_bytes"], resource["peak_reserved_bytes"]) <= MEMORY_LIMIT_BYTES,
            )
        main, replay = full_records["main"], full_records["replay"]
        accepted, diffs = compare_tensors(main["gradients"], replay["gradients"], TOLERANCES["unscaled_gradients"])
        result["details"]["full144_gradient_differences"] = diffs
        result["full144_valid_windows"] = int(full["mask"].sum())
        result["details"]["full144"] = {
            "input_array_sha256": full_hashes,
            "observed_labels": int(observed.sum()),
            "effective_observed_denominator": denominator,
            "main": {"resource": main["resource"], "bce_sum": main["bce"], "scale": main["scale"]},
            "replay": {"resource": replay["resource"], "bce_sum": replay["bce"], "scale": replay["scale"]},
            "logits_max_abs_diff": float((main["logits"] - replay["logits"]).abs().max()),
            "BCE_sum_abs_diff": abs(main["bce"] - replay["bce"]),
            "dropout_RNG_exact": same_rng(main["rng"], replay["rng"]),
        }
        full_checks = (
            ("full144_all_trainable_gradient_repeatability", accepted),
            ("full144_logit_repeatability", torch.allclose(main["logits"], replay["logits"], **TOLERANCES["logits"])),
            ("full144_masked_BCE_repeatability", bool(np.isclose(main["bce"], replay["bce"], **TOLERANCES["bce"]))),
            (
                "full144_dropout_RNG_once_and_scale_repeatability",
                same_rng(main["rng"], replay["rng"]) and main["scale"] == replay["scale"],
            ),
        )
        for name, passed in full_checks:
            result["checks"][name] = bool(passed)
        for name, passed in full_checks:
            require(result, name, passed)
        del full, full_records, main, replay, targets, observed
        model.zero_grad(set_to_none=True)

        sparse, sparse_hashes = synthetic_inputs(full=False, seed=config["seed"] + 1)
        group_targets = [[synthetic_targets(index) for index in range(size)] for size in GROUP_SIZES]
        group_rng = rng_state()
        arm_records = {}
        for path in ("main", "replay"):
            restore_trainable(model, initial)
            model.train()
            restore_rng(group_rng)
            optimizer = make_regional_optimizer(model)
            scaler = torch.amp.GradScaler("cuda", enabled=True)
            records = []
            for size, teachers in zip(GROUP_SIZES, group_targets):
                optimizer.zero_grad(set_to_none=True)
                denominator = sum(int(mask.sum()) for _, mask in teachers)
                logits, numerator = [], 0.0
                for targets, observed in teachers:
                    output, bce = backward_micro(model, sparse, targets, observed, denominator, scaler)
                    logits.append(output)
                    numerator += bce
                gradients = trainable_state(model, gradients=True, scale=scaler.get_scale())
                require(
                    result,
                    f"group{size}_{path}_frozen_prefix_no_grad",
                    prefix_digest(model) == prefix
                    and all(value.grad is None for value in model.parameters() if not value.requires_grad),
                )
                step = finish_regional_group(model, optimizer, scaler)
                require(
                    result,
                    f"group{size}_{path}_AMP_clip_AdamW_updated",
                    step["finite"] is True and step["updated"] is True and np.isfinite(float(step["grad_norm"])),
                )
                result["optimizer_steps_artificial"] += int(step["updated"])
                records.append(
                    {
                        "logits": torch.cat(logits),
                        "bce_sum": numerator,
                        "denominator": denominator,
                        "gradients": gradients,
                        "parameters": trainable_state(model),
                        "optimizer": optimizer_snapshot(optimizer),
                        "scaler": copy.deepcopy(scaler.state_dict()),
                        "rng": rng_state(),
                        "step": step,
                    }
                )
            arm_records[path] = records
            del optimizer, scaler
        group_details = []
        group_checks = []
        result["details"]["effective_groups"] = group_details
        result["details"]["sparse_input_array_sha256"] = sparse_hashes
        for size, main, replay in zip(GROUP_SIZES, arm_records["main"], arm_records["replay"]):
            gradient_ok, gradient_diffs = compare_tensors(
                main["gradients"], replay["gradients"], TOLERANCES["unscaled_gradients"]
            )
            parameter_ok, parameter_diffs = compare_tensors(
                main["parameters"], replay["parameters"], TOLERANCES["parameters"]
            )
            optimizer_ok, optimizer_diffs = compare_optimizers(main["optimizer"], replay["optimizer"])
            rng_ok = main["scaler"] == replay["scaler"] and same_rng(main["rng"], replay["rng"])
            output_ok = (
                main["denominator"] == replay["denominator"]
                and torch.allclose(main["logits"], replay["logits"], **TOLERANCES["logits"])
                and bool(np.isclose(main["bce_sum"], replay["bce_sum"], **TOLERANCES["bce"]))
            )
            group_checks.extend(
                (f"effective_group{size}_{name}", passed)
                for name, passed in (
                    ("all_gradient_repeatability", gradient_ok),
                    ("post_AdamW_parameter_repeatability", parameter_ok),
                    ("AdamW_state_repeatability", optimizer_ok),
                    ("AMP_scaler_and_dropout_RNG_repeatability", rng_ok),
                    ("logits_and_masked_BCE_repeatability", output_ok),
                )
            )
            group_details.append(
                {
                    "studies": size,
                    "observed_denominator": main["denominator"],
                    "gradient_differences": gradient_diffs,
                    "parameter_differences": parameter_diffs,
                    "optimizer_differences": optimizer_diffs,
                    "main_BCE_sum": main["bce_sum"],
                    "replay_BCE_sum": replay["bce_sum"],
                    "logits_max_abs_diff": float((main["logits"] - replay["logits"]).abs().max()),
                    "BCE_sum_abs_diff": abs(main["bce_sum"] - replay["bce_sum"]),
                    "post_AdamW_failed_coordinates": update_diagnostics(initial, main, replay),
                    "main_step": main["step"],
                    "replay_step": replay["step"],
                    "main_scaler": main["scaler"],
                    "replay_scaler": replay["scaler"],
                    "dropout_RNG_exact": same_rng(main["rng"], replay["rng"]),
                }
            )
        result["group_sizes"] = [row["studies"] for row in group_details]
        # Record every comparison before an assertion can terminate this artificial run.
        for name, passed in group_checks:
            result["checks"][name] = bool(passed)
        for name, passed in group_checks:
            require(result, name, passed)
        require(
            result,
            "sparse_masks_padding_NaN_positions_and_missing_teacher_excluded",
            int(sparse["mask"].sum()) == 3
            and bool(torch.isnan(sparse["positions"][~sparse["mask"]]).all())
            and all(bool(torch.isnan(target[~mask]).all()) for group in group_targets for target, mask in group),
        )
        require(result, "prefix_unchanged_after_six_artificial_updates", prefix_digest(model) == prefix)
        identity_checks(model, provenance, sparse, out, result)
        result["all_passed"] = bool(result["checks"]) and all(value is True for value in result["checks"].values())
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        dump_json(out / "checks.json", result)
    print(
        json.dumps(
            {
                "all_passed": result["all_passed"],
                "checks": len(result["checks"]),
                "optimizer_steps_artificial": result["optimizer_steps_artificial"],
                "output": str(out / "checks.json"),
            }
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "repo", "weights", "provenance", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    main()
