"""Fixed2x2regional1024 only; require exact parent3063 scientific parity."""
from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import os
import struct
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from pathlib import Path


def record_line(record):
    return (json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n").encode()


def fixed_regional_pool(feature, cat):
    """Channel256 per quadrant, TL/TR/BL/BR; no forward or random operation."""
    return cat([feature[:, :, :, :8, :8].float().mean(dim=(2, 3, 4)),
                feature[:, :, :, :8, 8:16].float().mean(dim=(2, 3, 4)),
                feature[:, :, :, 8:16, :8].float().mean(dim=(2, 3, 4)),
                feature[:, :, :, 8:16, 8:16].float().mean(dim=(2, 3, 4))], dim=1).cpu()


def require_parent_success(record, parent):
    """Exact source/geometry/tensor/bits parity, excluding runtime/RNG-start state."""
    fields = ("StudyInstanceUID", "status", "metadata", "noise_seed", "normalized_input",
              "normalization_min_max_float32", "noisy_input", "mid_2", "pooled", "pooled_float32_values",
              "forward_counts", "all_RNG_restored")
    if parent.get("status") != "success" or any(record.get(key) != parent.get(key) for key in fields):
        different = [key for key in fields if record.get(key) != parent.get(key)]
        raise ValueError("Parent-success scientific parity mismatch: " + ",".join(different))
    if struct.pack("<256f", *record["pooled_float32_values"]) != struct.pack("<256f", *parent["pooled_float32_values"]):
        raise ValueError("Parent global256 FP32 bits mismatch")


class RegionalJournal:
    """Reuse the original256/global/failure validation and add regional1024."""

    def __init__(self, original_journal):
        self.original = original_journal

    def __getattr__(self, name):
        return getattr(self.original, name)

    def append(self, record):
        if record.get("status") == "success":
            values = record.get("regional_2x2_float32_values")
            if not isinstance(values, list) or len(values) != 1024 or not all(
                isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in values
            ):
                raise ValueError("Exactly1024 finite regional FP32 values required")
            raw = struct.pack("<1024f", *values)
            if list(struct.unpack("<1024f", raw)) != values or record.get("regional_2x2") != {
                "shape": [1, 1024], "dtype": "torch.float32", "bytes": 4096, "sha256": hashlib.sha256(raw).hexdigest()
            } or record.get("parent_scientific_parity_passed") is not True:
                raise ValueError("Regional bits/hash or parent parity contract differs")
        elif any(key in record for key in ("regional_2x2", "regional_2x2_float32_values")):
            raise ValueError("Inherited failure cannot contain regional features")
        self.original.append(record)


def load_parent(config, base):
    """Pin the fixedV1 full output before importing decoders/model or any MRI."""
    expected = config["regional_parent_input"]
    folder = base["resolve_stage"](expected["candidates"])
    for name, value in expected["members"].items():
        path = folder / name
        base["require"](path.is_file() and not path.is_symlink() and base["file_pin"](path) == value,
                        "FixedV1 parent output pin differs: " + name)
    summary = json.loads((folder / "ORTHODIFF_FULLWEAK_SUMMARY.json").read_bytes())
    status = json.loads((folder / "ORTHODIFF_FULLWEAK_STATUS.json").read_bytes())
    base["require"](summary.get("complete") is True and status.get("complete") is True
                    and "error" not in summary and summary.get("phase") == status.get("phase") == "full4207_finished"
                    and summary["coverage"] == status["coverage"] == expected["coverage"]
                    and summary["cohort_UID_hash"] == config["UID_order_sha256"]
                    and summary["requested_UIDs"] == config["uids"] and summary["model_load_calls"] == 1,
                    "Complete qualified parent terminal binding differs")
    records, success, failure = {}, 0, 0
    success_uids, failure_uids = [], []
    with (folder / "ORTHODIFF_FULLWEAK_RECORDS.jsonl").open("rb") as stream:
        for index, line in enumerate(stream):
            record = json.loads(line)
            uid = record["StudyInstanceUID"]
            base["require"](index < len(config["uids"]) and uid == config["uids"][index] and uid not in records
                            and record_line(record) == line, "Parent record order/uniqueness/exact bytes differ")
            records[uid] = record
            if record.get("status") == "success":
                success += 1
                success_uids.append(uid)
            else:
                base["require"](record.get("status") == "data_failure" and record.get("feature_forward_calls") == 0
                                and record.get("fallback_features") is False
                                and not any(key in record for key in ("pooled", "pooled_float32_values", "mid_2", "normalized_input")),
                                "Parent failure must be featureless/zero forward")
                failure += 1
                failure_uids.append(uid)
    base["require"](len(records) == 4207 and success == 3063 and failure == 1144, "Exact3063/1144 parent partition required")
    for key, uids in (("success_UID_order_sha256", success_uids), ("failure_UID_order_sha256", failure_uids)):
        base["require"](base["sha"]("".join(uid + "\n" for uid in uids).encode()) == expected[key], "Parent partition UID hash differs")
    return records, {"root": str(folder), "members": expected["members"], "requested": 4207,
                     "recomputed_successes_required": success, "featureless_failures_inherited_without_image_access": failure,
                     "independent_actual_qualification_pin": expected["qualification_pin"], "fixed_Input": expected["Input"]}


def status_snapshot(state, journal, started):
    """Small constant-size status: never serialize requested4207 or old records."""
    return {"schema_version": "orthodiff_regional2x2_small_status_v1", "complete": state["complete"],
            "phase": state["phase"], "elapsed_seconds": time.monotonic() - started,
            "current_UID": state.get("current_UID"), "coverage": journal.coverage(),
            "journal": {"filename": journal.path.name, **journal.pin()},
            "durability": "append+flush+fsync before publishing this prefix digest",
            "error_type": state.get("error", {}).get("type")}


def run(config, vendor, base, pilot, full, output_dir=Path("/kaggle/working/orthodiff-regional2x2")):
    """Parent-success-only recomputation; any scientific difference is fatal."""
    output_dir = Path(output_dir)
    base["require"](not output_dir.exists() and not output_dir.is_symlink(), "Fresh fullweak output directory required")
    base["require"](len(config["uids"]) == len(set(config["uids"])) == 4207, "All4207 unique weak UIDs required")
    base["require"](config["operational_resource_limits"] == {
        "CUDA_reserved_bytes": 14 * 1024**3, "process_RSS_bytes": 8 * 1024**3,
        "total_seconds": 28800, "UID_boundary_cooperative_clock_check": True,
        "hard_C_decoder_timeout": False,
    }, "Frozen fullweak operational resource contract required")
    output_dir.mkdir(parents=True)
    output = output_dir / "ORTHODIFF_REGIONAL2X2_SUMMARY.json"
    journal = RegionalJournal(full["RecordJournal"](output_dir / "ORTHODIFF_REGIONAL2X2_RECORDS.jsonl", config["uids"]))
    status = output_dir / "ORTHODIFF_REGIONAL2X2_STATUS.json"
    PilotDataError = pilot["PilotDataError"]
    check = pilot["check"]
    choose_series, read_clip = pilot["choose_series"], pilot["read_clip"]
    strict_load, decoder_preflight = pilot["strict_load"], pilot["decoder_preflight"]
    start = time.monotonic()
    state = {"schema_version": "orthodiff_regional2x2_frozen_continuous16_v1", "complete": False,
             "protocol": config["regional_protocol"], "frozen_parent_protocol": config["preprocess"], "cohort_UID_hash": config["UID_order_sha256"], "requested_UIDs": config["uids"],
             "scope": {"Gold_labels": 0, "Report": 0, "training": 0, "network": 0, "Save_Submit": 0,
                       "Public_improvement_verified": False, "author_NIfTI_orientation_equivalent": False},
             "phase": "start", "model_load_calls": 0, "parent_scientific_parity_successes": 0,
             "parent_failures_inherited": 0, "source_bindings": config["regional_source_bindings"],
             "full_scope": {"requested_count": 4207, "original_supplied_report_groups_included_whole": True,
                            "frozen32_protocol_has_historical_cohort_output_resource_descriptions": True,
                            "current_operational_resource_limits": config["operational_resource_limits"],
                            "runtime_embeds_UIDs_not_groups_folds_Report_or_targets": True,
                            "outputs": [journal.path.name, status.name, output.name],
                            "resumption_or_skip_existing": False, "missing_feature_zero_fill": False,
                            "frozen_pilot_pins": config["frozen_pilot_pins"]}}

    def progress(phase):
        state["phase"], state["elapsed_seconds"] = phase, time.monotonic() - start
        base["write_receipt"](status, status_snapshot(state, journal, start))
        if phase in {"single_frozen_backbone_loaded", "regional4207_finished", "fatal_error"} or (
            phase in {"feature_success_recorded", "parent_failure_inherited"} and journal.count % 32 == 0
        ):
            print("ORTHODIFF_REGIONAL2X2_PROGRESS " + json.dumps({
                "phase": phase, "elapsed_seconds": state["elapsed_seconds"], "complete": state["complete"],
                "requested": len(journal.uids), "recorded": journal.count, "success": journal.success,
                "failure": journal.failure, "journal_prefix_SHA256": journal.pin()["sha256"],
            }, separators=(",", ":")), flush=True)

    try:
        uid_bytes = "".join(uid + "\n" for uid in config["uids"]).encode()
        base["require"](base["sha"](uid_bytes) == config["UID_order_sha256"], "Frozen UID order differs")
        parents, state["parent_input"] = load_parent(config, base)
        stage = base["resolve_stage"](config["assets"]["stage_candidates"])
        state["verified_generic_assets"] = base["guard_stage"](stage, config["assets"])
        root = base["resolve_stage"](config["competition_candidates"])
        series_path = root / "train_series.csv"
        base["require"](base["file_pin"](series_path) == config["series_metadata_pin"], "Official series metadata differs")
        wanted = set(config["uids"])
        series_by_uid = defaultdict(list)
        with series_path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["StudyInstanceUID"] in wanted:
                    series_by_uid[row["StudyInstanceUID"]].append(row)
        base["require"](set(series_by_uid) == wanted, "All4207 require exact official series metadata coverage")
        image_roots = [path for path in (root / "train_series", root / "train_images") if path.is_dir()]
        base["require"](len(image_roots) == 1, "Unique official train DICOM root required")
        progress("all_nine_assets_and_UID_metadata_guarded")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        with tempfile.TemporaryDirectory(prefix="od-regional2x2-") as temp:
            folder = Path(temp)
            np, pydicom, decoder_target = decoder_preflight(config["decoder"], folder, base, state)
            import resource

            import cv2
            import torch

            base["require"](torch.cuda.is_available(), "CUDA required; no CPU fallback")
            torch.cuda.set_device(0)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.use_deterministic_algorithms(True)
            state["environment"] = {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                                    "pydicom": pydicom.__version__, "cv2": cv2.__version__, "cuda": torch.version.cuda,
                                    "selected_GPU": torch.cuda.get_device_name(0), "visible_GPUs": torch.cuda.device_count()}
            package = base["install_vendor"](folder, vendor)
            sys.path.insert(0, temp)
            try:
                unet = importlib.import_module(base["PACKAGE"] + ".unet")
                features = importlib.import_module(base["PACKAGE"] + ".features")
                state["vendor_imports"] = {}
                for name in ("__init__", "unet", "modules", "fp16_util", "features"):
                    module = sys.modules[base["PACKAGE"] if name == "__init__" else base["PACKAGE"] + "." + name]
                    path = Path(module.__file__).resolve()
                    base["require"](path == (package / (name + ".py")).resolve()
                                    and base["file_pin"](path) == {key: vendor[name + ".py"][key] for key in ("bytes", "sha256")}, "Imported vendor origin/pin differs")
                    state["vendor_imports"][name] = {"path": str(path), **base["file_pin"](path)}
                begin = time.monotonic()
                state["model_load_calls"] = 1
                model, denoiser = strict_load(torch, unet, features, stage, base, state)
                state["strict_load_seconds"] = time.monotonic() - begin
                progress("single_frozen_backbone_loaded")
                counts = {"denoiser": 0, "decoder_blocks": 0, "final_out": 0}
                noisy = []

                def count_denoiser(_module, args):
                    counts["denoiser"] += 1
                    noisy.append(base["tensor_info"](args[0]))

                def count_decoder(_module, _args, _result):
                    counts["decoder_blocks"] += 1

                def count_out(_module, _args, _result):
                    counts["final_out"] += 1

                handles = [denoiser.register_forward_pre_hook(count_denoiser), denoiser.out.register_forward_hook(count_out)]
                handles.extend(block.register_forward_hook(count_decoder) for block in denoiser.output_blocks)
                torch.cuda.reset_peak_memory_stats(0)
                try:
                    for uid in config["uids"]:
                        base["require"](time.monotonic() - start < config["operational_resource_limits"]["total_seconds"], "Fullweak cooperative wall budget exceeded")
                        state["current_UID"] = uid
                        parent = parents[uid]
                        if parent["status"] == "data_failure":
                            journal.append(parent)
                            state["parent_failures_inherited"] += 1
                            progress("parent_failure_inherited")
                            continue
                        progress("metadata_and_continuous16_" + uid)
                        begin = time.monotonic()
                        selected = None
                        try:
                            selected, paths = choose_series(uid, series_by_uid[uid], image_roots[0], pydicom)
                            array, detail = read_clip(uid, selected, paths, pydicom, np, cv2, base)
                            x = torch.from_numpy(np.ascontiguousarray(array)).float()
                            lo, hi = x.min(), x.max()
                            check(bool(torch.isfinite(x).all().item()) and bool((hi > lo).item()), "constant_or_nonfinite_resized_clip")
                        except (MemoryError, SystemError):
                            raise
                        except (PilotDataError, OSError, ValueError, RuntimeError, NotImplementedError, pydicom.errors.InvalidDicomError) as error:
                            raise RuntimeError("Original parent-success preprocessing failed for " + uid + ":" + type(error).__name__ + ":" + str(error)) from error
                        data_seconds = time.monotonic() - begin
                        x = ((x - lo) / (hi - lo + 1e-8) * 2 - 1).unsqueeze(0).unsqueeze(0).to("cuda:0")
                        t = torch.tensor([100], dtype=torch.long, device="cuda:0")
                        seed = base["scoped_seed"](uid, detail["SeriesInstanceUID"], [detail["clip_start"], detail["clip_end"]], 20261017)
                        cpu_rng = torch.random.get_rng_state().clone()
                        cuda_rng = [value.clone() for value in torch.cuda.get_rng_state_all()]
                        before = dict(counts)
                        torch.cuda.synchronize(0)
                        begin = time.monotonic()
                        # Model/GPU exceptions are outside the data-failure catch and are fatal.
                        with torch.random.fork_rng(devices=[0]), torch.inference_mode():
                            torch.cuda.default_generators[0].manual_seed(seed)
                            feature = model.get_feature(x, t, name="mid_2", ret_in=False, ret_mid=True, ret_out=False, use_amp=False)
                            base["require"](list(feature.shape) == [1, 256, 1, 16, 16] and feature.dtype == torch.float32
                                            and torch.isfinite(feature).all().item(), "InvalidMRIfeature")
                            pooled = feature.float().mean(dim=(2, 3, 4)).cpu()
                        torch.cuda.synchronize(0)
                        feature_seconds = time.monotonic() - begin
                        base["require"](list(pooled.shape) == [1, 256] and torch.isfinite(pooled).all().item(), "InvalidMRIpool")
                        restored = torch.equal(cpu_rng, torch.random.get_rng_state()) and all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True))
                        delta = {key: counts[key] - before[key] for key in counts}
                        base["require"](restored and delta == {"denoiser": 1, "decoder_blocks": 0, "final_out": 0}, "RNG/forward contract differs")
                        record = {"StudyInstanceUID": uid, "status": "success", "metadata": detail, "noise_seed": seed,
                            "data_seconds": data_seconds, "feature_seconds": feature_seconds, "normalized_input": base["tensor_info"](x),
                            "normalization_min_max_float32": [float(lo), float(hi)], "noisy_input": noisy[-1], "mid_2": base["tensor_info"](feature),
                            "pooled": base["tensor_info"](pooled), "pooled_float32_values": pooled[0].tolist(), "forward_counts": delta,
                            "CPU_RNG_before_after_SHA": [base["tensor_info"](cpu_rng)["sha256"], base["tensor_info"](torch.random.get_rng_state())["sha256"]],
                            "all_CUDA_RNG_before_after_SHA": [[base["tensor_info"](a)["sha256"], base["tensor_info"](b)["sha256"]]
                                                             for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True)],
                            "all_RNG_restored": True}
                        require_parent_success(record, parent)
                        pool_begin = time.monotonic()
                        with torch.inference_mode():
                            regional = fixed_regional_pool(feature, torch.cat)
                        base["require"](list(regional.shape) == [1, 1024] and regional.dtype == torch.float32
                                        and torch.isfinite(regional).all().item(), "InvalidRegional1024")
                        extra_rng_restored = torch.equal(cpu_rng, torch.random.get_rng_state()) and all(
                            torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True))
                        base["require"](extra_rng_restored and {key: counts[key] - before[key] for key in counts} == delta,
                                        "Regional pooling changed RNG or forward counts")
                        record.update(regional_2x2=base["tensor_info"](regional), regional_2x2_float32_values=regional[0].tolist(),
                                      regional_pooling_seconds=time.monotonic() - pool_begin, parent_scientific_parity_passed=True,
                                      parent_record_line_sha256=base["sha"](record_line(parent)))
                        journal.append(record)
                        state["parent_scientific_parity_successes"] += 1
                        del x, feature, pooled, regional, array
                        progress("feature_success_recorded")
                finally:
                    for handle in handles:
                        handle.remove()
                state["coverage"] = journal.coverage()
                state["resources"] = {"peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(0),
                                      "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(0),
                                      "peak_process_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                      "total_seconds": time.monotonic() - start}
                state["forward_counts"] = counts
                base["require"](journal.count == 4207 and counts["denoiser"] == state["coverage"]["success"] == state["parent_scientific_parity_successes"] == 3063
                                and state["coverage"]["failure"] == state["parent_failures_inherited"] == 1144
                                and counts["decoder_blocks"] == counts["final_out"] == 0, "Final coverage/forward binding differs")
                base["require"](state["resources"]["peak_cuda_reserved_bytes"] <= 14 * 1024**3
                                and state["resources"]["peak_process_RSS_bytes"] <= 8 * 1024**3
                                and state["resources"]["total_seconds"] <= 28800, "Fullweak resource budget exceeded")
            finally:
                for path in (temp, str(decoder_target)):
                    if path in sys.path:
                        sys.path.remove(path)
                for name in list(sys.modules):
                    if name == base["PACKAGE"] or name.startswith(base["PACKAGE"] + "."):
                        del sys.modules[name]
        state["owned_temp_removed"] = not folder.exists()
        base["require"](state["owned_temp_removed"], "Owned temporary directory remains")
        state["journal"] = {"filename": journal.path.name, **journal.finalize(), "final_full_file_rehash_matches_incremental": True}
        state["complete"] = True
        state["all_requested_MRI_success"] = state["coverage"]["success"] == 4207
        progress("regional4207_finished")
    except Exception as error:
        journal.close()
        state["complete"] = False
        state["coverage"] = journal.coverage()
        state["journal"] = {"filename": journal.path.name, **journal.pin(), "final_full_file_rehash_matches_incremental": False}
        state["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        state["owned_temp_removed"] = not folder.exists() if "folder" in locals() else True
        print("ORTHODIFF_REGIONAL2X2_ERROR " + json.dumps(state["error"], separators=(",", ":")), file=sys.stderr, flush=True)
        progress("fatal_error")
    finally:
        journal.close()
    base["write_receipt"](output, state)
    print("ORTHODIFF_REGIONAL2X2 " + json.dumps({"complete": state["complete"], "receipt": str(output), "pin": base["file_pin"](output)}), flush=True)
    return state
