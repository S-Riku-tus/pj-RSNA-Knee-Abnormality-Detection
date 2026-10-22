"""Separate fixed1080 fluid-sensitive route; unchanged frozen numerical forward."""
from __future__ import annotations

import csv
import importlib
import json
import math
import os
import sys
import tempfile
import time
import traceback
import types
from collections import defaultdict
from pathlib import Path
from typing import Callable

# Literal reader template dependencies are bound only by bind_reader(), in its
# new function namespace. These inert placeholders do not alter pilot globals.
HEADER_TAGS: tuple | None = None
check: Callable | None = None
physical_order: Callable | None = None


def bind_reader(pilot):
    """New function namespace; never alter pilot globals or its PD classifier."""
    dependencies = {"__builtins__": __builtins__, "math": math,
                    "HEADER_TAGS": pilot["HEADER_TAGS"], "check": pilot["check"],
                    "physical_order": pilot["physical_order"]}
    return types.FunctionType(read_fluid_clip.__code__, dependencies, "read_fluid_clip")


def choose_fixed_series(uid, request, rows, image_root, check):
    """Use the preselected metadata series exactly once; never rank/retry here."""
    check(request["StudyInstanceUID"] == uid and request["route"] == "fluid_sensitive_fallback"
          and request["confirmed_PD"] is False, "fixed_fluid_request_binding_mismatch")
    selected = request["selected_series"]
    if selected is None:
        check(request["status"] == "metadata_unavailable", "unavailable_status_mismatch")
        check(False, "fluid_sensitive_metadata_unavailable")
    check(request["status"] == "metadata_candidate_selected", "selected_status_mismatch")
    sid = selected["SeriesInstanceUID"]
    actual = [row for row in rows if row["SeriesInstanceUID"] == sid]
    check(len(actual) == 1 and actual[0]["StudyInstanceUID"] == uid, "fixed_series_official_UID_mismatch")
    check({key: actual[0][key] for key in ("Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression")}
          == selected["metadata"], "fixed_series_official_metadata_mismatch")
    check(selected["Fluid_Sensitive"] is True and type(selected["Fat_Suppression"]) is bool,
          "fixed_fluid_flags_mismatch")
    entry = {"SeriesInstanceUID": sid, **selected["metadata"], "route": "fluid_sensitive_fallback",
             "confirmed_PD": False, "stratum": "fluid_sensitive_FS" if selected["Fat_Suppression"] else "fluid_sensitive_nonFS",
             "selection": "frozen subset: FluidSensitive1/FS1-before-FS0/SeriesUID; before headers/pixels",
             "retry_another_series_after_chosen_failure": False}
    # Returning an empty list still binds the chosen series in the failure record.
    return entry, sorted((image_root / uid / sid).glob("*.dcm"))


def read_fluid_clip(uid, selected, paths, pydicom, np, cv2, base):
    headers = [pydicom.dcmread(path, stop_before_pixels=True, specific_tags=HEADER_TAGS) for path in paths]
    first = headers[0]
    spacing = [float(value) for value in getattr(first, "PixelSpacing", [])]
    check(len(spacing) == 2 and all(math.isfinite(x) and x > 0 for x in spacing), "missing_or_invalid_PixelSpacing")
    shape = [int(getattr(first, "Rows", 0)), int(getattr(first, "Columns", 0))]
    check(all(value > 0 for value in shape) and shape[0] * shape[1] * 16 * 8 <= 512 * 1024**2, "invalid_matrix_or_clip_exceeds512MiB")
    photometric = str(getattr(first, "PhotometricInterpretation", ""))
    check(photometric in {"MONOCHROME1", "MONOCHROME2"}, "nonmonochrome_or_unknown_polarity")
    geometry, sop = [], []
    for header in headers:
        check(str(getattr(header, "StudyInstanceUID", "")) == uid and str(getattr(header, "SeriesInstanceUID", "")) == selected["SeriesInstanceUID"], "header_UID_binding_mismatch")
        check(int(getattr(header, "NumberOfFrames", 1)) == 1, "multiframe_unsupported")
        check([int(getattr(header, "Rows", 0)), int(getattr(header, "Columns", 0))] == shape, "inconsistent_matrix")
        other_spacing = [float(value) for value in getattr(header, "PixelSpacing", [])]
        check(len(other_spacing) == 2 and all(math.isfinite(x) for x in other_spacing)
              and max(abs(a - b) for a, b in zip(spacing, other_spacing, strict=True)) <= .00001, "inconsistent_PixelSpacing")
        check(str(getattr(header, "PhotometricInterpretation", "")) == photometric, "mixed_photometric_polarity")
        geometry.append({"IOP": list(getattr(header, "ImageOrientationPatient", [])), "IPP": list(getattr(header, "ImagePositionPatient", []))})
        sop.append(str(getattr(header, "SOPInstanceUID", "")))
    check(all(sop) and len(set(sop)) == len(sop), "missing_or_duplicate_SOP_UID")
    plan = physical_order(geometry)
    arrays, source_files = [], []
    allowed_compressed = {"1.2.840.10008.1.2.4.57", "1.2.840.10008.1.2.4.70", "1.2.840.10008.1.2.4.90", "1.2.840.10008.1.2.4.91"}
    for index in plan["selected_indices"]:
        header, path = headers[index], paths[index]
        check(hasattr(header, "file_meta") and hasattr(header.file_meta, "TransferSyntaxUID"), "missing_transfer_syntax")
        syntax = pydicom.uid.UID(str(header.file_meta.TransferSyntaxUID))
        check(not syntax.is_compressed or str(syntax) in allowed_compressed, "unsupported_unqualified_compressed_syntax")
        value = pydicom.pixels.pixel_array(path, decoding_plugin="pylibjpeg" if syntax.is_compressed else "")
        check(list(value.shape) == shape and value.ndim == 2, "decoded_matrix_differs")
        slope, intercept = float(getattr(header, "RescaleSlope", 1)), float(getattr(header, "RescaleIntercept", 0))
        check(math.isfinite(slope) and math.isfinite(intercept) and slope != 0, "nonfinite_or_zero_rescale")
        value = value.astype(np.float64) * slope + intercept
        check(np.isfinite(value).all(), "nonfinite_decoded_pixels")
        arrays.append(value)
        source_files.append({"SOPInstanceUID": sop[index], "filename": path.name, "syntax": str(syntax), "rescale": [slope, intercept],
                             **base["file_pin"](path)})
    volume = np.stack(arrays)
    retained_min, retained_max = float(volume.min()), float(volume.max())
    if photometric == "MONOCHROME1":
        volume = retained_min + retained_max - volume
    if plan["transpose"]:
        volume = volume.transpose(0, 2, 1)
    if plan["flip_rows"]:
        volume = volume[:, ::-1, :]
    if plan["flip_columns"]:
        volume = volume[:, :, ::-1]
    resized = np.stack([cv2.resize(np.ascontiguousarray(value), (256, 256), interpolation=cv2.INTER_LINEAR)
                        if value.shape != (256, 256) else value for value in volume])
    check(resized.shape == (16, 256, 256) and np.isfinite(resized).all(), "invalid_resized_clip")
    return resized, {**selected, **plan, "native_matrix": shape, "pixel_spacing_mm": spacing,
                     "photometric": photometric, "retained_pre_resize_range": [retained_min, retained_max],
                     "rescale_resize_dtype": str(resized.dtype), "selected_sources": source_files}


def status_snapshot(state, journal, started):
    """Small constant-size status: never serialize requested1080 or old records."""
    return {"schema_version": "orthodiff_fluid_fallback1080_small_status_v1", "complete": state["complete"],
            "phase": state["phase"], "elapsed_seconds": time.monotonic() - started,
            "current_UID": state.get("current_UID"), "coverage": journal.coverage(),
            "journal": {"filename": journal.path.name, **journal.pin()},
            "durability": "append+flush+fsync before publishing this prefix digest",
            "error_type": state.get("error", {}).get("type")}


def run(config, vendor, base, pilot, full, output_dir=Path("/kaggle/working/orthodiff-fluid-fallback1080")):
    """Kaggle-only separate fallback extraction; parent successes stay external."""
    RecordJournal = full["RecordJournal"]
    output_dir = Path(output_dir)
    base["require"](not output_dir.exists() and not output_dir.is_symlink(), "Fresh fluid fallback output directory required")
    base["require"](len(config["uids"]) == len(set(config["uids"])) == 1080, "Exactly1080 frozen unknown-family UIDs required")
    base["require"](config["operational_resource_limits"] == {
        "CUDA_reserved_bytes": 14 * 1024**3, "process_RSS_bytes": 8 * 1024**3,
        "total_seconds": 28800, "UID_boundary_cooperative_clock_check": True,
        "hard_C_decoder_timeout": False,
    }, "Frozen fluid fallback operational resource contract required")
    output_dir.mkdir(parents=True)
    output = output_dir / "ORTHODIFF_FLUID_FALLBACK_SUMMARY.json"
    journal = RecordJournal(output_dir / "ORTHODIFF_FLUID_FALLBACK_RECORDS.jsonl", config["uids"])
    status = output_dir / "ORTHODIFF_FLUID_FALLBACK_STATUS.json"
    PilotDataError = pilot["PilotDataError"]
    check = pilot["check"]
    read_clip = bind_reader(pilot)
    requests = config["fluid_requests"]
    base["require"]([row["StudyInstanceUID"] for row in requests] == config["uids"]
                    and sum(row["selected_series"] is not None for row in requests) == 1077, "Frozen1080/1077/3 subset differs")
    request_by_uid = {row["StudyInstanceUID"]: row for row in requests}
    strict_load, decoder_preflight = pilot["strict_load"], pilot["decoder_preflight"]
    start = time.monotonic()
    state = {"schema_version": "orthodiff_fluid_fallback1080_frozen_continuous16_v1", "complete": False,
             "protocol": config["fluid_protocol"], "frozen_parent_protocol": config["preprocess"],
             "cohort_UID_hash": config["UID_order_sha256"], "requested_UIDs": config["uids"],
             "scope": {"Gold_labels": 0, "Report": 0, "training": 0, "network": 0, "Save_Submit": 0,
                       "Public_improvement_verified": False, "author_NIfTI_orientation_equivalent": False},
             "phase": "start", "model_load_calls": 0,
             "route": "fluid_sensitive_fallback", "confirmed_PD": False,
             "source_bindings": config["fluid_source_bindings"],
             "fluid_scope": {"requested_count": 1080, "parent_success_UIDs_requested": 0,
                            "parent_success_features_modified": 0, "adopted": False,
                            "frozen32_protocol_has_historical_cohort_output_resource_descriptions": True,
                            "current_operational_resource_limits": config["operational_resource_limits"],
                            "runtime_embeds_UIDs_not_groups_folds_Report_or_targets": True,
                            "outputs": [journal.path.name, status.name, output.name],
                            "resumption_or_skip_existing": False, "missing_feature_zero_fill": False,
                            "frozen_pilot_pins": config["frozen_pilot_pins"]}}

    def progress(phase):
        state["phase"], state["elapsed_seconds"] = phase, time.monotonic() - start
        base["write_receipt"](status, status_snapshot(state, journal, start))
        if phase in {"single_frozen_backbone_loaded", "fluid1080_finished", "fatal_error"} or (
            phase in {"feature_success_recorded", "data_failure_recorded"} and journal.count % 32 == 0
        ):
            print("ORTHODIFF_FLUID_FALLBACK_PROGRESS " + json.dumps({
                "phase": phase, "elapsed_seconds": state["elapsed_seconds"], "complete": state["complete"],
                "requested": len(journal.uids), "recorded": journal.count, "success": journal.success,
                "failure": journal.failure, "journal_prefix_SHA256": journal.pin()["sha256"],
            }, separators=(",", ":")), flush=True)

    try:
        uid_bytes = "".join(uid + "\n" for uid in config["uids"]).encode()
        base["require"](base["sha"](uid_bytes) == config["UID_order_sha256"], "Frozen UID order differs")
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
        base["require"](set(series_by_uid) == wanted, "All1080 require exact official series metadata coverage")
        image_roots = [path for path in (root / "train_series", root / "train_images") if path.is_dir()]
        base["require"](len(image_roots) == 1, "Unique official train DICOM root required")
        progress("all_nine_assets_and_UID_metadata_guarded")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        with tempfile.TemporaryDirectory(prefix="od-fluid-fallback-") as temp:
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
                        base["require"](time.monotonic() - start < config["operational_resource_limits"]["total_seconds"], "Fluid fallback cooperative wall budget exceeded")
                        state["current_UID"] = uid
                        progress("metadata_and_continuous16_" + uid)
                        begin = time.monotonic()
                        selected = None
                        try:
                            selected, paths = choose_fixed_series(uid, request_by_uid[uid], series_by_uid[uid], image_roots[0], check)
                            check(bool(paths), "fixed_chosen_series_has_no_DICOM_files")
                            array, detail = read_clip(uid, selected, paths, pydicom, np, cv2, base)
                            x = torch.from_numpy(np.ascontiguousarray(array)).float()
                            lo, hi = x.min(), x.max()
                            check(bool(torch.isfinite(x).all().item()) and bool((hi > lo).item()), "constant_or_nonfinite_resized_clip")
                        except (MemoryError, SystemError):
                            raise
                        except (PilotDataError, OSError, ValueError, RuntimeError, NotImplementedError, pydicom.errors.InvalidDicomError) as error:
                            journal.append({"StudyInstanceUID": uid, "status": "data_failure", "selected_series": selected,
                                            "reason_type": type(error).__name__, "reason": str(error), "data_seconds": time.monotonic() - begin,
                                            "feature_forward_calls": 0, "fallback_features": False,
                                            "route": "fluid_sensitive_fallback", "confirmed_PD": False,
                                            "parent_record_canonical_sha256": request_by_uid[uid]["parent_record_canonical_sha256"]})
                            progress("data_failure_recorded")
                            continue
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
                        journal.append({"StudyInstanceUID": uid, "status": "success", "metadata": detail, "noise_seed": seed,
                            "data_seconds": data_seconds, "feature_seconds": feature_seconds, "normalized_input": base["tensor_info"](x),
                            "normalization_min_max_float32": [float(lo), float(hi)], "noisy_input": noisy[-1], "mid_2": base["tensor_info"](feature),
                            "pooled": base["tensor_info"](pooled), "pooled_float32_values": pooled[0].tolist(), "forward_counts": delta,
                            "CPU_RNG_before_after_SHA": [base["tensor_info"](cpu_rng)["sha256"], base["tensor_info"](torch.random.get_rng_state())["sha256"]],
                            "all_CUDA_RNG_before_after_SHA": [[base["tensor_info"](a)["sha256"], base["tensor_info"](b)["sha256"]]
                                                             for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True)],
                            "all_RNG_restored": True})
                        del x, feature, pooled, array
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
                base["require"](journal.count == 1080 and counts["denoiser"] == state["coverage"]["success"]
                                and counts["decoder_blocks"] == counts["final_out"] == 0, "Final coverage/forward binding differs")
                base["require"](state["resources"]["peak_cuda_reserved_bytes"] <= 14 * 1024**3
                                and state["resources"]["peak_process_RSS_bytes"] <= 8 * 1024**3
                                and state["resources"]["total_seconds"] <= 28800, "Fluid fallback resource budget exceeded")
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
        state["all_requested_MRI_success"] = state["coverage"]["success"] == 1080
        progress("fluid1080_finished")
    except Exception as error:
        journal.close()
        state["complete"] = False
        state["coverage"] = journal.coverage()
        state["journal"] = {"filename": journal.path.name, **journal.pin(), "final_full_file_rehash_matches_incremental": False}
        state["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        state["owned_temp_removed"] = not folder.exists() if "folder" in locals() else True
        print("ORTHODIFF_FLUID_FALLBACK_ERROR " + json.dumps(state["error"], separators=(",", ":")), file=sys.stderr, flush=True)
        progress("fatal_error")
    finally:
        journal.close()
    base["write_receipt"](output, state)
    print("ORTHODIFF_FLUID_FALLBACK " + json.dumps({"complete": state["complete"], "receipt": str(output), "pin": base["file_pin"](output)}), flush=True)
    return state
