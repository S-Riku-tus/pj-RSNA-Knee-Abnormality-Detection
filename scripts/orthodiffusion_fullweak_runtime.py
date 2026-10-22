"""Full4207 frozen-pilot features; only transport/control differ from fixed32."""
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
from collections import Counter, defaultdict
from pathlib import Path


class RecordJournal:
    """One durable JSON line per requested UID, incremental digest, no zero rows."""

    def __init__(self, path, uids):
        self.path = Path(path)
        if not uids or len(uids) != len(set(uids)):
            raise ValueError("Unique nonempty requested UID order required")
        self.uids = tuple(uids)
        self.handle = self.path.open("xb")
        self.digest = hashlib.sha256()
        self.bytes = 0
        self.count = 0
        self.success = 0
        self.failure = 0
        self.success_strata = Counter()
        self.failure_reason_types = Counter()

    def append(self, record):
        if self.count >= len(self.uids) or record.get("StudyInstanceUID") != self.uids[self.count]:
            raise ValueError("Record UID must exactly follow frozen requested order")
        status = record.get("status")
        if status == "success":
            values = record.get("pooled_float32_values")
            if not isinstance(values, list) or len(values) != 256 or not all(
                isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in values
            ):
                raise ValueError("Exactly256 finite successful features required")
            raw = struct.pack("<256f", *values)
            if list(struct.unpack("<256f", raw)) != values or record.get("pooled") != {
                "shape": [1, 256], "dtype": "torch.float32", "bytes": 1024, "sha256": hashlib.sha256(raw).hexdigest()
            }:
                raise ValueError("Successful feature vector must match FP32 pin exactly")
            if record.get("forward_counts") != {"denoiser": 1, "decoder_blocks": 0, "final_out": 0} or record.get("all_RNG_restored") is not True:
                raise ValueError("Success forward/RNG contract differs")
        elif status == "data_failure":
            if not record.get("reason") or record.get("feature_forward_calls") != 0 or record.get("fallback_features") is not False:
                raise ValueError("Failure reason/no-forward/no-fallback required")
            if any(key in record for key in ("pooled", "pooled_float32_values", "mid_2", "normalized_input")):
                raise ValueError("Failure rows must not contain fabricated features")
        else:
            raise ValueError("Only success or data_failure records permitted")
        encoded = (json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n").encode()
        self.handle.write(encoded)
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.digest.update(encoded)
        self.bytes += len(encoded)
        self.count += 1
        self.success += status == "success"
        self.failure += status == "data_failure"
        if status == "success":
            self.success_strata[record["metadata"]["stratum"]] += 1
        else:
            self.failure_reason_types[record["reason_type"]] += 1

    def pin(self):
        return {"bytes": self.bytes, "sha256": self.digest.hexdigest()}

    def coverage(self):
        return {"requested": len(self.uids), "recorded": self.count, "success": self.success,
                "failure": self.failure, "success_strata": dict(self.success_strata),
                "failure_reason_types": dict(self.failure_reason_types)}

    def close(self):
        if not self.handle.closed:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()

    def finalize(self):
        self.close()
        digest = hashlib.sha256()
        length = 0
        with self.path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
                length += len(block)
        if {"bytes": length, "sha256": digest.hexdigest()} != self.pin():
            raise ValueError("Final journal bytes/hash differ from append digest")
        return self.pin()


def status_snapshot(state, journal, started):
    """Small constant-size status: never serialize requested4207 or old records."""
    return {"schema_version": "orthodiff_full4207_small_status_v1", "complete": state["complete"],
            "phase": state["phase"], "elapsed_seconds": time.monotonic() - started,
            "current_UID": state.get("current_UID"), "coverage": journal.coverage(),
            "journal": {"filename": journal.path.name, **journal.pin()},
            "durability": "append+flush+fsync before publishing this prefix digest",
            "error_type": state.get("error", {}).get("type")}


def run(config, vendor, base, pilot, output_dir=Path("/kaggle/working/orthodiff-fullweak4207")):
    """Kaggle-only extraction. Frozen helpers and numerical forward are unchanged."""
    output_dir = Path(output_dir)
    base["require"](not output_dir.exists() and not output_dir.is_symlink(), "Fresh fullweak output directory required")
    base["require"](len(config["uids"]) == len(set(config["uids"])) == 4207, "All4207 unique weak UIDs required")
    base["require"](config["operational_resource_limits"] == {
        "CUDA_reserved_bytes": 14 * 1024**3, "process_RSS_bytes": 8 * 1024**3,
        "total_seconds": 28800, "UID_boundary_cooperative_clock_check": True,
        "hard_C_decoder_timeout": False,
    }, "Frozen fullweak operational resource contract required")
    output_dir.mkdir(parents=True)
    output = output_dir / "ORTHODIFF_FULLWEAK_SUMMARY.json"
    journal = RecordJournal(output_dir / "ORTHODIFF_FULLWEAK_RECORDS.jsonl", config["uids"])
    status = output_dir / "ORTHODIFF_FULLWEAK_STATUS.json"
    PilotDataError = pilot["PilotDataError"]
    check = pilot["check"]
    choose_series, read_clip = pilot["choose_series"], pilot["read_clip"]
    strict_load, decoder_preflight = pilot["strict_load"], pilot["decoder_preflight"]
    start = time.monotonic()
    state = {"schema_version": "orthodiff_full4207_frozen_continuous16_v1", "complete": False,
             "protocol": config["preprocess"], "cohort_UID_hash": config["UID_order_sha256"], "requested_UIDs": config["uids"],
             "scope": {"Gold_labels": 0, "Report": 0, "training": 0, "network": 0, "Save_Submit": 0,
                       "Public_improvement_verified": False, "author_NIfTI_orientation_equivalent": False},
             "phase": "start", "model_load_calls": 0,
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
        if phase in {"single_frozen_backbone_loaded", "full4207_finished", "fatal_error"} or (
            phase in {"feature_success_recorded", "data_failure_recorded"} and journal.count % 32 == 0
        ):
            print("ORTHODIFF_FULLWEAK_PROGRESS " + json.dumps({
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
        base["require"](set(series_by_uid) == wanted, "All4207 require exact official series metadata coverage")
        image_roots = [path for path in (root / "train_series", root / "train_images") if path.is_dir()]
        base["require"](len(image_roots) == 1, "Unique official train DICOM root required")
        progress("all_nine_assets_and_UID_metadata_guarded")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        with tempfile.TemporaryDirectory(prefix="od-fullweak-") as temp:
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
                            journal.append({"StudyInstanceUID": uid, "status": "data_failure", "selected_series": selected,
                                            "reason_type": type(error).__name__, "reason": str(error), "data_seconds": time.monotonic() - begin,
                                            "feature_forward_calls": 0, "fallback_features": False})
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
                base["require"](journal.count == 4207 and counts["denoiser"] == state["coverage"]["success"]
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
        progress("full4207_finished")
    except Exception as error:
        journal.close()
        state["complete"] = False
        state["coverage"] = journal.coverage()
        state["journal"] = {"filename": journal.path.name, **journal.pin(), "final_full_file_rehash_matches_incremental": False}
        state["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        state["owned_temp_removed"] = not folder.exists() if "folder" in locals() else True
        progress("fatal_error")
    finally:
        journal.close()
    base["write_receipt"](output, state)
    print("ORTHODIFF_FULLWEAK " + json.dumps({"complete": state["complete"], "receipt": str(output), "pin": base["file_pin"](output)}), flush=True)
    return state
