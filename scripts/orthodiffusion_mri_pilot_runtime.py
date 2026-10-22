"""Fixed32 Kaggle-only continuous16 pilot; stdlib-only until explicit run()."""
from __future__ import annotations

import csv
import importlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path


class PilotDataError(ValueError):
    """Chosen-series data problem; recorded without replacing its UID/series."""


def check(condition, reason):
    if not condition:
        raise PilotDataError(reason)


def classify_pd(description, sequence):
    text = re.sub(r"[_\-.]+", " ", (str(description) + " " + str(sequence)).casefold())
    contradiction = bool(re.search(r"(?<![a-z0-9])(?:t1|t1w|t1tse|t1fse|t2|t2w|t2tse|t2fse)(?![a-z0-9])", text))
    pd = bool(re.search(r"(?<![a-z0-9])(?:pd|pdw|pdtse|pdfse|proton\s*density)(?![a-z0-9])", text))
    return {"explicit_PD": pd, "T1_T2_contradiction": contradiction,
            "eligible": pd and not contradiction,
            "reason": "explicit_PD" if pd and not contradiction else "T1_T2_contradiction" if contradiction else "unknown_PD"}


def orientation_plan(iop):
    values = [float(value) for value in iop]
    check(len(values) == 6 and all(math.isfinite(value) for value in values), "missing_or_nonfinite_IOP")
    col, row = values[:3], values[3:]  # IOP first vector increases array columns.
    dot = sum(a * b for a, b in zip(col, row, strict=True))
    check(abs(sum(x * x for x in col) - 1) <= .001 and abs(sum(x * x for x in row) - 1) <= .001
          and abs(dot) <= .001, "nonorthonormal_IOP")
    normal = [col[1] * row[2] - col[2] * row[1], col[2] * row[0] - col[0] * row[2], col[0] * row[1] - col[1] * row[0]]
    length = math.sqrt(sum(value * value for value in normal))
    normal = [value / length for value in normal]
    check(abs(normal[0]) >= .90, "not_sagittal_or_oblique_normal")
    if abs(row[2]) >= .90 and abs(col[1]) >= .90:
        transpose, out_row, out_col = False, row, col
    elif abs(col[2]) >= .90 and abs(row[1]) >= .90:
        transpose, out_row, out_col = True, col, row
    else:
        raise PilotDataError("oblique_inplane_axes_not_permutation")
    if normal[0] > 0:
        normal = [-value for value in normal]
    return {"transpose": transpose, "flip_rows": out_row[2] > 0, "flip_columns": out_col[1] < 0,
            "depth_normal_LPS": normal, "raw_IOP": values,
            "dominance_cosines": [abs(normal[0]), abs(out_row[2]), abs(out_col[1])],
            "canonical_axes": "depthLtoR,rowStoI,columnAtoP; retained native obliquity"}


def physical_order(headers):
    check(len(headers) >= 16, "fewer_than16_actual_slices")
    iop = list(headers[0]["IOP"])
    plan = orientation_plan(iop)
    positions = []
    for index, header in enumerate(headers):
        other = list(header["IOP"])
        check(len(other) == 6 and all(math.isfinite(float(x)) for x in other)
              and max(abs(float(a) - float(b)) for a, b in zip(iop, other, strict=True)) <= .001, "inconsistent_IOP")
        ipp = [float(value) for value in header["IPP"]]
        check(len(ipp) == 3 and all(math.isfinite(value) for value in ipp), "missing_or_nonfinite_IPP")
        positions.append((sum(a * b for a, b in zip(ipp, plan["depth_normal_LPS"], strict=True)), index))
    positions.sort()
    gaps = [b[0] - a[0] for a, b in zip(positions, positions[1:])]
    check(all(value > .0001 for value in gaps), "duplicate_slice_positions")
    median = statistics.median(gaps)
    check(max(abs(value - median) for value in gaps) <= max(.1, .10 * median), "irregular_or_missing_slice_spacing")
    depth = len(headers)
    start = depth // 2 - 8
    order = [index for _, index in positions]
    return {**plan, "physical_order": order, "physical_positions_mm": [value for value, _ in positions],
            "spacing_mm": median, "minimum_gap_mm": min(gaps), "maximum_gap_mm": max(gaps),
            "depth": depth, "clip_start": start, "clip_end": start + 16, "selected_indices": order[start:start + 16]}


def decoder_preflight(config, folder, base, state):
    import importlib.metadata

    root = base["resolve_stage"](config["candidates"])
    for name, expected in config["members"].items():
        path = root / name
        base["require"](path.is_file() and not path.is_symlink() and base["file_pin"](path) == expected, "Decoder member pin differs: " + name)
    base["require"](sys.platform == "linux" and sys.version_info[:2] == (3, 13), "Pinned decoder requires LinuxPython3.13")
    manifest = json.loads((root / "decoder-manifest.json").read_bytes())
    target = folder / "decoder-env"
    before = {name: importlib.metadata.version(name) for name in ("numpy", "torch", "pydicom")}
    command = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--target", str(target),
               *[str(root / "wheels" / row["filename"]) for row in manifest["wheels"]]]
    installed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=180)
    sys.path.insert(0, str(target))
    import numpy as np
    import pydicom

    after = {name: importlib.metadata.version(name) for name in before}
    base["require"](before == after, "Decoder installation changed core environment")
    phantom = json.loads((root / "phantoms/phantom-manifest.json").read_bytes())
    expected = np.load(root / "phantoms" / phantom["expected_filename"], allow_pickle=False)
    verified = []
    for row in phantom["files"]:
        path = root / "phantoms" / row["filename"]
        ds = pydicom.dcmread(path)
        default = ds.pixel_array
        explicit = pydicom.pixels.pixel_array(ds, decoding_plugin="pylibjpeg")
        base["require"](str(ds.file_meta.TransferSyntaxUID) == row["syntax"] and default.shape == expected.shape
                        and explicit.shape == expected.shape and default.dtype == expected.dtype and explicit.dtype == expected.dtype
                        and np.array_equal(default, expected) and np.array_equal(explicit, expected), "Own synthetic decoder fixture mismatch")
        verified.append({"file": row["filename"], "syntax": row["syntax"], "shape": list(default.shape),
                         "dtype": str(default.dtype), "default_and_explicit_values_equal": True})
    base["require"](len(verified) == 4, "Four fixed decoder fixtures required")
    state["decoder"] = {"root": str(root), "verified_members": config["members"], "fixtures": verified,
                        "install_returncode": installed.returncode, "install_stdout_stderr": (installed.stdout + installed.stderr)[-12000:],
                        "no_index": True, "no_deps": True, "target": str(target), "core_versions_unchanged": before,
                        "packages": {row["name"]: importlib.metadata.version(row["name"]) for row in manifest["wheels"]}}
    return np, pydicom, target


HEADER_TAGS = ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "SeriesDescription", "SequenceName", "ImageOrientationPatient",
               "ImagePositionPatient", "PixelSpacing", "Rows", "Columns", "NumberOfFrames", "PhotometricInterpretation",
               "RescaleSlope", "RescaleIntercept", "InstanceNumber")


def choose_series(uid, rows, image_root, pydicom):
    candidates = []
    rejected = []
    for row in rows:
        if row["Anatomical_Plane"].strip().casefold() != "sagittal":
            continue
        sid = row["SeriesInstanceUID"]
        entry = {"SeriesInstanceUID": sid, "Fat_Suppression": row["Fat_Suppression"], "Fluid_Sensitive": row["Fluid_Sensitive"]}
        paths = sorted((image_root / uid / sid).glob("*.dcm"))
        if not paths:
            rejected.append({**entry, "reason": "no_DICOM_files"})
            continue
        try:
            first = pydicom.dcmread(paths[0], stop_before_pixels=True, specific_tags=HEADER_TAGS)
            classification = classify_pd(getattr(first, "SeriesDescription", ""), getattr(first, "SequenceName", ""))
            entry.update(SeriesDescription=str(getattr(first, "SeriesDescription", "")),
                         SequenceName=str(getattr(first, "SequenceName", "")), **classification)
            check(str(getattr(first, "StudyInstanceUID", "")) == uid and str(getattr(first, "SeriesInstanceUID", "")) == sid,
                  "header_UID_binding_mismatch")
            check(row["Fat_Suppression"] in {"0", "1"}, "unknown_FS_flag")
            check(classification["eligible"], classification["reason"])
        except (PilotDataError, OSError, ValueError, pydicom.errors.InvalidDicomError) as error:
            rejected.append({**entry, "reason": str(error)})
            continue
        candidates.append((int(row["Fat_Suppression"]), sid, entry, paths))
    check(bool(candidates), "no_explicit_noncontradictory_PD_sagittal_series:" + json.dumps(rejected, separators=(",", ":")))
    _, _, selected, paths = sorted(candidates, key=lambda value: (value[0], value[1]))[0]
    selected.update(stratum="PD_nonFS" if selected["Fat_Suppression"] == "0" else "PD_FS_fallback",
                    selection="FS0-before-FS1 then SeriesUID lexicographic; before geometry/pixels", rejected_candidates=rejected,
                    eligible_candidate_series=[value[1] for value in sorted(candidates, key=lambda value: (value[0], value[1]))])
    return selected, paths


def read_clip(uid, selected, paths, pydicom, np, cv2, base):
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
        classification = classify_pd(getattr(header, "SeriesDescription", ""), getattr(header, "SequenceName", ""))
        check(classification["eligible"], "mixed_or_unknown_PD_header")
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


def strict_load(torch, unet, features, stage, base, state):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1729)
        denoiser = unet.create_model(image_size=256, num_channels=64, num_res_blocks=1, in_channels=1, out_channels=1)
        model = features.GaussianDiffusion(denoiser, image_size=256, depth_size=16, channels=1, timesteps=1000, loss_type="l2")
    checkpoint = torch.load(stage / "weights/sagittal_model.pt", map_location="cpu", weights_only=True)
    base["require"](isinstance(checkpoint, dict) and "ema" in checkpoint, "Explicit checkpoint EMA required")
    ema, changes = base["normalize_ema"](checkpoint["ema"])
    expected = model.state_dict()
    base["require"](all(isinstance(value, torch.Tensor) for value in ema.values()), "Tensor-only EMA required")
    def metadata(rows):
        return {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in rows.items()}
    base["validate_metadata"](metadata(ema), metadata(expected))
    base["require"](all(torch.isfinite(value).all().item() for value in ema.values()), "Nonfinite EMA")
    base["require"](set(dict(model.named_buffers())) == set(base["BUFFER_NAMES"]), "Expected thirteen buffers required")
    checks = {key: bool(torch.equal(ema[key], expected[key])) for key in base["BUFFER_NAMES"]}
    base["require"](all(checks.values()), "Exact schedule buffer mismatch")
    loaded = model.load_state_dict(ema, strict=True)
    base["require"](not loaded.missing_keys and not loaded.unexpected_keys
                    and all(torch.equal(model.state_dict()[key], value) for key, value in ema.items()), "Full loaded EMA differs")
    base["require"](len(ema) == 257 and sum(value.numel() for value in model.parameters()) == 34524545, "Artificial-qualified architecture coverage differs")
    state["checkpoint"] = {"load_calls": 1, "weights_only": True, "strict": True, "arm": "ema", "keys": len(ema),
                           "parameters": 34524545, "prefix_changes": changes, "all_key_shapes_dtypes": metadata(ema),
                           "schedule_buffers_exact": checks, "all_loaded_tensors_equal": True}
    model.eval().requires_grad_(False).to("cuda:0")
    return model, denoiser


def run(config, vendor, base, output=Path("/kaggle/working/ORTHODIFF_MRI_PILOT.json")):
    output = Path(output)
    base["require"](not output.exists(), "Fresh pilot output required")
    start = time.monotonic()
    state = {"schema_version": "orthodiff_sagittal_fixed32_continuous16_pilot_v1", "complete": False,
             "protocol": config["preprocess"], "cohort_UID_hash": config["UID_order_sha256"], "requested_UIDs": config["uids"],
             "records": [], "scope": {"Gold_labels": 0, "Report": 0, "training": 0, "network": 0, "Save_Submit": 0,
                                        "Public_improvement_verified": False, "author_NIfTI_orientation_equivalent": False},
             "phase": "start", "model_load_calls": 0}
    def progress(phase):
        state["phase"], state["elapsed_seconds"] = phase, time.monotonic() - start
        base["write_receipt"](output, state)
    try:
        base["require"](len(config["uids"]) == len(set(config["uids"])) == 32, "Fixed32 unique UIDs required")
        uid_bytes = "".join(uid + "\n" for uid in config["uids"]).encode()
        base["require"](base["sha"](uid_bytes) == config["UID_order_sha256"], "Frozen UID order differs")
        stage = base["resolve_stage"](config["assets"]["stage_candidates"])
        state["verified_generic_assets"] = base["guard_stage"](stage, config["assets"])
        root = base["resolve_stage"](config["competition_candidates"])
        series_path = root / "train_series.csv"
        base["require"](base["file_pin"](series_path) == config["series_metadata_pin"], "Official series metadata differs")
        with series_path.open(encoding="utf-8-sig", newline="") as stream:
            series = [row for row in csv.DictReader(stream) if row["StudyInstanceUID"] in set(config["uids"])]
        image_roots = [path for path in (root / "train_series", root / "train_images") if path.is_dir()]
        base["require"](len(image_roots) == 1, "Unique official train DICOM root required")
        progress("all_nine_assets_and_UID_metadata_guarded")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        with tempfile.TemporaryDirectory(prefix="od-mri-pilot-") as temp:
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
                        state["current_UID"] = uid
                        progress("metadata_and_continuous16_" + uid)
                        begin = time.monotonic()
                        selected = None
                        try:
                            selected, paths = choose_series(uid, [row for row in series if row["StudyInstanceUID"] == uid], image_roots[0], pydicom)
                            array, detail = read_clip(uid, selected, paths, pydicom, np, cv2, base)
                            x = torch.from_numpy(np.ascontiguousarray(array)).float()
                            lo, hi = x.min(), x.max()
                            check(bool(torch.isfinite(x).all().item()) and bool((hi > lo).item()), "constant_or_nonfinite_resized_clip")
                        except (MemoryError, SystemError):
                            raise
                        except (PilotDataError, OSError, ValueError, RuntimeError, NotImplementedError, pydicom.errors.InvalidDicomError) as error:
                            state["records"].append({"StudyInstanceUID": uid, "status": "data_failure", "selected_series": selected,
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
                        state["records"].append({"StudyInstanceUID": uid, "status": "success", "metadata": detail, "noise_seed": seed,
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
                state["coverage"] = {"requested": 32, "success": sum(row["status"] == "success" for row in state["records"]),
                                     "failure": sum(row["status"] != "success" for row in state["records"]),
                                     "success_strata": dict(Counter(row["metadata"]["stratum"] for row in state["records"] if row["status"] == "success"))}
                state["resources"] = {"peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(0),
                                      "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(0),
                                      "peak_process_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                      "total_seconds": time.monotonic() - start}
                state["forward_counts"] = counts
                base["require"](len(state["records"]) == 32 and counts["denoiser"] == state["coverage"]["success"]
                                and counts["decoder_blocks"] == counts["final_out"] == 0, "Final coverage/forward binding differs")
                base["require"](state["resources"]["peak_cuda_reserved_bytes"] <= 14 * 1024**3
                                and state["resources"]["peak_process_RSS_bytes"] <= 8 * 1024**3
                                and state["resources"]["total_seconds"] <= 1800, "Pilot resource budget exceeded")
            finally:
                for path in (temp, str(decoder_target)):
                    if path in sys.path:
                        sys.path.remove(path)
                for name in list(sys.modules):
                    if name == base["PACKAGE"] or name.startswith(base["PACKAGE"] + "."):
                        del sys.modules[name]
        state["owned_temp_removed"] = not folder.exists()
        state["complete"] = True
        state["all_requested_MRI_success"] = state["coverage"]["success"] == 32
        progress("fixed32_pilot_finished")
    except Exception as error:
        state["complete"] = False
        state["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        progress("fatal_error")
    print("ORTHODIFF_MRI_PILOT " + json.dumps({"complete": state["complete"], "receipt": str(output), "pin": base["file_pin"](output)}), flush=True)
    return state
