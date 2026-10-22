"""Source-only prepared test feature/CPU head boundary, with no Kaggle API.

Import is standard-library only. Actual entry is disabled by default and needs
the frozen model's separate, source-bound actual qualification. No CSV overlay,
parent selection, fit, upload, Save or Submit is performed here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import struct
import sys
import tempfile
import time
from pathlib import Path

ID = "StudyInstanceUID"
TARGETS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")
SCHEMA = "orthodiff_innerC_dynamic_test_transport_v1"
MODEL_SHA = "c9b5ae26d0ad5895903002ff6bfd2c4947bed152c070c749588be8c76710a714"


def require(value, message):
    if not value:
        raise ValueError(message)


def pin(path):
    digest, count = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            count += len(block)
            digest.update(block)
    return {"bytes": count, "sha256": digest.hexdigest()}


def parse_json(raw):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON field")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON token")))


def checked_json(path, expected_sha, maximum=4 * 1024**2):
    require(isinstance(expected_sha, str) and len(expected_sha) == 64
            and all(c in "0123456789abcdef" for c in expected_sha), "Full SHA256 required")
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= maximum,
            "Bounded regular JSON required")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha, "JSON SHA256 differs")
    return parse_json(raw)


def safe_uid(value):
    require(isinstance(value, str) and value and value.strip() == value
            and value not in {".", ".."} and not any(c in value for c in "/\\:\x00\r\n"),
            "Unsafe or empty UID path segment")
    return value


def uid_hash(uids):
    return hashlib.sha256("".join(uid + "\n" for uid in uids).encode()).hexdigest()


def read_test_metadata(test_csv, series_csv):
    """Project only test UID and the five original selector metadata fields."""
    with Path(test_csv).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        require(header is not None and len(header) == len(set(header)) and ID in header,
                "Unique test CSV header with StudyInstanceUID required")
        index = header.index(ID)
        uids = []
        for row in reader:
            require(len(row) == len(header), "Malformed test row width")
            uids.append(safe_uid(row[index]))
    require(uids and len(uids) == len(set(uids)), "Unique nonempty requested test UIDs required")
    fields = (ID, "SeriesInstanceUID", "Anatomical_Plane", "Fat_Suppression", "Fluid_Sensitive")
    by_uid, seen, wanted = {uid: [] for uid in uids}, set(), set(uids)
    with Path(series_csv).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        require(header is not None and len(header) == len(set(header)) and set(fields) <= set(header),
                "Original series selector metadata header required")
        indices = [header.index(field) for field in fields]
        for row in reader:
            require(len(row) == len(header), "Malformed series row width")
            uid = row[indices[0]]
            if uid not in wanted:
                continue
            values = {field: row[index] for field, index in zip(fields, indices)}
            sid = safe_uid(values["SeriesInstanceUID"])
            require((uid, sid) not in seen, "Duplicate requested UID/series metadata")
            seen.add((uid, sid))
            by_uid[uid].append(values)
    return uids, by_uid, {"test_csv": pin(test_csv), "test_series_csv": pin(series_csv),
                          "requested_count": len(uids), "UID_order_sha256": uid_hash(uids)}


def validate_partition(uids, records):
    require(len(uids) == len(set(uids)) and uids and len(records) == len(uids),
            "Complete unique requested UID partition required")
    successes, failures = [], []
    for uid, record in zip(uids, records):
        require(record.get(ID) == uid, "Exactly-once ordered result UID required")
        if record.get("status") == "success":
            require(record.get("forward_counts") == {"denoiser": 1, "decoder_blocks": 0, "final_out": 0}
                    and record.get("all_RNG_restored") is True, "Success forward/RNG contract differs")
            values = record.get("regional_2x2_float32_values")
            require(isinstance(values, list) and len(values) == 1024 and all(
                type(value) in (int, float) and math.isfinite(value) for value in values),
                "Finite width1024 feature required")
            raw = struct.pack("<1024f", *values)
            require(list(struct.unpack("<1024f", raw)) == values
                    and record.get("regional_2x2") == {"shape": [1, 1024], "dtype": "torch.float32",
                                                       "bytes": 4096, "sha256": hashlib.sha256(raw).hexdigest()},
                    "Exact FP32 regional shape/hash required")
            successes.append(uid)
        else:
            require(record.get("status") == "data_failure" and record.get("reason")
                    and record.get("feature_forward_calls") == 0 and record.get("fallback_features") is False
                    and not any(key in record for key in ("regional_2x2_float32_values", "regional_2x2", "mid_2")),
                    "Typed no-forward failure without invented features required")
            failures.append(uid)
    return successes, failures


def make_failure(uid, selected, error, seconds):
    return {ID: uid, "status": "data_failure", "selected_series": selected,
            "reason_type": type(error).__name__, "reason": str(error), "data_seconds": seconds,
            "feature_forward_calls": 0, "fallback_features": False}


def process_uids(uids, series_by_uid, prepare, forward, data_error_types, progress=None, boundary=None):
    """Shared real/fake loop: data catch ends before the model is called."""
    records, digest, success = [], hashlib.sha256(), 0
    for uid in uids:
        if boundary is not None:
            boundary()  # Clock/resource contracts must never become data failures.
        started = time.monotonic()
        try:
            prepared = prepare(uid, series_by_uid.get(uid, []))
        except (MemoryError, SystemError):
            raise
        except data_error_types as error:
            records.append(make_failure(uid, getattr(error, "selected_series", None), error,
                                        time.monotonic() - started))
        else:
            record = forward(uid, prepared)  # Model/RNG/resource errors are fatal.
            require(record.get(ID) == uid and record.get("status") == "success", "Forward result differs")
            records.append(record)
            success += 1
        digest.update((json.dumps(records[-1], allow_nan=False, separators=(",", ":")) + "\n").encode())
        if progress is not None and len(records) % 32 == 0:
            progress({"recorded": len(records), "success": success, "failure": len(records) - success,
                      "record_prefix_sha256": digest.hexdigest()})
    validate_partition(uids, records)
    return records


def load_bundle(path, expected_sha):
    bundle = checked_json(path, expected_sha)
    require(bundle.get("schema_version") == SCHEMA and bundle.get("model_sha256") == MODEL_SHA
            and bundle.get("feature_dimension") == 1024 and bundle.get("base_seed") == 20261017,
            "Only the frozen innerC test feature transport is supported")
    require(pin(__file__) == bundle["transport_sources"]["runtime"], "Actual runtime source differs")
    for name, fragment in bundle["helper_closures"].items():
        raw = fragment["source"].encode()
        require({"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} == fragment["pin"],
                "Scientific helper closure changed: " + name)
    raw = bundle["adapter"]["source"].encode()
    require({"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} == bundle["adapter"]["pin"]
            and bundle["adapter"]["pin"]["sha256"] == "33af760be7c33672f780529cf42af67782dd9eac2215fbedb705b94a8a1983dd",
            "Frozen NumPy adapter differs")
    return bundle


def actual_entry(bundle_path, bundle_sha, model_path, model_sha, qualification_path, qualification_sha):
    """Every actual mode passes this before opening test metadata or DICOMs."""
    bundle = load_bundle(bundle_path, bundle_sha)
    require(bundle.get("actual_entry_enabled") is True, "Source-only transport: actual entry disabled")
    qualification = checked_json(qualification_path, qualification_sha)
    require(qualification.get("feature_transport_bundle_sha256") == bundle_sha,
            "Root qualification must bind this new feature transport")
    namespace = {"__name__": "_sealed_innerC_cpu_adapter"}
    exec(compile(bundle["adapter"]["source"], "<pinned-innerC-numpy>", "exec"), namespace)
    head = namespace["InnerCNumpyHead"](model_path, expected_model_sha256=model_sha,
                                      qualification_path=qualification_path,
                                      expected_qualification_sha256=qualification_sha)
    require(head.actual_model_qualified is True and head.public_submission_contract_qualified is False,
            "Only root-qualified single-model CPU predictions allowed")
    bundle["_qualified_head"] = head
    return bundle, head


def helper_namespaces(bundle):
    spaces = {}
    for name, fragment in bundle["helper_closures"].items():
        namespace = {"__name__": "_innerC_test_helpers_" + name}
        exec(compile(fragment["source"], "<pinned-helper-" + name + ">", "exec"), namespace)
        spaces[name] = namespace
    return spaces


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes((json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2) + "\n").encode())
    os.replace(temporary, path)


def write_feature_handoff(folder, uids, records, numpy_module, bundle_sha, metadata_pins):
    """Local intermediate only: successful UID matrix plus complete metadata."""
    np = numpy_module
    success, failed = validate_partition(uids, records)
    values = [record["regional_2x2_float32_values"] for record in records if record["status"] == "success"]
    matrix = np.asarray(values, dtype=np.float32).reshape(len(success), 1024)
    feature_path = Path(folder) / "success-features.npz"
    require(not feature_path.exists(), "Fresh local feature handoff required")
    with feature_path.open("xb") as stream:
        np.savez(stream, success_uids=np.asarray(success, dtype=str), features=matrix)
    receipt = {"schema_version": "orthodiff_innerC_test_features_complete_v1", "complete": True,
               "feature_transport_bundle_sha256": bundle_sha, "requested_uids": uids,
               "UID_order_sha256": uid_hash(uids), "success_uids": success, "failure_uids": failed,
               "metadata_inputs": metadata_pins, "matrix": {"filename": feature_path.name, **pin(feature_path)},
               "records": [{key: value for key, value in record.items() if key != "regional_2x2_float32_values"}
                           for record in records], "feature_dimension": 1024,
               "zero_filled_rows": 0, "research_features_or_labels_used": False}
    write_json(Path(folder) / "feature-receipt.json", receipt)
    return receipt


def load_feature_handoff(folder, expected_receipt_sha, numpy_module, expected_bundle_sha):
    np = numpy_module
    folder = Path(folder)
    receipt = checked_json(folder / "feature-receipt.json", expected_receipt_sha, maximum=64 * 1024**2)
    require(receipt.get("schema_version") == "orthodiff_innerC_test_features_complete_v1"
            and receipt.get("complete") is True and receipt.get("feature_transport_bundle_sha256") == expected_bundle_sha
            and receipt.get("feature_dimension") == 1024 and receipt.get("zero_filled_rows") == 0,
            "Complete source-bound feature handoff required")
    uids = receipt["requested_uids"]
    require(uids and len(uids) == len(set(uids)) and all(safe_uid(uid) for uid in uids)
            and uid_hash(uids) == receipt["UID_order_sha256"], "Requested UID order differs")
    row_path = folder / "success-features.npz"
    require(receipt["matrix"]["filename"] == row_path.name and not row_path.is_symlink()
            and pin(row_path) == {key: receipt["matrix"][key] for key in ("bytes", "sha256")},
            "Feature handoff matrix pin differs")
    with np.load(row_path, allow_pickle=False) as archive:
        require(set(archive.files) == {"success_uids", "features"}, "Unexpected handoff arrays")
        success, values = archive["success_uids"], archive["features"]
    require(success.ndim == 1 and success.dtype.kind == "U" and success.tolist() == receipt["success_uids"]
            and values.dtype == np.dtype("float32") and values.shape == (len(success), 1024)
            and np.isfinite(values).all(), "UID-aligned finite FP32 features required")
    rows, index = [], 0
    for record in receipt["records"]:
        record = dict(record)
        if record.get("status") == "success":
            require(index < len(success) and record.get(ID) == success[index], "Success matrix UID order differs")
            record["regional_2x2_float32_values"] = values[index].tolist()
            index += 1
        rows.append(record)
    actual_success, actual_failed = validate_partition(uids, rows)
    require(index == len(success) and actual_success == receipt["success_uids"]
            and actual_failed == receipt["failure_uids"], "Complete handoff partition differs")
    return receipt, values


def read_parent_rows(path, requested_uids):
    """Guard for a future external overlay; never chooses or writes a parent."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        require(next(reader, None) == [ID, *TARGETS], "Exact canonical parent CSV header order required")
        rows = []
        for row in reader:
            require(len(row) == 13, "Exactly thirteen parent fields required")
            rows.append(dict(zip([ID, *TARGETS], row)))
    parent_uids = [row[ID] for row in rows]
    require(len(parent_uids) == len(set(parent_uids)) == len(requested_uids)
            and set(parent_uids) == set(requested_uids), "Exact parent/test UID join required")
    return rows  # Original row order and cell text are retained.


def extract_test_features(bundle, competition_root, output_dir, seconds_budget):
    """Actual T4 child entry. The caller has already passed actual_entry()."""
    require(bundle.get("actual_entry_enabled") is True
            and getattr(bundle.get("_qualified_head"), "actual_model_qualified", False) is True
            and isinstance(bundle.get("_actual_bundle_sha"), str), "Actual adapter gate required before test metadata")
    require(0 < seconds_budget <= 9 * 3600, "Explicit remaining whole-notebook time budget required")
    started = time.monotonic()
    output_dir, root = Path(output_dir), Path(competition_root)
    require(not output_dir.exists() and not output_dir.is_symlink(), "Fresh dedicated feature directory required")
    output_dir.mkdir(parents=True)
    state = {"schema_version": "orthodiff_innerC_test_feature_execution_v1", "complete": False,
             "actual_feature_path_qualified": False, "phase": "source_guarded", "model_load_calls": 0}
    spaces = helper_namespaces(bundle)
    base, pilot, regional = spaces["base"], spaces["pilot"], spaces["regional"]
    counts = {"denoiser": 0, "decoder_blocks": 0, "final_out": 0}
    try:
        uids, series, metadata_pins = read_test_metadata(root / "test.csv", root / "test_series.csv")
        roots = [path for path in (root / "test_series", root / "test_images") if path.is_dir()]
        require(len(roots) == 1, "Unique official test image root required")
        stage = base["resolve_stage"](bundle["assets"]["stage_candidates"])
        state["assets"] = base["guard_stage"](stage, bundle["assets"])
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        with tempfile.TemporaryDirectory(prefix="innerC-test-science-") as temporary:
            folder = Path(temporary)
            np, pydicom, decoder_target = pilot["decoder_preflight"](bundle["decoder"], folder, base, state)
            import importlib
            import resource

            import cv2
            import torch

            require(torch.cuda.is_available(), "CUDA/T4 feature extraction required; no CPU fallback")
            torch.cuda.set_device(0)
            require("T4" in torch.cuda.get_device_name(0), "Pinned T4 feature runtime required")
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.use_deterministic_algorithms(True)
            require(sys.version_info[:2] == (3, 13) and np.__version__ == "2.1.3", "Pinned CP313/NumPy2.1.3 required")
            state["environment"] = {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
                                    "cv2": cv2.__version__, "pydicom": pydicom.__version__,
                                    "CUDA": torch.version.cuda, "device": torch.cuda.get_device_name(0)}
            package = base["install_vendor"](folder, bundle["vendor"])
            sys.path.insert(0, temporary)
            try:
                unet = importlib.import_module(base["PACKAGE"] + ".unet")
                features = importlib.import_module(base["PACKAGE"] + ".features")
                for name in ("__init__", "unet", "modules", "fp16_util", "features"):
                    module = sys.modules[base["PACKAGE"] if name == "__init__" else base["PACKAGE"] + "." + name]
                    path = Path(module.__file__).resolve()
                    require(path == (package / (name + ".py")).resolve()
                            and pin(path) == {key: bundle["vendor"][name + ".py"][key] for key in ("bytes", "sha256")},
                            "Scientific import origin/hash differs")
                model, denoiser = pilot["strict_load"](torch, unet, features, stage, base, state)
                state["model_load_calls"] = 1
                print("INNER_C_TEST_PROGRESS " + json.dumps({"phase": "model_loaded", "requested": len(uids)}), flush=True)
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

                def prepare(uid, rows):
                    selected = None
                    try:
                        selected, paths = pilot["choose_series"](uid, rows, roots[0], pydicom)
                        array, detail = pilot["read_clip"](uid, selected, paths, pydicom, np, cv2, base)
                        x = torch.from_numpy(np.ascontiguousarray(array)).float()
                        lo, hi = x.min(), x.max()
                        pilot["check"](bool(torch.isfinite(x).all().item()) and bool((hi > lo).item()), "constant_or_nonfinite_resized_clip")
                        return x, lo, hi, detail
                    except (MemoryError, SystemError):
                        raise
                    except Exception as error:
                        error.selected_series = selected
                        raise

                def forward(uid, prepared):
                    x, lo, hi, detail = prepared
                    x = ((x - lo) / (hi - lo + 1e-8) * 2 - 1).unsqueeze(0).unsqueeze(0).to("cuda:0")
                    t = torch.tensor([100], dtype=torch.long, device="cuda:0")
                    seed = base["scoped_seed"](uid, detail["SeriesInstanceUID"], [detail["clip_start"], detail["clip_end"]], 20261017)
                    cpu_rng = torch.random.get_rng_state().clone()
                    cuda_rng = [value.clone() for value in torch.cuda.get_rng_state_all()]
                    before = dict(counts)
                    torch.cuda.synchronize(0)
                    with torch.random.fork_rng(devices=[0]), torch.inference_mode():
                        torch.cuda.default_generators[0].manual_seed(seed)
                        feature = model.get_feature(x, t, name="mid_2", ret_in=False, ret_mid=True, ret_out=False, use_amp=False)
                        base["require"](list(feature.shape) == [1, 256, 1, 16, 16] and feature.dtype == torch.float32
                                        and torch.isfinite(feature).all().item(), "InvalidMRIfeature")
                    torch.cuda.synchronize(0)
                    restored = torch.equal(cpu_rng, torch.random.get_rng_state()) and all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True))
                    delta = {key: counts[key] - before[key] for key in counts}
                    base["require"](restored and delta == {"denoiser": 1, "decoder_blocks": 0, "final_out": 0}, "RNG/forward contract differs")
                    with torch.inference_mode():
                        pooled = regional["fixed_regional_pool"](feature, torch.cat)
                    require(list(pooled.shape) == [1, 1024] and pooled.dtype == torch.float32
                            and torch.isfinite(pooled).all().item(), "InvalidRegional1024")
                    require(torch.equal(cpu_rng, torch.random.get_rng_state()) and all(
                        torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True))
                        and {key: counts[key] - before[key] for key in counts} == delta, "Pooling changed RNG/forward")
                    require(torch.cuda.max_memory_reserved(0) <= 14 * 1024**3
                            and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= 8 * 1024**3,
                            "Feature resource budget exceeded")
                    return {ID: uid, "status": "success", "metadata": detail, "noise_seed": seed,
                            "normalization_min_max_float32": [float(lo), float(hi)],
                            "normalized_input": base["tensor_info"](x), "noisy_input": noisy[-1],
                            "mid_2": base["tensor_info"](feature), "regional_2x2": base["tensor_info"](pooled),
                            "regional_2x2_float32_values": pooled[0].tolist(), "forward_counts": delta,
                            "all_RNG_restored": True}

                def progress(summary):
                    print("INNER_C_TEST_PROGRESS " + json.dumps({"phase": "UIDs", **summary,
                        "elapsed": time.monotonic() - started}), flush=True)

                def boundary():
                    require(time.monotonic() - started < seconds_budget, "Remaining feature time budget exceeded")

                errors = (pilot["PilotDataError"], OSError, ValueError, RuntimeError,
                          NotImplementedError, pydicom.errors.InvalidDicomError)
                try:
                    records = process_uids(uids, series, prepare, forward, errors, progress, boundary)
                finally:
                    for handle in handles:
                        handle.remove()
                success, _ = validate_partition(uids, records)
                require(counts == {"denoiser": len(success), "decoder_blocks": 0, "final_out": 0},
                        "Final feature forward counts differ")
                state["resources"] = {"CUDA_reserved_bytes": torch.cuda.max_memory_reserved(0),
                    "RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    "elapsed_seconds": time.monotonic() - started}
                require(state["resources"]["elapsed_seconds"] < seconds_budget, "Final clock budget exceeded")
                require(state["resources"]["CUDA_reserved_bytes"] <= 14 * 1024**3
                        and state["resources"]["RSS_bytes"] <= 8 * 1024**3, "Final resource budget exceeded")
            finally:
                for path in (temporary, str(decoder_target)):
                    if path in sys.path:
                        sys.path.remove(path)
                for name in list(sys.modules):
                    if name == base["PACKAGE"] or name.startswith(base["PACKAGE"] + "."):
                        del sys.modules[name]
        require(not folder.exists(), "Owned scientific temporary directory remains")
        state["owned_scientific_temp_removed"] = True
        handoff = write_feature_handoff(output_dir, uids, records, np,
                                        bundle["_actual_bundle_sha"], metadata_pins)
        state["handoff"] = {"receipt": pin(output_dir / "feature-receipt.json"),
                            "success": len(handoff["success_uids"]), "failure": len(handoff["failure_uids"])}
        state["complete"] = True
        print("INNER_C_TEST_PROGRESS " + json.dumps({"phase": "complete", "complete": True,
            "requested": len(uids), "success": len(handoff["success_uids"]),
            "failure": len(handoff["failure_uids"]), "elapsed": time.monotonic() - started}), flush=True)
    except Exception as error:
        state.update(complete=False, error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        write_json(output_dir / "execution-receipt.json", state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("features", "head"))
    for name in ("bundle", "bundle-sha", "model", "model-sha", "qualification", "qualification-sha", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--competition-root")
    parser.add_argument("--seconds-budget", type=float)
    parser.add_argument("--feature-dir")
    parser.add_argument("--feature-receipt-sha")
    args = parser.parse_args(argv)
    bundle, head = actual_entry(args.bundle, args.bundle_sha, args.model, args.model_sha,
                                args.qualification, args.qualification_sha)
    bundle["_actual_bundle_sha"] = args.bundle_sha
    if args.mode == "features":
        require(args.competition_root is not None and args.seconds_budget is not None,
                "Explicit official test root/remaining time required")
        extract_test_features(bundle, args.competition_root, args.output_dir, args.seconds_budget)
    else:
        require(args.feature_dir is not None and args.feature_receipt_sha is not None, "Pinned local feature handoff required")
        import numpy as np

        receipt, values = load_feature_handoff(args.feature_dir, args.feature_receipt_sha, np, args.bundle_sha)
        require(np.__version__ == "2.1.3", "Qualified CPU NumPy2.1.3 required")
        output = Path(args.output_dir)
        require(not output.exists() and not output.is_symlink(), "Fresh CPU-head output directory required")
        output.mkdir(parents=True)
        probabilities = head.predict(values)
        require(probabilities.shape == (len(receipt["success_uids"]), 12) and np.isfinite(probabilities).all(),
                "Complete success-only head probabilities required")
        with (output / "head-success.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow([ID, *TARGETS])
            for uid, row in zip(receipt["success_uids"], probabilities):
                writer.writerow([uid, *[format(float(value), ".17g") for value in row]])
        write_json(output / "head-receipt.json", {"schema_version": "orthodiff_innerC_test_head_predictions_v1",
            "complete": True, "feature_receipt": pin(Path(args.feature_dir) / "feature-receipt.json"),
            "bundle_sha256": args.bundle_sha, "model_sha256": args.model_sha,
            "qualification_sha256": args.qualification_sha, "UID_order_sha256": receipt["UID_order_sha256"],
            "success_uids": receipt["success_uids"], "failure_uids": receipt["failure_uids"],
            "probabilities": pin(output / "head-success.csv"), "numpy": np.__version__,
            "parent_selected_or_overlaid": False, "Public_submission_contract_qualified": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
