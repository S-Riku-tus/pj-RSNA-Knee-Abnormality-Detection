"""Observe an unchanged public0033 run; invoked only by a private Kaggle profile.

No labels, Reports, downloads, training or remote operations are used here.
The released runtime and its inference batch/precision are not rewritten.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import threading
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import public0033_streaming_adapter as adapter

BUNDLE_MANIFEST_SHA256 = "b610229f046fe0f27ec8dcf661c5532d4991427428378975d0443da671607945"
DATASET_VERSION_ID = 19356288
PROFILE_COUNT = 64


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def uid_sha256(uids):
    return hashlib.sha256("".join(f"{uid}\n" for uid in uids).encode()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def validate_cohort(cohort):
    uids = cohort["uids"]
    if len(uids) != PROFILE_COUNT or len(set(uids)) != PROFILE_COUNT or uids != sorted(uids):
        raise ValueError("fixed sorted 64-study cohort required")
    for uid in uids:
        adapter._safe_child(Path("/unused"), uid)
    if uid_sha256(uids) != cohort["study_uid_sha256"]:
        raise ValueError("cohort UID hash mismatch")
    if cohort["gold_overlap_count"] != 0 or not cohort["gold_excluded"]:
        raise ValueError("Gold exclusion was not proved at preparation")
    if cohort["Report_or_targets_consumed"] or cohort["accuracy_evaluated"]:
        raise ValueError("profile must not consume targets/Reports or evaluate accuracy")
    return uids


def read_series(input_root, uids):
    """Read only series metadata; train.csv and its Report/targets are never opened."""
    root = Path(input_root)
    roots = (root / "competitions/rsna-knee-abnormality-detection", root / "rsna-knee-abnormality-detection")
    candidates = [path for path in roots if (path / "train_series.csv").is_file()]
    if len(candidates) != 1:
        raise ValueError("exactly one competition training metadata mount required")
    official = candidates[0]
    trees = [official / name for name in ("train_series", "train_images") if (official / name).is_dir()]
    if len(trees) != 1:
        raise ValueError("exactly one training DICOM directory required")
    selected = set(uids)
    rows = []
    with (official / "train_series.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["StudyInstanceUID"] in selected:
                rows.append({key: row[key] for key in ("StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane")})
    if {row["StudyInstanceUID"] for row in rows} != selected:
        raise ValueError("some fixed profile studies are missing from mounted series metadata")
    return rows, trees[0]


def find_bundle(input_root):
    """Search shallow known dataset mounts, without traversing any DICOM tree."""
    root = Path(input_root)
    candidates = (
        root / "rsna-knee-public0033-meniscus-bag-v1",
        root / "datasets/renta0426/rsna-knee-public0033-meniscus-bag-v1",
    )
    found = [path for path in candidates if (path / "bundle_manifest.json").is_file()]
    if len(found) != 1:
        raise ValueError("exactly one pinned renta public0033 Dataset mount required")
    bundle = found[0]
    if sha256(bundle / "bundle_manifest.json") != BUNDLE_MANIFEST_SHA256:
        raise ValueError("renta Dataset V2 manifest drift")
    return bundle


class Resources:
    """Sample RSS/device memory; exact PyTorch allocator peaks are captured separately."""

    def __init__(self, scratch):
        import psutil

        self.process = psutil.Process()
        self.scratch = Path(scratch)
        self.stop = threading.Event()
        self.peak_rss = 0
        self.peak_gpu_mib = []
        self.samples = 0
        self.gpu_samples = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self):
        import psutil

        rss = 0
        for process in [self.process, *self.process.children(recursive=True)]:
            try:
                rss += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        self.peak_rss = max(self.peak_rss, rss)
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
            )
            values = [int(value.strip()) for value in result.stdout.splitlines() if value.strip()]
            if len(values) == 2:
                self.peak_gpu_mib = (
                    values
                    if not self.peak_gpu_mib
                    else [max(a, b) for a, b in zip(self.peak_gpu_mib, values, strict=True)]
                )
                self.gpu_samples += 1
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        self.samples += 1

    def _loop(self):
        while not self.stop.wait(0.5):
            self._sample()

    def __enter__(self):
        self.started = time.monotonic()
        self._sample()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=5)
        self._sample()
        self.elapsed = time.monotonic() - self.started

    def result(self):
        return {
            "elapsed_seconds": self.elapsed,
            "sampled_peak_process_tree_rss_bytes": self.peak_rss,
            "sampled_gpu_peak_used_mib": self.peak_gpu_mib,
            "samples": self.samples,
            "gpu_samples": self.gpu_samples,
            "disk_free_bytes_after": shutil.disk_usage(self.scratch).free,
            "rss_note": "Sampled sum of RSS; shared pages/page cache may inflate it and short peaks may be missed.",
            "gpu_note": "Sampled device-wide nvidia-smi memory, including allocations outside this process.",
        }


@contextmanager
def decoder_observer():
    """Return original decoded pixels unchanged and preserve original error policy."""
    import numpy as np
    import pydicom

    original_read, original_slot = pydicom.dcmread, adapter.prepare_slot
    counts = Counter()
    syntaxes = Counter()
    native_shapes = Counter()
    statistics = {"pixel_decode_seconds": 0.0}

    def read(*args, **kwargs):
        kind = "header" if kwargs.get("stop_before_pixels", False) else "pixel_dataset"
        counts[f"{kind}_attempts"] += 1
        try:
            value = original_read(*args, **kwargs)
        except Exception:
            counts[f"{kind}_read_errors"] += 1
            raise
        if kind == "pixel_dataset":
            syntaxes[str(getattr(getattr(value, "file_meta", None), "TransferSyntaxUID", "missing"))] += 1
        return value

    def slot(file_count, read_pixels, pixel_spacing, *, plane, laterality=None):
        counts["slots_attempted"] += 1

        def pixels(index):
            started = time.monotonic()
            counts["selected_frame_decode_attempts"] += 1
            try:
                value = read_pixels(index)
                native_shapes["x".join(map(str, np.asarray(value).shape))] += 1
                if np.asarray(value).ndim != 2 or not np.isfinite(value).all():
                    counts["selected_frame_invalid"] += 1
                else:
                    counts["selected_frame_success"] += 1
                return value
            except Exception:
                counts["selected_frame_decode_errors"] += 1
                raise
            finally:
                statistics["pixel_decode_seconds"] += time.monotonic() - started

        value = original_slot(file_count, pixels, pixel_spacing, plane=plane, laterality=laterality)
        counts["slots_valid" if value is not None else "slots_invalid"] += 1
        return value

    pydicom.dcmread, adapter.prepare_slot = read, slot
    try:
        yield {
            "counts": counts,
            "transfer_syntax_counts": syntaxes,
            "selected_native_shape_counts": native_shapes,
            "timing": statistics,
        }
    finally:
        pydicom.dcmread, adapter.prepare_slot = original_read, original_slot


@contextmanager
def stock_observer(runtime):
    """Observe unchanged load, forward and worker return values; no prediction edits."""
    import numpy as np
    import torch

    original_load, original_worker = runtime._load_fullfit_model, runtime._worker_predict
    observed = {"loads": [], "forwards": [], "workers": [], "full12": {}}
    lock = threading.Lock()
    handles = []

    def load(root, device):
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        model = original_load(root, device)
        parameters = list(model.parameters())
        if len(model.state_dict()) != runtime.EXPECTED_STATE_TENSORS:
            raise ValueError("strict-loaded model state tensor count changed")
        if any(parameter.dtype != torch.float32 for parameter in parameters):
            raise ValueError("stock loaded parameters are not FP32")
        with lock:
            observed["loads"].append(
                {
                    "device": str(device),
                    "elapsed_seconds": time.monotonic() - started,
                    "state_tensor_count": len(model.state_dict()),
                    "parameters_fp32": True,
                    "stock_strict_load": True,
                }
            )

        def forward(module, inputs, output):
            images, masks = inputs
            expected = (len(images), 6, 3, 336, 336)
            if tuple(images.shape) != expected or tuple(masks.shape) != (len(images), 6):
                raise ValueError("actual stock forward input shape drift")
            if tuple(output.shape) != (len(images), 12) or output.dtype != torch.float32:
                raise ValueError("actual stock 12-logit FP32 output shape drift")
            finite = bool(torch.isfinite(output).all().item())
            if not finite:
                raise ValueError("actual stock logits contain nonfinite values")
            with lock:
                observed["forwards"].append(
                    {
                        "device": str(device),
                        "batch_studies": len(images),
                        "images_shape": list(images.shape),
                        "images_dtype": str(images.dtype),
                        "mask_shape": list(masks.shape),
                        "logits_shape": list(output.shape),
                        "logits_dtype": str(output.dtype),
                        "finite": finite,
                    }
                )

        handles.append(model.register_forward_hook(forward))
        return model

    def worker(**kwargs):
        index = kwargs["device_index"]
        device = torch.device(f"cuda:{index}")
        started = time.monotonic()
        result = original_worker(**kwargs)
        indices, values, receipt = result
        torch.cuda.synchronize(device)
        if values.shape != (len(indices), 12) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("actual stock worker 12-probability contract failed")
        with lock:
            observed["workers"].append(
                {
                    "device_index": index,
                    "indices": indices.tolist(),
                    "probability_shape": list(values.shape),
                    "finite": True,
                    "runtime_seconds_with_observer": time.monotonic() - started,
                    "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    "stock_receipt": receipt,
                }
            )
            for position, row in zip(indices.tolist(), values, strict=True):
                if position in observed["full12"]:
                    raise ValueError("stock worker returned a duplicate study index")
                observed["full12"][position] = row.copy()
        return result

    runtime._load_fullfit_model, runtime._worker_predict = load, worker
    try:
        yield observed
    finally:
        runtime._load_fullfit_model, runtime._worker_predict = original_load, original_worker
        for handle in handles:
            handle.remove()


def validate_observations(observed, uids):
    """Require both complete stock batch32 partitions and all ten windows per GPU."""
    if len(observed["loads"]) != 2 or len(observed["workers"]) != 2:
        raise ValueError("both checkpoint replicas must complete")
    if set(observed["full12"]) != set(range(PROFILE_COUNT)):
        raise ValueError("incomplete actual 12-column predictions")
    for device in ("cuda:0", "cuda:1"):
        forwards = [record for record in observed["forwards"] if record["device"] == device]
        if len(forwards) != 10 or any(record["batch_studies"] != 32 or not record["finite"] for record in forwards):
            raise ValueError("actual stock maximum batch32/ten-window execution not observed")
    workers = sorted(observed["workers"], key=lambda record: record["device_index"])
    if [record["indices"] for record in workers] != [list(range(32)), list(range(32, 64))]:
        raise ValueError("stock contiguous GPU partitions changed")
    if len(uids) != PROFILE_COUNT:
        raise ValueError("fixed cohort changed during inference")


def write_full12(path, uids, observed):
    """Diagnostic CSV; it is intentionally never named submission.csv."""
    import numpy as np

    values = np.stack([observed["full12"][position] for position in range(len(uids))])
    if values.shape != (PROFILE_COUNT, 12) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("full12 diagnostic contract failed")
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["StudyInstanceUID", *adapter.LABELS])
        for uid, row in zip(uids, values, strict=True):
            writer.writerow([uid, *[repr(float(value)) for value in row]])
    rows = adapter._read_csv(path, adapter.LABELS)
    fields = ["StudyInstanceUID", *adapter.LABELS]
    if [row["StudyInstanceUID"] for row in rows] != uids:
        raise ValueError("diagnostic UID/12-column CSV roundtrip failed")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "rows": len(rows),
        "columns": fields,
        "finite_in_range": True,
        "minimum_by_target": dict(zip(adapter.LABELS, values.min(axis=0).tolist(), strict=True)),
        "maximum_by_target": dict(zip(adapter.LABELS, values.max(axis=0).tolist(), strict=True)),
        "unique_values_by_target": {
            label: len(set(values[:, index].tolist())) for index, label in enumerate(adapter.LABELS)
        },
    }


def environment():
    import torch

    packages = {}
    for name in (
        "torch",
        "numpy",
        "pydicom",
        "transformers",
        "safetensors",
        "psutil",
        "pylibjpeg",
        "pylibjpeg-libjpeg",
        "pylibjpeg-openjpeg",
        "python-gdcm",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda_runtime": torch.version.cuda,
        "devices": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ],
    }


def run_profile(
    cohort,
    *,
    input_root="/kaggle/input",
    output_directory="/kaggle/working/public0033_profile64",
    scratch="/kaggle/temp",
):
    """The Notebook's single entry point; final passed summary is written last."""
    import numpy as np

    uids = validate_cohort(cohort)
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError("a fresh profile output directory is required")
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=False)
    events = output / "PUBLIC0033_PROFILE_EVENTS.jsonl"

    def event(stage, **values):
        with events.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps({"stage": stage, "elapsed_seconds": time.monotonic() - started, **values}, allow_nan=False)
                + "\n"
            )
        print("[public0033 profile]", stage, values, flush=True)

    started = time.monotonic()
    try:
        event("preflight")
        bundle = find_bundle(input_root)
        runtime = adapter.load_pinned_runtime(bundle)
        manifest = runtime.verify_bundle(bundle)
        devices = runtime._require_t4x2()
        if (
            tuple(runtime.TARGETS) != adapter.LABELS
            or runtime.BATCH_STUDIES != 32
            or tuple(runtime.WINDOW_STARTS) != tuple(range(10))
        ):
            raise ValueError("stock runtime 12-target/batch32/ten-window drift")
        file_hashes = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(bundle.iterdir())
            if path.is_file()
        }
        rows, tree = read_series(input_root, uids)
        build_study = adapter.make_dicom_study_builder(rows, tree)
        mask_records = []
        decode_started = time.monotonic()

        def build(uid):
            before = time.monotonic()
            pixels, mask = build_study(uid)
            mask_records.append(
                {
                    "uid": uid,
                    "mask": mask.tolist(),
                    "cache_shape": list(pixels.shape),
                    "cache_dtype": str(pixels.dtype),
                    "elapsed_seconds": time.monotonic() - before,
                }
            )
            event(
                "study_decoded",
                index=len(mask_records),
                valid_slots=int(mask.sum()),
                seconds=mask_records[-1]["elapsed_seconds"],
            )
            return pixels, mask

        with (
            Resources(scratch) as resources,
            decoder_observer() as decoder,
            adapter.memmap_cache(uids, build, scratch=scratch) as (cache, mask, ordered),
        ):
            decode_seconds = time.monotonic() - decode_started
            if ordered != uids or not isinstance(cache, np.memmap):
                raise ValueError("memmap UID/cache contract drift")
            runtime._validate_cache_inputs(cache, mask, ordered)
            if len(mask_records) != PROFILE_COUNT or any(sum(record["mask"]) <= 0 for record in mask_records):
                raise ValueError("a profile study has no real valid slot")
            cache_record = {
                "shape": list(cache.shape),
                "dtype": str(cache.dtype),
                "data_bytes": cache.nbytes,
                "file_bytes": Path(cache.filename).stat().st_size,
                "file_sha256": sha256(cache.filename),
                "mask_sha256": hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest(),
                "valid_slots_per_uid": mask_records,
                "disk_backed": True,
                "maximum_anonymous_cohort_studies_during_decode": 1,
            }
            event("stock_call_started", devices=devices, studies=len(uids), batch_studies=32)
            inference_started = time.monotonic()
            with stock_observer(runtime) as observed:
                result = runtime.run_cached_inference(
                    cache,
                    mask,
                    ordered,
                    {
                        "bag_raw_csv": output / "public0033_bag_raw.csv",
                        "receipt_json": output / "public0033_cached_inference_receipt.json",
                        "work_dir": output,
                    },
                )
            inference_seconds = time.monotonic() - inference_started
            validate_observations(observed, uids)
        full12 = write_full12(output / "diagnostic_public0033_full12.csv", uids, observed)
        raw_rows = adapter._read_csv(output / "public0033_bag_raw.csv", adapter.MENISCUS)
        raw_fields = ["StudyInstanceUID", *adapter.MENISCUS]
        if [row["StudyInstanceUID"] for row in raw_rows] != uids:
            raise ValueError("stock raw CSV UID order drift")
        for position, row in enumerate(raw_rows):
            for target in adapter.MENISCUS:
                if float(row[target]) != float(observed["full12"][position][adapter.LABELS.index(target)]):
                    raise ValueError("stock raw Meniscus output differs from actual 12-column worker values")
        receipt = json.loads((output / "public0033_cached_inference_receipt.json").read_text(encoding="utf-8"))
        if (
            result["status"] != "passed"
            or receipt["status"] != "passed"
            or receipt["fallback"] != 0
            or receipt["neutralized_predictions"] != 0
        ):
            raise ValueError("stock receipt failed/fallback predictions detected")
        if (
            receipt["study_uid_sha256"] != uid_sha256(uids)
            or receipt["mask_sha256"] != cache_record["mask_sha256"]
            or not receipt["checkpoint"]["strict_load"]
        ):
            raise ValueError("stock receipt UID/mask/strict-load contract drift")
        write_json(output / "PUBLIC0033_PROFILE_COHORT.json", cohort)
        report = {
            "schema_version": "public0033_single_branch_profile64_v1",
            "status": "passed",
            "studies": PROFILE_COUNT,
            "gold_excluded": True,
            "Report_or_targets_consumed": False,
            "accuracy_evaluated": False,
            "fold_safe_oof": False,
            "fusion_applied": False,
            "submission_created": False,
            "stock_call_count": 1,
            "dataset": {
                "ref": "renta0426/rsna-knee-public0033-meniscus-bag-v1",
                "version": 2,
                "internal_version_id_expected": DATASET_VERSION_ID,
                "mounted_payload_hashes_verified": True,
                "control_plane_version_id_observed_in_runtime": False,
            },
            "stock_source_unchanged": True,
            "stock_precision": "fp32",
            "stock_batch_studies": 32,
            "stock_windows": list(range(10)),
            "study_uid_sha256": uid_sha256(uids),
            "weak_manifest_sha256": cohort["weak_manifest_sha256"],
            "gold_manifest_sha256": cohort["gold_manifest_sha256"],
            "environment": environment(),
            "bundle_files": file_hashes,
            "model_provenance": {
                "checkpoint": manifest["checkpoint"],
                "config_and_provenance": manifest["artifacts"],
                "runtime": manifest["runtime"],
            },
            "cache": cache_record,
            "decoder": {key: dict(value) for key, value in decoder.items()},
            "timing": {
                "decode_and_cache_seconds": decode_seconds,
                "stock_inference_with_observer_seconds": inference_seconds,
                "total_seconds_to_contract_complete": time.monotonic() - started,
                "note": "Includes observation synchronization and repeated bundle/cache hash I/O; not hidden-test runtime.",
            },
            "resources": resources.result(),
            "actual_loads": observed["loads"],
            "actual_forwards": observed["forwards"],
            "actual_workers": observed["workers"],
            "full12_contract": full12,
            "stock_raw_contract": {
                "columns": raw_fields,
                "rows": len(raw_rows),
                "sha256": sha256(output / "public0033_bag_raw.csv"),
            },
            "stock_receipt": receipt,
            "helper_sha256": {path.name: sha256(path) for path in (Path(__file__), Path(adapter.__file__))},
            "limits": [
                "Fixed64 weak-fullfit studies, not OOF or accuracy evaluation.",
                "Healthy reference preprocessing reproduced; selected-slot decoder errors retain stock cache policy and are counted.",
                "Batch32 is measured, but a larger/heterogeneous hidden cohort and repeated batches remain untested.",
                "Distinct representation does not establish improvement over own Public0.950.",
            ],
        }
        if (
            (scratch / "submission.csv").exists()
            or (output / "submission.csv").exists()
            or (output.parent / "submission.csv").exists()
        ):
            raise ValueError("diagnostic must not generate submission.csv")
        event("contracts_passed", actual_forwards=len(observed["forwards"]), raw_rows=len(raw_rows))
        # A failed or partially completed run has no final summary claiming passed.
        write_json(output / "PUBLIC0033_PROFILE_SUMMARY.json", report)
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in ("status", "studies", "stock_batch_studies", "timing", "resources", "submission_created")
                },
                indent=2,
            ),
            flush=True,
        )
        return report
    except Exception as error:
        event("failed", exception_type=type(error).__name__, message=str(error))
        raise
