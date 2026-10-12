"""Append the frozen Meniscus10 branch after an unchanged Apex prediction.

Only the two Meniscus columns receive the prespecified 90:10 rank overlay.
The incumbent CSV is backed up before optional branch work. A branch failure
retains it byte-for-byte. This module never downloads, trains, launches or submits.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import shutil
import tempfile
import time
import traceback
from contextlib import ExitStack, contextmanager
from pathlib import Path

import public0033_streaming_adapter as adapter

MANIFEST_SHA256 = "b610229f046fe0f27ec8dcf661c5532d4991427428378975d0443da671607945"
CHECKPOINT_SHA256 = "e1443bb0f518418a31428cd7eb153f15af4c48c7175baf1f9d21c84f1c2a93e7"
BASELINE_SHA256 = "75c3abe77e8f0ccdaec2f57a8946d2baf297db8cc6fbdb275e5fcb1153b57546"


class ExpertInputContractError(ValueError):
    """Explicit unsupported live input; the valid Apex prediction may be retained."""


class ExpertSlotsUnavailable(ExpertInputContractError):
    def __init__(self, uid):
        self.uid_sha256 = hashlib.sha256(uid.encode()).hexdigest()
        super().__init__("no valid recovered Meniscus slot for a live study; UIDsha256=" + self.uid_sha256)


@contextmanager
def branch_observer():
    """Observe masks/full12 results without rewriting any existing helper/runtime."""
    from public0033_profile_runtime import stock_observer

    original_builder = adapter.make_dicom_study_builder
    original_load = adapter.load_pinned_runtime
    observed = {"study_masks": {}, "stock": []}

    def builder(*args, **kwargs):
        original = original_builder(*args, **kwargs)

        def build(uid):
            try:
                result = original(uid)
            except ValueError as error:
                if str(error).startswith("no valid Meniscus10 slot:"):
                    raise ExpertSlotsUnavailable(uid) from error
                raise
            except FileNotFoundError as error:
                raise ExpertInputContractError(
                    "live study has no DICOM directory; UIDsha256=" + hashlib.sha256(uid.encode()).hexdigest()
                ) from error
            if float(result[1].sum()) <= 0:
                raise ExpertSlotsUnavailable(uid)
            observed["study_masks"][uid] = result[1].tolist()
            return result

        return build

    with ExitStack() as stack:

        def load(root):
            module = original_load(root)
            observed["stock"].append(stack.enter_context(stock_observer(module)))
            return module

        adapter.make_dicom_study_builder, adapter.load_pinned_runtime = builder, load
        try:
            yield observed
        finally:
            adapter.make_dicom_study_builder, adapter.load_pinned_runtime = original_builder, original_load


def verify_actual_full12(observed, uids, raw_csv):
    """Require variable-N FP32 forward/worker completion and exact raw2 projection."""
    import numpy as np

    if set(observed["study_masks"]) != set(uids) or len(observed["stock"]) != 1:
        raise ValueError("not all live study masks/one stock call were observed")
    stock = observed["stock"][0]
    if len(stock["loads"]) != 2 or len(stock["workers"]) != 2 or set(stock["full12"]) != set(range(len(uids))):
        raise ValueError("complete actual two-replica/full12 stock outputs not observed")
    midpoint = (len(uids) + 1) // 2
    for index, size in enumerate((midpoint, len(uids) - midpoint)):
        forwards = [row for row in stock["forwards"] if row["device"] == f"cuda:{index}"]
        expected_batches = [min(32, size - offset) for offset in range(0, size, 32) for _ in range(10)]
        if [row["batch_studies"] for row in forwards] != expected_batches or any(
            not row["finite"] or row["logits_dtype"] != "torch.float32" for row in forwards
        ):
            raise ValueError("actual FP32/window10 variable-N forward coverage drift")
    values = np.stack([stock["full12"][index] for index in range(len(uids))])
    if (
        values.shape != (len(uids), 12)
        or values.dtype != np.float32
        or not np.isfinite(values).all()
        or np.any((values < 0) | (values > 1))
    ):
        raise ValueError("actual full12 worker probabilities failed")
    raw = adapter._read_csv(raw_csv, adapter.MENISCUS)
    ordered = sorted(uids)
    if [row["StudyInstanceUID"] for row in raw] != ordered or any(
        float(row[label]) != float(values[index, adapter.LABELS.index(label)])
        for index, row in enumerate(raw)
        for label in adapter.MENISCUS
    ):
        raise ValueError("raw2 CSV differs from actual full12 probabilities")
    return {
        "studies": len(uids),
        "columns": list(adapter.LABELS),
        "actual_forward_count": len(stock["forwards"]),
        "actual_loads": stock["loads"],
        "actual_workers": stock["workers"],
        "all_probabilities_finite_FP32_in_range": True,
        "raw2_exact_projection": True,
        "mask_UID_count": len(observed["study_masks"]),
        "valid_slots_histogram": {
            str(count): sum(sum(mask) == count for mask in observed["study_masks"].values()) for count in range(1, 7)
        },
        "mask_sha256": hashlib.sha256(
            np.asarray([observed["study_masks"][uid] for uid in ordered], dtype=np.float32).tobytes()
        ).hexdigest(),
    }


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def uid_sha256(uids):
    return hashlib.sha256("".join(f"{uid}\n" for uid in uids).encode()).hexdigest()


def gpu_memory():
    """Observe this process's allocator; retain any live parent model references."""
    import torch

    if not torch.cuda.is_available():
        return []
    return [
        {
            "device_index": index,
            "device_name": torch.cuda.get_device_name(index),
            "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            "allocated_bytes": torch.cuda.memory_allocated(index),
            "reserved_bytes": torch.cuda.memory_reserved(index),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(index),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(index),
        }
        for index in range(torch.cuda.device_count())
    ]


def release_inactive_cache():
    """Drop only unused allocator blocks after Apex output has been fixed on disk."""
    import torch

    before = gpu_memory()
    gc.collect()
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(index)
    return before, gpu_memory()


def find_bundle(input_root):
    root = Path(input_root)
    paths = (
        root / "rsna-knee-public0033-meniscus-bag-v1",
        root / "datasets/renta0426/rsna-knee-public0033-meniscus-bag-v1",
    )
    found = [path for path in paths if (path / "bundle_manifest.json").is_file()]
    if len(found) != 1 or sha256(found[0] / "bundle_manifest.json") != MANIFEST_SHA256:
        raise ValueError("exact pinned public0033 DatasetV2 mount/manifest required")
    return found[0]


def read_live_metadata(input_root, incumbent_uids):
    """Use the current test metadata; training files/Reports are never read."""
    root = Path(input_root)
    candidates = [
        path
        for path in (root / "competitions/rsna-knee-abnormality-detection", root / "rsna-knee-abnormality-detection")
        if (path / "test.csv").is_file() and (path / "test_series.csv").is_file()
    ]
    if len(candidates) != 1:
        raise ExpertInputContractError("exactly one live competition test metadata mount required")
    competition = candidates[0]
    with (competition / "test.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if "StudyInstanceUID" not in (reader.fieldnames or []):
            raise ExpertInputContractError("test.csv lacks required StudyInstanceUID column")
        test_uids = [row["StudyInstanceUID"] for row in reader]
    if (
        len(set(test_uids)) != len(test_uids)
        or set(test_uids) != set(incumbent_uids)
        or len(test_uids) != len(incumbent_uids)
    ):
        raise ExpertInputContractError("incumbent UID coverage differs from the live hidden/public test.csv")
    selected = set(test_uids)
    with (competition / "test_series.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane"} <= set(reader.fieldnames or []):
            raise ExpertInputContractError("test_series.csv lacks required UID/plane columns")
        rows = [
            {key: row[key] for key in ("StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane")}
            for row in reader
            if row["StudyInstanceUID"] in selected
        ]
    if {row["StudyInstanceUID"] for row in rows} != selected:
        raise ExpertInputContractError("live test series metadata does not cover all incumbent studies")
    trees = [competition / name for name in ("test_series", "test_images") if (competition / name).is_dir()]
    if len(trees) != 1:
        raise ExpertInputContractError("exactly one live test DICOM directory required")
    return rows, trees[0]


def atomic_publish(source, destination):
    """Replace only a fully validated CSV, through a new temporary sibling file."""
    destination = Path(destination)
    descriptor, name = tempfile.mkstemp(prefix=".p008_", suffix=".csv.tmp", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        shutil.copyfile(source, temporary)
        if sha256(temporary) != sha256(source):
            raise ValueError("publish staging copy differs from validated candidate")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_stock_outputs(output, result, uids):
    """Require this fresh run's complete pinned stock output before any fusion."""
    output = Path(output)
    raw = output / "public0033_bag_raw.csv"
    receipt_path = output / "public0033_cached_inference_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    rows = adapter._read_csv(raw, adapter.MENISCUS)
    ordered = sorted(uids)
    if [row["StudyInstanceUID"] for row in rows] != ordered:
        raise ValueError("fresh stock raw CSV UID coverage/order differs")
    if (
        result.get("status") != "passed"
        or receipt.get("status") != "passed"
        or receipt.get("fallback") != 0
        or receipt.get("neutralized_predictions") != 0
    ):
        raise ValueError("stock specialist did not complete without fallback")
    if (
        result.get("study_count") != len(uids)
        or receipt.get("study_count") != len(uids)
        or receipt.get("study_uid_sha256") != uid_sha256(ordered)
    ):
        raise ValueError("stock receipt live study count/UID hash differs")
    if (
        Path(result.get("bag_raw_csv", "")).resolve() != raw.resolve()
        or Path(result.get("receipt_json", "")).resolve() != receipt_path.resolve()
    ):
        raise ValueError("stock returned artifacts from a different/stale output directory")
    if (
        receipt.get("runtime_sha256") != adapter.PINNED_RUNTIME_SHA256
        or receipt.get("bundle_manifest_sha256") != MANIFEST_SHA256
    ):
        raise ValueError("stock source or manifest differs from audited DatasetV2")
    checkpoint = receipt.get("checkpoint", {})
    if (
        checkpoint.get("sha256") != CHECKPOINT_SHA256
        or checkpoint.get("bytes") != 347965736
        or checkpoint.get("state_tensor_count") != 233
        or not checkpoint.get("strict_load")
    ):
        raise ValueError("stock strict checkpoint identity drift")
    if (
        receipt.get("precision") != "fp32"
        or receipt.get("autocast") is not False
        or receipt.get("batch_studies") != 32
        or receipt.get("window_starts") != list(range(10))
    ):
        raise ValueError("stock FP32/batch32/window10 contract drift")
    if receipt.get("cache") != {"shape": [len(uids), 6, 12, 336, 336], "dtype": "uint8"}:
        raise ValueError("stock live cache shape/dtype drift")
    if [device.get("device_name") for device in receipt.get("devices", [])] != ["Tesla T4", "Tesla T4"]:
        raise ValueError("stock runtime devices are not both T4")
    if receipt.get("raw_output", {}).get("sha256") != sha256(raw):
        raise ValueError("stock raw CSV bytes do not match its receipt")
    partitions = sorted(receipt.get("partitions", []), key=lambda row: row["device_index"])
    midpoint = (len(uids) + 1) // 2
    if (
        len(partitions) != 2
        or [
            (row.get("device_index"), row.get("index_start"), row.get("index_stop_exclusive"), row.get("study_count"))
            for row in partitions
        ]
        != [(0, 0, midpoint, midpoint), (1, midpoint, len(uids), len(uids) - midpoint)]
        or any(not row.get("strict_load") for row in partitions)
    ):
        raise ValueError("stock complete contiguous partitions changed")
    return receipt


def verify_candidate(backup, candidate, uids):
    base = adapter._read_csv(backup, adapter.LABELS)
    proposed = adapter._read_csv(candidate, adapter.LABELS)
    if [row["StudyInstanceUID"] for row in proposed] != uids:
        raise ValueError("candidate row/UID order changed")
    others = [label for label in adapter.LABELS if label not in adapter.MENISCUS]
    if len(proposed) != len(base) or any(
        left[label] != right[label] for left, right in zip(base, proposed, strict=True) for label in others
    ):
        raise ValueError("non-Meniscus CSV value strings changed")
    return {
        "rows": len(proposed),
        "columns": ["StudyInstanceUID", *adapter.LABELS],
        "uid_order_preserved": True,
        "other10_value_strings_preserved": True,
        "finite_in_range": True,
    }


def append_specialist(
    *,
    submission="/kaggle/working/submission.csv",
    input_root="/kaggle/input",
    output_directory="/kaggle/working/p008_meniscus10",
    scratch="/kaggle/temp",
):
    """Run optional specialist once; known unsupported inputs retain the incumbent.

    The initial CSV contract is mandatory. A stale output directory is rejected,
    never used for predictions. N<2 explicitly skips the unchanged stock runtime.
    """
    submission = Path(submission)
    baseline_rows = adapter._read_csv(submission, adapter.LABELS)
    baseline_uids = [row["StudyInstanceUID"] for row in baseline_rows]
    baseline_sha = sha256(submission)
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError("stale branch output directory refused; Apex CSV unchanged")
    output.mkdir(parents=True, exist_ok=False)
    backup = output / "apex_incumbent.csv"
    with backup.open("xb") as stream:
        stream.write(submission.read_bytes())
    if sha256(backup) != baseline_sha:
        raise RuntimeError("incumbent backup bytes differ")
    started = time.monotonic()
    summary = {
        "schema_version": "p008_apex_fixed_meniscus10_v1",
        "status": "started",
        "source_p005_notebook_sha256": BASELINE_SHA256,
        "incumbent_sha256": baseline_sha,
        "studies": len(baseline_uids),
        "study_uid_sha256": uid_sha256(baseline_uids),
        "specialist_used": False,
        "fixed_branch_rank_weight": 0.10,
        "fixed_parent_rank_weight": 0.90,
        "targets": list(adapter.MENISCUS),
        "Report_or_targets_consumed": False,
        "accuracy_evaluated": False,
        "fold_safe_oof": False,
        "Public_improvement_verified": False,
        "scratch_pixel_cache_exported": False,
    }
    pending_error = None
    try:
        if len(baseline_uids) < 2:
            summary.update(
                status="skipped_incumbent_retained",
                reason="stock_requires_two_nonempty_GPU_partitions_N_at_least2",
                stock_call_count=0,
            )
        else:
            summary["stage"] = "live_test_input_contract"
            rows, tree = read_live_metadata(input_root, baseline_uids)
            summary["stage"] = "pinned_bundle_contract"
            bundle = find_bundle(input_root)
            # Apex has already finished; releasing its inactive allocator cache
            # changes no predictions. The new branch retains original FP32 math.
            memory_before, memory_after_cleanup = release_inactive_cache()
            summary.update(gpu_memory_before_cleanup=memory_before, gpu_memory_after_cleanup=memory_after_cleanup)
            summary["stage"] = "separate_stock_decode_and_inference"
            with branch_observer() as observed:
                result = adapter.run_separate_branch(
                    bundle_root=bundle,
                    series_rows=rows,
                    test_series_root=tree,
                    study_ids=baseline_uids,
                    output_directory=output / "branch",
                    scratch=scratch,
                )
            summary["gpu_memory_stock_branch_peaks"] = gpu_memory()
            summary["stage"] = "complete_stock_outputs_contract"
            stock_receipt = verify_stock_outputs(output / "branch", result, baseline_uids)
            full12_contract = verify_actual_full12(observed, baseline_uids, output / "branch/public0033_bag_raw.csv")
            if full12_contract["mask_sha256"] != stock_receipt["mask_sha256"]:
                raise ValueError("actual input masks differ from stock cache receipt")
            candidate = output / "apex_meniscus10_candidate.csv"
            summary["stage"] = "fixed_two_target_rank_overlay"
            overlay = adapter.fixed_meniscus_overlay(backup, output / "branch/public0033_bag_raw.csv", candidate)
            contract = verify_candidate(backup, candidate, baseline_uids)
            if sha256(submission) != baseline_sha:
                raise RuntimeError("incumbent was modified before specialist publication")
            atomic_publish(candidate, submission)
            if sha256(submission) != sha256(candidate):
                raise RuntimeError("published CSV differs from validated candidate")
            summary.update(
                status="passed",
                stage="candidate_published",
                specialist_used=True,
                stock_call_count=1,
                stock_receipt=stock_receipt,
                overlay_receipt=overlay,
                candidate_contract=contract,
                actual_full12_contract=full12_contract,
                candidate_sha256=sha256(candidate),
            )
    except Exception as error:
        # Explicit input unavailability can retain Apex. Unexpected code,
        # checkpoint or CUDA errors are recorded and re-raised, never disguised
        # as a successful/fallback specialist. All cases retain the original CSV.
        if sha256(submission) != baseline_sha:
            atomic_publish(backup, submission)
        if sha256(submission) != baseline_sha:
            raise RuntimeError("failed to restore the original incumbent") from error
        expected_input = isinstance(error, ExpertInputContractError)
        summary.update(
            status="fallback_incumbent_retained" if expected_input else "failed_incumbent_preserved",
            specialist_used=False,
            reason=type(error).__name__,
            message=str(error),
            exception_traceback=traceback.format_exc(),
        )
        if not expected_input:
            pending_error = error
        try:
            summary["gpu_memory_on_branch_failure"] = gpu_memory()
        except Exception as memory_error:
            summary["gpu_memory_observation_error"] = str(memory_error)
        print(f"[p008 specialist {summary['status']}] {type(error).__name__}: {error}; Apex CSV retained.", flush=True)
    summary.update(
        elapsed_seconds=time.monotonic() - started,
        submission_sha256=sha256(submission),
        incumbent_backup_sha256=sha256(backup),
    )
    try:
        adapter.save_receipt(output / "P008_MENISCUS_SUMMARY.json", summary)
    except Exception:
        if sha256(submission) != baseline_sha:
            atomic_publish(backup, submission)
        raise
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key not in ("stock_receipt", "overlay_receipt")},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if pending_error is not None:
        raise pending_error
    return summary
