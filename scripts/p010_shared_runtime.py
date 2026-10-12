"""Bounded transport and fixed rank blending for the unqualified p010 candidate.

This module never downloads, decodes, trains, uploads or submits anything.
Transport is a separately controlled execution change, not an accuracy claim.
"""

import csv
import hashlib
import io
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path


class PipelineFailure(RuntimeError):
    """Preparation or inference failed; no partial predictions may be submitted."""


class FusionObserver:
    """Transparent stdout provenance guard around the literal original fusion."""

    def __init__(self):
        self.original = sys.stdout
        self.buffer = io.StringIO()
        self.restored = False
        sys.stdout = self

    def write(self, value):
        self.buffer.write(value)
        return self.original.write(value)

    def flush(self):
        return self.original.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)

    def restore(self):
        if self.restored or sys.stdout is not self:
            raise PipelineFailure("Fusion stdout observer identity or lifetime changed")
        sys.stdout = self.original
        self.restored = True
        return self.buffer.getvalue()

    def require_success(self):
        value = self.restore()
        counts = {
            "success": value.count("[Apex Fusion SUCCESS] Successfully fused"),
            "exception": value.count("[Apex Fusion EXCEPTION]"),
            "fallback": value.count("[Apex Fusion FALLBACK]"),
        }
        if counts != {"success": 1, "exception": 0, "fallback": 0}:
            raise PipelineFailure(f"Original incumbent fusion did not finish exactly once: {counts}")
        return {
            "original_fusion_stdout_guard": counts,
            "original_fusion_success_verified": True,
            "counts_from_parent_inprocess_stdout_not_frontend_repeated_display": True,
        }


def run_bounded_pipeline(
    pool_factory,
    task,
    items,
    devices,
    consume,
    *,
    queue_max=48,
    outstanding_limit=48,
    shutdown_seconds=120.0,
    poll_seconds=0.05,
):
    """Deliver completed prep tasks without Pool.imap's unbounded result buffer.

    At most outstanding_limit task results are pending delivery, at most
    queue_max are queued for inference, and one is owned by each consumer.
    Callbacks never block the Pool result handler. A new task is submitted only
    after a previous result has been accepted by the inference queue.
    """
    if not devices or queue_max < 1 or outstanding_limit < 1 or shutdown_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("Positive capacities, shutdown time and devices are required")
    values = list(items)
    if len(values) != len(set(values)):
        raise ValueError("Task identities must be unique")
    delivery = queue.Queue(maxsize=outstanding_limit)
    gpu_queue = queue.Queue(maxsize=queue_max)
    cancel = threading.Event()
    lock = threading.Lock()
    first_error = []
    counters = {
        "submitted": 0,
        "delivered": 0,
        "consumed": 0,
        "discarded_after_cancel": 0,
        "max_outstanding": 0,
        "max_delivery_queue": 0,
        "max_inference_queue": 0,
    }

    def fail(origin, exception):
        with lock:
            if not first_error:
                first_error.append((origin, type(exception).__name__, str(exception)))
        cancel.set()

    def callback(value):
        try:
            delivery.put_nowait((True, value))
            with lock:
                counters["max_delivery_queue"] = max(counters["max_delivery_queue"], delivery.qsize())
        except BaseException as exception:
            fail("result_callback", exception)

    def error_callback(exception):
        fail("preparation_worker", exception)

    def consumer(device):
        while True:
            item = gpu_queue.get()
            try:
                if item is None:
                    return
                if cancel.is_set():
                    with lock:
                        counters["discarded_after_cancel"] += 1
                    continue
                try:
                    consume(device, item)
                    with lock:
                        counters["consumed"] += 1
                except BaseException as exception:
                    fail(f"inference_consumer:{device}", exception)
            finally:
                gpu_queue.task_done()
                del item

    threads = [threading.Thread(target=consumer, args=(device,), daemon=True) for device in devices]
    for thread in threads:
        thread.start()
    pool = None
    exceptional = None
    next_index = 0
    outstanding = 0
    try:
        pool = pool_factory()

        def submit_one():
            nonlocal next_index, outstanding
            value = values[next_index]
            pool.apply_async(task, (value,), callback=callback, error_callback=error_callback)
            next_index += 1
            outstanding += 1
            with lock:
                counters["submitted"] += 1
                counters["max_outstanding"] = max(counters["max_outstanding"], outstanding)

        while next_index < len(values) and outstanding < outstanding_limit:
            submit_one()
        while outstanding:
            if cancel.is_set():
                raise PipelineFailure("Worker or inference consumer failed")
            try:
                succeeded, payload = delivery.get(timeout=poll_seconds)
            except queue.Empty:
                continue
            assert succeeded
            try:
                accepted = False
                while not accepted:
                    if cancel.is_set():
                        raise PipelineFailure("Worker or inference consumer failed")
                    try:
                        gpu_queue.put(payload, timeout=poll_seconds)
                        accepted = True
                        with lock:
                            counters["max_inference_queue"] = max(counters["max_inference_queue"], gpu_queue.qsize())
                            counters["delivered"] += 1
                    except queue.Full:
                        continue
            finally:
                del payload
                delivery.task_done()
            outstanding -= 1
            if next_index < len(values):
                submit_one()
        pool.close()
        pool.join()
    except BaseException as exception:
        exceptional = exception
        fail("producer", exception)
        if pool is not None:
            pool.terminate()
            pool.join()
    finally:
        deadline = time.monotonic() + shutdown_seconds
        for _ in threads:
            while True:
                if time.monotonic() >= deadline:
                    raise PipelineFailure("Inference consumers did not drain before shutdown deadline")
                try:
                    gpu_queue.put(None, timeout=poll_seconds)
                    break
                except queue.Full:
                    continue
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in threads):
            raise PipelineFailure("Inference consumers did not stop before shutdown deadline")
        if gpu_queue.unfinished_tasks != 0:
            raise PipelineFailure("Inference queue task_done contract failed")
        while True:
            try:
                delivery.get_nowait()
                delivery.task_done()
            except queue.Empty:
                break
        if delivery.unfinished_tasks != 0:
            raise PipelineFailure("Delivery queue task_done contract failed")
    receipt = {
        "schema_version": "p010_bounded_transport_v1",
        "tasks": len(values),
        "queue_max": queue_max,
        "outstanding_limit": outstanding_limit,
        "consumer_count": len(devices),
        "prepared_parent_payload_upper_bound": queue_max + outstanding_limit + len(devices) + 1,
        "pool_result_handler_last_payload_reserve": 1,
        "bound_excludes_worker_and_IPC_serialisation_memory": True,
        "all_tasks_consumed": counters["consumed"] == len(values),
        "queue_unfinished_tasks": gpu_queue.unfinished_tasks,
        "delivery_unfinished_tasks": delivery.unfinished_tasks,
        "errors": len(first_error),
        "cancelled": cancel.is_set(),
        "consumer_threads_joined": True,
        "delivery": "bounded_apply_async_single_study_tasks",
        **counters,
    }
    if first_error:
        origin, kind, message = first_error[0]
        failure = PipelineFailure(f"{origin}: {kind}: {message}")
        failure.receipt = receipt
        raise failure from exceptional
    assert counters["submitted"] == counters["delivered"] == counters["consumed"] == len(values)
    assert counters["discarded_after_cancel"] == 0
    return receipt


def average_percentile(values):
    """Average ties, ranks 1..N divided by N, as pandas rank(pct=True)."""
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("Finite, nonempty probabilities are required")
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    left = 0
    while left < len(order):
        right = left + 1
        while right < len(order) and values[order[right]] == values[order[left]]:
            right += 1
        rank = ((left + 1) + right) / (2 * len(order))
        for position in order[left:right]:
            result[position] = rank
        left = right
    return result


def shared_input_receipt(volume, mask):
    """Hash exactly the shared native uint8 bytes; never cast, resize or decode."""
    import numpy as np

    if volume.shape != (96, 384, 384) or volume.dtype != np.uint8 or not volume.flags.c_contiguous:
        raise ValueError("Original contiguous 96x384x384 uint8 volume is required")
    if mask.shape != (96,) or mask.dtype != np.uint8 or not mask.flags.c_contiguous:
        raise ValueError("Original contiguous 96-entry uint8 mask is required")
    if not bool(np.isin(mask, [0, 1]).all()):
        raise ValueError("Mask must retain the original binary slot contract")
    return {
        "volume_sha256": hashlib.sha256(memoryview(volume).cast("B")).hexdigest(),
        "mask_sha256": hashlib.sha256(memoryview(mask).cast("B")).hexdigest(),
        "volume_shape": list(volume.shape),
        "volume_dtype": str(volume.dtype),
        "mask_shape": list(mask.shape),
        "mask_dtype": str(mask.dtype),
        "valid_slices": int(mask.sum()),
        "volume_bytes": int(volume.nbytes),
        "mask_bytes": int(mask.nbytes),
    }


def read_submission(path, labels, expected_uids=None):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["StudyInstanceUID", *labels] or len(labels) != 12:
            raise ValueError("Exactly twelve findings in the original label order are required")
        rows = list(reader)
        if any(set(row) != set(reader.fieldnames) or any(value is None for value in row.values()) for row in rows):
            raise ValueError("Every row must retain the exact thirteen-column schema")
    uids = [row["StudyInstanceUID"] for row in rows]
    if not uids or any(not uid for uid in uids) or len(set(uids)) != len(uids):
        raise ValueError("Nonempty unique study identities are required")
    if expected_uids is not None and uids != list(expected_uids):
        raise ValueError("Study order or identity changed")
    matrix = [[float(row[label]) for label in labels] for row in rows]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for row in matrix for value in row):
        raise ValueError("Finite probabilities in [0,1] are required")
    return uids, matrix


def write_submission_exclusive(path, uids, matrix, labels):
    path = Path(path)
    if len(matrix) != len(uids) or any(len(row) != 12 for row in matrix):
        raise ValueError("Submission matrix shape changed")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["StudyInstanceUID", *labels])
        for uid, row in zip(uids, matrix, strict=True):
            writer.writerow([uid, *[format(float(value), ".17g") for value in row]])
    read_submission(path, labels, uids)


def fixed_rank10(incumbent, expert):
    if len(incumbent) != len(expert) or not incumbent:
        raise ValueError("Branches must cover the same nonempty studies")
    if any(len(row) != 12 for matrix in (incumbent, expert) for row in matrix):
        raise ValueError("Both branches must retain all twelve original findings")
    columns = []
    for label in range(12):
        left = average_percentile([row[label] for row in incumbent])
        right = average_percentile([row[label] for row in expert])
        columns.append(average_percentile([0.9 * a + 0.1 * b for a, b in zip(left, right, strict=True)]))
    return [[column[index] for column in columns] for index in range(len(incumbent))]


def finalise_fixed_rank10(incumbent_path, expert_path, labels, uids, *, working=None):
    """Retain exact incumbent bytes and atomically replace a validated candidate."""
    incumbent_path, expert_path = Path(incumbent_path), Path(expert_path)
    root = Path(working) if working is not None else incumbent_path.parent
    if incumbent_path.resolve().parent != root.resolve() or expert_path.resolve().parent != root.resolve():
        raise ValueError("Both prediction files must belong to the exact working directory")
    _, incumbent = read_submission(incumbent_path, labels, uids)
    backup = root / "p010_incumbent.csv"
    stage = root / "p010_candidate.partial.csv"
    if backup.exists() or stage.exists():
        raise FileExistsError("Preserve earlier candidate or baseline evidence")
    payload = incumbent_path.read_bytes()
    with backup.open("xb") as handle:
        handle.write(payload)
    receipt = {
        "schema_version": "p010_fixed_rank10_v1",
        "studies": len(uids),
        "incumbent_sha256": hashlib.sha256(payload).hexdigest(),
        "coefficients_predeclared": [0.9, 0.1],
        "labels": list(labels),
        "rank_ties": "average",
        "normalisation": "rank_1_to_N_divided_by_N",
        "accuracy_improvement_measured": False,
    }
    if len(uids) < 2:
        receipt.update(
            {"expert_used": False, "reason": "fewer_than_two_studies", "final_sha256": receipt["incumbent_sha256"]}
        )
        return receipt
    _, expert = read_submission(expert_path, labels, uids)
    result = fixed_rank10(incumbent, expert)
    write_submission_exclusive(stage, uids, result, labels)
    read_submission(stage, labels, uids)
    os.replace(stage, incumbent_path)
    receipt.update(
        {
            "expert_used": True,
            "expert_sha256": hashlib.sha256(expert_path.read_bytes()).hexdigest(),
            "final_sha256": hashlib.sha256(incumbent_path.read_bytes()).hexdigest(),
        }
    )
    return receipt
