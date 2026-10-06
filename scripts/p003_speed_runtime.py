"""One-factor Raptor scheduling experiment; no data access at module import."""

import functools
import gc
import hashlib
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from p003_guard import rank_pct, save_json


def tensor_sha256(value):
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def compare_arrays(reference, candidate):
    import numpy as np

    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if reference.shape != candidate.shape or reference.ndim != 2 or not len(reference):
        raise ValueError("Comparison requires equal, nonempty study-by-label matrices")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Nonfinite reference or candidate predictions")
    return {
        "exact_equal": bool(np.array_equal(reference, candidate)),
        "max_absolute_difference": float(np.max(np.abs(reference.astype(float) - candidate))),
        "rank_changed_values": int(np.count_nonzero(rank_pct(reference) != rank_pct(candidate))),
    }


def paired_prefetch(ids, prepare):
    """At most current + one future study pair; emit the original UID order."""
    ids = list(ids)
    if not ids:
        return
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(prepare, ids[0])
        for index, uid in enumerate(ids):
            current = future
            if index + 1 < len(ids):
                future = executor.submit(prepare, ids[index + 1])
            yield index, uid, current


class RaptorSchedule:
    def __init__(self, work, profile):
        self.work = Path(work)
        self.profile = bool(profile)
        self.lock = threading.Lock()
        self.metrics = {}
        self.input_hashes = {}
        self.report = {"status": "started", "single_factor": "native_raptor_study_interleaving"}

    def add(self, name, seconds):
        with self.lock:
            item = self.metrics.setdefault(name, {"calls": 0, "seconds": 0.0})
            item["calls"] += 1
            item["seconds"] += seconds

    def install(self, namespace):
        self.original_prepare = namespace["_ke_prepare_windows"]
        self.original_reader = namespace["_ke_base_make_reader"]
        original_cached_reader = namespace["_KE_NS"]["_make_reader"]
        original_infer = namespace["_KE_NS"]["infer_probs"]

        @functools.wraps(self.original_prepare)
        def prepare(arm, uid, *args):
            started = time.monotonic()
            try:
                value = self.original_prepare(arm, uid, *args)
                if self.profile and arm["name"] in ("native384dense-v10", "native384-v8"):
                    self.input_hashes[(arm["name"], str(uid))] = tensor_sha256(value)
                return value
            finally:
                self.add("prepare/" + arm["name"], time.monotonic() - started)

        def reader_factory():
            order, pixel, resize = original_cached_reader()

            def timed_pixel(path):
                started = time.monotonic()
                try:
                    return pixel(path)
                finally:
                    self.add("cached_pixel_requests", time.monotonic() - started)

            return order, timed_pixel, resize

        def raw_reader_factory():
            order, pixel, resize = self.original_reader()

            def timed_pixel(path):
                started = time.monotonic()
                try:
                    return pixel(path)
                finally:
                    self.add("actual_pixel_decodes", time.monotonic() - started)

            return order, timed_pixel, resize

        def infer(model, windows, device):
            started = time.monotonic()
            try:
                return original_infer(model, windows, device)
            finally:
                # Original infer returns a CPU NumPy array, synchronizing its CUDA work.
                self.add("forward/" + str(device), time.monotonic() - started)

        namespace["_ke_prepare_windows"] = prepare
        namespace["_ke_base_make_reader"] = raw_reader_factory
        namespace["_KE_NS"]["_make_reader"] = reader_factory
        namespace["_KE_NS"]["infer_probs"] = infer

    def run_native_pair(self, namespace, arms, ids, series, series_root, outputs, device):
        """Keep both unchanged native models resident and visit each study once."""
        import numpy as np

        ns = namespace["_KE_NS"]
        torch = ns["torch"]
        started = time.monotonic()
        models = {}
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        try:
            for index in (1, 3):
                arm = arms[index]
                loaded = time.monotonic()
                model, resolution = ns["load_model"](
                    ns["find_weight_file"](arm["file"]), arm["arch"], arm["res"], device
                )
                if int(resolution) != int(arm["res"]):
                    raise RuntimeError("Native Raptor checkpoint resolution changed")
                models[index] = model
                self.add("model_load/" + arm["name"], time.monotonic() - loaded)
            reader = ns["_make_reader"]()

            def prepare_pair(uid):
                return {
                    index: namespace["_ke_prepare_windows"](arms[index], uid, series, series_root, reader)
                    for index in (1, 3)
                }

            for study_index, uid, future in paired_prefetch(ids, prepare_pair):
                namespace["rsna_deadline"]("Raptor interleaved native inference")
                waited = time.monotonic()
                windows = future.result()
                self.add("native_wait_for_preparation", time.monotonic() - waited)
                for index in (1, 3):
                    value = ns["infer_probs"](models[index], windows[index], device)
                    if not np.isfinite(value).all():
                        raise RuntimeError(f"Nonfinite native Raptor output: {uid}")
                    outputs[index][study_index] = value
                del windows
            self.report.update(
                native_candidate_seconds=time.monotonic() - started,
                native_candidate_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
                studies=len(ids),
            )
        finally:
            # The local loop variable also owns the last model.
            if "model" in locals():
                del model
            models.clear()
            gc.collect()
            torch.cuda.empty_cache()
            self.flush()
        if self.profile:
            self.run_reference(namespace, arms, ids, series, series_root, outputs, device)

    def run_reference(self, namespace, arms, ids, series, series_root, outputs, device):
        """Full original native-arm order, uncached input reads, isolated audit IDs."""
        import numpy as np

        ns = namespace["_KE_NS"]
        local = dict(self.original_prepare.__globals__)
        local["_ke_input_ids"] = set()
        local["_ke_input_lock"] = threading.Lock()
        local["_ke_input_audit_path"] = self.work / "raptor_reference_inputs.jsonl"
        prepare = types.FunctionType(self.original_prepare.__code__, local, self.original_prepare.__name__)
        reference = []
        comparisons = []
        started = time.monotonic()
        for index in (1, 3):
            arm = arms[index]
            reader = self.original_reader()
            model, resolution = ns["load_model"](ns["find_weight_file"](arm["file"]), arm["arch"], arm["res"], device)
            if int(resolution) != int(arm["res"]):
                raise RuntimeError("Reference checkpoint resolution changed")
            values = np.empty_like(outputs[index])
            input_equal = True
            try:
                # Each arm processes the complete cohort, as in original gpu_one.
                for study_index, uid in enumerate(ids):
                    namespace["rsna_deadline"]("Raptor scheduling parity reference")
                    windows = prepare(arm, uid, series, series_root, reader)
                    input_equal &= tensor_sha256(windows) == self.input_hashes[(arm["name"], str(uid))]
                    values[study_index] = namespace["_ke_infer_input"](model, windows, device)
                    del windows
            finally:
                del model
                gc.collect()
                ns["torch"].cuda.empty_cache()
            item = compare_arrays(values, outputs[index])
            item.update(arm=arm["name"], all_input_tensor_hashes_equal=bool(input_equal))
            comparisons.append(item)
            reference.append(values)
        np.savez_compressed(
            self.work / "raptor_schedule_reference_raw.npz",
            study_uids=np.asarray(ids),
            labels=np.asarray(ns["LAB"]),
            arm_indices=np.asarray([1, 3]),
            raw_probabilities=np.stack(reference),
        )
        self.report.update(
            reference_seconds=time.monotonic() - started,
            reference_timing_is_unprefetched=True,
            comparisons=comparisons,
            parity_passed=all(item["exact_equal"] and item["all_input_tensor_hashes_equal"] for item in comparisons),
        )
        self.flush()
        if not self.report["parity_passed"]:
            raise RuntimeError("Raptor scheduling/input parity failed; candidate must not be adopted")

    def flush(self):
        with self.lock:
            report = {**self.report, "metrics": {name: dict(value) for name, value in self.metrics.items()}}
        report["timing_note"] = "CPU preparation overlaps GPU; metric seconds must not be summed as wall time."
        report["speedup_verified"] = False
        save_json(self.work / "P003_SPEED_RAPTOR.json", report)
