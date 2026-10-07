"""Observations for a larger, label-free P003 cohort; no prediction changes."""

import functools
import hashlib
import json
import shutil
import threading
import time
import traceback
from pathlib import Path

from p003_guard import read_json, save_json, sha256
from p003_speed_guard import P003SpeedProfileGuard


def array_fingerprint(value):
    """Hash contiguous uint8 pixels in bounded views, without a full-size copy."""
    import numpy as np

    if value.dtype != np.uint8 or not value.flags.c_contiguous or not value.size:
        raise ValueError("Expected nonempty contiguous uint8 pixel cache")
    view = memoryview(value).cast("B")
    digest = hashlib.sha256()
    for start in range(0, len(view), 8 << 20):
        digest.update(view[start : start + (8 << 20)])
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "bytes": int(value.nbytes),
        "storage": "memmap" if isinstance(value, np.memmap) else "ram",
        "minimum": int(value.min()),
        "maximum": int(value.max()),
        "sha256": digest.hexdigest(),
    }


def process_rss():
    """Linux parent RSS/HWM; child-wide pressure is separately observed in cgroup."""
    result = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in ("VmRSS", "VmHWM"):
                result[key + "_bytes"] = int(value.split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return result


def read_cgroup(root=Path("/sys/fs/cgroup")):
    """Read v2 or v1 counters; unavailable counters are not model failures."""
    root = Path(root)
    alternatives = [
        (
            "v2",
            root,
            {
                "memory.current": "memory.current",
                "memory.max": "memory.max",
                "memory.peak": "memory.peak",
                "memory.events": "memory.events",
            },
        ),
        (
            "v1",
            root / "memory",
            {
                "memory.current": "memory.usage_in_bytes",
                "memory.max": "memory.limit_in_bytes",
                "memory.peak": "memory.max_usage_in_bytes",
                "memory.failcnt": "memory.failcnt",
            },
        ),
        (
            "v1",
            root,
            {
                "memory.current": "memory.usage_in_bytes",
                "memory.max": "memory.limit_in_bytes",
                "memory.peak": "memory.max_usage_in_bytes",
                "memory.failcnt": "memory.failcnt",
            },
        ),
    ]
    for version, directory, names in alternatives:
        result = {"version": version, "available": False, "sources": {}}
        for name, filename in names.items():
            try:
                text = (directory / filename).read_text().strip()
                if name == "memory.events":
                    result[name] = {key: int(value) for key, value in (line.split() for line in text.splitlines())}
                else:
                    result[name] = int(text) if text != "max" else text
                result["sources"][name] = str(directory / filename)
            except (OSError, ValueError):
                pass
        if isinstance(result.get("memory.current"), int):
            result["available"] = True
            return result
    return {"version": "unavailable", "available": False, "sources": {}}


def filesystem_kind(path):
    """Longest matching mount point, so a /tmp tmpfs assumption can be checked."""
    path = str(Path(path).resolve())
    candidates = []
    try:
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            left, right = line.split(" - ", 1)
            fields = left.split()
            mount = fields[4].replace("\\040", " ")
            if path == mount or path.startswith(mount.rstrip("/") + "/"):
                candidates.append((len(mount), mount, right.split()[0]))
    except (OSError, ValueError, IndexError):
        pass
    if not candidates:
        return None
    _, mount, kind = max(candidates)
    return {"mount": mount, "filesystem": kind}


def event_category(kind):
    """Describe existing source events without changing their accept/reject policy."""
    if kind in {"empty_study_rows", "a5_empty_study", "a5_no_metadata"}:
        return "layout_absence_observed_check_other_errors"
    if kind.endswith("fp16_nonfinite_retry_fp32"):
        return "numeric_precision_retry"
    if kind in {"raptor_acquisition_empty", "a5_slot_no_files", "slot_no_files"}:
        return "selected_source_missing"
    if any(word in kind for word in ("decode", "crop", "slot_unfilled", "slot_read_error", "shape_mismatch")):
        return "decoding_or_pixel_preparation"
    if any(word in kind for word in ("order", "geometry")):
        return "ordering_or_geometry"
    if any(word in kind for word in ("child_failed", "study_failed", "nonfinite", "fallback_studies")):
        return "model_or_required_component_failure"
    if kind.startswith("coat_") and any(word in kind for word in ("unavailable", "partial", "rejected")):
        return "recipe_degraded"
    return "other_source_event"


class ResourceRecorder:
    def __init__(self, work, interval=2.0):
        self.work = Path(work)
        self.interval = interval
        self.started = time.monotonic()
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.count = 0
        self.peak_parent_rss = 0
        self.peak_sampled_cgroup = 0
        self.disk_first, self.disk_last, self.disk_min_free = {}, {}, {}
        self.resource_first, self.resource_last = None, None
        self.allocations, self.cache_fingerprints, self.children = [], [], []
        self.errors = []
        self.source_event_categories = {}
        self.cgroup_observed = False
        self.parent_rss_observed = False

    def emit(self, kind, **fields):
        with self.lock:
            event = {"kind": kind, "elapsed_seconds": time.monotonic() - self.started, **fields}
            try:
                with (self.work / "P003_STRESS_EVENTS.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, allow_nan=False, default=str) + "\n")
                self.count += 1
            except (OSError, TypeError, ValueError) as exc:
                self.errors.append(f"Observation write: {type(exc).__name__}: {exc}")

    def sample(self, tag):
        result = {"tag": tag, **process_rss()}
        cgroup = read_cgroup()
        result["cgroup"] = cgroup
        disks = {}
        for location in (str(self.work), "/kaggle/temp", "/tmp"):
            path = Path(location)
            if path.is_dir():
                try:
                    usage = shutil.disk_usage(path)
                    disks[location] = {"total_bytes": usage.total, "free_bytes": usage.free}
                except OSError as exc:
                    self.errors.append(f"Disk observation: {type(exc).__name__}: {exc}")
        result["disk"] = disks
        with self.lock:
            if self.resource_first is None:
                self.resource_first = result
            self.resource_last = result
            self.cgroup_observed |= cgroup["available"]
            self.parent_rss_observed |= "VmHWM_bytes" in result
            self.peak_parent_rss = max(self.peak_parent_rss, result.get("VmHWM_bytes", 0))
            current = cgroup.get("memory.current", 0)
            if isinstance(current, int):
                self.peak_sampled_cgroup = max(self.peak_sampled_cgroup, current)
            for location, usage in disks.items():
                self.disk_first.setdefault(location, usage)
                self.disk_last[location] = usage
                self.disk_min_free[location] = min(
                    self.disk_min_free.get(location, usage["free_bytes"]), usage["free_bytes"]
                )
        self.emit("resource", **result)
        return result

    def start(self):
        self.sample("initial")
        self.emit("filesystem", locations={p: filesystem_kind(p) for p in (str(self.work), "/kaggle/temp", "/tmp")})

        def observe():
            while not self.stop_event.wait(self.interval):
                try:
                    self.sample("periodic")
                except Exception as exc:
                    self.errors.append(f"{type(exc).__name__}: {exc}")

        self.thread = threading.Thread(target=observe, name="p003-resource-observer", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=10)
        self.sample("final")

    def summary(self):
        with self.lock:
            return {
                "event_count": self.count,
                "sample_interval_seconds": self.interval,
                "parent_rss_observation_available": self.parent_rss_observed,
                "cgroup_memory_observation_available": self.cgroup_observed,
                "parent_peak_rss_bytes": self.peak_parent_rss if self.parent_rss_observed else None,
                "sampled_cgroup_memory_current_peak_bytes": self.peak_sampled_cgroup if self.cgroup_observed else None,
                "resource_initial": self.resource_first,
                "resource_final": self.resource_last,
                "cgroup_peak_is_sampled_except_separate_kernel_memory_peak": True,
                "cgroup_includes_child_processes_parent_rss_does_not": True,
                "disk_initial": dict(self.disk_first),
                "disk_final": dict(self.disk_last),
                "disk_min_free_sampled_bytes": dict(self.disk_min_free),
                "cache_allocations": list(self.allocations),
                "pixel_cache_fingerprints": list(self.cache_fingerprints),
                "child_processes": list(self.children),
                "observer_errors": list(self.errors),
                "source_event_category_counts": dict(self.source_event_categories),
                "gpu_memory_evidence": "Unchanged phase events and per-family worker receipts; parent CUDA stats omit child allocations.",
            }

    def wrap_child(self, original):
        @functools.wraps(original)
        def child(command, environment, logfile):
            started = time.monotonic()
            self.sample("child-start")
            item = {"logfile": str(logfile), "status": "started"}
            self.emit("child_start", **item)
            try:
                result = original(command, environment, logfile)
                item.update(status="complete", returncode=int(result))
                return result
            except BaseException as exc:
                item.update(status="failed", error=f"{type(exc).__name__}: {str(exc)[:5000]}")
                raise
            finally:
                item["seconds"] = time.monotonic() - started
                with self.lock:
                    self.children.append(item)
                self.emit("child_finish", **item)
                self.sample("child-finish")

        return child

    def wrap_array(self, original):
        @functools.wraps(original)
        def array(shape, tag):
            self.sample("cache-allocation-start:" + str(tag))
            value = original(shape, tag)
            import numpy as np

            item = {
                "tag": str(tag),
                "shape": list(value.shape),
                "bytes": int(value.nbytes),
                "storage": "memmap" if isinstance(value, np.memmap) else "ram",
            }
            if isinstance(value, np.memmap):
                item["filename"] = str(value.filename)
            with self.lock:
                self.allocations.append(item)
            self.emit("cache_allocation", **item)
            self.sample("cache-allocation-finish:" + str(tag))
            return value

        return array

    def wrap_cache(self, original):
        @functools.wraps(original)
        def cache(*args, **kwargs):
            value = original(*args, **kwargs)
            ids, pixels, masks = value
            item = {
                "tag": str(kwargs.get("tag", args[3] if len(args) > 3 else "unknown")),
                "study_count": len(ids),
                "mask_shape": list(masks.shape),
            }
            try:
                item.update(array_fingerprint(pixels), fingerprint_available=True)
            except Exception as exc:
                item.update(storage="unavailable", fingerprint_available=False, error=f"{type(exc).__name__}: {exc}")
                self.errors.append("Cache fingerprint observation: " + item["error"])
            with self.lock:
                self.cache_fingerprints.append(item)
            self.emit("cache_fingerprint", **item)
            self.sample("cache-prepared:" + item["tag"])
            return value

        return cache

    def wrap_event(self, original):
        @functools.wraps(original)
        def event(kind, **details):
            result = original(kind, **details)
            category = event_category(kind)
            with self.lock:
                self.source_event_categories[category] = self.source_event_categories.get(category, 0) + 1
            self.emit("source_event", source_kind=kind, category=category, details=details)
            return result

        return event


class P003StressProfileGuard(P003SpeedProfileGuard):
    def prepare(self):
        super().prepare()
        self.recorder = ResourceRecorder(self.work)
        self.recorder.start()

    def run_cell(self, number, source, namespace):
        try:
            self.recorder.sample(f"cell-{number}-start")
            super().run_cell(number, source, namespace)
            if number == 2:
                namespace["_run_required_child"] = self.recorder.wrap_child(namespace["_run_required_child"])
            if number == 4:
                namespace["rsna_array"] = self.recorder.wrap_array(namespace["rsna_array"])
                namespace["rsna_event"] = self.recorder.wrap_event(namespace["rsna_event"])
            if number == 11:
                namespace["build_cache"] = self.recorder.wrap_cache(namespace["build_cache"])
            self.recorder.sample(f"cell-{number}-complete")
        except BaseException:
            self.recorder.stop()
            save_json(
                self.work / "P003_STRESS_FAILURE.json",
                {
                    "status": "failed",
                    "cell": number,
                    "traceback": traceback.format_exc(),
                    "resources": self.recorder.summary(),
                    "public_score": None,
                },
            )
            raise

    def finish(self, namespace):
        try:
            if self.contract["speed"]["mode"] != "stress_profile" or namespace["_p003_speed_schedule"].profile:
                raise ValueError("Stress mode must omit reference replay and submission publication")
            parent = super().finish(namespace)
            self.recorder.stop()
            resources = self.recorder.summary()
            memmap_observed = any(item["storage"] == "memmap" for item in resources["pixel_cache_fingerprints"])
            observations_complete = (
                memmap_observed
                and all(item.get("fingerprint_available") for item in resources["pixel_cache_fingerprints"])
                and resources["parent_rss_observation_available"]
                and resources["cgroup_memory_observation_available"]
                and not resources["observer_errors"]
            )
            result = {
                "status": "profile_complete_not_for_submission",
                "studies": len(self.ids),
                "selection": parent["parent_completion"]["selection"],
                "elapsed_seconds_including_preflight": parent["parent_completion"][
                    "elapsed_seconds_including_preflight"
                ],
                "reference_replay_performed": False,
                "weights_preprocessing_and_blend_unchanged": True,
                "measurement_status": "complete" if observations_complete else "incomplete",
                "memmap_path_observed": memmap_observed,
                "measurement_limitations": []
                if observations_complete
                else [
                    "Missing observations do not establish safe memory/disk use and are not model failures. Inspect resources availability and observer_errors."
                ],
                "resources": resources,
                "raptor": parent["raptor"],
                "phase_events": parent["phase_events"],
                "coat_receipts": parent["coat_receipts"],
                "fallback_studies": 0,
                "profile_predictions_sha256": sha256(self.work / "profile_predictions.csv"),
                "speed_summary_sha256": sha256(self.work / "P003_SPEED_SUMMARY.json"),
                "profile_contract": read_json(self.work / "P003_SPEED_CONTRACT.json"),
                "public_score": None,
                "hidden_test_runtime_verified": False,
                "note": "Larger train cohort without labels; resource diagnosis only. Full-cache fingerprints add overhead; this is not a fair speed benchmark.",
            }
            save_json(self.work / "P003_STRESS_SUMMARY.json", result)
            return result
        except BaseException:
            self.recorder.stop()
            save_json(
                self.work / "P003_STRESS_FAILURE.json",
                {
                    "status": "failed",
                    "cell": "finish",
                    "traceback": traceback.format_exc(),
                    "resources": self.recorder.summary(),
                    "public_score": None,
                },
            )
            raise
