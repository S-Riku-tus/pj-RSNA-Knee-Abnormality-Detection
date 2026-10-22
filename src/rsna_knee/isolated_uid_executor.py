"""Bounded, observable CPU preparation: one owned subprocess per study.

This implements the Pool subset used by p010: apply_async/close/join/terminate.
Results cross a private regular-file transport, never a large multiprocessing
pipe. It does not import NumPy, Torch, readers, models, or any network client.
Only a positively pinned top-level task (optionally functools.partial) runs.

Default operational failures use error_callback, preserving the enclosing
pipeline's fatal-error contract. Explicit ``operational_failure_mode='return'``
delivers a NEW OperationalUIDFailure for exit/lost-result/deadline failures.
That mode requires a separately reviewed consumer/projection adapter; it never
constructs or claims the old source-bound StructuralReaderPolicy/FailedUID.
Ordinary task exceptions, source mismatches and corrupt transports stay fatal.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.machinery
import inspect
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _function(task):
    while isinstance(task, functools.partial):
        task = task.func
    if not inspect.isfunction(task) or task.__module__ == "__main__" or "<locals>" in task.__qualname__:
        raise ValueError("A positively pinned importable top-level function is required")
    return task


def _fresh_task_origin(module_name, search_path):
    """Resolve each dotted component without sys.modules or package execution.

    Only the final task source is positively pinned here. Package initializers
    and transitive dependencies remain within the trusted source/path boundary;
    this transport is not a sandbox for untrusted Python or pickle inputs.
    """
    path = search_path
    components = module_name.split(".")
    for index in range(len(components)):
        spec = importlib.machinery.PathFinder.find_spec(".".join(components[:index + 1]), path)
        if spec is None:
            raise ValueError("Fresh task module cannot be resolved")
        if index == len(components) - 1:
            if spec.origin is None:
                raise ValueError("Fresh task module needs a regular source origin")
            return Path(spec.origin).resolve(strict=True)
        if spec.submodule_search_locations is None:
            raise ValueError("Fresh task parent is not a package")
        path = list(spec.submodule_search_locations)


@dataclass(frozen=True)
class TaskSourcePin:
    module: str
    qualname: str
    origin: str
    sha256: str

    @classmethod
    def from_callable(cls, task, *, expected_sha256=None):
        function = _function(task)
        path = Path(inspect.getsourcefile(function)).resolve(strict=True)
        digest = _file_sha(path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("Pinned preparation source SHA differs")
        return cls(function.__module__, function.__qualname__, str(path), digest)

    def verify(self, task):
        if self != type(self).from_callable(task, expected_sha256=self.sha256):
            raise ValueError("Preparation task identity/origin differs")


class OperationalUIDFailure(RuntimeError):
    """Explicit NEW operational evidence, never an old structural failure."""

    def __init__(self, uid, reason, *, pid=None, returncode=None, elapsed_seconds=None,
                 request_sha256=None, task_sha256=None, detail=None):
        self.uid, self.reason, self.pid = uid, reason, pid
        self.returncode, self.elapsed_seconds = returncode, elapsed_seconds
        self.request_sha256, self.task_sha256, self.detail = request_sha256, task_sha256, detail
        super().__init__(f"UID {uid}: {reason}" + (f": {detail}" if detail else ""))

    def receipt(self):
        return {"schema_version": "isolated_uid_operational_failure_v1", "classification": "operational",
                "old_structural_failure_claimed": False,
                **{name: getattr(self, name) for name in ("uid", "reason", "pid", "returncode", "elapsed_seconds",
                                                        "request_sha256", "task_sha256", "detail")}}


class UIDAsyncResult:
    """Pool-compatible readiness/get facade without executor-retained result history."""

    def __init__(self):
        self._event = threading.Event()
        self._value = self._error = None

    def ready(self):
        return self._event.is_set()

    def wait(self, timeout=None):
        self._event.wait(timeout)

    def successful(self):
        if not self.ready():
            raise ValueError("Result is not ready")
        return self._error is None

    def get(self, timeout=None):
        if not self._event.wait(timeout):
            raise TimeoutError("UID result is not ready")
        if self._error is not None:
            raise self._error
        return self._value

    def _set(self, value=None, error=None):
        self._value, self._error = value, error
        self._event.set()


class _LimitedWriter:
    def __init__(self, stream, limit):
        self.stream, self.limit, self.count = stream, limit, 0

    def write(self, value):
        if self.count + memoryview(value).nbytes > self.limit:
            raise ValueError("Preparation result exceeds the declared transport byte limit")
        count = self.stream.write(value)
        self.count += count
        return count


def _atomic_json(path, value):
    temporary = path.with_suffix(".partial")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _worker_main(metadata_path):
    """Fresh CPU worker; request pickle is private, local and source-pinned."""
    path = Path(metadata_path).resolve(strict=True)
    metadata = json.loads(path.read_bytes())
    work = path.parent
    envelope = {"uid": metadata["uid"], "request_sha256": metadata["request_sha256"],
                "pid": os.getpid(), "process_group": os.getpgrp() if hasattr(os, "getpgrp") else None}
    try:
        sys.path[:] = metadata["sys_path"]
        pin = TaskSourcePin(**metadata["task_pin"])
        fresh = _fresh_task_origin(pin.module, sys.path)
        if str(fresh) != pin.origin or _file_sha(fresh) != pin.sha256:
            raise ValueError("Pre-import fresh task origin/SHA differs")
        # Import before unpickling so a fresh interpreter cannot silently bind a
        # same-named shadow module hidden by the parent's sys.modules cache.
        module = importlib.import_module(pin.module)
        if str(Path(module.__file__).resolve()) != pin.origin or _file_sha(module.__file__) != pin.sha256:
            raise ValueError("Fresh-worker task module origin/SHA differs")
        envelope.update(fresh_task_origin=str(Path(module.__file__).resolve()), fresh_task_sha256=pin.sha256)
        request = work / "request.pkl"
        if _file_sha(request) != metadata["request_sha256"]:
            raise ValueError("Private request SHA differs")
        with request.open("rb") as stream:
            task, args, kwds = pickle.load(stream)
        # This module runs as __main__, so compare dictionaries, not class identity.
        if asdict(TaskSourcePin.from_callable(task)) != metadata["task_pin"]:
            raise ValueError("Unpickled task identity differs")
        value = task(*args, **kwds)
        temporary = work / "result.partial"
        with temporary.open("xb") as stream:
            pickle.dump(value, _LimitedWriter(stream, metadata["max_result_bytes"]), protocol=5)
        result = work / "result.pkl"
        os.replace(temporary, result)
        envelope.update(kind="result", result_bytes=result.stat().st_size, result_sha256=_file_sha(result))
    except BaseException as error:
        envelope.update(kind="task_exception", error_type=type(error).__name__, error=str(error)[:4000],
                        traceback=traceback.format_exc()[-12000:])
    _atomic_json(work / "completion.json", envelope)


class IsolatedUIDExecutor:
    """One CPU child per UID; max four active+pending jobs, deadline includes startup.

    Callbacks must be nonblocking, as are p010's callbacks. No callback runs
    after terminate begins. Each active PID is a direct Popen child, stays in
    the current process group, and is reaped before its owned files disappear.
    Only this executor's newly created temporary root is recursively removed.
    ``workspace_item_index=4`` isolates the native reader's scratch namespace;
    all scientific task inputs and returned objects otherwise remain unchanged.
    """

    def __init__(self, task_pin, *, max_workers=2, max_pending=4, uid_deadline_seconds=180.0,
                 cleanup_seconds=3.0, poll_seconds=.02, temporary_parent=None, max_result_bytes=96 * 1024**2,
                 workspace_item_index=4, operational_failure_mode="fatal"):
        if not isinstance(task_pin, TaskSourcePin) or type(max_workers) is not int or not 1 <= max_workers <= 4:
            raise ValueError("A source pin and 1..4 CPU workers are required")
        if type(max_pending) is not int or not max_workers <= max_pending <= 4:
            raise ValueError("Active+pending UID capacity must be between worker count and four")
        if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0
               for x in (uid_deadline_seconds, cleanup_seconds, poll_seconds)):
            raise ValueError("Finite positive deadlines/polling are required")
        if type(max_result_bytes) is not int or max_result_bytes < 1 or operational_failure_mode not in ("fatal", "return"):
            raise ValueError("Invalid transport limit/failure mode")
        parent = Path(temporary_parent or tempfile.gettempdir()).resolve(strict=True)
        if parent == Path("/kaggle/working") or Path("/kaggle/working") in parent.parents:
            raise ValueError("Reader/result temporary files cannot be placed under Kaggle Output")
        self.root = Path(tempfile.mkdtemp(prefix="isolated-uid-", dir=parent)).resolve(strict=True)
        self._owned_parent = parent
        self.pin, self.max_workers, self.max_pending = task_pin, max_workers, max_pending
        self.uid_deadline, self.cleanup_seconds, self.poll_seconds = uid_deadline_seconds, cleanup_seconds, poll_seconds
        self.max_result_bytes, self.workspace_index = max_result_bytes, workspace_item_index
        self.failure_mode = operational_failure_mode
        self._condition = threading.Condition(threading.RLock())
        self._pending, self._active, self._seen_uids = deque(), {}, set()
        self._closed = self._cancelled = self._finished = False
        self._serial = 0
        self._records, self._internal_errors = [], []
        self._peak_active = 0
        self._root_cleaned = False
        self._thread = threading.Thread(target=self._monitor, name="isolated-uid-monitor", daemon=True)
        self._thread.start()

    def apply_async(self, func, args=(), kwds=None, callback=None, error_callback=None):
        self.pin.verify(func)
        if len(args) != 1 or not isinstance(args[0], tuple) or len(args[0]) < 2 or type(args[0][1]) is not str:
            raise ValueError("Expected one native-style tuple item with UID at index 1")
        uid = args[0][1]
        if not uid or kwds not in (None, {}):
            raise ValueError("Nonempty UID and no additional task keyword arguments required")
        with self._condition:
            if self._closed or self._cancelled:
                raise ValueError("Executor is closed/terminated")
            if uid in self._seen_uids:
                raise ValueError("A UID may be submitted only once")
            if len(self._pending) + len(self._active) >= self.max_pending:
                raise RuntimeError("Bounded UID submission capacity exceeded")
            self._serial += 1
            folder = self.root / f"uid-{self._serial:08d}"
            folder.mkdir()
            ticket = UIDAsyncResult()
            job = {"uid": uid, "folder": folder, "ticket": ticket, "callback": callback,
                   "error_callback": error_callback, "process": None}
            try:
                item = list(args[0])
                if self.workspace_index is not None:
                    if type(self.workspace_index) is not int or not 0 <= self.workspace_index < len(item):
                        raise ValueError("Native workspace tuple index differs")
                    scratch = folder / "reader-scratch"
                    scratch.mkdir()
                    item[self.workspace_index] = str(scratch)
                request = folder / "request.pkl"
                with request.open("xb") as stream:
                    pickle.dump((func, (tuple(item),), {}), _LimitedWriter(stream, 2 * 1024**2), protocol=5)
                job["request_sha256"] = _file_sha(request)
                metadata = {"uid": uid, "request_sha256": job["request_sha256"], "task_pin": asdict(self.pin),
                            "sys_path": list(sys.path), "max_result_bytes": self.max_result_bytes}
                _atomic_json(folder / "request.json", metadata)
            except BaseException:
                self._remove_folder(folder)
                raise
            self._seen_uids.add(uid)
            self._pending.append(job)
            self._condition.notify_all()
            return ticket

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def terminate(self):
        with self._condition:
            self._closed = self._cancelled = True
            self._condition.notify_all()

    def join(self, timeout=None):
        with self._condition:
            if not self._closed:
                raise ValueError("Close or terminate the executor before join")
        natural_bound = math.ceil(self.max_pending / self.max_workers) * self.uid_deadline + 4 * self.cleanup_seconds + 5
        self._thread.join(natural_bound if timeout is None else timeout)
        if self._thread.is_alive():
            self.terminate()
            self._thread.join(4 * self.cleanup_seconds + 5)
        if self._thread.is_alive():
            raise RuntimeError("UID monitor failed to join within bounded cleanup")
        if self._internal_errors:
            raise RuntimeError("UID monitor/cleanup failed: " + "; ".join(self._internal_errors))

    def receipt(self):
        with self._condition:
            return {"schema_version": "isolated_uid_executor_v1", "failure_mode": self.failure_mode,
                    "worker_limit": self.max_workers, "active_plus_pending_limit": self.max_pending,
                    "uid_deadline_seconds": self.uid_deadline, "peak_active": self._peak_active,
                    "same_process_group": True, "uses_result_pipe": False, "ordinary_task_errors_fatal": True,
                    "process_group_observation_supported": hasattr(os, "getpgrp"),
                    "old_structural_failure_constructed": False, "task_pin": asdict(self.pin),
                    "monitor_joined": not self._thread.is_alive(), "cancelled": self._cancelled,
                    "temporary_root_cleaned": self._root_cleaned,
                    "records": [dict(row) for row in self._records], "internal_errors": list(self._internal_errors)}

    def _remove_folder(self, folder):
        # Verify the resolved absolute target before every recursive deletion.
        if folder.is_symlink() or folder.resolve().parent != self.root:
            raise ValueError("Refusing to remove an unowned UID directory")
        shutil.rmtree(folder)

    def _stop(self, job):
        process = job["process"]
        if process is None:
            return
        if process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=self.cleanup_seconds)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                process.wait(timeout=self.cleanup_seconds)
        else:
            process.wait(timeout=self.cleanup_seconds)

    def _failure(self, job, reason, detail=None):
        process = job["process"]
        return OperationalUIDFailure(job["uid"], reason, pid=process.pid if process else None,
                                     returncode=process.returncode if process else None,
                                     elapsed_seconds=time.monotonic() - job.get("started", time.monotonic()),
                                     request_sha256=job["request_sha256"], task_sha256=self.pin.sha256, detail=detail)

    def _finish(self, job, *, value=None, error=None, cancelled=False):
        cancelled = cancelled or self._cancelled
        if cancelled:
            value, error = None, self._failure(job, "cancelled")
        record = {"uid": job["uid"], "pid": job["process"].pid if job["process"] else None,
                  "returncode": job["process"].returncode if job["process"] else None,
                  "request_sha256": job["request_sha256"], "cancelled": cancelled,
                  "elapsed_seconds": time.monotonic() - job.get("started", time.monotonic()),
                  "callback_count": 0, "direct_child_reaped": job["process"] is None or job["process"].poll() is not None}
        for name in ("worker_process_group", "fresh_task_origin", "fresh_task_sha256",
                     "result_transport_bytes", "result_transport_sha256"):
            if name in job:
                record[name] = job[name]
        if error is not None:
            record["failure"] = error.receipt()
        try:
            self._remove_folder(job["folder"])
            record["owned_files_cleaned"] = True
        except BaseException as cleanup_error:
            # A cleanup error must still resolve the async result/error_callback.
            # Losing that notification would recreate the old endless poll.
            self._internal_errors.append(repr(cleanup_error))
            error = self._failure(job, "cleanup_error", repr(cleanup_error))
            record.update(owned_files_cleaned=False, cleanup_exception=repr(cleanup_error), failure=error.receipt())
        recoverable = error is not None and error.reason in ("deadline_exceeded", "worker_exit", "missing_result")
        returned_failure = recoverable and self.failure_mode == "return" and not cancelled
        result_error = None if returned_failure else error
        result_value = error if returned_failure else value
        # Share the condition with terminate, so cancelled generations cannot
        # race a late callback. Known p010 callbacks are put_nowait/event only.
        with self._condition:
            if not self._cancelled and not cancelled:
                callback = job["callback"] if result_error is None else job["error_callback"]
                if callback is not None:
                    record["callback_count"] = 1
                    try:
                        callback(result_value if result_error is None else result_error)
                    except BaseException as callback_error:
                        result_error = self._failure(job, "callback_exception", repr(callback_error))
                        record["callback_exception"] = repr(callback_error)
            job["ticket"]._set(value=result_value, error=result_error)
            record["returned_operational_failure"] = returned_failure
            self._records.append(record)

    def _read_result(self, job):
        completion = job["folder"] / "completion.json"
        if not completion.is_file():
            return None, self._failure(job, "missing_result")
        try:
            if completion.stat().st_size > 32 * 1024 or completion.is_symlink():
                raise ValueError("Invalid completion envelope")
            envelope = json.loads(completion.read_bytes())
            if (envelope["uid"] != job["uid"] or envelope["pid"] != job["process"].pid
                    or envelope["request_sha256"] != job["request_sha256"]):
                raise ValueError("Completion UID/PID/request binding differs")
            if hasattr(os, "getpgrp") and envelope["process_group"] != os.getpgrp():
                raise ValueError("UID worker escaped its owner's process group")
            job["worker_process_group"] = envelope["process_group"]
            if envelope["kind"] == "task_exception":
                return None, self._failure(job, "task_exception", json.dumps(envelope, ensure_ascii=True))
            result = job["folder"] / "result.pkl"
            if (envelope["kind"] != "result" or result.is_symlink() or not result.is_file()
                    or not 0 < envelope["result_bytes"] == result.stat().st_size <= self.max_result_bytes
                    or envelope["result_sha256"] != _file_sha(result)):
                raise ValueError("Result size/SHA envelope differs")
            if envelope["fresh_task_origin"] != self.pin.origin or envelope["fresh_task_sha256"] != self.pin.sha256:
                raise ValueError("Fresh-worker positive source pin differs")
            job.update(fresh_task_origin=envelope["fresh_task_origin"], fresh_task_sha256=envelope["fresh_task_sha256"],
                       result_transport_bytes=envelope["result_bytes"], result_transport_sha256=envelope["result_sha256"])
            with result.open("rb") as stream:
                value = pickle.load(stream)
                if stream.read(1):
                    raise ValueError("Trailing bytes after private result")
            payload_uid = value[0] if isinstance(value, tuple) and len(value) == 3 else getattr(value, "uid", None)
            if type(payload_uid) is not str or payload_uid != job["uid"]:
                raise ValueError("Preparation payload UID differs from its owned job UID")
            return value, None
        except BaseException as error:
            return None, self._failure(job, "transport_error", repr(error))

    def _monitor(self):
        try:
            while True:
                with self._condition:
                    if self._cancelled:
                        jobs = [*self._pending, *self._active.values()]
                        self._pending.clear()
                        self._active.clear()
                        for job in jobs:
                            self._stop(job)
                            self._finish(job, error=self._failure(job, "cancelled"), cancelled=True)
                        break
                    while self._pending and len(self._active) < self.max_workers:
                        job = self._pending.popleft()
                        job["started"] = time.monotonic()
                        try:
                            # Windows venv python.exe may be a redirector that
                            # creates an unowned grandchild. Launch its real
                            # interpreter directly; restore venv import paths
                            # from the pinned private request before task import.
                            executable = getattr(sys, "_base_executable", sys.executable) if os.name == "nt" else sys.executable
                            job["process"] = subprocess.Popen(
                                [executable, "-B", "-u", str(Path(__file__).resolve()), "--worker",
                                 str(job["folder"] / "request.json")],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=False,
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
                                     "HF_HUB_DISABLE_TELEMETRY": "1", "TMPDIR": str(job["folder"]),
                                     "TMP": str(job["folder"]), "TEMP": str(job["folder"])})
                            self._active[job["uid"]] = job
                            self._peak_active = max(self._peak_active, len(self._active))
                        except BaseException as error:
                            self._finish(job, error=self._failure(job, "launch_error", repr(error)))
                    active = list(self._active)
                for uid in active:
                    job = self._active[uid]
                    process = job["process"]
                    error = None
                    if process.poll() is not None:
                        process.wait(timeout=self.cleanup_seconds)
                        if time.monotonic() - job["started"] > self.uid_deadline:
                            error = self._failure(job, "deadline_exceeded")
                        elif process.returncode != 0:
                            error = self._failure(job, "worker_exit")
                        value, error = (None, error) if error is not None else self._read_result(job)
                    elif time.monotonic() - job["started"] >= self.uid_deadline:
                        self._stop(job)
                        value, error = None, self._failure(job, "deadline_exceeded")
                    else:
                        continue
                    with self._condition:
                        self._active.pop(job["uid"])
                        self._finish(job, value=value, error=error)
                    # Do not retain completed large arrays in monitor history.
                    value = job = None
                active = []
                with self._condition:
                    if self._closed and not self._pending and not self._active:
                        break
                    self._condition.wait(self.poll_seconds)
        except BaseException as error:
            self._internal_errors.append(repr(error))
        finally:
            with self._condition:
                jobs = [*self._pending, *self._active.values()]
                self._pending.clear()
                self._active.clear()
                for job in jobs:
                    try:
                        self._stop(job)
                        self._finish(job, error=self._failure(job, "monitor_error"), cancelled=True)
                    except BaseException as error:
                        self._internal_errors.append(repr(error))
                        job["ticket"]._set(error=self._failure(job, "cleanup_error", repr(error)))
                try:
                    if self.root.is_symlink() or self.root.resolve().parent != self._owned_parent:
                        raise ValueError("Executor temporary root identity changed")
                    self.root.rmdir()
                    self._root_cleaned = True
                except BaseException as error:
                    self._internal_errors.append(repr(error))
                self._finished = True
                self._condition.notify_all()


class IsolatedUIDMultiprocessingProxy:
    """Optional narrow runtime hook; never patch the shared multiprocessing module.

    Assign this object only to the sealed serving_router module's global name
    ``multiprocessing``. Its existing get_context('spawn').Pool(processes=2)
    factory then uses this executor. All other multiprocessing APIs delegate.
    The pinned source file itself remains unchanged; this is an explicit new
    operational runtime factor and requires new execution qualification.
    """

    def __init__(self, original, task_pin, **executor_options):
        self.original, self.task_pin, self.options = original, task_pin, executor_options
        self.executors = []

    def __getattr__(self, name):
        return getattr(self.original, name)

    def get_context(self, method=None):
        if method != "spawn":
            raise ValueError("The isolated UID hook only supports the explicit spawn context")
        owner = self
        original_context = self.original.get_context(method)

        class Context:
            def __getattr__(self, name):
                return getattr(original_context, name)

            def Pool(self, processes=None, initializer=None, initargs=(), maxtasksperchild=None):
                if initializer is not None or initargs or maxtasksperchild is not None:
                    raise ValueError("Pool initializer/recycling is outside the audited UID API subset")
                executor = IsolatedUIDExecutor(owner.task_pin, max_workers=processes, **owner.options)
                owner.executors.append(executor)
                return executor

        return Context()


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("This file is only a private UID worker entrypoint")
    _worker_main(sys.argv[2])
