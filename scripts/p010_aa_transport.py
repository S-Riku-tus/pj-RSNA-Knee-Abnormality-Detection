"""Diagnostic-only owned dispatch; original serving transport stays untouched."""

import hashlib
import inspect
import queue
import threading
import time


class OwnedQueue:
    """One bounded payload container, selectable by the assigned consumer."""

    def __init__(self, devices, maxsize, assignment):
        self.devices = tuple(devices)
        self.maxsize = maxsize
        self.assignment = assignment
        self.condition = threading.Condition()
        self.values = []
        self.unfinished_tasks = 0
        if not self.devices or maxsize < 1:
            raise ValueError("Positive queue bound and device identities are required")
        if assignment is not None and not set(assignment.values()) <= set(self.devices):
            raise ValueError("Unknown assigned device")

    def qsize(self):
        with self.condition:
            return len(self.values)

    def _put(self, owner, item, timeout):
        deadline = None if timeout is None else time.monotonic() + timeout
        with self.condition:
            while len(self.values) >= self.maxsize:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise queue.Full
                self.condition.wait(remaining)
            self.values.append((owner, item))
            self.unfinished_tasks += 1
            self.condition.notify_all()

    def put(self, item, timeout=None):
        if item is None:
            raise ValueError("Terminal markers need an explicit consumer owner")
        owner = None if self.assignment is None else self.assignment[item[0]]
        self._put(owner, item, timeout)

    def put_terminal(self, owner, timeout=None):
        if owner not in self.devices:
            raise ValueError("Unknown terminal consumer")
        self._put(owner, None, timeout)

    def get_for(self, device):
        if device not in self.devices:
            raise ValueError("Unknown consumer")
        with self.condition:
            while True:
                for index, (owner, item) in enumerate(self.values):
                    if owner is None or owner == device:
                        del self.values[index]
                        self.condition.notify_all()
                        return item
                self.condition.wait()

    def task_done(self):
        with self.condition:
            if self.unfinished_tasks < 1:
                raise ValueError("task_done called too often")
            self.unfinished_tasks -= 1
            self.condition.notify_all()


def fixed_pipeline(original, assignment):
    """Bind only consumer ownership in a fresh copy of the original pipeline."""
    source = inspect.getsource(original)
    replacements = {
        "gpu_queue = queue.Queue(maxsize=queue_max)": (
            "gpu_queue = _AA_OwnedQueue(devices, maxsize=queue_max, assignment=_AA_assignment)"
        ),
        "item = gpu_queue.get()": "item = gpu_queue.get_for(device)",
        "for _ in threads:": "for _AA_device in devices:",
        "gpu_queue.put(None, timeout=poll_seconds)": "gpu_queue.put_terminal(_AA_device, timeout=poll_seconds)",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError("Original bounded transport insertion point changed")
        source = source.replace(old, new, 1)
    scope = dict(original.__globals__)
    scope.update({"_AA_OwnedQueue": OwnedQueue, "_AA_assignment": dict(assignment)})
    exec(compile(source, "<p010-AA-fixed-dispatch>", "exec"), scope)
    function = scope[original.__name__]

    def run(*args, **kwargs):
        try:
            receipt = function(*args, **kwargs)
        except BaseException as exception:
            if hasattr(exception, "receipt"):
                exception.receipt["diagnostic_assignment_policy"] = "UID_ordinal_modulo_device_count"
                exception.receipt["diagnostic_owned_queue_source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
                exception.receipt["original_shared_pipeline_function_unmodified"] = True
            raise
        receipt["diagnostic_assignment_policy"] = "UID_ordinal_modulo_device_count"
        receipt["diagnostic_owned_queue_source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        receipt["original_shared_pipeline_function_unmodified"] = True
        receipt["consumer_ownership_changed_not_a_serving_candidate"] = True
        return receipt

    return run
