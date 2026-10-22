"""Importable real-process artificial tasks; no MRI/model/library dependency."""

import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path

LARGE_BYTES = 3 * 48 * 3 * 392 * 392


@dataclass(frozen=True)
class ArtificialUIDRecord:
    """Identity carrier only; no old structural provenance is claimed."""

    uid: str


def prepare_artificial_uid(item):
    _, uid, mode, _, workspace = item
    scratch = Path(workspace)
    scratch.mkdir(exist_ok=True)
    (scratch / "artificial-reader.tmp").write_bytes(b"synthetic only")
    if mode == "wrong_uid_failure":
        return ArtificialUIDRecord("DIFFERENT-UID")
    if mode == "correct_uid_record":
        return ArtificialUIDRecord(uid)
    if mode == "exception":
        raise ValueError("synthetic non-whitelisted task exception")
    if mode == "exit":
        os._exit(73)
    if mode == "lost":
        os._exit(0)
    if mode in ("hang", "ignore_term"):
        if mode == "ignore_term" and os.name != "nt":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(60)
    if mode == "delay":
        time.sleep(.25)
    if mode == "numpy":
        import numpy as np

        arrays = {"images": np.full((3, 48, 3, 392, 392), 17, np.uint8), "mask": np.ones((3, 48), np.bool_),
                  "positions": np.zeros((3, 48), np.float32), "planes": np.arange(3, dtype=np.int64),
                  "flags": np.zeros((3, 2), np.float32)}
        size = arrays["images"].nbytes
    else:
        payload = bytes(range(256)) * (LARGE_BYTES // 256) if mode == "large" else b"artificial-array-bytes"
        arrays, size = {"images": payload}, len(payload)
    return "DIFFERENT-UID" if mode == "wrong_uid_tuple" else uid, arrays, {
        "pid": os.getpid(), "parent_pid": os.getppid(),
        "process_group": os.getpgrp() if hasattr(os, "getpgrp") else None,
        "workspace": workspace, "payload_bytes": size, "scientific_value": .125,
    }
