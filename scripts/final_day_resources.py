"""WindowsのメモリとGPUを読み取り、資格判断用の実測値を返す。"""
from __future__ import annotations

import ctypes
import json
import subprocess
from ctypes import wintypes
from datetime import datetime, timezone


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD),
                *[(name, ctypes.c_ulonglong) for name in
                  ("total_phys", "avail_phys", "total_page", "avail_page",
                   "total_virtual", "avail_virtual", "avail_extended")]]


def observe():
    memory = MemoryStatus()
    memory.length = ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise ctypes.WinError()
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"],
        text=True, timeout=10).strip()
    return {"at_UTC": datetime.now(timezone.utc).isoformat(), "GPU": gpu,
            "memory_source": "Windows GlobalMemoryStatusEx",
            "physical_available_GiB": memory.avail_phys / 1024**3,
            "physical_total_GiB": memory.total_phys / 1024**3,
            "commit_available_GiB": memory.avail_page / 1024**3,
            "commit_limit_GiB": memory.total_page / 1024**3}


if __name__ == "__main__":
    print(json.dumps(observe(), ensure_ascii=False))
