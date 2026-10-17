"""Append one isolated native child to an already completed P029M notebook.

This module never uploads or submits. Any failed native extension retains the
completed parent CSV and emits an explicit no-comparison receipt.
"""

from __future__ import annotations

import base64
import gc
import hashlib
import io
import json
import lzma
import os
import signal
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_file(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def parent_progress(stage, state, *, mode, **details):
    print("P032_PARENT_PROGRESS", json.dumps({
        "stage": stage, "state": state, "mode": mode,
        "monotonic_seconds": time.monotonic(), **details,
    }, allow_nan=False), flush=True)


def restore_runtime(encoded, expected):
    raw = lzma.decompress(base64.b85decode(encoded))
    if len(raw) > 8 * 1024**2 or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("P032 runtime bundle SHA/size differs")
    root = Path("/kaggle/working/P032_RUNTIME")
    root.mkdir(exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(
            name.startswith("/") or ":" in name or "\\" in name or
            any(part in ("", ".", "..") for part in name.split("/")) for name in names
        ):
            raise ValueError("Unsafe P032 runtime member")
        for name in names:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(archive.read(name))
    return root


def relay_child_log(log, offset):
    """Relay at most 2 KB of newly appended child output, never old output."""
    with Path(log).open("rb") as stream:
        end = stream.seek(0, os.SEEK_END)
        offset = min(offset, end)
        if end > offset:
            start = max(offset, end - 2000)
            stream.seek(start)
            data = stream.read(end - start).decode("utf-8", errors="replace")
            print("P032_CHILD_LOG", json.dumps({
                "log_path": str(log), "start_byte": start, "end_byte": end,
                "omitted_new_bytes": start - offset, "data": data,
            }), flush=True)
    return end


def run_child(argv, log, budget, *, mode="unknown"):
    """Watch the child process group, including OS-killed decoder deadlocks."""
    start = time.monotonic()
    record = {"argv": argv, "mode": mode, "budget_seconds": budget, "complete": False,
              "timeout": False, "joined": False, "process_group_scoped": True}
    offset, next_progress = 0, start
    with Path(log).open("xb") as stream:
        process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT,
                                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                                        "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"},
                                   start_new_session=True)
        record["pid"] = process.pid
        try:
            while True:
                now = time.monotonic()
                observed_returncode = process.poll()
                if observed_returncode is not None:
                    record["returncode"] = observed_returncode
                    break
                if now >= next_progress:
                    print("P032_CHILD_HEARTBEAT", json.dumps({
                        "mode": mode, "pid": process.pid, "monotonic_seconds": now,
                        "elapsed_seconds": now - start, "budget_seconds": budget,
                        "log_path": str(log), "status": "WAITING_FOR_CHILD",
                    }), flush=True)
                    offset = relay_child_log(log, offset)
                    next_progress = now + 30
                remaining = budget - (now - start)
                if remaining <= 0:
                    if process.poll() is None:
                        raise subprocess.TimeoutExpired(argv, budget)
                    record["returncode"] = process.returncode
                    break
                try:
                    record["returncode"] = process.wait(timeout=min(1.0, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
        except subprocess.TimeoutExpired:
            record["timeout"] = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
            record["returncode"] = process.returncode
        record["joined"] = process.poll() is not None
        # A terminated top-level child may have left decoder grandchildren.
        # Bound cleanup too; only this newly created process group is signaled.
        def group_exists():
            try:
                os.killpg(process.pid, 0)
                return True
            except ProcessLookupError:
                return False

        if group_exists():
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            limit = time.monotonic() + 3
            while group_exists() and time.monotonic() < limit:
                time.sleep(0.05)
            if group_exists():
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                limit = time.monotonic() + 2
                while group_exists() and time.monotonic() < limit:
                    time.sleep(0.05)
        record["owned_process_group_empty"] = not group_exists()
        relay_child_log(log, offset)
    record["elapsed_seconds"] = time.monotonic() - start
    record["complete"] = (record["returncode"] == 0 and not record["timeout"]
                          and record["joined"] and record["owned_process_group_empty"])
    return record


def parent_contract(path, test, sample, package):
    sys.path.insert(0, str(package))
    from p032blend.contracts import validate_submission

    return validate_submission(path, test, sample)


def execute(namespace, *, runtime, started, mode="serving", diagnostic_count=192):
    parent_progress("execute", "begin", mode=mode, runtime=str(runtime))
    work = Path("/kaggle/working/P032")
    work.mkdir(exist_ok=False)
    package = Path(runtime)
    receipt = {
        "schema_version": "p032_p029m_f006_fixed20_execution_v1", "mode": mode,
        "complete": False, "native_complete": False, "all_native_disabled": True,
        "native_contribution_used": False,
        "accuracy_comparison_status": "NOT_MEASURED_NATIVE_EXTENSION_NOT_COMPLETE",
        "Public_accuracy_verified": False, "Submit_calls": 0,
        "whole_notebook_budget_seconds": 30600, "final_write_reserve_seconds": 300,
        "native_stage_budget_limit_seconds": 3600 if mode == "diagnostic" else 28800,
        "earlier_hidden_failure_cause_confirmed": False,
        "historical_GPU_native64_is_not_new_parent_hidden_qualification": True,
    }
    final = Path("/kaggle/working/submission.csv")
    parent = work / "parent-submission.csv"
    preserved = False
    try:
        if mode == "serving":
            if not final.is_file() or final.is_symlink():
                raise ValueError("P029M must complete a regular submission.csv first")
            parent.write_bytes(final.read_bytes())
            preserved = True
            receipt["parent_csv_sha256"] = sha(parent)
            receipt["parent_source_scientific_cells_unchanged"] = True
            data = Path(namespace["DATA"])
            receipt["parent_contract"] = parent_contract(parent, data / "test.csv",
                                                         data / "sample_submission.csv", package)
            original_blend = json.loads((final.parent / "blend_receipt.json").read_text())
            if original_blend.get("weights") != {
                "raptor": 0.5, "ours": 0.5,
                "raptor_per_class": {"Medial OA": 0.5, "Lateral OA": 0.5, "PF OA": 0.5}, "a224": 0.35,
            }:
                raise ValueError("Original parent blend weight receipt differs")
            receipt["parent_blend_receipt_sha256"] = sha(final.parent / "blend_receipt.json")
            parent_controller = json.loads((final.parent / "submission_receipt.json").read_text())
            if (parent_controller.get("status") != "PASS"
                    or parent_controller.get("bundle_id") != "fe055406fad6a12f"
                    or parent_controller.get("worker_exit_codes") != {"0": 0, "1": 0}):
                raise RuntimeError("Parent Compact controller workers did not exit successfully")
            receipt["parent_controller_receipt_sha256"] = sha(final.parent / "submission_receipt.json")
            receipt["parent_process_completion"] = {
                "Raptor384_and_Raptor224": "unchanged synchronous launch cells completed without raising",
                "Compact_worker_exit_codes": parent_controller["worker_exit_codes"],
                "old_S6_two_controller_owner_used": False,
            }
        # The public parent runs all models in joined subprocesses. Do not
        # reuse the old two-controller S6 ownership adapter for its three arms.
        import multiprocessing

        parent_progress("import_torch", "begin", mode=mode)
        import torch

        parent_progress("import_torch", "end", mode=mode)
        if multiprocessing.active_children():
            raise RuntimeError("Parent decoder children remain")
        gc.collect()
        parent_progress("CUDA_available", "begin", mode=mode)
        cuda_available = torch.cuda.is_available()
        parent_progress("CUDA_available", "end", mode=mode, available=cuda_available)
        if not cuda_available:
            raise RuntimeError("P032 requires two CUDA devices")
        parent_progress("CUDA_device_count", "begin", mode=mode)
        cuda_count = torch.cuda.device_count()
        parent_progress("CUDA_device_count", "end", mode=mode, device_count=cuda_count)
        if cuda_count != 2:
            raise RuntimeError("P032 requires two CUDA devices")
        cleanup = []
        for index in (0, 1):
            parent_progress("GPU_cleanup", "begin", mode=mode, device=index)
            before = {"allocated": torch.cuda.memory_allocated(index),
                      "reserved": torch.cuda.memory_reserved(index)}
            with torch.cuda.device(index):
                torch.cuda.empty_cache()
            after = {"allocated": torch.cuda.memory_allocated(index),
                     "reserved": torch.cuda.memory_reserved(index)}
            parent_progress("GPU_cleanup", "end", mode=mode, device=index, before=before, after=after)
            if after["allocated"] != 0:
                raise RuntimeError("Parent retains live CUDA allocations")
            cleanup.append({"device": index, "before": before, "after": after})
        receipt["parent_GPU_cleanup"] = cleanup
        remaining = 30600 - (time.monotonic() - started) - 300
        budget = min(receipt["native_stage_budget_limit_seconds"], remaining)
        if budget < 120:
            raise TimeoutError("Insufficient whole-notebook time for native extension")
        result = work / "native-result.json"
        argv = [sys.executable, "-B", "-u", str(package / "p032_native_child.py"),
                "--mode", mode, "--output", str(result), "--budget-seconds", str(budget),
                "--diagnostic-count", str(diagnostic_count)]
        if mode == "serving":
            argv += ["--baseline", str(parent)]
        parent_progress("native_child", "begin", mode=mode, budget_seconds=budget,
                        child_script=str(package / "p032_native_child.py"), result_path=str(result))
        child = run_child(argv, work / "native-child.log", budget, mode=mode)
        parent_progress("native_child", "end", mode=mode, complete=child["complete"],
                        returncode=child.get("returncode"), timeout=child.get("timeout"))
        receipt["child_watchdog"] = child
        json_file(work / "child-watchdog.json", child)
        if child["complete"] is not True:
            raise RuntimeError("Native child failed, timed out or retained owned processes")
        adapter = json.loads((work / "adapter-receipt.json").read_text())
        if adapter.get("complete") is not True:
            raise RuntimeError("Native adapter did not complete")
        receipt["native_complete"] = True
        receipt["native_adapter_receipt_sha256"] = sha(work / "adapter-receipt.json")
        receipt["native_result_sha256"] = sha(result)
        receipt["native_studies"] = adapter["studies"]
        receipt["native_structural_failures"] = adapter["structural_failure_count"]
        if mode == "diagnostic":
            receipt.update(complete=True,
                           accuracy_comparison_status="ENGINEERING_DIAGNOSTIC_ONLY_NO_SUBMISSION",
                           no_submission_csv_written=True)
            return receipt
        from p032blend.fixed20_blend import blend_csv

        blended = work / "fixed20"
        numerical = blend_csv(parent, adapter["native_csv"], blended,
                              failure_mask_csv=adapter["failure_mask_csv"], alpha=0.2,
                              test_csv=data / "test.csv", sample_csv=data / "sample_submission.csv")
        receipt["numeric_blend"] = numerical
        receipt["all_native_disabled"] = numerical["all_native_disabled"]
        receipt["native_contribution_used"] = not numerical["all_native_disabled"]
        receipt["accuracy_comparison_status"] = numerical["accuracy_comparison_status"]
        candidate = blended / "submission.csv"
        parent_contract(candidate, data / "test.csv", data / "sample_submission.csv", package)
        if sha(final) != receipt["parent_csv_sha256"]:
            raise RuntimeError("Completed parent changed during independent native child")
        temporary = final.with_name("P032_final.tmp")
        with temporary.open("xb") as stream:
            stream.write(candidate.read_bytes())
        os.replace(temporary, final)
        receipt.update(complete=True, submission_csv_sha256=sha(final),
                       parent_csv_preserved_sha256=sha(parent))
        return receipt
    except Exception as error:
        receipt.update(error_type=type(error).__name__, error=str(error),
                       traceback=traceback.format_exc(), all_native_disabled=True,
                       native_contribution_used=False,
                       accuracy_comparison_status="NOT_AN_ACCURACY_COMPARISON_NATIVE_DISABLED")
        if preserved:
            final.write_bytes(parent.read_bytes())
            receipt.update(parent_fallback_preserved=True, submission_csv_sha256=sha(final))
        if mode == "diagnostic":
            receipt["no_submission_csv_written"] = not final.exists()
        return receipt
    finally:
        receipt["whole_notebook_elapsed_seconds"] = time.monotonic() - started
        json_file(work / "execution-receipt.json", receipt)
        print("P032_COMPLETE", json.dumps({key: receipt.get(key) for key in
              ("mode", "complete", "native_complete", "all_native_disabled", "native_contribution_used",
               "accuracy_comparison_status", "error_type", "whole_notebook_elapsed_seconds")} ), flush=True)
