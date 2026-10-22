"""Run artificial CPU-only process QA into a fresh local directory; no APIs/MRI/models."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from isolated_uid_fixtures import LARGE_BYTES, prepare_artificial_uid  # noqa: E402

from rsna_knee.isolated_uid_executor import (  # noqa: E402
    IsolatedUIDExecutor,
    OperationalUIDFailure,
    TaskSourcePin,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def item(uid, mode):
    return ("artificial-input", uid, mode, "artificial-settings", "owned-workspace-replacement-required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Use a fresh artificial-QA directory")
    args.output.mkdir(parents=True)
    console = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_isolated_uid_executor.py")
    result = unittest.TextTestRunner(stream=console, verbosity=2).run(suite)
    (args.output / "tests.log").write_text(console.getvalue(), encoding="utf-8")
    if not result.wasSuccessful():
        raise RuntimeError("Artificial executor tests failed; inspect tests.log")
    pin = TaskSourcePin.from_callable(prepare_artificial_uid)
    scenarios = {}
    with tempfile.TemporaryDirectory(prefix="isolated-uid-qa-") as temporary:
        for scenario, modes, failure_mode in (
            ("four_success", ["success"] * 4, "fatal"),
            ("large_bytes", ["large"], "fatal"),
            ("large_numpy", ["numpy"], "fatal"),
            ("ordinary_exception", ["exception"], "return"),
            ("hard_exit", ["exit"], "fatal"),
            ("zero_exit_lost_result", ["lost"], "fatal"),
            ("deadline_then_success", ["hang", "success"], "return"),
            ("cancel_active_pending", ["hang"] * 4, "fatal"),
        ):
            start = time.monotonic()
            callbacks, errors = [], []
            executor = IsolatedUIDExecutor(
                pin, max_workers=1 if scenario == "deadline_then_success" else 2,
                uid_deadline_seconds=1.5 if scenario == "deadline_then_success" else 20,
                cleanup_seconds=.3, temporary_parent=temporary, operational_failure_mode=failure_mode)
            tickets = [executor.apply_async(prepare_artificial_uid, (item(str(index), mode),),
                                           callback=callbacks.append, error_callback=errors.append)
                       for index, mode in enumerate(modes)]
            if scenario == "cancel_active_pending":
                until = time.monotonic() + 5
                while executor.receipt()["peak_active"] < 2 and time.monotonic() < until:
                    time.sleep(.01)
                executor.terminate()
            else:
                executor.close()
            executor.join()
            outcomes = []
            for ticket in tickets:
                try:
                    value = ticket.get(0)
                    if type(value) is OperationalUIDFailure:
                        outcomes.append({"returned_operational_failure": value.receipt()})
                    else:
                        uid, arrays, metadata = value
                        summaries = {}
                        for name, array in arrays.items():
                            raw = array if isinstance(array, bytes) else memoryview(array).cast("B")
                            summaries[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                            if not isinstance(array, bytes):
                                summaries[name].update(shape=list(array.shape), dtype=str(array.dtype))
                        outcomes.append({"uid": uid, "arrays": summaries, "metadata": metadata})
                except OperationalUIDFailure as error:
                    outcomes.append({"raised_operational_failure": error.receipt()})
            scenarios[scenario] = {"elapsed_seconds": time.monotonic() - start,
                                   "callback_count": len(callbacks), "error_callback_count": len(errors),
                                   "outcomes": outcomes, "executor": executor.receipt()}
            if executor.root.exists():
                raise AssertionError("Owned temporary artifacts remain")
            del callbacks, tickets, errors
    candidate = ROOT / "artifacts/kaggle/p034-p029m-f006-success20-20261017-v2"
    pins = json.loads((candidate / "manifest.json").read_bytes())
    if sha(candidate / "source.ipynb") != pins["source_sha256"]:
        raise AssertionError("P034-v2 immutable notebook changed")
    static = ROOT / "artifacts/research/20261016-f006-execution-v1/static-native-source"
    for name, pin_entry in pins["native"]["source_members"].items():
        if sha(static / name) != pin_entry["sha256"]:
            raise AssertionError("An existing sealed native source changed: " + name)
    sources = ("src/rsna_knee/isolated_uid_executor.py", "scripts/verify_isolated_uid_executor.py",
               "tests/test_isolated_uid_executor.py", "tests/isolated_uid_fixtures.py")
    verification = {
        "schema_version": "isolated_uid_artificial_real_process_QA_v1",
        "at_UTC": datetime.now(timezone.utc).isoformat(),
        "qualification": "PASS_ARTIFICIAL_WINDOWS_CPU_ONLY_NOT_KAGGLE_RUNTIME_QUALIFICATION",
        "environment": {"python": platform.python_version(), "system": platform.system()},
        "tests": {"run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
                  "skipped": [{"test": str(test), "reason": reason} for test, reason in result.skipped],
                  "log_sha256": sha(args.output / "tests.log")},
        "source_sha256": {name: sha(ROOT / name) for name in sources},
        "native_sized_image_payload_bytes": LARGE_BYTES,
        "scenarios": scenarios,
        "preserved": {"P034_v2_source_sha256": pins["source_sha256"], "sealed_native_sources_unchanged": 89,
                      "P032_child_parent_builder_edited_by_this_task": False},
        "minimal_connection": {
            "task_pin": "TaskSourcePin.from_callable(serving_router.prepare_hidden_task, expected_sha256=sealed_SHA)",
            "hook": "router.multiprocessing = IsolatedUIDMultiprocessingProxy(original, task_pin, ...)",
            "scope": "Only router global binding; original multiprocessing module/source stays unchanged.",
            "existing_pipeline": "run_bounded_pipeline uses its original factory/task/consume/queue arguments.",
            "default_mode": "fatal: all ordinary and operational failures enter original error_callback/global parent fallback.",
            "return_mode": "NEW OperationalUIDFailure only for timeout/exit/lost result; new explicit consume/projection contract required.",
            "fixed_science": "Parent, native four models, reader/scientific inputs, alpha20 and success-subset rank policy fixed.",
        },
        "unqualified": [
            "Linux process-group observation and SIGTERM-ignored/SIGKILL test: unavailable locally; Windows-only skip.",
            "Real MRI reader import/decoder startup, per-UID 180s deadline and scratch I/O budget remain unqualified.",
            "No MRI/model/CUDA/Kaggle Save/Submit/cancel or network was executed by this verifier.",
            "No new consumer/projector for OperationalUIDFailure is installed in the existing sealed serving stack.",
            "Callbacks must remain nonblocking; p010 uses put_nowait and Event callbacks.",
            "Only direct CPU workers are joined here; outer native process group remains final descendant containment.",
        ],
    }
    raw = (json.dumps(verification, ensure_ascii=False, indent=2) + "\n").encode()
    (args.output / "verification.json").write_bytes(raw)
    print(json.dumps({"verification": str(args.output / "verification.json"),
                      "sha256": hashlib.sha256(raw).hexdigest(), "tests_run": result.testsRun,
                      "skipped": len(result.skipped)}))


if __name__ == "__main__":
    main()
