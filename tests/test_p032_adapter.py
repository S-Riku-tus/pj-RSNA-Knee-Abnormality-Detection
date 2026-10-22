"""P032 boundary fixtures: no model, MRI, CUDA or external operations."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rsna_knee.contracts import ID, SERIES_COLUMNS, SUBMISSION_COLUMNS, TARGETS  # noqa: E402


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


extension = load("p032_extension_fixture", "p032_parent_extension.py")
builder = load("p032_builder_fixture", "prepare_p032_p029m_f006.py")
native = load("p032_native_fixture", "p032_native_child.py")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, SUBMISSION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


class ParentBoundaryTests(unittest.TestCase):
    def run_extension(self, child_outcome, *, blend_policy="fullN-column", native_executor="legacy-pool"):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        work, package, data = base / "working", base / "runtime", base / "data"
        work.mkdir()
        data.mkdir()
        (package / "p032blend").mkdir(parents=True)
        (package / "p032blend/__init__.py").write_text("")
        for name in ("contracts.py", "fixed20_blend.py", "success20_blend.py"):
            shutil.copyfile(ROOT / "src/rsna_knee" / name, package / "p032blend" / name)
        # Real saved parent's CSV field strings; change native order independently.
        rows = [{ID: str(i + 1), **{target: value for target in TARGETS}}
                for i, value in enumerate(("0.16666667", "0.50000000", "0.83333331"))]
        final = work / "submission.csv"
        write_csv(final, rows)
        before = final.read_bytes()
        write_csv(data / "sample_submission.csv", rows)
        (data / "test.csv").write_text(ID + "\n1\n2\n3\n")
        (work / "blend_receipt.json").write_text(json.dumps({"weights": {
            "raptor": 0.5, "ours": 0.5,
            "raptor_per_class": {"Medial OA": 0.5, "Lateral OA": 0.5, "PF OA": 0.5}, "a224": 0.35}}))
        (work / "submission_receipt.json").write_text(json.dumps({
            "status": "PASS", "bundle_id": "fe055406fad6a12f", "worker_exit_codes": {"0": 0, "1": 0}}))
        cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
                                     memory_allocated=lambda _: 0, memory_reserved=lambda _: 0,
                                     device=lambda _: nullcontext(), empty_cache=lambda: None)

        def child(argv, log, budget, *, mode):
            target = work / "P032"
            if child_outcome == "error":
                return {"complete": False, "timeout": True, "returncode": -15}
            native = [{ID: str(i + 1), **{target: value for target in TARGETS}}
                      for i, value in enumerate(("0.9", "0.1", "0.5"))]
            write_csv(target / "native-mean.csv", native)
            mask = target / "native-failures.csv"
            mask.write_text(ID + "\n" + ("2\n" if child_outcome in ("structural", "operational") else ""))
            result_path = target / "native-result.json"
            result_path.write_text(json.dumps({"schema_version": "native_isolated_uid_operational_recovery_v1",
                                              "complete": True, "operational_recovery_contract_passed": True})
                                   if native_executor == "isolated-uid" and child_outcome != "bad-typed" else "{}")
            adapter = {
                "complete": True, "studies": 3,
                "structural_failure_count": int(child_outcome == "structural"),
                "native_csv": str(target / "native-mean.csv"), "failure_mask_csv": str(mask)}
            if native_executor == "isolated-uid":
                self.assertIn("--native-executor", argv)
                self.assertEqual(argv[argv.index("--native-executor") + 1], "isolated-uid")
                adapter.update(schema_version="p035_f006_isolated_uid_adapter_v1", native_executor="isolated-uid",
                               operational_recovery_contract_passed=True,
                               failure_mask_is_structural_operational_union=True,
                               operational_failure_count=int(child_outcome == "operational"),
                               failure_UID_union=["2"] if child_outcome in ("structural", "operational") else [],
                               native_result_schema="native_isolated_uid_operational_recovery_v1",
                               native_result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest())
            (target / "adapter-receipt.json").write_text(json.dumps(adapter))
            return {"complete": True, "timeout": False, "returncode": 0}

        def mapped(value):
            text = str(value)
            prefix = "/kaggle/working"
            return work / text[len(prefix):].lstrip("/") if text.startswith(prefix) else Path(value)

        with patch.object(extension, "Path", side_effect=mapped), \
                patch.object(extension, "run_child", side_effect=child), \
                patch("multiprocessing.active_children", return_value=[]), \
                patch.dict(sys.modules, {"torch": types.SimpleNamespace(cuda=cuda)}):
            result = extension.execute({"DATA": data}, runtime=package,
                                       started=extension.time.monotonic(), mode="serving", blend_policy=blend_policy,
                                       native_executor=native_executor)
        actual = final.read_bytes()
        for name in tuple(sys.modules):
            if name == "p032blend" or name.startswith("p032blend."):
                del sys.modules[name]
        sys.path[:] = [item for item in sys.path if item != str(package)]
        return result, before, actual

    def test_timeout_retains_exact_parent_and_marks_no_comparison(self):
        result, before, actual = self.run_extension("error")
        self.assertEqual(before, actual)
        self.assertFalse(result["complete"])
        self.assertTrue(result["all_native_disabled"])
        self.assertFalse(result["native_contribution_used"])
        self.assertIn("NOT_AN_ACCURACY_COMPARISON", result["accuracy_comparison_status"])

    def test_structural_failure_mask_disables_every_native_target(self):
        result, before, actual = self.run_extension("structural")
        self.assertTrue(result["complete"])
        self.assertTrue(result["native_complete"])
        self.assertTrue(result["all_native_disabled"])
        self.assertFalse(result["native_contribution_used"])
        self.assertEqual(before, actual)

    def test_success_uses_native_and_retains_two_excluded_targets(self):
        result, before, actual = self.run_extension("success")
        self.assertTrue(result["complete"])
        self.assertFalse(result["all_native_disabled"])
        self.assertTrue(result["native_contribution_used"])
        self.assertNotEqual(before, actual)
        import io

        old, new = [list(csv.DictReader(io.StringIO(body.decode()))) for body in (before, actual)]
        for left, right in zip(old, new):
            self.assertEqual(left["MCL"], right["MCL"])
            self.assertEqual(left["Synovitis"], right["Synovitis"])

    def test_success_subset_policy_retains_failed_uid_and_uses_other_uids(self):
        result, before, actual = self.run_extension("structural", blend_policy="success-subset")
        self.assertTrue(result["complete"])
        self.assertFalse(result["all_native_disabled"])
        self.assertTrue(result["native_contribution_used"])
        self.assertEqual(result["blend_policy"], "success-subset")
        old, new = [list(csv.DictReader(io.StringIO(body.decode()))) for body in (before, actual)]
        self.assertEqual(new[1], old[1])
        self.assertNotEqual(new[0]["ACL"], old[0]["ACL"])
        for left, right in zip(old, new):
            self.assertEqual(left["MCL"], right["MCL"])
            self.assertEqual(left["Synovitis"], right["Synovitis"])


class SourceIntegrityTests(unittest.TestCase):
    def test_spawn_unpickling_uses_sealed_router_after_common_path_is_prepended(self):
        # The parent cache hides this collision. Reproduce it in a fresh Python,
        # then actually unpickle the named task in two fresh spawn processes.
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            correct, shadow = base / "serving", base / "common"
            correct.mkdir()
            shadow.mkdir()
            good_source = "def prepare_hidden_task(item, *, policy):\n    return 'sealed-router'\n"
            (correct / "serving_router.py").write_text(good_source, encoding="utf-8", newline="\n")
            (shadow / "serving_router.py").write_text(
                "def prepare_hidden_task(item):\n    return 'wrong-common-router'\n", encoding="utf-8")
            runner = base / "spawn_fixture.py"
            runner.write_text(textwrap.dedent('''\
                import functools
                import importlib.machinery
                import importlib.util
                import json
                import multiprocessing
                import sys
                from pathlib import Path

                def invoke(connection, task):
                    import serving_router
                    try:
                        result = {"value": task(("synthetic",))}
                    except Exception as error:
                        result = {"error_type": type(error).__name__}
                    result["origin"] = serving_router.__file__
                    connection.send(result)
                    connection.close()

                def spawned(task):
                    context = multiprocessing.get_context("spawn")
                    receive, send = context.Pipe(duplex=False)
                    process = context.Process(target=invoke, args=(send, task))
                    process.start()
                    send.close()
                    try:
                        if not receive.poll(10):
                            raise RuntimeError("Artificial spawn did not return")
                        result = receive.recv()
                        process.join(10)
                        if process.exitcode != 0:
                            raise RuntimeError("Artificial spawn did not exit successfully")
                        return result
                    finally:
                        receive.close()
                        if process.is_alive():
                            process.terminate()
                            process.join(5)
                        process.close()

                if __name__ == "__main__":
                    adapter_path, correct, shadow, expected_sha = sys.argv[1:]
                    sys.path.insert(0, correct)
                    import serving_router
                    cached_origin = serving_router.__file__
                    task = functools.partial(serving_router.prepare_hidden_task, policy={"synthetic": True})
                    sys.path.insert(0, shadow)
                    before_spec = importlib.machinery.PathFinder.find_spec("serving_router", sys.path).origin
                    before = spawned(task)
                    spec = importlib.util.spec_from_file_location("artificial_p032_adapter", adapter_path)
                    adapter = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(adapter)
                    receipt = adapter.pin_spawn_router_source(correct, expected_sha)
                    after = spawned(task)
                    print(json.dumps({"cached_origin": cached_origin, "before_spec": before_spec,
                                      "before": before, "after": after, "binding": receipt}))
                '''), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-B", str(runner), str(ROOT / "scripts/p032_native_child.py"),
                 str(correct), str(shadow), hashlib.sha256(good_source.encode()).hexdigest()],
                capture_output=True, text=True, check=True, timeout=35,
            )
            receipt = json.loads(result.stdout)
            self.assertEqual(Path(receipt["cached_origin"]), correct / "serving_router.py")
            self.assertEqual(Path(receipt["before_spec"]), shadow / "serving_router.py")
            self.assertEqual(Path(receipt["before"]["origin"]), shadow / "serving_router.py")
            self.assertEqual(receipt["before"]["error_type"], "TypeError")
            self.assertEqual(receipt["after"]["value"], "sealed-router")
            self.assertEqual(Path(receipt["after"]["origin"]), correct / "serving_router.py")
            self.assertTrue(receipt["binding"]["spawn_sys_path_binding_verified"])

    def test_diagnostic_sampling_excludes_old64_and_uses_only_series_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            original, work = base / "competition", base / "work"
            original.mkdir()
            work.mkdir()
            temporary_root = base / "temporary"
            temporary_root.mkdir()
            with (original / "train_series.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, SERIES_COLUMNS)
                writer.writeheader()
                for key in range(1, 14):
                    for series in range(1, key % 4 + 2):
                        writer.writerow({ID: str(key), "SeriesInstanceUID": f"{key}.{series}",
                                         "Fluid_Sensitive": str(series % 2), "Fat_Suppression": "0",
                                         "Anatomical_Plane": ("sagittal", "coronal", "axial")[key % 3]})
            io = types.SimpleNamespace(ID=ID, TARGETS=TARGETS, SERIES_COLUMNS=SERIES_COLUMNS,
                                       resolve_test_input=lambda _: (original, [], [], {}, {}), sha=lambda _: "metadata-sha")
            c = types.SimpleNamespace(read_pin=lambda _: ({"source_cohort": {"uids": ["1", "2"]}}, b""))
            # Windows may not permit symlinks. This fixture verifies the exact
            # proposed target and creates an empty stand-in; no DICOM is read.
            targets = []

            def fake_link(path, target, **kwargs):
                targets.append(target)
                path.mkdir()

            with patch.object(native, "_p18_contracts", c, create=True), \
                    patch.object(native, "_p18_source", base, create=True), \
                    patch.object(Path, "symlink_to", fake_link):
                root, baseline, cleanup = native.diagnostic_fixture(
                    {"io": io, "p": types.SimpleNamespace(COHORTS={"native64": "pin"})}, work, 8,
                    temporary_root=temporary_root)
            result = json.loads((work / "diagnostic-selection.json").read_text())
            self.assertEqual(result["studies"], 8)
            self.assertFalse(set(result["UIDs"]) & {"1", "2"})
            self.assertEqual(len(result["UIDs"]), len(set(result["UIDs"])))
            self.assertFalse(result["train_csv_Report_labels_read"])
            self.assertFalse(result["Gold_labels_fit_select_or_evaluate"])
            self.assertFalse(result["MRI_directory_link_under_Output_working"])
            self.assertNotIn(work, root.parents)
            self.assertEqual(targets, [original / "train_series"])
            self.assertTrue(baseline.is_file())
            self.assertTrue((root / "rsna-knee-abnormality-detection/test.csv").is_file())
            cleanup()
            self.assertFalse(root.exists())

    def test_recovered_scientific_cells_and_all_native_sources_verified_offline(self):
        parent = builder.RECOVERY / "P029M-source.ipynb"
        native = builder.RECOVERY / "F006-native64-source.ipynb"
        if not parent.is_file() or not native.is_file():
            self.skipTest("Ignored recovered Kaggle source is not present in this checkout")
        _, review = builder.inspect_parent(parent)
        _, component = builder.inspect_native(native)
        self.assertTrue(review["all_seven_match_archived_original_after_single_mount_branch_normalization"])
        self.assertEqual(set(component["model_fingerprints"]), {"CLS:0", "CLS:1", "REG:0", "REG:1"})
        self.assertEqual(component["model_fingerprints"]["CLS:0"]["selected_epoch"], 12)
        self.assertEqual(component["model_fingerprints"]["CLS:1"]["selected_epoch"], 12)


class ObservabilityTests(unittest.TestCase):
    def test_log_relay_caps_new_bytes_and_never_replays_previous_output(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "child.log"
            path.write_bytes(b"x" * 2600)
            console = io.StringIO()
            with redirect_stdout(console):
                offset = extension.relay_child_log(path, 0)
                self.assertEqual(extension.relay_child_log(path, offset), offset)
                with path.open("ab") as stream:
                    stream.write(b"next-event\n")
                offset = extension.relay_child_log(path, offset)
            events = [json.loads(line.split(" ", 1)[1]) for line in console.getvalue().splitlines()]
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0]["data"], "x" * 2000)
            self.assertEqual(events[0]["omitted_new_bytes"], 600)
            self.assertEqual(events[1]["data"], "next-event\n")
            self.assertEqual(events[1]["omitted_new_bytes"], 0)
            self.assertEqual(offset, path.stat().st_size)

    def watch_fixture(self, *, duration, budget):
        clock, signals = [0.0], []
        console = io.StringIO()

        class Process:
            pid = 321
            returncode = None

            def __init__(self, argv, *, stdout, **kwargs):
                self.stream = stdout
                self.stream.write(b"bootstrap-begin\n")
                self.stream.flush()

            def poll(self):
                return self.returncode

            def wait(self, timeout):
                if self.returncode is not None:
                    return self.returncode
                clock[0] += timeout
                if clock[0] >= duration:
                    self.stream.write(b"native-done\n")
                    self.stream.flush()
                    self.returncode = 0
                    return 0
                raise extension.subprocess.TimeoutExpired(["fixture"], timeout)

        processes = []

        def launch(*args, **kwargs):
            process = Process(*args, **kwargs)
            processes.append(process)
            return process

        def kill_group(pid, signal):
            self.assertEqual(pid, 321)
            process = processes[0]
            if process.returncode is not None:
                raise ProcessLookupError
            if signal:
                signals.append(signal)
                process.returncode = -signal

        timer = types.SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda value: clock.__setitem__(0, clock[0] + value))
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(extension, "time", timer), \
                patch.object(extension.subprocess, "Popen", side_effect=launch), \
                patch.object(extension.os, "killpg", side_effect=kill_group, create=True), \
                redirect_stdout(console):
            result = extension.run_child(["fixture"], Path(folder) / "child.log", budget, mode="diagnostic")
        events = [(line.split(" ", 1)[0], json.loads(line.split(" ", 1)[1]))
                  for line in console.getvalue().splitlines()]
        return result, events, signals

    def test_watchdog_emits_live_heartbeat_and_reports_real_completion(self):
        result, events, signals = self.watch_fixture(duration=32, budget=50)
        self.assertTrue(result["complete"])
        self.assertFalse(result["timeout"])
        self.assertEqual(result["returncode"], 0)
        beats = [event for kind, event in events if kind == "P032_CHILD_HEARTBEAT"]
        self.assertEqual([event["elapsed_seconds"] for event in beats], [0.0, 30.0])
        self.assertTrue(all(event["status"] == "WAITING_FOR_CHILD" for event in beats))
        logs = [event["data"] for kind, event in events if kind == "P032_CHILD_LOG"]
        self.assertEqual(logs, ["bootstrap-begin\n", "native-done\n"])
        self.assertEqual(signals, [])

    def test_watchdog_timeout_stays_failure_and_signals_only_owned_group(self):
        result, events, signals = self.watch_fixture(duration=500, budget=31)
        self.assertFalse(result["complete"])
        self.assertTrue(result["timeout"])
        self.assertEqual(result["returncode"], -extension.signal.SIGTERM)
        self.assertTrue(result["joined"])
        self.assertTrue(result["owned_process_group_empty"])
        self.assertEqual(signals, [extension.signal.SIGTERM])
        logs = [event["data"] for kind, event in events if kind == "P032_CHILD_LOG"]
        self.assertEqual(logs, ["bootstrap-begin\n"])

    def test_diagnostic_budget_is_one_hour_and_failure_creates_no_submission(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            work = base / "working"
            work.mkdir()
            console = io.StringIO()
            cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
                                         memory_allocated=lambda _: 0, memory_reserved=lambda _: 0,
                                         device=lambda _: nullcontext(), empty_cache=lambda: None)

            def mapped(value):
                text = str(value)
                prefix = "/kaggle/working"
                return work / text[len(prefix):].lstrip("/") if text.startswith(prefix) else Path(value)

            with patch.object(extension, "Path", side_effect=mapped), \
                    patch.object(extension, "run_child", return_value={"complete": False}) as child, \
                    patch("multiprocessing.active_children", return_value=[]), \
                    patch.dict(sys.modules, {"torch": types.SimpleNamespace(cuda=cuda)}), \
                    redirect_stdout(console):
                result = extension.execute({}, runtime=base / "runtime", started=extension.time.monotonic(),
                                           mode="diagnostic")
            self.assertEqual(child.call_args.args[2], 3600)
            self.assertEqual(child.call_args.kwargs, {"mode": "diagnostic"})
            self.assertEqual(result["native_stage_budget_limit_seconds"], 3600)
            self.assertFalse(result["complete"])
            self.assertFalse(result["native_complete"])
            self.assertTrue(result["no_submission_csv_written"])
            self.assertFalse((work / "submission.csv").exists())
            events = [json.loads(line.split(" ", 1)[1]) for line in console.getvalue().splitlines()
                      if line.startswith("P032_PARENT_PROGRESS ")]
            self.assertEqual([(event["stage"], event["state"]) for event in events], [
                ("execute", "begin"), ("import_torch", "begin"), ("import_torch", "end"),
                ("CUDA_available", "begin"), ("CUDA_available", "end"),
                ("CUDA_device_count", "begin"), ("CUDA_device_count", "end"),
                ("GPU_cleanup", "begin"), ("GPU_cleanup", "end"),
                ("GPU_cleanup", "begin"), ("GPU_cleanup", "end"),
                ("native_child", "begin"), ("native_child", "end"),
            ])
            self.assertTrue(all(event["mode"] == "diagnostic" for event in events))
            self.assertEqual(events[6]["device_count"], 2)
            self.assertEqual(events[-1]["complete"], False)

    def test_child_fault_tracebacks_are_cancelled_when_execution_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            argv = ["p032-native", "--mode", "diagnostic", "--output", str(Path(folder) / "result.json"),
                    "--budget-seconds", "3600"]
            with patch.object(sys, "argv", argv), \
                    patch.object(native, "run", side_effect=RuntimeError("fixture failure")), \
                    patch("faulthandler.enable") as enable, \
                    patch("faulthandler.dump_traceback_later") as later, \
                    patch("faulthandler.cancel_dump_traceback_later") as cancel:
                with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                    native.main()
            enable.assert_called_once_with()
            later.assert_called_once_with(120, repeat=True)
            cancel.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
