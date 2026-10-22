"""Fake Popen/clock and synthetic artifacts; no actual signal/GPU/model/MRI."""
import csv
import importlib.util
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rsna_knee import orthodiff_inner_c_supervisor as S  # noqa: E402


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


B = module("supervisor_fake_builder", "scripts/prepare_orthodiffusion_inner_c_test.py")
F = module("supervisor_fake_data", "tests/test_orthodiffusion_inner_c_test.py")


class Clock:
    def __init__(self):
        self.now = 100.

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Process:
    def __init__(self, pid, status, events):
        self.pid, self.status, self.events = pid, status, events

    def wait(self, timeout):
        self.events.append(("wait", self.pid, timeout))
        if self.status is None:
            raise subprocess.TimeoutExpired("FAKE", timeout)
        return self.status

    def kill(self):
        self.events.append(("direct_owned_child_kill", self.pid))
        self.status = -9


class Operations:
    def __init__(self, events, processes):
        self.events, self.processes = events, processes

    def ids(self, pid):
        return pid, pid

    def own_group(self):
        return 998

    def peek(self, process):
        self.events.append(("peek_without_reap", process.pid))
        if isinstance(process.status, BaseException):
            raise process.status
        return process.status

    def group_signal(self, pid, value):
        self.events.append(("signal", pid, int(value)))
        if int(value) == 9:
            self.processes[pid].status = -9


def write(path, value):
    S.save_json(path, value)
    return S.pin(path)["sha256"]


def setup(folder):
    pack = folder / "source"
    manifest = B.build(pack, True)
    competition = folder / "competition"
    competition.mkdir()
    (competition / "test.csv").write_text("StudyInstanceUID,UNUSED\nu3,POISON\nu1,POISON\nu2,POISON\n", encoding="utf-8")
    parent = folder / "parent.csv"
    with parent.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([S.ID, *S.TARGETS])
        writer.writerows([[uid, *["0.50000"] * 12] for uid in ("u1", "u2", "u3")])
    model = folder / "FAKE-NOT-A-MODEL.json"
    model.write_bytes(b"ARTIFICIAL_OPAQUE_FILE_NOT_A_REAL_MODEL")
    q = folder / "FAKE-NOT-AN-ACTUAL-QUALIFICATION.json"
    q_sha = write(q, {"schema_version": "orthodiff_innerC_numpy_transport_qualification_v1",
        "passed": True, "scope": "actual_sealed_single_model", "feature_transport_bundle_sha256": manifest["bundle"]["sha256"],
        "bindings": {"model_sha256": S.MODEL_SHA}, "actual_model_qualified": True,
        "runtime_numerical_audit_passed": True, "qualified_numpy_version": "2.1.3",
        "outer2_engineering_and_fixed10_positive": True, "root_candidate_approved": True,
        "outer2_result_sha256": "a" * 64, "root_candidate_review_sha256": "b" * 64})
    return S.BoundRun(pack / "bundle.json", manifest["bundle"]["sha256"], model, q, q_sha,
                      competition, parent, S.pin(parent)["sha256"], folder / "owned-output",
                      S.TimeBudget(100., 100., 10., 10., 1., 2., .1), 8 * 1024**2)


class Spawn:
    def __init__(self, spec, feature="normal", head="normal"):
        self.spec, self.feature, self.head = spec, feature, head
        self.events, self.processes, self.argvs = [], {}, []

    def __call__(self, argv, **kwargs):
        self.argvs.append(argv)
        assert argv[:3] == [sys.executable, "-I", str(self.spec.runtime.absolute())]
        assert kwargs["start_new_session"] is True and kwargs["shell"] is False and kwargs["close_fds"] is True
        assert kwargs["stdin"] == subprocess.DEVNULL
        mode = argv[3]
        case = self.feature if mode == "features" else self.head
        pid = 1000 + len(self.processes)
        process = Process(pid, None if case == "timeout" else 7 if case == "nonzero" else 0, self.events)
        if case in {"interrupt", "exit"}:
            process.status = KeyboardInterrupt() if case == "interrupt" else SystemExit(2)
        self.processes[pid] = process
        target = Path(argv[argv.index("--output-dir") + 1])
        if mode == "features" and case not in {"timeout", "nonzero"}:
            target.mkdir()
            records = [F.fake_success("u3"), F.fake_success("u1", .5),
                       F.R.make_failure("u2", None, ValueError("synthetic_PD_missing"), 0.)]
            receipt = F.R.write_feature_handoff(target, ["u3", "u1", "u2"], records, F.np,
                                                self.spec.bundle_sha256, {})
            if case == "incomplete":
                receipt["complete"] = False
            if case == "record_sha":
                receipt["records"][0]["regional_2x2"]["sha256"] = "0" * 64
            if case in {"incomplete", "record_sha"}:
                write(target / "feature-receipt.json", receipt)
            write(target / "execution-receipt.json", {"schema_version": "orthodiff_innerC_test_feature_execution_v1",
                "complete": True, "model_load_calls": 1, "owned_scientific_temp_removed": True,
                "resources": {"elapsed_seconds": .25, "CUDA_reserved_bytes": 0, "RSS_bytes": 0},
                "handoff": {"receipt": S.pin(target / "feature-receipt.json"), "success": 2, "failure": 1}})
            if case == "matrix_sha":
                (target / "success-features.npz").write_bytes(b"tampered")
            if case == "parent_changed":
                Path(self.spec.parent_csv).write_bytes(b"MUST_BE_RESTORED_EXACTLY")
        elif mode == "head" and case not in {"timeout", "nonzero"}:
            target.mkdir()
            feature = Path(argv[argv.index("--feature-dir") + 1])
            with (target / "head-success.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow([S.ID, *S.TARGETS])
                writer.writerows([[uid, *["0.25"] * 12] for uid in ("u3", "u1")])
            write(target / "head-receipt.json", {"schema_version": "orthodiff_innerC_test_head_predictions_v1",
                "complete": True, "bundle_sha256": self.spec.bundle_sha256, "model_sha256": S.MODEL_SHA,
                "qualification_sha256": self.spec.qualification_sha256,
                "feature_receipt": S.pin(feature / "feature-receipt.json"),
                "UID_order_sha256": S.uid_hash(["u3", "u1", "u2"]), "success_uids": ["u3", "u1"],
                "failure_uids": ["u2"], "numpy": "2.1.3", "parent_selected_or_overlaid": False,
                "Public_submission_contract_qualified": False, "probabilities": S.pin(target / "head-success.csv")})
            if case == "matrix_sha":
                with (target / "head-success.csv").open("ab") as stream:
                    stream.write(b"bad")
        return process

    def run(self):
        clock = Clock()
        return S._pipeline(self.spec, spawn=self, ops=Operations(self.events, self.processes), clock=clock, sleep=clock.sleep)


class SupervisorTests(unittest.TestCase):
    def test_normal_fixed_source_feature_then_head_complete_and_parent_untouched(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = setup(Path(temporary))
            raw = spec.parent_csv.read_bytes()
            spawn = Spawn(spec)
            result = spawn.run()
            self.assertTrue(result["complete"])
            self.assertTrue(result["head_artifacts_verified"])
            self.assertFalse(result["Public_submission_contract_qualified"])
            self.assertEqual([argv[3] for argv in spawn.argvs], ["features", "head"])
            self.assertEqual(float(spawn.argvs[0][-1]), 77.)
            self.assertEqual(result["automatic_retry_count"], 0)
            self.assertEqual(spec.parent_csv.read_bytes(), raw)
            self.assertTrue(all(trace["leader_reaped"] for trace in result["children"]))
            self.assertFalse(any(item[0] == "signal" for item in spawn.events))

    def test_nonzero_timeout_incomplete_and_hash_failures_stop_before_head(self):
        for case in ("nonzero", "timeout", "incomplete", "matrix_sha", "record_sha", "interrupt", "exit"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                spec = setup(Path(temporary))
                raw, spawn = spec.parent_csv.read_bytes(), Spawn(spec, feature=case)
                result = spawn.run()
                self.assertFalse(result["complete"])
                self.assertFalse(result["head_artifacts_verified"])
                self.assertTrue(result["upper_driver_must_fail_when_incomplete"])
                self.assertEqual([argv[3] for argv in spawn.argvs], ["features"])
                signals = [item[2] for item in spawn.events if item[0] == "signal"]
                self.assertEqual(signals, [15, 9])
                self.assertEqual(spawn.events[-1][0], "wait")
                self.assertTrue(result["children"][0]["leader_reaped"])
                self.assertEqual(result["automatic_retry_count"], 0)
                self.assertEqual(spec.parent_csv.read_bytes(), raw)

    def test_head_failure_or_parent_corruption_never_becomes_parent_only_success(self):
        for kwargs in ({"head": "nonzero"}, {"head": "timeout"}, {"head": "matrix_sha"}, {"feature": "parent_changed"}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as temporary:
                spec = setup(Path(temporary))
                raw, spawn = spec.parent_csv.read_bytes(), Spawn(spec, **kwargs)
                result = spawn.run()
                self.assertFalse(result["complete"])
                self.assertFalse(result["head_artifacts_verified"])
                self.assertEqual(spec.parent_csv.read_bytes(), raw)
                if "feature" in kwargs:
                    self.assertTrue(result["parent_restored"])
                    self.assertEqual(len(spawn.argvs), 1)
                with self.assertRaises(S.SupervisionFailure):
                    fake_os = types.SimpleNamespace(name="posix", **{name: True for name in (
                        "waitid", "WNOWAIT", "WEXITED", "WNOHANG", "P_PID", "getpgid", "getsid", "killpg")})
                    with mock.patch.object(S, "os", fake_os), mock.patch.object(S, "_pipeline", return_value=result):
                        S.supervise(spec)

    def test_binding_reserve_and_Windows_reject_before_launch(self):
        with mock.patch.object(S, "os", types.SimpleNamespace(name="nt")):
            with self.assertRaisesRegex(ValueError, "POSIX"):
                S.supervise(None)
        with tempfile.TemporaryDirectory() as temporary:
            spec = setup(Path(temporary))
            spawn = Spawn(spec)
            spec.runtime.write_bytes(spec.runtime.read_bytes() + b"\n# wrong-source\n")
            with self.assertRaisesRegex(ValueError, "frozen innerC"):
                spawn.run()
            self.assertEqual(spawn.argvs, [])
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            S.TimeBudget(100., 20., 10., 10., 1., 2., .1).allowance("features", 100.)
        self.assertNotIn("torch", sys.modules)

    def test_wrong_owned_group_never_gets_a_group_signal(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = setup(Path(temporary))
            spawn = Spawn(spec)
            ops = Operations(spawn.events, spawn.processes)
            ops.ids = lambda pid: (pid + 100, pid + 100)
            clock = Clock()
            result = S._pipeline(spec, spawn=spawn, ops=ops, clock=clock, sleep=clock.sleep)
            self.assertFalse(result["complete"])
            self.assertFalse(any(event[0] == "signal" for event in spawn.events))
            self.assertTrue(any(event[0] == "direct_owned_child_kill" for event in spawn.events))
            self.assertEqual(spawn.events[-1][0], "wait")


if __name__ == "__main__":
    unittest.main()
