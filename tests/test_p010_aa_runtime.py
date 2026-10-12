"""Artificial routing/identity/persistence checks; no MRI, real weights or API."""

import ast
import importlib.util
import sys
import tempfile
import threading
import unittest
from multiprocessing.pool import ThreadPool
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


transport = load("_p010_AA_transport_test", ROOT / "scripts/p010_aa_transport.py")
shared = load("_p010_AA_shared_test", ROOT / "scripts/p010_shared_runtime.py")
with mock.patch.dict(sys.modules, {"p010_aa_transport": transport}):
    runtime = load("_p010_AA_runtime_test", ROOT / "scripts/p010_aa_runtime.py")
builder = load("_p010_AA_builder_test", ROOT / "scripts/build_p010_profile_v4.py")
dependencies = load("_p010_AA_dependencies_test", ROOT / "tests/test_p010_profile.py")
with mock.patch.dict(
    sys.modules,
    {
        "p006_metrics": dependencies.p006,
        "p007_metrics": dependencies.p007,
        "p009_metrics": dependencies.p009,
        "p010_shared_metrics": dependencies.shared,
        "p010_author224_metrics": dependencies.author,
    },
):
    previous = load("_p010_AA_previous_test", ROOT / "scripts/p010_profile_runtime_v3.py")


class AARuntimeTest(unittest.TestCase):
    def test_fixed1307_studies_exact_once_owned_devices_and_bounds(self):
        identities = [f"artificial-{index}" for index in range(1307)]
        devices = ["fake0", "fake1"]
        assignment = {uid: devices[index % 2] for index, uid in enumerate(identities)}
        consumed, lock = [], threading.Lock()

        def consume(device, payload):
            with lock:
                consumed.append((payload[0], device))

        receipt = transport.fixed_pipeline(shared.run_bounded_pipeline, assignment)(
            lambda: ThreadPool(4),
            lambda uid: (uid, bytearray(4096)),
            identities,
            devices,
            consume,
            queue_max=5,
            outstanding_limit=7,
            shutdown_seconds=3,
        )
        self.assertEqual(dict(consumed), assignment)
        self.assertEqual(len(consumed), len(identities))
        self.assertLessEqual(receipt["max_inference_queue"], 5)
        self.assertLessEqual(receipt["max_outstanding"], 7)
        self.assertEqual(receipt["delivery_unfinished_tasks"], 0)
        self.assertEqual(receipt["queue_unfinished_tasks"], 0)
        self.assertTrue(receipt["consumer_threads_joined"])

    def test_owned_consumer_failure_drains_and_retains_receipt(self):
        values = [f"artificial-{index}" for index in range(100)]
        assignment = {uid: f"fake{index % 2}" for index, uid in enumerate(values)}

        def consume(_device, payload):
            if payload[0] == values[0]:
                raise RuntimeError("artificial consumer fault")

        with self.assertRaises(shared.PipelineFailure) as caught:
            transport.fixed_pipeline(shared.run_bounded_pipeline, assignment)(
                lambda: ThreadPool(2),
                lambda uid: (uid, None),
                values,
                ["fake0", "fake1"],
                consume,
                queue_max=1,
                outstanding_limit=2,
                shutdown_seconds=3,
            )
        receipt = caught.exception.receipt
        self.assertEqual(receipt["delivery_unfinished_tasks"], 0)
        self.assertEqual(receipt["queue_unfinished_tasks"], 0)
        self.assertTrue(receipt["consumer_threads_joined"])
        self.assertTrue(receipt["cancelled"])
        self.assertIn("diagnostic_owned_queue_source_sha256", receipt)

    def test_owned_worker_failure_drains_all_queues(self):
        values = [f"artificial-{index}" for index in range(20)]
        assignment = {uid: f"fake{index % 2}" for index, uid in enumerate(values)}

        def task(uid):
            if uid == values[1]:
                raise ValueError("artificial worker fault")
            return uid, None

        with self.assertRaises(shared.PipelineFailure) as caught:
            transport.fixed_pipeline(shared.run_bounded_pipeline, assignment)(
                lambda: ThreadPool(2),
                task,
                values,
                ["fake0", "fake1"],
                lambda *_: None,
                queue_max=1,
                outstanding_limit=2,
                shutdown_seconds=3,
            )
        self.assertEqual(caught.exception.receipt["delivery_unfinished_tasks"], 0)
        self.assertEqual(caught.exception.receipt["queue_unfinished_tasks"], 0)

    def test_queue_assignment_terminal_and_acknowledgement_refusal(self):
        queue = transport.OwnedQueue(["fake0", "fake1"], 2, {"a": "fake1"})
        queue.put(("a", 1))
        self.assertEqual(queue.get_for("fake1"), ("a", 1))
        queue.task_done()
        with self.assertRaises(ValueError):
            queue.task_done()
        with self.assertRaises(ValueError):
            queue.put(None)
        with self.assertRaises(ValueError):
            queue.put_terminal("wrong")
        queue.put_terminal("fake0")
        self.assertIsNone(queue.get_for("fake0"))
        queue.task_done()

    def test_two_simultaneous_consumers_keep_UIDs_and_devices_thread_local(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = runtime.PhaseRecorder(
                directory,
                runtime.PHASES[0][0],
                ["artificial-a", "artificial-b"],
                [f"label{index}" for index in range(12)],
                lambda: {"artificial": True},
            )
            barrier = threading.Barrier(2)

            def record(uid, device, value):
                recorder.begin(uid, None, None, device, {"artificial_input": uid})
                barrier.wait(timeout=2)
                recorder.complete([value] * 12, device)

            threads = [
                threading.Thread(target=record, args=("artificial-a", "fake0", 0.25)),
                threading.Thread(target=record, args=("artificial-b", "fake1", 0.75)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(3)
                self.assertFalse(thread.is_alive())
            self.assertEqual(recorder.raw["artificial-a"]["device"], "fake0")
            self.assertEqual(recorder.raw["artificial-b"]["device"], "fake1")
            self.assertEqual(recorder.raw["artificial-a"]["raw_probabilities"], [0.25] * 12)
            self.assertEqual(recorder.raw["artificial-b"]["raw_probabilities"], [0.75] * 12)

    def test_completed_phase_persisted_before_later_comparison_failure(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(sys.modules, {"p010_shared_metrics": shared}):
            recorder = runtime.PhaseRecorder(
                directory,
                runtime.PHASES[0][0],
                ["artificial-a", "artificial-b"],
                [f"label{index}" for index in range(12)],
                lambda: {"artificial": True},
            )
            for uid, value in [("artificial-a", 0.25), ("artificial-b", 0.75)]:
                recorder.begin(uid, None, None, "fake0", {"artificial_input": uid})
                recorder.complete([value] * 12, "fake0")
            result = recorder.finish(state={"artificial": True}, resources={}, peaks=[])
            receipt = Path(directory) / f"P010_AA_{recorder.phase}.json"
            original = receipt.read_bytes()
            with self.assertRaises(AssertionError):
                assert False, "artificial later mismatch"
            self.assertEqual(receipt.read_bytes(), original)
            self.assertEqual(result["status"], "phase_complete_before_comparison")
            self.assertTrue(result["qualification_guard_not_yet_evaluated"])
            self.assertFalse(result["candidate_gate_passed"])
            with self.assertRaises(ValueError):
                recorder.finish()
            with self.assertRaises(FileExistsError):
                runtime.PhaseRecorder(directory, recorder.phase, ["artificial-a"], recorder.labels, lambda: {})

    def test_partial_failure_raw_already_saved_and_missing_UIDs_recorded(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(sys.modules, {"p010_shared_metrics": shared}):
            recorder = runtime.PhaseRecorder(
                directory,
                runtime.PHASES[0][0],
                ["artificial-a", "artificial-b"],
                [f"label{index}" for index in range(12)],
                lambda: {"artificial": True},
            )
            recorder.begin("artificial-a", None, None, "fake1", {"artificial_input": True})
            recorder.complete([0.125] * 12, "fake1")
            self.assertTrue((recorder.directory / "study-000000.json").exists())
            result = recorder.finish(exception=RuntimeError("artificial failed second study"), resources={}, peaks=[])
            self.assertEqual(result["status"], "phase_failed")
            self.assertEqual(result["completed_UIDs"], ["artificial-a"])
            self.assertEqual(len(result["missing_UID_sha256"]), 1)
            self.assertEqual(result["exception"]["type"], "RuntimeError")
            self.assertTrue((Path(directory) / result["raw_csv"]["path"]).exists())

    def test_float32_probabilities_reject_nonfinite_badsize_and_badidentity(self):
        for values in ([0.5] * 11, [float("nan")] * 12, [float("inf")] * 12, [1.01] * 12):
            with self.assertRaises(ValueError):
                runtime.float32_values(values)
        with tempfile.TemporaryDirectory() as directory:
            recorder = runtime.PhaseRecorder(
                directory, runtime.PHASES[0][0], ["artificial-a"], [f"label{index}" for index in range(12)], lambda: {}
            )
            with self.assertRaises(ValueError):
                recorder.begin("unknown", None, None, "fake0", {})
            recorder.begin("artificial-a", None, None, "fake0", {})
            with self.assertRaises(ValueError):
                recorder.complete([0.5] * 12, "wrongdevice")

    def test_observer_preserves_all_original_numerical_definitions(self):
        book, inherited = builder.previous.make_profile()
        self.assertEqual(len(book["cells"]), 19)
        with mock.patch.dict(sys.modules, {"p010_previous_metrics": previous}):
            for key, original_transport in [("original", True), ("bounded", False)]:
                original = inherited[key]
                observed = runtime.observed_main_source(
                    original, runtime.PHASES[0][0], original_transport=original_transport
                )

                def definitions(source):
                    lines = source.splitlines(keepends=True)
                    return {
                        node.name: "".join(
                            lines[
                                min([node.lineno, *[item.lineno for item in node.decorator_list]]) - 1 : node.end_lineno
                            ]
                        )
                        for node in ast.parse(source).body
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name != "main"
                    }

                originals, observations = definitions(original), definitions(observed)
                self.assertTrue(all(observations[name] == value for name, value in originals.items()))
                self.assertIn("_AA_recorder.complete(value.tolist(),args[3])", observed)
                self.assertIn("torch.backends.cudnn.benchmark = True", observed)

    def test_builder_keeps19_previous_cells_except_explicit_changes_and_all5_literals(self):
        book, inherited = builder.make_profile()
        previous_book, _ = builder.previous.make_profile()
        self.assertEqual(len(book["cells"]), 21)
        self.assertEqual(sum(cell["cell_type"] == "code" for cell in book["cells"]), 18)
        for index, cell in enumerate(previous_book["cells"]):
            if index not in (0, 10, 18):
                self.assertIn(cell, book["cells"])
        for name in ("original", "bounded", "shared", "reader", "fusion"):
            self.assertIn(repr(inherited[name]), builder.serving.text(book["cells"][-1]))
        self.assertIn('"status":"diagnostic_complete"', builder.serving.text(book["cells"][-1]))
        self.assertEqual(inherited["helpers"]["p010_previous_metrics.py"], builder.previous.HELPER)


if __name__ == "__main__":
    unittest.main()
