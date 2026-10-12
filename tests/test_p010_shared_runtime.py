"""No MRI, labels, weights, GPU training or remote service in transport/CSV tests."""

import csv
import importlib.util
import io
import queue
import sys
import tempfile
import threading
import time
import unittest
from multiprocessing.pool import ThreadPool
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p010_shared_runtime", ROOT / "scripts/p010_shared_runtime.py")
RUNTIME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME)
LABELS = [f"finding_{index}" for index in range(12)]


class Payload:
    live = 0
    peak = 0
    lock = threading.Lock()

    def __init__(self, identity):
        self.identity = identity
        self.synthetic_bytes = bytearray(4096)
        with self.lock:
            type(self).live += 1
            type(self).peak = max(type(self).peak, type(self).live)

    def __del__(self):
        with self.lock:
            type(self).live -= 1


class P010TransportTest(unittest.TestCase):
    def test_cancel_with_held_delivery_payload_still_acknowledges_both_queues(self):
        actual_queue = queue.Queue
        queues = []
        allow_failure = threading.Event()

        class CapturingQueue(actual_queue):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.delivery = not queues
                queues.append(self)

            def get(self, *args, **kwargs):
                value = super().get(*args, **kwargs)
                if self.delivery and value == (True, 1):
                    allow_failure.set()
                    time.sleep(0.05)
                return value

        def consume(_device, value):
            if value == 0:
                if not allow_failure.wait(2):
                    raise AssertionError("Held delivery coordination did not occur")
                raise RuntimeError("Synthetic consumer fault with producer-held next payload")

        with mock.patch.object(RUNTIME.queue, "Queue", CapturingQueue):
            with self.assertRaises(RUNTIME.PipelineFailure) as caught:
                RUNTIME.run_bounded_pipeline(
                    lambda: ThreadPool(1),
                    lambda value: value,
                    range(8),
                    ["synthetic"],
                    consume,
                    queue_max=1,
                    outstanding_limit=2,
                    shutdown_seconds=2,
                )
        self.assertEqual([item.unfinished_tasks for item in queues], [0, 0])
        self.assertEqual(caught.exception.receipt["delivery_unfinished_tasks"], 0)
        self.assertEqual(caught.exception.receipt["queue_unfinished_tasks"], 0)
        self.assertTrue(caught.exception.receipt["cancelled"])

    def test_more_than_1300_studies_are_bounded_and_exactly_once(self):
        Payload.live = Payload.peak = 0
        consumed = []
        lock = threading.Lock()

        def consume(device, payload):
            time.sleep(0.0002)
            with lock:
                consumed.append((device, payload.identity))

        receipt = RUNTIME.run_bounded_pipeline(
            lambda: ThreadPool(4),
            Payload,
            range(1307),
            ["gpu0", "gpu1"],
            consume,
            queue_max=5,
            outstanding_limit=7,
            shutdown_seconds=3,
        )
        self.assertEqual(sorted(identity for _, identity in consumed), list(range(1307)))
        self.assertEqual(receipt["consumed"], 1307)
        self.assertLessEqual(receipt["max_outstanding"], 7)
        self.assertLessEqual(receipt["max_delivery_queue"], 7)
        self.assertLessEqual(receipt["max_inference_queue"], 5)
        # ThreadPool test workers retain up to four last-result references.
        # Real process workers and IPC serialization are explicitly outside
        # the parent payload-count bound and remain independently bounded.
        self.assertLessEqual(Payload.peak, receipt["prepared_parent_payload_upper_bound"] + 4)
        self.assertEqual(Payload.live, 0)
        self.assertEqual(receipt["queue_unfinished_tasks"], 0)
        self.assertTrue(receipt["consumer_threads_joined"])

    def test_worker_and_consumer_errors_cancel_drain_and_fail_closed(self):
        for origin in ("worker", "consumer"):
            with self.subTest(origin=origin):

                def task(identity):
                    if origin == "worker" and identity == 11:
                        raise RuntimeError("synthetic preparation fault")
                    return identity

                def consume(_device, identity):
                    if origin == "consumer" and identity == 9:
                        raise RuntimeError("synthetic expert fault")
                    time.sleep(0.0001)

                started = time.monotonic()
                with self.assertRaises(RUNTIME.PipelineFailure) as caught:
                    RUNTIME.run_bounded_pipeline(
                        lambda: ThreadPool(4),
                        task,
                        range(1307),
                        ["gpu0", "gpu1"],
                        consume,
                        queue_max=5,
                        outstanding_limit=7,
                        shutdown_seconds=3,
                    )
                self.assertLess(time.monotonic() - started, 3)
                receipt = caught.exception.receipt
                self.assertTrue(receipt["cancelled"])
                self.assertTrue(receipt["consumer_threads_joined"])
                self.assertEqual(receipt["queue_unfinished_tasks"], 0)
                self.assertEqual(receipt["errors"], 1)
                self.assertFalse(receipt["all_tasks_consumed"])

    def test_pool_creation_fault_and_invalid_inputs(self):
        def bad_pool():
            raise RuntimeError("synthetic pool startup fault")

        with self.assertRaises(RUNTIME.PipelineFailure) as caught:
            RUNTIME.run_bounded_pipeline(bad_pool, lambda value: value, [1], ["cpu"], lambda *_: None)
        self.assertTrue(caught.exception.receipt["consumer_threads_joined"])
        for items, devices, capacity in (([1, 1], ["cpu"], 1), ([1], [], 1), ([1], ["cpu"], 0)):
            with self.assertRaises(ValueError):
                RUNTIME.run_bounded_pipeline(
                    lambda: ThreadPool(1),
                    lambda value: value,
                    items,
                    devices,
                    lambda *_: None,
                    outstanding_limit=capacity,
                )


class P010CSVTest(unittest.TestCase):
    def test_fusion_success_guard_restores_stdout_and_rejects_fallback_or_duplicates(self):
        for value, passes in (
            ("[Apex Fusion SUCCESS] Successfully fused\n", True),
            ("[Apex Fusion FALLBACK]\n", False),
            ("[Apex Fusion EXCEPTION]\n[Apex Fusion SUCCESS] Successfully fused\n", False),
            ("[Apex Fusion SUCCESS] Successfully fused\n" * 2, False),
        ):
            with self.subTest(value=value), mock.patch.object(sys, "stdout", io.StringIO()) as stream:
                observer = RUNTIME.FusionObserver()
                print(value, end="")
                if passes:
                    self.assertTrue(observer.require_success()["original_fusion_success_verified"])
                else:
                    with self.assertRaises(RUNTIME.PipelineFailure):
                        observer.require_success()
                self.assertIs(sys.stdout, stream)
                self.assertEqual(stream.getvalue(), value)

    def test_average_ties_all_twelve_columns_and_atomic_baseline_retention(self):
        self.assertEqual(RUNTIME.average_percentile([3.0, 1.0, 1.0, 2.0]), [1.0, 0.375, 0.375, 0.75])
        uids = ["study_c", "study_a", "study_b", "study_d"]
        left = [[value] * 12 for value in (0.8, 0.1, 0.1, 0.4)]
        right = [[value] * 12 for value in (0.2, 0.9, 0.6, 0.5)]
        expected = [[value] * 12 for value in (1.0, 0.5, 0.25, 0.75)]
        self.assertEqual(RUNTIME.fixed_rank10(left, right), expected)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            base, expert = directory / "submission.csv", directory / "p010_expert_raw.csv"
            RUNTIME.write_submission_exclusive(base, uids, left, LABELS)
            before = base.read_bytes()
            RUNTIME.write_submission_exclusive(expert, uids, right, LABELS)
            receipt = RUNTIME.finalise_fixed_rank10(base, expert, LABELS, uids)
            self.assertEqual((directory / "p010_incumbent.csv").read_bytes(), before)
            self.assertEqual(RUNTIME.read_submission(base, LABELS, uids)[1], expected)
            self.assertEqual(receipt["coefficients_predeclared"], [0.9, 0.1])
            self.assertFalse((directory / "p010_candidate.partial.csv").exists())
            with self.assertRaises(FileExistsError):
                RUNTIME.finalise_fixed_rank10(base, expert, LABELS, uids)

    def test_single_study_skips_expert_and_contract_faults_preserve_base(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            base = directory / "submission.csv"
            RUNTIME.write_submission_exclusive(base, ["one"], [[0.5] * 12], LABELS)
            before = base.read_bytes()
            receipt = RUNTIME.finalise_fixed_rank10(base, directory / "missing_expert.csv", LABELS, ["one"])
            self.assertFalse(receipt["expert_used"])
            self.assertEqual(base.read_bytes(), before)
            for value in (float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    RUNTIME.average_percentile([value])
            with self.assertRaises(ValueError):
                RUNTIME.read_submission(base, LABELS[::-1], ["one"])
            with self.assertRaises(ValueError):
                RUNTIME.read_submission(base, LABELS, ["other"])
            with base.open("a", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerow(["two", *([0.5] * 12), "extra"])
            with self.assertRaises(ValueError):
                RUNTIME.read_submission(base, LABELS)


if __name__ == "__main__":
    unittest.main()
