"""Real CPU processes verify isolated UID failure, deadlines and bounded cleanup."""

import hashlib
import importlib.util
import multiprocessing
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from isolated_uid_fixtures import LARGE_BYTES, prepare_artificial_uid  # noqa: E402

from rsna_knee.isolated_uid_executor import (  # noqa: E402
    IsolatedUIDExecutor,
    IsolatedUIDMultiprocessingProxy,
    OperationalUIDFailure,
    TaskSourcePin,
)


def item(uid, mode="success"):
    return ("artificial-input", uid, mode, "artificial-settings", "must-be-replaced-by-owned-scratch")


class ExecutorTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.pools = []
        self.addCleanup(self.stop_pools)
        self.pin = TaskSourcePin.from_callable(prepare_artificial_uid)

    def stop_pools(self):
        for pool in self.pools:
            pool.terminate()
            pool.join()

    def pool(self, **kwargs):
        result = IsolatedUIDExecutor(self.pin, temporary_parent=self.temporary.name, **kwargs)
        self.pools.append(result)
        return result

    def assert_clean(self, pool):
        receipt = pool.receipt()
        self.assertTrue(receipt["monitor_joined"])
        self.assertTrue(receipt["temporary_root_cleaned"])
        self.assertFalse(pool.root.exists())
        self.assertFalse(receipt["internal_errors"])
        self.assertTrue(all(row["direct_child_reaped"] and row["owned_files_cleaned"] for row in receipt["records"]))
        self.assertTrue(all(row["callback_count"] <= 1 for row in receipt["records"]))
        self.assertLessEqual(receipt["peak_active"], receipt["worker_limit"])
        return receipt

class IsolatedUIDExecutorTests(ExecutorTestCase):
    def test_success_same_parent_group_exact_bytes_owned_scratch_and_single_callback(self):
        pool = self.pool(max_workers=2)
        callbacks, errors = [], []
        tickets = [pool.apply_async(prepare_artificial_uid, (item(str(i)),),
                                   callback=callbacks.append, error_callback=errors.append) for i in range(4)]
        pool.close()
        pool.join()
        values = [ticket.get(0) for ticket in tickets]
        self.assertEqual(len(callbacks), 4)
        self.assertFalse(errors)
        self.assertEqual({value[0] for value in values}, {"0", "1", "2", "3"})
        for uid, arrays, receipt in values:
            self.assertEqual(arrays["images"], b"artificial-array-bytes")
            self.assertEqual(receipt["scientific_value"], .125)
            self.assertEqual(receipt["parent_pid"], os.getpid())
            self.assertNotEqual(receipt["pid"], os.getpid())
            self.assertIn(pool.root, Path(receipt["workspace"]).parents)
            if hasattr(os, "getpgrp"):
                self.assertEqual(receipt["process_group"], os.getpgrp())
        self.assertEqual(len({value[2]["pid"] for value in values}), 4)
        self.assertEqual(self.assert_clean(pool)["peak_active"], 2)

    def test_large_native_sized_result_never_waits_for_an_undrained_pipe(self):
        pool = self.pool(max_workers=1, uid_deadline_seconds=20)
        start = time.monotonic()
        ticket = pool.apply_async(prepare_artificial_uid, (item("large", "large"),))
        pool.close()
        pool.join()
        _, arrays, receipt = ticket.get(0)
        self.assertEqual(receipt["payload_bytes"], LARGE_BYTES)
        expected = hashlib.sha256(bytes(range(256)) * (LARGE_BYTES // 256)).hexdigest()
        self.assertEqual(hashlib.sha256(arrays["images"]).hexdigest(), expected)
        self.assertLess(time.monotonic() - start, 20)
        self.assertFalse(self.assert_clean(pool)["uses_result_pipe"])

    def test_ordinary_task_exception_is_fatal_even_in_return_mode(self):
        for mode in ("fatal", "return"):
            with self.subTest(mode=mode):
                pool = self.pool(max_workers=1, operational_failure_mode=mode)
                callbacks, errors = [], []
                ticket = pool.apply_async(prepare_artificial_uid, (item(mode, "exception"),),
                                          callback=callbacks.append, error_callback=errors.append)
                pool.close()
                pool.join()
                with self.assertRaises(OperationalUIDFailure) as caught:
                    ticket.get(0)
                self.assertEqual(caught.exception.reason, "task_exception")
                self.assertIn("ValueError", caught.exception.detail)
                self.assertFalse(callbacks)
                self.assertEqual(len(errors), 1)
                self.assertFalse(caught.exception.receipt()["old_structural_failure_claimed"])
                self.assert_clean(pool)

    def test_five_numpy_arrays_keep_shapes_dtypes_and_all_bytes(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is not installed")
        pool = self.pool(max_workers=1, uid_deadline_seconds=20)
        ticket = pool.apply_async(prepare_artificial_uid, (item("arrays", "numpy"),))
        pool.close()
        pool.join()
        _, arrays, receipt = ticket.get(0)
        self.assertEqual(set(arrays), {"images", "mask", "positions", "planes", "flags"})
        self.assertEqual(receipt["payload_bytes"], LARGE_BYTES)
        for name, shape, dtype, expected in (
            ("images", (3, 48, 3, 392, 392), np.uint8, 17), ("mask", (3, 48), np.bool_, True),
            ("positions", (3, 48), np.float32, 0), ("flags", (3, 2), np.float32, 0)):
            self.assertEqual(arrays[name].shape, shape)
            self.assertEqual(arrays[name].dtype, dtype)
            self.assertTrue(np.all(arrays[name] == expected))
        self.assertEqual(arrays["planes"].dtype, np.int64)
        self.assertTrue(np.array_equal(arrays["planes"], np.arange(3, dtype=np.int64)))
        self.assert_clean(pool)

    def test_fresh_worker_rejects_cached_parent_shadow_module(self):
        shadow = Path(self.temporary.name) / "shadow"
        shadow.mkdir()
        marker = shadow / "unpinned-top-level-executed"
        (shadow / "isolated_uid_fixtures.py").write_text(
            "from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('artificial')\n"
            "def prepare_artificial_uid(item):\n    return 'wrong-module'\n", encoding="utf-8")
        pool = self.pool(max_workers=1)
        sys.path.insert(0, str(shadow))
        try:
            ticket = pool.apply_async(prepare_artificial_uid, (item("shadow"),))
        finally:
            sys.path.remove(str(shadow))
        pool.close()
        pool.join()
        with self.assertRaises(OperationalUIDFailure) as caught:
            ticket.get(0)
        self.assertEqual(caught.exception.reason, "task_exception")
        self.assertIn("origin/SHA differs", caught.exception.detail)
        self.assertFalse(marker.exists(), "Reject task source before executing its unpinned top level")
        self.assert_clean(pool)

    def test_owned_job_uid_matches_tuple_or_uid_record_before_success_callback(self):
        for mode in ("wrong_uid_tuple", "wrong_uid_failure", "correct_uid_record"):
            with self.subTest(mode=mode):
                pool = self.pool(max_workers=1, operational_failure_mode="return")
                callbacks, errors = [], []
                ticket = pool.apply_async(prepare_artificial_uid, (item("EXPECTED-UID", mode),),
                                          callback=callbacks.append, error_callback=errors.append)
                pool.close()
                pool.join()
                if mode.startswith("wrong_"):
                    with self.assertRaises(OperationalUIDFailure) as caught:
                        ticket.get(0)
                    self.assertEqual(caught.exception.reason, "transport_error")
                    self.assertIn("payload UID differs", caught.exception.detail)
                    self.assertFalse(callbacks)
                    self.assertEqual(len(errors), 1)
                else:
                    self.assertEqual(ticket.get(0).uid, "EXPECTED-UID")
                    self.assertEqual(len(callbacks), 1)
                    self.assertFalse(errors)
                self.assert_clean(pool)

    def test_hard_exit_and_zero_exit_missing_result_finish_without_lost_result_wait(self):
        for mode, expected in (("exit", "worker_exit"), ("lost", "missing_result")):
            with self.subTest(mode=mode):
                pool = self.pool(max_workers=1, uid_deadline_seconds=10)
                errors = []
                start = time.monotonic()
                ticket = pool.apply_async(prepare_artificial_uid, (item(mode, mode),), error_callback=errors.append)
                pool.close()
                pool.join()
                with self.assertRaises(OperationalUIDFailure) as caught:
                    ticket.get(0)
                self.assertEqual(caught.exception.reason, expected)
                self.assertEqual(len(errors), 1)
                self.assertLess(time.monotonic() - start, 10)
                self.assert_clean(pool)

    def test_deadline_returns_new_operational_type_and_next_uid_recovers(self):
        pool = self.pool(max_workers=1, uid_deadline_seconds=1.5, cleanup_seconds=.3,
                         operational_failure_mode="return")
        callbacks, errors = [], []
        hung = pool.apply_async(prepare_artificial_uid, (item("hung", "hang"),),
                               callback=callbacks.append, error_callback=errors.append)
        good = pool.apply_async(prepare_artificial_uid, (item("good"),),
                                callback=callbacks.append, error_callback=errors.append)
        pool.close()
        pool.join()
        failure = hung.get(0)
        self.assertIs(type(failure), OperationalUIDFailure)
        self.assertEqual(failure.uid, "hung")
        self.assertEqual(failure.reason, "deadline_exceeded")
        self.assertEqual(good.get(0)[0], "good")
        self.assertEqual(len(callbacks), 2)
        self.assertFalse(errors)
        receipt = self.assert_clean(pool)
        self.assertTrue(receipt["records"][0]["returned_operational_failure"])

    @unittest.skipIf(os.name == "nt", "Windows TerminateProcess has no ignorable SIGTERM")
    def test_ignored_sigterm_is_hard_killed_and_joined(self):
        pool = self.pool(max_workers=1, uid_deadline_seconds=1.5, cleanup_seconds=.15)
        ticket = pool.apply_async(prepare_artificial_uid, (item("hard", "ignore_term"),))
        pool.close()
        pool.join()
        with self.assertRaises(OperationalUIDFailure) as caught:
            ticket.get(0)
        self.assertEqual(caught.exception.returncode, -9)
        self.assert_clean(pool)

    def test_cancel_active_and_pending_jobs_reaps_all_and_suppresses_late_callbacks(self):
        pool = self.pool(max_workers=2, uid_deadline_seconds=20, cleanup_seconds=.3)
        callbacks, errors = [], []
        tickets = [pool.apply_async(prepare_artificial_uid, (item(str(i), "hang"),),
                                   callback=callbacks.append, error_callback=errors.append) for i in range(4)]
        deadline = time.monotonic() + 5
        while pool.receipt()["peak_active"] != 2 and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertEqual(pool.receipt()["peak_active"], 2)
        pool.terminate()
        pool.join()
        for ticket in tickets:
            with self.assertRaises(OperationalUIDFailure) as caught:
                ticket.get(0)
            self.assertEqual(caught.exception.reason, "cancelled")
        self.assertFalse(callbacks)
        self.assertFalse(errors)
        self.assertEqual(len(self.assert_clean(pool)["records"]), 4)
        pool.terminate()
        pool.join()

    def test_bound_four_uid_identity_and_pin_guard(self):
        with self.assertRaises(ValueError):
            self.pool(max_workers=5)
        with self.assertRaises(ValueError):
            TaskSourcePin.from_callable(prepare_artificial_uid, expected_sha256="0" * 64)
        pool = self.pool(max_workers=1)
        for i in range(4):
            pool.apply_async(prepare_artificial_uid, (item(str(i), "hang"),))
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            pool.apply_async(prepare_artificial_uid, (item("fifth"),))
        with self.assertRaisesRegex(ValueError, "only once"):
            pool.apply_async(prepare_artificial_uid, (item("0"),))
        pool.terminate()
        pool.join()
        self.assert_clean(pool)

    def test_callback_exception_does_not_kill_the_monitor_or_lose_other_results(self):
        pool = self.pool(max_workers=1)

        def broken(_):
            raise RuntimeError("artificial callback error")

        bad = pool.apply_async(prepare_artificial_uid, (item("bad-callback"),), callback=broken)
        good = pool.apply_async(prepare_artificial_uid, (item("good"),))
        pool.close()
        pool.join()
        with self.assertRaises(OperationalUIDFailure) as caught:
            bad.get(0)
        self.assertEqual(caught.exception.reason, "callback_exception")
        self.assertEqual(good.get(0)[0], "good")
        self.assert_clean(pool)


class ExistingPipelineIntegrationTests(ExecutorTestCase):
    def test_narrow_context_proxy_keeps_original_multiprocessing_unmodified(self):
        original = multiprocessing.get_context
        proxy = IsolatedUIDMultiprocessingProxy(
            multiprocessing, self.pin, temporary_parent=self.temporary.name, uid_deadline_seconds=5)
        pool = proxy.get_context("spawn").Pool(processes=2)
        self.pools.append(pool)
        ticket = pool.apply_async(prepare_artificial_uid, (item("proxy"),))
        pool.close()
        pool.join()
        self.assertIs(multiprocessing.get_context, original)
        self.assertEqual(ticket.get(0)[0], "proxy")
        self.assertEqual(proxy.executors, [pool])
        self.assert_clean(pool)

    def test_unmodified_old_pipeline_accepts_executor_factory(self):
        path = ROOT / ("artifacts/research/20261016-f006-execution-v1/static-native-source/"
                       "common/reused/code/scripts/p010_shared_runtime.py")
        if not path.exists():
            self.skipTest("Ignored sealed historical pipeline is not available")
        spec = importlib.util.spec_from_file_location("artificial_old_p010", path)
        pipeline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pipeline)
        for mode in ("success", "exit", "hang"):
            with self.subTest(mode=mode):
                pool = self.pool(max_workers=2, uid_deadline_seconds=1.5, cleanup_seconds=.3)
                consumed, lock = [], threading.Lock()

                def consume(device, payload):
                    with lock:
                        consumed.append(payload[0])

                start = time.monotonic()
                if mode == "success":
                    receipt = pipeline.run_bounded_pipeline(
                        lambda: pool, prepare_artificial_uid, [item(str(i), mode) for i in range(6)],
                        ("artificial-device-0", "artificial-device-1"), consume, queue_max=2, outstanding_limit=2,
                        shutdown_seconds=5)
                    self.assertEqual(receipt["submitted"], 6)
                    self.assertEqual(receipt["consumed"], 6)
                    self.assertEqual(set(consumed), {str(i) for i in range(6)})
                else:
                    with self.assertRaises(pipeline.PipelineFailure) as caught:
                        pipeline.run_bounded_pipeline(
                            lambda: pool, prepare_artificial_uid, [item("bad", mode), item("good")],
                            ("artificial-device-0",), consume, queue_max=2, outstanding_limit=2, shutdown_seconds=5)
                    self.assertTrue(caught.exception.receipt["cancelled"])
                    self.assertEqual(caught.exception.receipt["queue_unfinished_tasks"], 0)
                    self.assertEqual(caught.exception.receipt["delivery_unfinished_tasks"], 0)
                self.assertLess(time.monotonic() - start, 10)
                self.assert_clean(pool)


if __name__ == "__main__":
    unittest.main()
