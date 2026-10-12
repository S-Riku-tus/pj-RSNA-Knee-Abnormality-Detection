"""Pure CSV/queue/source/lifecycle checks; synthetic data only."""

import bisect
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import pickle
import struct
import sys
import tempfile
import types
import unittest
from multiprocessing.pool import ThreadPool
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


contracts = load("p011_contracts", ROOT / "scripts/p011_contracts.py")
runtime = load("_p011_test_runtime", ROOT / "scripts/p011_independent_runtime.py")
builder = load("_p011_test_builder", ROOT / "scripts/build_p011_independent224.py")
profile = load("_p011_test_profile", ROOT / "scripts/build_p011_profile.py")


def reference_rank(values):
    ordered = sorted(values)
    return [(bisect.bisect_left(ordered, v) + bisect.bisect_right(ordered, v) + 1) / (2 * len(values)) for v in values]


class CSVAndBranchContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="p011-inert-csv-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def csv(self, count, name="submission.csv"):
        uids = [f"artificial-{i:05d}" for i in range(count)]
        matrix = [
            [struct.unpack("<f", struct.pack("<f", ((i * (j + 1)) % count) / max(1, count - 1)))[0] for j in range(12)]
            for i in range(count)
        ]
        path = self.root / name
        contracts.write_submission_exclusive(path, uids, matrix, runtime.LABELS)
        return path, uids, matrix

    def test_9_32_1301_fixed_all12_ties_f32_CSV_and_seal_identity(self):
        for count in (9, 32, 1301):
            with self.subTest(count=count), tempfile.TemporaryDirectory(dir=self.root) as child:
                folder = Path(child)
                uids = [f"artificial-{i}" for i in range(count)]
                base = [
                    [
                        struct.unpack("<f", struct.pack("<f", ((i + j) % count) / max(1, count - 1)))[0]
                        for j in range(12)
                    ]
                    for i in range(count)
                ]
                expert = [[((i * 7 + j) % 17) / 16 for j in range(12)] for i in range(count)]
                path, extra = folder / "submission.csv", folder / "extra.csv"
                contracts.write_submission_exclusive(path, uids, base, runtime.LABELS)
                original = path.read_bytes()
                seal = runtime.cpu_csv_seal(path, working=folder)
                self.assertFalse(seal["receipt"]["extra_model_or_namespace_created"])
                contracts.write_submission_exclusive(extra, uids, expert, runtime.LABELS)
                runtime.fixed10_from_seal(seal, extra)
                self.assertEqual((folder / "p011_baseline_sealed.csv").read_bytes(), original)
                got = contracts.read_submission(path, runtime.LABELS, uids)[1]
                for j in range(12):
                    left = reference_rank([r[j] for r in base])
                    right = reference_rank([r[j] for r in expert])
                    expected = reference_rank([0.9 * a + 0.1 * b for a, b in zip(left, right, strict=True)])
                    self.assertEqual([r[j] for r in got], expected)

    def test_nonfinite_out_of_range_duplicate_UID_or_wrong_header_rejected(self):
        path, uids, _ = self.csv(9)
        text = path.read_text()
        lines = text.splitlines()
        for changed in (
            text.replace(",0,", ",nan,", 1),
            text.replace(",0,", ",1.1,", 1),
            "\n".join([*lines, lines[1]]) + "\n",
            text.replace("ACL,MCL", "MCL,ACL", 1),
        ):
            path.write_text(changed)
            with self.assertRaises(ValueError):
                contracts.read_submission(path, runtime.LABELS, uids)
        path.write_text(text)

    def test_baseline_mutation_rejected_before_any_extra_or_replace(self):
        path, _, _ = self.csv(9)
        seal = runtime.cpu_csv_seal(path, working=self.root)
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "changed after sealing"):
            runtime.assert_sealed(seal)
        self.assertFalse((self.root / "p011_candidate.partial.csv").exists())

    def test_stale_baseline_backup_rejected_exclusive(self):
        path, _, _ = self.csv(9)
        runtime.cpu_csv_seal(path, working=self.root)
        with self.assertRaises(FileExistsError):
            runtime.cpu_csv_seal(path, working=self.root)

    def test_new7_cells_literal_order_namespace_after_seal(self):
        base = json.loads(builder.BASE.read_text(encoding="utf-8"))
        for cohort in (None, profile.cohort64()):
            book, stock = builder.make_book(cohort=cohort)
            original = [builder.text(c) for c in base["cells"] if c["cell_type"] == "code"]
            actual = [builder.text(c) for c in book["cells"] if c["cell_type"] == "code"]
            self.assertEqual(len(original), 7)
            self.assertTrue(all(actual.count(s) == 1 for s in original))
            self.assertEqual([actual.index(s) for s in original], sorted(actual.index(s) for s in original))
            self.assertEqual(
                hashlib.sha256(stock.encode()).hexdigest(),
                "b61046e344425c446370d3df1ca019cf278f13fba385117035f849484a5a85f0",
            )
            tail = actual[-1]
            self.assertLess(tail.index("cpu_csv_seal("), tail.index("run_standalone224("))
            self.assertNotIn("make_stock_namespace(", tail)

    def test_cohort64_whole_groups_Gold_excluded(self):
        cohort = profile.cohort64()
        plan = json.loads(profile.PLAN.read_text(encoding="utf-8"))
        chosen = set(cohort["uids"])
        groups = set(cohort["groups"].values())
        self.assertEqual(len(chosen), 64)
        self.assertFalse(chosen & set(cohort["gold_UIDs"]))
        self.assertFalse(groups & set(cohort["gold_groups"]))
        self.assertEqual({r["StudyInstanceUID"] for r in plan["studies"] if r["group_id"] in groups}, chosen)

    def test_generated_package_stock_bytes_match_embedded_LF_source_on_Windows(self):
        package, notebook, audit = self.root / "package", self.root / "test.ipynb", self.root / "audit.json"
        receipt = builder.build(package, notebook, audit)
        stock_bytes = (package / "p011_author224_stock.py").read_bytes()
        self.assertNotIn(b"\r\n", stock_bytes)
        self.assertEqual(hashlib.sha256(stock_bytes).hexdigest(), receipt["stock_source_sha256"])
        self.assertTrue(receipt["checks"]["package_stock_file_exact_LF_source_bytes"])
        book = json.loads(notebook.read_text(encoding="utf-8"))
        body = next(
            builder.text(c)
            for c in book["cells"]
            if builder.text(c).startswith("%%writefile /kaggle/working/p011_src/p011_author224_stock.py\n")
        ).split("\n", 1)[1]
        self.assertEqual(stock_bytes, body.encode("utf-8"))

    def test_benign_observer_two_exec_cells_without_success_fabrication(self):
        with contextlib.redirect_stdout(io.StringIO()):
            ns = {"FusionObserver": contracts.FusionObserver}
            exec("probe=FusionObserver()\nprint('[benign only]')", ns)
            exec("captured=probe.restore()", ns)
        self.assertEqual(ns["captured"].count("[benign only]"), 1)
        self.assertNotIn("SUCCESS", ns["captured"])

    def test_observer_missing_or_rebound_lifetime_fails(self):
        destination = io.StringIO()
        with contextlib.redirect_stdout(destination):
            observer = contracts.FusionObserver()
            with self.assertRaises(contracts.PipelineFailure):
                observer.require_success()
            observer = contracts.FusionObserver()
            sys.stdout = destination
            with self.assertRaisesRegex(contracts.PipelineFailure, "lifetime"):
                observer.restore()

    def test_registered_profile_function_pickle_identity_and_cleanup(self):
        ns = {"__name__": "__main__"}
        p = runtime.BaselineProfile(ns, self.root, self.root)
        try:
            exec("def _pre_one(uid):\n    return uid\n", ns)
            p.module.__dict__.update(ns)
            self.assertIs(pickle.loads(ForkingPickler.dumps(ns["_pre_one"])), ns["_pre_one"])
        finally:
            p.close()
        self.assertEqual(ns["__name__"], "__main__")
        self.assertNotIn(p.name, sys.modules)

    def test_exact_reader_command_routes_once_wrong_split_output_models_data_rejected(self):
        helper = types.SimpleNamespace(observed_reader_source=lambda s, mode: s + "\n# inert synthetic observer\n")

        class DummyResources:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def result(self):
                return {"synthetic": True}

        resources = types.SimpleNamespace(Resources=DummyResources)
        infer = self.root / "own_src/infer.py"
        infer.parent.mkdir()
        original = json.loads(builder.BASE.read_text(encoding="utf-8"))
        infer.write_text(builder.text(original["cells"][8]).split("\n", 1)[1], encoding="utf-8")
        ns = {"__name__": "__main__"}
        p = runtime.BaselineProfile(ns, self.root / "alias", self.root)
        calls = []
        p.original_subprocess_run = lambda args, *a, **kw: calls.append(args)
        original_real_subprocess_run = runtime.subprocess.run
        command = [
            sys.executable,
            "/kaggle/working/own_src/infer.py",
            "/kaggle/input/competitions/rsna-knee-abnormality-detection",
            "test",
            "/kaggle/working/_own.csv",
            *[f"/synthetic/cnxt_v0_fold{i}.pt" for i in range(3)],
        ]
        try:
            with patch.dict(sys.modules, {"p007_metrics": helper, "p006_metrics": resources}):
                p.prepare_reader()
            for pos, wrong in ((2, "/wrong/data"), (3, "train"), (4, "/wrong/output.csv"), (5, "/synthetic/other.pt")):
                changed = command.copy()
                changed[pos] = wrong
                with self.assertRaises(ValueError):
                    runtime.subprocess.run(changed, check=True, env={})
            with patch.dict(sys.modules, {"p006_metrics": resources}):
                runtime.subprocess.run(command, check=True, env={})
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0], [*command[:2], str(p.data_root), *command[3:]])
            with self.assertRaisesRegex(ValueError, "Repeated reader"):
                runtime.subprocess.run(command, check=True, env={})
        finally:
            p.close()
            runtime.subprocess.run = original_real_subprocess_run

    def test_decoder_exception_is_recorded_and_reraised_without_model_catch(self):
        def failed(*args):
            raise RuntimeError("artificial CPU decoder failure")

        wrapped = runtime.ObservedBuilder(failed, self.root)
        with self.assertRaisesRegex(RuntimeError, "CPU decoder"):
            wrapped("artificial-UID", {}, "/synthetic/tree")
        receipt = json.loads(next(self.root.glob("*.json")).read_text())
        self.assertFalse(receipt["native_input_available"])
        self.assertEqual(receipt["decoder_error_type"], "RuntimeError")

    def test_native_worker_receipt_missing_corrupt_wrong_UID_schema_or_hash_rejected(self):
        uid = "artificial-native"
        key = hashlib.sha256(uid.encode()).hexdigest()
        path = self.root / (key + ".json")
        inputs = {
            "volume_sha256": "a" * 64,
            "mask_sha256": "b" * 64,
            "valid_slices": 1,
            "volume_shape": [96, 384, 384],
            "mask_shape": [96],
            "volume_dtype": "uint8",
            "mask_dtype": "uint8",
            "volume_bytes": 96 * 384 * 384,
            "mask_bytes": 96,
        }
        record = {"uid_sha256": key, "native_input_available": True, **inputs}
        with self.assertRaisesRegex(ValueError, "Missing, corrupt"):
            runtime.validate_native_receipt(self.root, uid, inputs)
        for bad in (
            b"{broken",
            json.dumps({**record, "uid_sha256": "f" * 64}).encode(),
            json.dumps({**record, "native_input_available": 1}).encode(),
            json.dumps({**record, "volume_sha256": "c" * 64}).encode(),
            json.dumps({**record, "unknown_key": True}).encode(),
            b" " * 4097,
        ):
            path.write_bytes(bad)
            with self.subTest(payload=bad[:70]), self.assertRaises(ValueError):
                runtime.validate_native_receipt(self.root, uid, inputs)
        path.write_bytes(json.dumps(record).encode())
        accepted = runtime.validate_native_receipt(self.root, uid, inputs)
        self.assertTrue(accepted["worker_receipt_verified_before_model"])
        self.assertEqual(accepted["native_worker_receipt_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_recorded_decoder_failure_keeps_original_zero_input_contract_and_finite_stock_policy(self):
        uid = "artificial-decode-failure"
        key = hashlib.sha256(uid.encode()).hexdigest()
        path = self.root / (key + ".json")
        inputs = {
            "volume_sha256": hashlib.sha256(bytes(96 * 384 * 384)).hexdigest(),
            "mask_sha256": hashlib.sha256(bytes(96)).hexdigest(),
            "valid_slices": 0,
        }
        path.write_bytes(
            json.dumps(
                {"uid_sha256": key, "native_input_available": False, "decoder_error_type": "RuntimeError"}
            ).encode()
        )
        self.assertTrue(runtime.validate_native_receipt(self.root, uid, inputs)["worker_receipt_verified_before_model"])
        for bad in (
            {**inputs, "valid_slices": 1},
            {**inputs, "volume_sha256": "a" * 64},
            {**inputs, "mask_sha256": "b" * 64},
        ):
            with self.assertRaisesRegex(ValueError, "zero volume and mask"):
                runtime.validate_native_receipt(self.root, uid, bad)

    def test_failed_extra_initializer_sets_shared_fatal_flag_before_any_model_call(self):
        flag = types.SimpleNamespace(value=0)
        state = {
            "namespace": {},
            "original_init": lambda: None,
            "workers": self.root,
            "inputs": self.root,
            "data_root": None,
            "initializer_failed": flag,
        }
        with (
            patch.object(runtime, "_EXTRA", state),
            patch.object(runtime, "init_observed_builder", side_effect=ValueError("synthetic source guard")),
        ):
            runtime.extra_pre_init()
        self.assertEqual(flag.value, 1)
        with self.assertRaisesRegex(contracts.PipelineFailure, "initializer failed"):
            state["namespace"]["_READER"]("synthetic", {}, "/none")
        self.assertTrue(json.loads(next(self.root.glob("*-FAILED.json")).read_text())["initializer_failed"])

    def test_failed_profile_initializer_returns_and_parent_rejects_before_original_pair(self):
        class DummyResources:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def result(self):
                return {"artificial": True}

        ns = {"__name__": "__main__", "PipelineFailure": contracts.PipelineFailure}
        p = runtime.BaselineProfile(ns, self.root / "alias", self.root)
        exec(
            "def _pre_one(uid):\n    return uid\n"
            "def _pre_init():\n    pass\n"
            "def find_test_root():\n    return '/none'\n"
            "def load_model(*args):\n    raise AssertionError('no artificial GPU load')\n"
            "def infer_pair_probs(*args):\n    raise AssertionError('failed init must prevent original GPU pair')\n"
            "def main():\n"
            "    _pre_init()\n"
            "    try:\n        infer_pair_probs('synthetic')\n"
            "    except PipelineFailure:\n        pass\n",
            ns,
        )
        try:
            with (
                patch.dict(
                    sys.modules,
                    {"torch": types.SimpleNamespace(), "p006_metrics": types.SimpleNamespace(Resources=DummyResources)},
                ),
                patch.object(runtime, "init_observed_builder", side_effect=ValueError("synthetic initializer guard")),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with self.assertRaisesRegex(contracts.PipelineFailure, "baseline preprocessing"):
                    p.execute_main()
            self.assertEqual(p.initializer_failed.value, 1)
            self.assertEqual(p.state["pair_calls"], 0)
            self.assertTrue(p.state["preprocessing_initializer_failed"])
            self.assertFalse(p.state["FAILED_or_FALLBACK_in_main"])
            with self.assertRaises(contracts.PipelineFailure):
                ns["_READER"]("synthetic", {}, "/none")
            self.assertTrue(json.loads(next(p.base_workers.glob("*-FAILED.json")).read_text())["initializer_failed"])
        finally:
            p.close()

    def test_resource_gate_requires_two_T4_finite_positive_complete_observations(self):
        phase = {
            "sampled_gpu_peak_used_mib": [4500, 4700],
            "elapsed_seconds": 12.3,
            "sampled_peak_process_tree_rss_bytes": 123456,
            "samples": 3,
        }
        GPU = {
            "GPU": [
                {"index": i, "name": "Tesla T4", "allocated_mib": 2000, "reserved_mib": 4000, "peak_reserved_mib": 4500}
                for i in range(2)
            ]
        }
        self.assertTrue(runtime.strict_profile_resources([phase] * 3, GPU)["positive_finite_peaks"])
        for field, values in (
            ("sampled_gpu_peak_used_mib", ([], [1], [1, float("nan")], [1, -1], [1, 0], [1, 13000])),
            ("elapsed_seconds", (float("nan"), float("inf"), -1, 0)),
            ("sampled_peak_process_tree_rss_bytes", (float("nan"), -1, 0)),
            ("samples", (0, -1, True, 1.5)),
        ):
            for value in values:
                changed = copy.deepcopy(phase)
                changed[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    runtime.strict_profile_resources([phase, changed, phase], GPU)
        for field, value in (
            ("name", "NVIDIA RTX4090"),
            ("index", 0),
            ("peak_reserved_mib", float("nan")),
            ("peak_reserved_mib", 0),
            ("reserved_mib", -1),
            ("allocated_mib", float("inf")),
        ):
            changed = copy.deepcopy(GPU)
            changed["GPU"][1][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                runtime.strict_profile_resources([phase] * 3, changed)
        for devices in ([], GPU["GPU"][:1]):
            with self.assertRaises(ValueError):
                runtime.strict_profile_resources([phase] * 3, {"GPU": devices})

    def test_large1301_queue_bounded_and_all_taskdone(self):
        seen = []
        result = contracts.run_bounded_pipeline(
            lambda: ThreadPool(4),
            lambda x: x,
            list(range(1301)),
            ["artificial0", "artificial1"],
            lambda d, x: seen.append(x),
            queue_max=3,
            outstanding_limit=4,
        )
        self.assertEqual(set(seen), set(range(1301)))
        self.assertEqual(len(seen), 1301)
        self.assertLessEqual(result["max_outstanding"], 4)
        self.assertLessEqual(result["max_delivery_queue"], 4)
        self.assertLessEqual(result["max_inference_queue"], 3)
        self.assertEqual(result["queue_unfinished_tasks"], 0)
        self.assertEqual(result["delivery_unfinished_tasks"], 0)

    def test_worker_and_consumer_failures_fatal_drained_no_silent_zero_prediction(self):
        def failing(x):
            if x == 2:
                raise RuntimeError("synthetic failure")
            return x

        for worker, consumer in ((failing, lambda d, x: None), (lambda x: x, lambda d, x: failing(x))):
            with self.assertRaises(contracts.PipelineFailure) as failure:
                contracts.run_bounded_pipeline(
                    lambda: ThreadPool(2),
                    worker,
                    list(range(20)),
                    ["synthetic0", "synthetic1"],
                    consumer,
                    queue_max=1,
                    outstanding_limit=2,
                )
            self.assertTrue(failure.exception.receipt["cancelled"])
            self.assertEqual(failure.exception.receipt["queue_unfinished_tasks"], 0)
            self.assertEqual(failure.exception.receipt["delivery_unfinished_tasks"], 0)


if __name__ == "__main__":
    unittest.main()
