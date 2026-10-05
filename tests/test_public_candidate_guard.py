"""Exercise submission refusal using artificial files and artificial source cells only."""

import contextlib
import csv
import hashlib
import io
import json
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import public_candidate_guard as guard

try:
    import numpy as np
except ImportError:
    np = None


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def write_csv(path, rows, header=guard.HEADER):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


class GuardFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.work = self.root / "work"
        self.work.mkdir()
        self.input = self.root / "input"
        self.competition = self.input / "competition"
        self.competition.mkdir(parents=True)
        (self.competition / "test_series").mkdir()
        self.ids = ["artificial-z", "artificial-a"]
        write_csv(self.competition / "test.csv", [[uid] for uid in self.ids], ["StudyInstanceUID"])
        write_csv(self.competition / "train.csv", [], ["StudyInstanceUID"])
        write_csv(self.competition / "test_series.csv", [], ["StudyInstanceUID", "SeriesInstanceUID"])
        self.rows = [[uid, *([str(value)] * 12)] for uid, value in zip(self.ids, [0.25, 0.75])]
        write_csv(self.competition / "sample_submission.csv", self.rows)
        self.asset = self.input / "asset"
        self.asset.mkdir()
        (self.asset / "synthetic.bin").write_bytes(b"artificial input, not model weights")
        self.contract = {
            "candidate_id": "p002-artificial-fixture",
            "inputs": [
                {"key": "competition", "ref": "fixture/competition", "mounts": ["competition"], "files": []},
                {
                    "key": "asset",
                    "ref": "fixture/asset",
                    "mounts": ["asset"],
                    "files": [
                        {
                            "path": "synthetic.bin",
                            "bytes": (self.asset / "synthetic.bin").stat().st_size,
                            "sha256": guard.sha256(self.asset / "synthetic.bin"),
                        }
                    ],
                },
            ],
            "source_cells": {},
            "dino_member_ids": [f"member-{i}" for i in range(20)],
        }
        self.candidate = guard.CandidateGuard(self.contract, work=self.work, input_root=self.input)

    def prepared(self):
        self.candidate.prepare_files()
        self.candidate.ready = True
        return self.candidate

    def run_stub(self, number, source, **namespace):
        source = textwrap.dedent(source)
        self.contract["source_cells"][str(number)] = digest(source.encode())
        self.candidate.completed = list(guard.CELL_ORDER[: guard.CELL_ORDER.index(number)])
        context = {
            "__name__": "__main__",
            "np": np,
            "write_csv": write_csv,
            "work": self.work,
            "fixture_rows": self.rows,
            "fixture_ids": self.ids,
            "member_ids": self.contract["dino_member_ids"],
            **namespace,
        }
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.candidate.run_cell(number, source, context)
        return context

    def assert_rejected(self):
        self.assertFalse((self.work / "submission.csv").exists())
        self.assertFalse((self.work / "P002_READY.json").exists())
        self.assertTrue((self.work / "P002_FAILED.json").is_file())
        self.assertFalse(self.candidate.ready)


class GuardInputTests(GuardFixture):
    def test_preflight_files_are_hashed_and_test_order_preserved(self):
        records = self.candidate.prepare_files()
        self.assertEqual(self.candidate.test_ids, self.ids)
        self.assertEqual(len(records), 1)
        self.assertTrue(self.candidate.owns_outputs)
        self.assertFalse(self.candidate.ready)

    def test_existing_outputs_are_preserved_and_refused(self):
        for filename in ("submission.csv", "public0033_bag_raw.csv", "old_receipt.json", "P002_READY.json"):
            with self.subTest(filename=filename):
                target = self.work / filename
                target.write_bytes(b"preserve existing experiment")
                try:
                    with self.assertRaises(RuntimeError):
                        self.candidate.prepare_files()
                    self.candidate.reject("Refuse old run")
                    self.assertEqual(target.read_bytes(), b"preserve existing experiment")
                    self.assertFalse((self.work / "P002_FAILED.json").exists())
                finally:
                    target.unlink()

    def test_same_size_input_corruption_is_rejected(self):
        path = self.asset / "synthetic.bin"
        payload = path.read_bytes()
        path.write_bytes(b"X" + payload[1:])
        with self.assertRaisesRegex(RuntimeError, "SHA-256"):
            self.candidate.prepare_files()

    def test_member_path_cannot_escape_mount(self):
        outside = self.input / "outside.bin"
        outside.write_bytes(b"outside")
        self.contract["inputs"][1]["files"] = [{"path": "../outside.bin", "bytes": 7, "sha256": guard.sha256(outside)}]
        with self.assertRaises((ValueError, RuntimeError)):
            self.candidate.prepare_files()

    def test_input_mount_cannot_escape_input_root(self):
        self.contract["inputs"][1]["mounts"] = ["../work"]
        self.contract["inputs"][1]["files"] = []
        with self.assertRaises((ValueError, RuntimeError)):
            self.candidate.prepare_files()

    def test_strict_inputs_refuse_extra_raptor_mount(self):
        self.contract["strict_input_mounts"] = True
        (self.input / "raptor-knee-maxspan").mkdir()
        with self.assertRaisesRegex(RuntimeError, "unlisted Input mount"):
            self.candidate.prepare_files()

    def test_strict_inputs_refuse_another_owner_inside_namespace(self):
        self.contract["strict_input_mounts"] = True
        nested = self.input / "datasets" / "fixture-owner" / "asset"
        nested.parent.mkdir(parents=True)
        self.asset.rename(nested)
        self.asset = nested
        self.contract["inputs"][1]["mounts"] = ["datasets/fixture-owner/asset"]
        (self.input / "datasets" / "unexpected-owner" / "raptor-knee-maxspan").mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "unlisted Input mount"):
            self.candidate.prepare_files()

    def test_strict_inputs_allow_nested_mount_without_walking_payload(self):
        self.contract["strict_input_mounts"] = True
        nested = self.input / "datasets" / "fixture-owner" / "asset"
        nested.parent.mkdir(parents=True)
        self.asset.rename(nested)
        self.asset = nested
        self.contract["inputs"][1]["mounts"] = ["datasets/fixture-owner/asset"]
        (self.asset / "internal" / "arbitrary-payload").mkdir(parents=True)
        (self.competition / "test_series" / "artificial-study" / "artificial-series").mkdir(parents=True)
        declared = [self.asset.resolve(), self.competition.resolve()]
        visited = []
        real_iterdir = Path.iterdir

        def guarded_iterdir(path):
            resolved = path.resolve()
            self.assertFalse(
                any(resolved.is_relative_to(mount) for mount in declared),
                "Preflight must not enumerate model or MRI payload directories",
            )
            visited.append(resolved)
            return real_iterdir(path)

        with patch.object(Path, "iterdir", guarded_iterdir):
            self.candidate.prepare_files()
        self.assertEqual(
            set(visited),
            {
                self.input.resolve(),
                (self.input / "datasets").resolve(),
                nested.parent.resolve(),
            },
        )

    def test_missing_test_series_does_not_use_train(self):
        (self.competition / "test_series").rmdir()
        (self.competition / "train_series").mkdir()
        with self.assertRaisesRegex(RuntimeError, "training-data fallback"):
            self.candidate.prepare_files()

    def test_schema_uid_and_numeric_submission_refusals(self):
        cases = [
            (list(reversed(self.rows)), guard.HEADER),
            ([self.rows[0], self.rows[0]], guard.HEADER),
            ([["", *self.rows[0][1:]], self.rows[1]], guard.HEADER),
            (self.rows, tuple(reversed(guard.HEADER))),
            ([self.rows[0][:-1], self.rows[1]], guard.HEADER),
        ]
        for value in ("nan", "inf", "-inf", "1.0001", "-0.01", ""):
            rows = [list(row) for row in self.rows]
            rows[0][2] = value
            cases.append((rows, guard.HEADER))
        path = self.work / "synthetic.csv"
        for index, (rows, header) in enumerate(cases):
            with self.subTest(index=index):
                write_csv(path, rows, header)
                with self.assertRaises(ValueError):
                    guard.read_table(path, self.ids)

    def test_tee_buffers_split_lines_and_preserves_output(self):
        observed = []
        stream = io.StringIO()
        tee = guard.Tee(stream, observed.append)
        tee.write("part")
        self.assertEqual(observed, [])
        tee.write(" one\nsecond")
        tee.write("\n")
        self.assertEqual(observed, ["part one", "second"])
        self.assertEqual(stream.getvalue(), "part one\nsecond\n")


class GuardExecutionTests(GuardFixture):
    def test_source_hash_change_refuses_execution_and_removes_partial_submission(self):
        self.prepared()
        self.contract["source_cells"]["2"] = digest(b"expected source")
        write_csv(self.work / "submission.csv", self.rows)
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.candidate.run_cell(2, "raise AssertionError('must never execute')", {})
        self.assert_rejected()
        self.assertTrue((self.work / "p002-rejected-submission.csv.disabled").is_file())

    def test_cell_order_mismatch_refuses_partial_submission(self):
        self.prepared()
        write_csv(self.work / "submission.csv", self.rows)
        with self.assertRaisesRegex(RuntimeError, "order drift"):
            self.candidate.run_cell(3, "raise AssertionError('must never execute')", {})
        self.assert_rejected()

    def test_source_exception_and_keyboard_interrupt_disable_outputs(self):
        for exception in ("RuntimeError", "KeyboardInterrupt"):
            with self.subTest(exception=exception):
                directory = self.work / exception
                directory.mkdir()
                self.candidate = guard.CandidateGuard(self.contract, work=directory, input_root=self.input)
                self.prepared()
                source = "write_csv(work / 'submission.csv', fixture_rows)\nraise " + exception + "('synthetic')\n"
                original = self.work
                self.work = directory
                try:
                    with self.assertRaises((RuntimeError, KeyboardInterrupt)):
                        self.run_stub(2, source)
                    self.assert_rejected()
                finally:
                    self.work = original

    def test_successful_stub_runs_once_and_keeps_shared_namespace(self):
        self.prepared()
        source = "counter = 17\n"
        namespace = self.run_stub(2, source)
        self.assertEqual(namespace["counter"], 17)
        self.assertEqual(self.candidate.completed, [2])
        with self.assertRaisesRegex(RuntimeError, "order drift"):
            self.candidate.run_cell(2, source, namespace)
        self.assert_rejected()


@unittest.skipIf(np is None, "numpy is optional; observer tests use artificial arrays")
class GuardMemberTests(GuardFixture):
    DINO = """
        def infer_from_package():
            per_member = [dict(id=mid, ids=list(reversed(fixture_ids)), pred=np.full((2, 12), .4))
                          for mid in member_ids]
            public_frontier_members = [dict(m, soft_pred=np.full((2, 12), .6)) for m in per_member]
            mutate(per_member, public_frontier_members)
            write_csv(work / 'submission_public_0899.csv', fixture_rows)
            print('final submission.csv = weighted rank mean of 20 member(s)')
            (work / 'submission_public_0899.csv').replace(work / 'submission.csv')
        infer_from_package()
    """
    A5 = """
        studies = list(reversed(fixture_ids))
        preds = np.full((5, 2, 12), .4)
        mutate(preds)
        print('inference done in 0.1 min')
    """
    RAPTOR = """
        def main():
            test_ids = list(fixture_ids)
            arm_probs = [np.full((2, 12), .5) for _ in range(4)]
            mutate(arm_probs)
            for arm in range(4):
                print(f'[arm {arm}] 2/2 | 1s')
                print(f'[arm {arm}] done + freed | 1s')
            if fallback:
                print('  [arm 2] study 1 artificial FALLBACK (ValueError: example)')
            write_csv(work / 'submission.csv', fixture_rows)
            print('wrote /kaggle/working/submission_coatnet.csv | 2 rows x 13 cols')
        main()
    """

    def test_dino_full_frontier_and_renamed_output_pass(self):
        self.prepared()
        self.run_stub(3, self.DINO, mutate=lambda *_: None)
        self.assertTrue(self.candidate.observations["dino_raw_valid"])
        self.assertTrue((self.work / "submission.csv").is_file())
        self.assertFalse((self.work / "submission_public_0899.csv").exists())

    def test_dino_dropped_frontier_is_refused(self):
        self.prepared()
        with self.assertRaisesRegex(RuntimeError, "20 unique"):
            self.run_stub(3, self.DINO, mutate=lambda members, frontier: frontier.pop())
        self.assert_rejected()

    def test_dino_nan_cannot_hide_behind_finite_final_csv(self):
        self.prepared()

        def corrupt(members, frontier):
            frontier[0]["soft_pred"][0, 0] = np.nan

        with self.assertRaisesRegex(RuntimeError, "finite"):
            self.run_stub(3, self.DINO, mutate=corrupt)
        self.assert_rejected()

    def test_a5_all_five_models_have_finite_predictions(self):
        self.prepared()
        self.run_stub(5, self.A5, mutate=lambda _: None)
        self.assertTrue(self.candidate.observations["a5_raw_valid"])

    def test_a5_missing_study_fails(self):
        self.prepared()

        def corrupt(predictions):
            predictions[:, 1, :] = np.nan

        with self.assertRaisesRegex(RuntimeError, "finite"):
            self.run_stub(5, self.A5, mutate=corrupt)
        self.assert_rejected()

    def test_duplicate_completion_event_is_refused(self):
        self.prepared()
        source = textwrap.dedent(self.A5) + "\nprint('inference done in 0.1 min')\n"
        with self.assertRaisesRegex(RuntimeError, "Repeated or missing"):
            self.run_stub(5, source, mutate=lambda _: None)
        self.assert_rejected()

    def test_raptor_all_four_arms_pass(self):
        self.prepared()
        self.run_stub(7, self.RAPTOR, mutate=lambda _: None, fallback=False)
        self.assertTrue(self.candidate.observations["raptor_raw_valid"])
        self.assertEqual(self.candidate.observations["raptor_done"], [0, 1, 2, 3])

    def test_raptor_fallback_is_refused_even_if_values_are_finite(self):
        self.prepared()
        with self.assertRaisesRegex(RuntimeError, "Raptor fallback"):
            self.run_stub(7, self.RAPTOR, mutate=lambda _: None, fallback=True)
        self.assert_rejected()

    def test_raptor_nonfinite_raw_is_refused_even_if_csv_is_finite(self):
        self.prepared()

        def corrupt(predictions):
            predictions[2][1, 3] = np.inf

        with self.assertRaisesRegex(RuntimeError, "finite"):
            self.run_stub(7, self.RAPTOR, mutate=corrupt, fallback=False)
        self.assert_rejected()

    def test_raptor_incomplete_progress_is_refused(self):
        self.prepared()
        source = self.RAPTOR.replace("range(4):", "range(3):")
        with self.assertRaisesRegex(RuntimeError, "four full-study"):
            self.run_stub(7, source, mutate=lambda _: None, fallback=False)
        self.assert_rejected()


class GuardFinishTests(GuardFixture):
    def setUp(self):
        super().setUp()
        self.specialist = self.input / "specialist"
        self.specialist.mkdir()
        (self.specialist / "bundle_manifest.json").write_text('{"fixture": true}\n', encoding="utf-8")
        self.contract["inputs"].append(
            {
                "key": "specialist",
                "ref": "fixture/specialist",
                "mounts": ["specialist"],
                "files": [],
            }
        )
        self.contract["check_receipt_inputs"] = True
        self.contract["receipt_values"] = {
            "public0033_cached_inference_receipt.json": {"fallback": 0, "checkpoint.strict_load": True},
            "dinosaur_v4_v6_receipt.json": {"new_models_loaded": False},
        }
        self.contract["receipt_shapes"] = {
            "public0033_cached_inference_receipt.json": {"cache.shape": ["N", 6, 12, 336, 336]},
            "infra0021_runtime_receipt.json": {"output.shape": ["N", 13]},
        }
        self.prepared()
        self.candidate.completed = list(guard.CELL_ORDER)
        self.candidate.observations.update(
            {
                "dino_raw_valid": True,
                "a5_raw_valid": True,
                "raptor_raw_valid": True,
                "dino_observations": 1,
                "a5_observations": 1,
                "raptor_observations": 1,
                "raptor_done": [0, 1, 2, 3],
                "raptor_full_progress": [0, 1, 2, 3],
            }
        )
        self.parent = [list(row) for row in self.rows]
        self.control = [list(row) for row in self.parent]
        self.control[0][3], self.control[1][3] = "0.4", "0.6"
        self.final = [list(row) for row in self.control]
        for target in guard.WEIGHTS:
            column = guard.HEADER.index(target)
            self.final[0][column], self.final[1][column] = "0.5", "1.0"
        write_csv(self.work / "submission_parent_exact.csv", self.parent)
        write_csv(self.work / "submission_dinosaur_v4_0937_control.csv", self.control)
        write_csv(self.work / "submission.csv", self.final)
        write_csv(self.work / "submission_dinosaur_v4_v6_candidate.csv", self.final)
        self.bag = [[self.ids[1], "0.2", "0.3"], [self.ids[0], "0.8", "0.7"]]
        write_csv(
            self.work / "public0033_bag_raw.csv", self.bag, ["StudyInstanceUID", "Medial Meniscus", "Lateral Meniscus"]
        )
        uid_hash = digest("".join(uid + "\n" for uid in self.ids).encode())
        bag_uid_hash = digest("".join(row[0] + "\n" for row in self.bag).encode())
        self.receipts = {
            "public0033_cached_inference_receipt.json": {
                "schema_version": "public0033_cached_inference_receipt_v1",
                "status": "passed",
                "fallback": 0,
                "checkpoint": {"strict_load": True},
                "study_count": 2,
                "study_uid_sha256": bag_uid_hash,
                "cache": {"shape": [2, 6, 12, 336, 336]},
                "bundle_manifest_sha256": guard.sha256(self.specialist / "bundle_manifest.json"),
                "raw_output": {"sha256": guard.sha256(self.work / "public0033_bag_raw.csv")},
            },
            "infra0021_runtime_receipt.json": {
                "schema_version": "infra0021_runtime_receipt_v1",
                "status": "passed",
                "input": {
                    "uid_sha256": uid_hash,
                    "study_count": 2,
                    "test_csv_sha256": guard.sha256(self.competition / "test.csv"),
                    "test_series_csv_sha256": guard.sha256(self.competition / "test_series.csv"),
                },
                "output": {
                    "submission_sha256": guard.sha256(self.work / "submission_parent_exact.csv"),
                    "backup_sha256": guard.sha256(self.work / "submission_parent_exact.csv"),
                    "shape": [2, 13],
                },
            },
            "public0033_overlay_receipt.json": {
                "schema_version": "public0033_medial_t30_r60_b10_overlay_v1",
                "status": "passed",
                "parent_uid_sha256": uid_hash,
                "parent_submission_sha256": guard.sha256(self.work / "submission_parent_exact.csv"),
                "bag_raw_sha256": guard.sha256(self.work / "public0033_bag_raw.csv"),
                "output_sha256": guard.sha256(self.work / "submission_dinosaur_v4_0937_control.csv"),
            },
            "dinosaur_v4_v6_receipt.json": {
                "schema_version": "dinosaur_v4_v6_outer_weights_v1",
                "status": "passed",
                "output_sha256": guard.sha256(self.work / "submission.csv"),
                "fixed_outer_weights": guard.WEIGHTS,
                "medial_meniscus_route_preserved": True,
                "new_models_loaded": False,
            },
        }
        self.persist_receipts()

    def persist_receipts(self):
        for name, receipt in self.receipts.items():
            guard.save_json(self.work / name, receipt)

    def update_final(self):
        write_csv(self.work / "submission.csv", self.final)
        write_csv(self.work / "submission_dinosaur_v4_v6_candidate.csv", self.final)
        self.receipts["dinosaur_v4_v6_receipt.json"]["output_sha256"] = guard.sha256(self.work / "submission.csv")
        self.persist_receipts()

    def reject_finish(self, expression):
        with self.assertRaisesRegex((ValueError, RuntimeError, KeyError), expression):
            self.candidate.finish()
        self.assert_rejected()

    def test_complete_receipts_and_reversed_bag_order_pass_without_changing_csv(self):
        before = (self.work / "submission.csv").read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.candidate.finish()
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["scoring_performed"])
        self.assertIsNone(result["public_score"])
        self.assertEqual((self.work / "submission.csv").read_bytes(), before)
        ready = json.loads((self.work / "P002_READY.json").read_text(encoding="utf-8"))
        self.assertEqual(ready["sha256"], digest(before))

    def test_wrong_receipt_schema_refuses_ready(self):
        self.receipts["public0033_cached_inference_receipt.json"]["schema_version"] = "unknown"
        self.persist_receipts()
        self.reject_finish("wrong-schema")

    def test_missing_receipt_refuses_ready(self):
        (self.work / "infra0021_runtime_receipt.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.candidate.finish()
        self.assert_rejected()

    def test_incomplete_source_execution_refuses_ready(self):
        self.candidate.completed.pop()
        self.reject_finish("Incomplete source")

    def test_completed_cells_without_observation_refuse_ready(self):
        self.candidate.observations.pop("dino_raw_valid")
        self.reject_finish("(?i)observ|member|completion")

    def test_previous_observer_failure_refuses_ready(self):
        self.candidate.failures.append("artificial hidden failure")
        self.reject_finish("(?i)observ|fail|member")

    def test_wrong_final_candidate_hash_refuses_ready(self):
        with (self.work / "submission_dinosaur_v4_v6_candidate.csv").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        self.reject_finish("Wrong final candidate")

    def test_wrong_control_hash_refuses_ready(self):
        self.receipts["public0033_overlay_receipt.json"]["output_sha256"] = "0" * 64
        self.persist_receipts()
        self.reject_finish("Control does not match")

    def test_consistent_hashes_do_not_allow_wrong_final_uid_order(self):
        self.final.reverse()
        self.update_final()
        self.reject_finish("UID order")

    def test_consistent_hashes_do_not_allow_nan_final(self):
        self.final[0][1] = "nan"
        self.update_final()
        self.reject_finish("Nonfinite")

    def test_consistent_hashes_do_not_allow_change_to_untouched_target(self):
        self.final[0][guard.HEADER.index("MCL")] = "0.2"
        self.update_final()
        self.reject_finish("Untouched target changed")

    def test_consistent_hashes_do_not_allow_specialist_to_change_mcl(self):
        column = guard.HEADER.index("MCL")
        self.control[0][column] = self.final[0][column] = "0.2"
        write_csv(self.work / "submission_dinosaur_v4_0937_control.csv", self.control)
        self.receipts["public0033_overlay_receipt.json"]["output_sha256"] = guard.sha256(
            self.work / "submission_dinosaur_v4_0937_control.csv"
        )
        self.update_final()
        self.reject_finish("non-target column")

    def test_boolean_receipt_gate_does_not_accept_integer(self):
        self.receipts["public0033_cached_inference_receipt.json"]["checkpoint"]["strict_load"] = 1
        self.persist_receipts()
        self.reject_finish("Receipt setting mismatch")

    def test_receipt_shape_drift_is_refused(self):
        self.receipts["public0033_cached_inference_receipt.json"]["cache"]["shape"][0] = 1
        self.persist_receipts()
        self.reject_finish("Receipt shape mismatch")

    def test_changed_competition_csv_refuses_ready(self):
        (self.competition / "test_series.csv").write_text("different metadata\n", encoding="utf-8")
        self.reject_finish("competition CSV hash mismatch")

    def test_bag_hash_cannot_use_test_order_when_bag_order_differs(self):
        self.receipts["public0033_cached_inference_receipt.json"]["study_uid_sha256"] = self.receipts[
            "infra0021_runtime_receipt.json"
        ]["input"]["uid_sha256"]
        self.persist_receipts()
        self.reject_finish("Specialist UID receipt mismatch")

    def test_changed_specialist_bundle_refuses_ready(self):
        (self.specialist / "bundle_manifest.json").write_text("{}", encoding="utf-8")
        self.reject_finish("Specialist bundle mismatch")

    def test_bag_wrong_uid_set_is_refused_even_with_updated_hashes(self):
        self.bag[0][0] = "artificial-outsider"
        write_csv(
            self.work / "public0033_bag_raw.csv", self.bag, ["StudyInstanceUID", "Medial Meniscus", "Lateral Meniscus"]
        )
        digest_now = guard.sha256(self.work / "public0033_bag_raw.csv")
        self.receipts["public0033_overlay_receipt.json"]["bag_raw_sha256"] = digest_now
        self.receipts["public0033_cached_inference_receipt.json"]["raw_output"]["sha256"] = digest_now
        self.persist_receipts()
        self.reject_finish("Specialist CSV study set drift")

    def test_bag_nan_is_refused_even_with_updated_hashes(self):
        self.bag[0][1] = "nan"
        write_csv(
            self.work / "public0033_bag_raw.csv",
            self.bag,
            ["StudyInstanceUID", "Medial Meniscus", "Lateral Meniscus"],
        )
        digest_now = guard.sha256(self.work / "public0033_bag_raw.csv")
        self.receipts["public0033_overlay_receipt.json"]["bag_raw_sha256"] = digest_now
        self.receipts["public0033_cached_inference_receipt.json"]["raw_output"]["sha256"] = digest_now
        self.persist_receipts()
        self.reject_finish("(?i)finite|range|prediction")


if __name__ == "__main__":
    unittest.main()
