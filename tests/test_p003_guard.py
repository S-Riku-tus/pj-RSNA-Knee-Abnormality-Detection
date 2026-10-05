"""Artificial, label-free submission checks; no public model or MRI execution."""

import ast
import copy
import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p003_guard", ROOT / "scripts/p003_guard.py")
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


class P003Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ids = ["a", "b", "c", "d"]

    def tearDown(self):
        self.tmp.cleanup()

    def csv(self, path, values, ids=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["StudyInstanceUID", *guard.LABELS])
            writer.writerows([[uid, *row] for uid, row in zip(ids or self.ids, values)])

    def npz(self, path, values, ids=None, **extra):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, study_uids=np.asarray(ids or self.ids), labels=guard.LABELS, values=values, **extra)

    def blend(self):
        return {
            "members": list(guard.MEMBERS),
            "study_count": 4,
            "inner_rerank": False,
            "private_alpha": 0.4,
            "public_raptor_alpha": 0.6,
            "within_coat": dict.fromkeys(guard.MEMBERS, 0.25),
            "finding_specific_weights": False,
            "family_reduction": "rank_of_member_probability_mean",
        }

    def test_partial_or_alternate_blend_is_rejected(self):
        guard.check_blend_receipt(self.blend(), 4)
        for key, value in (
            ("members", list(guard.MEMBERS)[:3]),
            ("inner_rerank", True),
            ("family_reduction", "rank_sum_fallback_probabilities_unavailable"),
            ("private_alpha", 0.45),
            ("study_count", 3),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                guard.check_blend_receipt({**self.blend(), key: value}, 4)

    def test_ranks_must_cover_whole_cohort_and_average_ties(self):
        scores = np.asarray([[0.1], [0.2], [0.8], [0.9]])
        np.testing.assert_array_equal(guard.rank_pct(scores).ravel(), [0.25, 0.5, 0.75, 1])
        chunked = np.concatenate([guard.rank_pct(scores[:2]), guard.rank_pct(scores[2:])])
        self.assertFalse(np.array_equal(chunked, guard.rank_pct(scores)))
        np.testing.assert_array_equal(guard.rank_pct([[1], [1], [3]]).ravel(), [0.5, 0.5, 1])

    def test_probability_mean_order_differs_from_mean_rank(self):
        # A confident second member reverses the first one's near tie.
        first = np.asarray([[0.49], [0.50]])
        second = np.asarray([[0.99], [0.01]])
        probability_first = guard.rank_pct((first + second) / 2)
        rank_first = (guard.rank_pct(first) + guard.rank_pct(second)) / 2
        self.assertGreater(probability_first[0, 0], probability_first[1, 0])
        self.assertEqual(rank_first[0, 0], rank_first[1, 0])

    def test_npz_aligns_uid_and_rejects_duplicates_extra_nonfinite_and_wrong_labels(self):
        values = np.arange(48).reshape(4, 12) / 50
        path = self.root / "raw.npz"
        self.npz(path, values[::-1], self.ids[::-1])
        np.testing.assert_array_equal(guard.aligned_npz(path, self.ids), values)
        for ids in (["a", "b", "c", "c"], ["a", "b", "c", "x"]):
            self.npz(path, values, ids)
            with self.assertRaises(ValueError):
                guard.aligned_npz(path, self.ids)
        for value in (np.nan, np.inf, -0.01, 1.01):
            bad = values.copy()
            bad[0, 0] = value
            self.npz(path, bad)
            with self.assertRaises(ValueError):
                guard.aligned_npz(path, self.ids)
        np.savez(path, study_uids=self.ids, labels=guard.LABELS[::-1], values=values)
        with self.assertRaises(ValueError):
            guard.aligned_npz(path, self.ids)

    def test_csv_cannot_silently_reorder_labels_or_studies(self):
        path = self.root / "submission.csv"
        self.csv(path, np.full((4, 12), 0.5), self.ids[::-1])
        with self.assertRaises(ValueError):
            guard.read_scores(path, self.ids)

    def test_raw_member_nonfinite_before_averaging_is_rejected(self):
        path = self.root / "raw.npz"
        values = np.ones((3, 4, 12)) / 2
        values[1, 0, 0] = np.nan
        np.savez(path, study_uids=self.ids, raw_probabilities=values)
        with self.assertRaises(ValueError):
            guard.aligned_npz(path, self.ids, "probability_mean", False)

    def test_a5_absence_is_distinct_from_inference_failure(self):
        obj = guard.P003Guard({}, self.root)
        obj.ids = self.ids
        raw = np.ones((5, 4, 12)) / 2
        raw[:, 1] = np.nan
        path = self.root / "speed_a5_raw.npz"
        np.savez(path, study_uids=self.ids, raw_probabilities=raw, applicable=[True, False, True, True])
        self.npz(self.root / "diagnostics/a5_rank_mean.npz", np.ones((4, 12)) / 2)
        obj.check_a5()
        raw[:, 0] = np.nan
        np.savez(path, study_uids=self.ids, raw_probabilities=raw, applicable=[True, False, True, True])
        with self.assertRaises(ValueError):
            obj.check_a5()
        with self.assertRaises(ValueError):
            obj.check_events({"_RSNA_AUDIT": {"events": [{"kind": "a5_study_failed"}]}})

    def test_fallback_events_and_unknown_events_are_rejected(self):
        obj = guard.P003Guard({}, self.root)
        obj.check_events(
            {"_RSNA_AUDIT": {"events": [{"kind": "cache_complete"}, {"kind": "raptor_fp16_nonfinite_retry_fp32"}]}}
        )
        for event in ("raptor_neutral_fill", "coat_d4_child_failed", "coat_family_partial", "unexpected"):
            with self.subTest(event=event), self.assertRaises(ValueError):
                obj.check_events({"_RSNA_AUDIT": {"events": [{"kind": event}]}})

    def test_reject_removes_only_owned_submission(self):
        path = self.root / "submission.csv"
        path.write_text("existing")
        obj = guard.P003Guard({}, self.root)
        obj.reject("preflight failed")
        self.assertEqual(path.read_text(), "existing")
        obj.owns_outputs = True
        obj.reject("invalid recipe")
        self.assertFalse(path.exists())
        self.assertEqual((self.root / "P003_REJECTED_submission.csv.disabled").read_text(), "existing")

    def test_source_hash_and_cell_order_fail_before_execution(self):
        source = "raise AssertionError('source must not run')"
        contract = {"cell_order": [2], "source_cells": {"2": hashlib.sha256(source.encode()).hexdigest()}}
        for number, value in ((3, source), (2, source + "\n")):
            obj = guard.P003Guard(contract, self.root)
            obj.ready = True
            with self.assertRaises((ValueError, RuntimeError)):
                obj.run_cell(number, value, {})

    def test_fresh_work_directory_required(self):
        (self.root / "old-experiment.json").write_text("keep")
        obj = guard.P003Guard({"inputs": []}, self.root)
        with self.assertRaises(RuntimeError):
            obj.prepare_files()
        self.assertEqual((self.root / "old-experiment.json").read_text(), "keep")

    def test_preflight_checks_hash_mounts_and_path_boundaries(self):
        inputs, work = self.root / "input", self.root / "working"
        asset = inputs / "model"
        comp = inputs / "competition"
        asset.mkdir(parents=True)
        (comp / "test_series").mkdir(parents=True)
        (asset / "weights.bin").write_bytes(b"artificial payload")
        for name in ("train.csv", "test.csv", "test_series.csv"):
            (comp / name).write_text("StudyInstanceUID\na\nb\nc\nd\n", encoding="utf-8")
        self.csv(comp / "sample_submission.csv", np.full((4, 12), 0.5))
        contract = {
            "inputs": [
                {
                    "key": "model",
                    "ref": "model",
                    "version": 1,
                    "mounts": ["model"],
                    "files": [{"path": "weights.bin", "bytes": 18, "sha256": guard.sha256(asset / "weights.bin")}],
                },
                {"key": "competition", "ref": "competition", "version": None, "mounts": ["competition"], "files": []},
            ]
        }
        obj = guard.P003Guard(contract, work, inputs)
        obj.prepare_files()
        self.assertEqual(obj.ids, self.ids)
        (asset / "weights.bin").write_bytes(b"corrupted payload!")
        with self.assertRaises(ValueError):
            guard.P003Guard(contract, work, inputs).prepare_files()
        (asset / "weights.bin").write_bytes(b"artificial payload")
        (inputs / "shadow").mkdir()
        with self.assertRaises(ValueError):
            guard.P003Guard(contract, work, inputs).prepare_files()
        bad = copy.deepcopy(contract)
        bad["inputs"][0]["files"][0]["path"] = "../../outside"
        with self.assertRaises(ValueError):
            guard.P003Guard(bad, work, inputs).prepare_files()

    def test_generator_preserves_all_code_and_refuses_overwrite(self):
        spec = importlib.util.spec_from_file_location("p003_builder", ROOT / "scripts/build_p003_candidate.py")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        first, second = self.root / "one", self.root / "two"
        builder.build(first)
        builder.build(second)
        path = first / "02_submit_p003.ipynb"
        self.assertEqual(path.read_bytes(), (second / path.name).read_bytes())
        original = json.loads(builder.SOURCE.read_text(encoding="utf-8"))
        generated = json.loads(path.read_text(encoding="utf-8"))
        observed = []
        for cell in generated["cells"]:
            if cell["cell_type"] != "code":
                continue
            self.assertEqual(cell["outputs"], [])
            self.assertIsNone(cell["execution_count"])
            tree = ast.parse("".join(cell["source"]))
            if cell["id"].startswith("p003-source-"):
                call = tree.body[-1].value
                number, source = (ast.literal_eval(arg) for arg in call.args[:2])
                self.assertEqual(source, "".join(original["cells"][number]["source"]))
                observed.append(number)
        self.assertEqual(observed, [i for i, c in enumerate(original["cells"]) if c["cell_type"] == "code"])
        with self.assertRaises(FileExistsError):
            builder.build(first)

    def test_full_recipe_replay_and_corruption_gates(self):
        rng = np.random.default_rng(12)
        parent, raptor = rng.random((4, 12)), rng.random((4, 12))
        coats = [rng.random((4, 12)) for _ in range(4)]
        final, family = guard.replay_final(parent, raptor, coats)
        self.npz(self.root / "btk_v32_rest_input.npz", parent)
        self.csv(self.root / "raptor_input_before_coat.csv", raptor)
        self.csv(self.root / "_coat_family_rank.csv", family)
        self.csv(self.root / "_pipeline_stage.csv", final)
        guard.save_json(self.root / "_coat_raptor_blend_receipt.json", self.blend())
        for name in ("resgated_runtime.log", "global96_runtime.log", "d4_input/d4.log", "repairv1_runtime.log"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("success", encoding="utf-8")
        for filename, value in zip(guard.PROB_FILES, coats):
            path = self.root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, study_uids=self.ids[::-1], probability_mean=value[::-1])
        for name, status, epochs, models in zip(
            guard.RECEIPT_FILES, guard.RECEIPT_STATUS, ([4, 6, 8], [16, 23, 18], [], [12, 7, 11]), (3, 3, 1, 3)
        ):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            guard.save_json(
                path,
                {
                    "status": status,
                    "fallback_studies": 0,
                    "models": models,
                    "checkpoints": [{"epoch": e} for e in epochs],
                },
            )
        namespace = {
            "_coatnet_weight": guard.OUTER,
            "_blend_cr": 0.6 * guard.rank_pct(raptor) + 0.4 * family,
            "_blend_tr": guard.rank_pct(parent),
            "_ke_input_ids": {
                (name, uid) for name in ("maxspan-v5", "native384dense-v10", "native384-v8") for uid in self.ids
            },
        }
        obj = guard.P003Guard({}, self.root)
        obj.ids = self.ids
        obj.check_coats(namespace)
        log = self.root / "resgated_runtime.log"
        log.write_text("[dense-fallback] use different input", encoding="utf-8")
        with self.assertRaises(ValueError):
            obj.check_coats(namespace)
        log.write_text("success", encoding="utf-8")
        wrong = copy.deepcopy(namespace)
        wrong["_ke_input_ids"].pop()
        with self.assertRaises(ValueError):
            obj.check_coats(wrong)
        self.csv(self.root / "_pipeline_stage.csv", final[::-1])
        with self.assertRaises(ValueError):
            obj.check_coats(namespace)


if __name__ == "__main__":
    unittest.main()
