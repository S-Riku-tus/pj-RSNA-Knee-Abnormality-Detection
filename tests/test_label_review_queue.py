"""Synthetic standard-library contracts for training-only report evidence sampling."""

import json
import tempfile
import unittest
from pathlib import Path

from rsna_knee.contracts import ID, SUBMISSION_COLUMNS, TARGETS, read_csv, sha256, write_csv
from rsna_knee.prepare import prepare
from scripts.build_label_review_queue import REPOSITORY, REVIEW_FIELDS, build_queue


class LabelReviewQueueTests(unittest.TestCase):
    def setUp(self):
        (REPOSITORY / "artifacts").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="label-review-test-", dir=REPOSITORY / "artifacts")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.train = self.root / "train.csv"
        self.weak = self.root / "weak-labels.csv"
        self.provenance = self.root / "provenance.json"
        self.states = self.root / "label_states.csv"
        self.groups = self.root / "groups.csv"
        self.config = self.root / "config.json"
        self.manifests = self.root / "manifests"
        self.train_rows = [
            {
                ID: f"synthetic-{i:03d}",
                "Report": f"Artificial report {i}\n架空の所見です。",
                **dict.fromkeys(TARGETS, ""),
            }
            for i in range(120)
        ]
        self.train_rows[0]["ACL"] = "1"
        self.train_rows[1]["Report"] = self.train_rows[0]["Report"]
        self.train_rows[3]["Report"] = self.train_rows[4]["Report"]
        write_csv(self.train, (ID, "Report", *TARGETS), self.train_rows)
        write_csv(
            self.groups,
            (ID, "group_id"),
            [
                {ID: row[ID], "group_id": "gold-bridge" if i in (1, 2) else f"pair-{i // 2}"}
                for i, row in enumerate(self.train_rows)
            ],
        )
        self.state_rows = [
            {ID: row[ID], **{t: ("P", "N", "B", "U", "M")[(i + j) % 5] for j, t in enumerate(TARGETS)}}
            for i, row in enumerate(self.train_rows[1:], start=1)
        ]
        self.weak_rows = [
            {ID: row[ID], **{t: {"P": "1", "N": "0", "B": "0", "U": "", "M": ""}[row[t]] for t in TARGETS}}
            for row in self.state_rows
        ]
        write_csv(self.weak, SUBMISSION_COLUMNS, self.weak_rows)
        write_csv(self.states, SUBMISSION_COLUMNS, self.state_rows)
        self.provenance.write_text(
            json.dumps(
                {
                    "source_url": "synthetic://review-tests",
                    "source_version": "1",
                    "license": "synthetic-only",
                    "method": "synthetic",
                    "created_at": "2026-10-06",
                    "gold_used_for_tuning": False,
                    "source_artifact_sha256": {"label_states.csv": sha256(self.states)},
                }
            ),
            encoding="utf-8",
        )
        self.config.write_text(json.dumps({"seed": 20261002, "n_folds": 5}), encoding="utf-8")

    def prepare(self):
        return prepare(self.train, self.weak, self.provenance, self.manifests, groups_path=self.groups)

    def build(self, name="queue", **kwargs):
        return build_queue(
            self.train,
            self.weak,
            self.provenance,
            self.manifests,
            self.config,
            self.root / name,
            groups_path=self.groups,
            **kwargs,
        )

    def test_protects_multiple_holdouts_gold_and_transitive_components(self):
        self.prepare()
        result = self.build(size=100, holdout_folds=(0, 1), states_path=self.states)
        _, rows = read_csv(self.root / "queue/studies.csv")
        selected = {r[ID] for r in rows}
        self.assertFalse(selected & {r[ID] for r in self.train_rows[:3]})
        _, prepared = read_csv(self.manifests / "weak.csv")
        withheld = {r[ID] for r in prepared if r["fold"] in ("0", "1")}
        self.assertFalse(selected & withheld)
        self.assertEqual(len(rows), len({r["component_id"] for r in rows}))
        self.assertEqual(result["selected_studies"], result["eligible_components"])
        self.assertTrue(result["fewer_than_requested"])
        self.assertEqual(result["holdout_folds"], [0, 1])
        self.assertFalse(result["patient_independence_verified"])

    def test_review_preserves_reports_missing_and_source_state_without_changing_inputs(self):
        self.prepare()
        before = {p: sha256(p) for p in self.root.glob("*.csv")}
        result = self.build(size=25, states_path=self.states)
        _, study_rows = read_csv(self.root / "queue/studies.csv")
        _, review_rows = read_csv(self.root / "queue/review.csv")
        original = {r[ID]: r for r in self.train_rows}
        source = {r[ID]: r for r in self.state_rows}
        self.assertEqual(len(review_rows), 25 * 12)
        self.assertEqual(sum(r["primary_review"] == "1" for r in review_rows), 25)
        for row in study_rows:
            self.assertEqual(row["Report"], original[row[ID]]["Report"])
        for row in review_rows:
            self.assertTrue(all(row[field] == "" for field in REVIEW_FIELDS))
            self.assertEqual(row["existing_source_state"], source[row[ID]][row["target"]])
            if row["existing_source_state"] in ("M", "U"):
                self.assertEqual(row["existing_label"], "")
                self.assertEqual(row["existing_label_kind"], "missing")
        self.assertEqual({p: sha256(p) for p in self.root.glob("*.csv")}, before)
        self.assertFalse(result["labels_changed"])
        self.assertNotIn("Artificial report", json.dumps(result))
        self.assertNotIn("synthetic-0", json.dumps(result))

    def test_sample_reproducible_across_input_order_and_seed_changes_selection(self):
        self.prepare()
        self.build("first", size=12, seed=17)
        header, rows = read_csv(self.manifests / "weak.csv")
        write_csv(self.manifests / "weak.csv", header, rows[::-1])
        self.build("reordered", size=12, seed=17)
        self.build("different-seed", size=12, seed=18)
        for name in ("studies.csv", "review.csv"):
            self.assertEqual(sha256(self.root / "first" / name), sha256(self.root / "reordered" / name))
        self.assertNotEqual(sha256(self.root / "first/studies.csv"), sha256(self.root / "different-seed/studies.csv"))

    def test_rare_negative_is_sampled_first_and_missing_is_never_negative(self):
        for row in self.weak_rows:
            row["Synovitis"] = ""
        write_csv(self.weak, SUBMISSION_COLUMNS, self.weak_rows)
        self.prepare()
        header, prepared = read_csv(self.manifests / "weak.csv")
        rare = next(r for r in prepared if r["fold"] != "0")
        # Preserve the same grouping/fold assignment when adding one observed source label.
        next(r for r in self.weak_rows if r[ID] == rare[ID])["Synovitis"] = "0"
        write_csv(self.weak, SUBMISSION_COLUMNS, self.weak_rows)
        rare["Synovitis"] = "0"
        write_csv(self.manifests / "weak.csv", header, prepared)
        metadata = json.loads((self.manifests / "manifest.json").read_text(encoding="utf-8"))
        metadata["inputs_sha256"]["weak"] = sha256(self.weak)
        (self.manifests / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
        result = self.build(size=1)
        _, rows = read_csv(self.root / "queue/studies.csv")
        self.assertEqual(rows[0][ID], rare[ID])
        self.assertEqual(rows[0]["primary_target"], "Synovitis")
        self.assertEqual(rows[0]["primary_label_kind"], "negative")
        synovitis = [r for r in result["stratum_coverage"] if r["target"] == "Synovitis"]
        self.assertEqual({r["label_kind"] for r in synovitis}, {"negative", "missing"})

    def test_soft_values_are_not_thresholded(self):
        for row in self.weak_rows:
            row["Synovitis"] = "0.37"
        write_csv(self.weak, SUBMISSION_COLUMNS, self.weak_rows)
        self.prepare()
        self.build(size=3)
        _, rows = read_csv(self.root / "queue/review.csv")
        for row in rows:
            if row["target"] == "Synovitis":
                self.assertEqual(row["existing_label"], "0.37")
                self.assertEqual(row["existing_label_kind"], "soft")

    def test_unobserved_target_still_allows_semantic_audit(self):
        for row in self.weak_rows:
            row["Synovitis"] = ""
        write_csv(self.weak, SUBMISSION_COLUMNS, self.weak_rows)
        self.prepare()
        self.build(size=3)
        audit = json.loads((self.root / "queue/fold-audit.json").read_text(encoding="utf-8"))
        self.assertFalse(audit["ready_for_training"])

    def test_rejects_report_or_group_leakage_before_writing(self):
        self.prepare()
        header, rows = read_csv(self.manifests / "weak.csv")
        members = [r for r in rows if r[ID] in ("synthetic-006", "synthetic-007")]
        members[0]["group_id"] = "fabricated"
        members[0]["fold"] = str((int(members[1]["fold"]) + 1) % 5)
        write_csv(self.manifests / "weak.csv", header, rows)
        with self.assertRaisesRegex(ValueError, "Report/supplied group"):
            self.build()
        self.assertFalse((self.root / "queue").exists())

    def test_rejects_changed_source_hash_and_unreviewed_states(self):
        self.prepare()
        self.states.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "states must match"):
            self.build(states_path=self.states)
        self.assertFalse((self.root / "queue").exists())
        self.weak.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "input hashes"):
            self.build()
        self.assertFalse((self.root / "queue").exists())

    def test_rejects_gold_tuned_source(self):
        provenance = json.loads(self.provenance.read_text(encoding="utf-8"))
        provenance["gold_used_for_tuning"] = True
        self.provenance.write_text(json.dumps(provenance), encoding="utf-8")
        self.prepare()
        with self.assertRaisesRegex(ValueError, "no gold tuning"):
            self.build()
        self.assertFalse((self.root / "queue").exists())

    def test_rejects_overwrite_public_output_and_invalid_holdout(self):
        self.prepare()
        self.build(size=2)
        original_hash = sha256(self.root / "queue/review.csv")
        with self.assertRaisesRegex(ValueError, "new output"):
            self.build(size=3)
        self.assertEqual(original_hash, sha256(self.root / "queue/review.csv"))
        with self.assertRaisesRegex(ValueError, "Git-ignored"):
            self.build(REPOSITORY / "docs/unsafe-report-review")
        for folds in ((), (-1,), (5,), (0, 1, 2, 3, 4)):
            with self.subTest(folds=folds), self.assertRaises(ValueError):
                self.build("invalid-fold", holdout_folds=folds)
        self.assertFalse((self.root / "invalid-fold").exists())


if __name__ == "__main__":
    unittest.main()
