"""Synthetic CSV checks for missing-label semantics and independent leakage auditing."""

import json
import tempfile
import unittest
from pathlib import Path

from rsna_knee.contracts import ID, SUBMISSION_COLUMNS, TARGETS, read_csv, sha256, write_csv
from rsna_knee.prepare import prepare
from scripts.audit_training_labels import audit_folds, audit_labels, convert_pilkwang, import_vmohitrao, target_summary


class LabelAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.train = self.root / "train.csv"
        self.labels = self.root / "labels.csv"
        self.provenance = self.root / "provenance.json"
        self.config = self.root / "config.json"
        self.manifests = self.root / "manifests"
        self.rows = [
            {ID: f"s{i:02d}", "Report": f"synthetic report {i}", **{t: "" for t in TARGETS}} for i in range(40)
        ]
        self.rows[0]["ACL"] = "1"
        self.rows[1]["Report"] = self.rows[0]["Report"]
        self.rows[3]["Report"] = self.rows[2]["Report"]
        write_csv(self.train, (ID, "Report", *TARGETS), self.rows)
        self.label_rows = [{ID: row[ID], **{t: "0.7" for t in TARGETS}} for row in self.rows]
        write_csv(self.labels, SUBMISSION_COLUMNS, self.label_rows)
        self.provenance.write_text(
            json.dumps(
                {
                    "source_url": "synthetic://tests",
                    "source_version": "1",
                    "license": "synthetic-only",
                    "method": "synthetic",
                    "created_at": "2026-10-03",
                    "gold_used_for_tuning": False,
                }
            ),
            encoding="utf-8",
        )
        self.config.write_text(json.dumps({"seed": 20261002, "n_folds": 5}), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def fold_audit(self, groups=None):
        return audit_folds(
            self.train,
            self.labels,
            self.provenance,
            self.manifests,
            self.root / "audit.json",
            self.config,
            groups,
        )

    def prepare(self, groups=None):
        return prepare(self.train, self.labels, self.provenance, self.manifests, groups_path=groups)

    def public_rows(self):
        return [
            {
                ID: "0001",
                **{
                    column: value
                    for t in TARGETS
                    for column, value in ((t, "0.5"), (t + "__conf", "0.2"), (t + "__verdict", "UNK"))
                },
            }
        ]

    def test_converter_preserves_missing_zero_soft_and_identifier(self):
        rows = self.public_rows()
        rows[0]["ACL"], rows[0]["ACL__verdict"] = "0", "NO"
        rows[0]["MCL"], rows[0]["MCL__verdict"] = "0.84", "YES"
        source, output = self.root / "public.csv", self.root / "canonical.csv"
        write_csv(source, rows[0].keys(), rows)
        result = convert_pilkwang(source, output, self.root / "conversion.json")
        header, converted = read_csv(output)
        self.assertEqual(tuple(header), SUBMISSION_COLUMNS)
        self.assertEqual(converted[0][ID], "0001")
        self.assertEqual(converted[0]["ACL"], "0.0")
        self.assertEqual(converted[0]["MCL"], "0.84")
        self.assertEqual(converted[0]["Synovitis"], "")
        self.assertEqual(result["masked_unknown_by_target"]["Synovitis"], 1)

    def test_converter_rejects_unrecognized_schema_before_writing(self):
        source, output = self.root / "public.csv", self.root / "canonical.csv"
        for invalid in ("MAYBE", "yes"):
            rows = self.public_rows()
            rows[0]["ACL__verdict"] = invalid
            write_csv(source, rows[0].keys(), rows)
            with self.assertRaisesRegex(ValueError, "verdict"):
                convert_pilkwang(source, output, self.root / "conversion.json")
            self.assertFalse(output.exists())
        rows = self.public_rows()
        rows[0]["ACL"] = "-1"
        write_csv(source, rows[0].keys(), rows)
        with self.assertRaises(ValueError):
            convert_pilkwang(source, output, self.root / "conversion.json")
        self.assertFalse(output.exists())

    def test_summary_separates_empty_binary_and_soft(self):
        rows = [{t: value for t in TARGETS} for value in ("", "0", "1", "0.5", "NA")]
        stat = target_summary(rows)["ACL"]
        self.assertEqual(
            (stat["observed"], stat["missing"], stat["exact_zero"], stat["exact_one"], stat["soft"]), (3, 2, 1, 1, 1)
        )
        self.assertEqual(stat["mean"], 0.5)

    def test_label_audit_flags_gold_matches_without_claiming_copy(self):
        self.label_rows[0]["ACL"] = "1"
        write_csv(self.labels, SUBMISSION_COLUMNS, self.label_rows)
        result = audit_labels(self.train, self.labels, self.root / "audit.json")
        self.assertEqual(result["official_rows_matching_all_observed_labels"], 1)
        self.assertIn("not proof", result["official_match_note"])
        self.assertNotIn("synthetic report", json.dumps(result))
        with self.assertRaisesRegex(ValueError, "new output"):
            audit_labels(self.train, self.labels, self.root / "audit.json")

    def test_fold_audit_checks_soft_distributions_and_gold_exposure(self):
        provenance = json.loads(self.provenance.read_text())
        provenance["gold_used_for_tuning"] = True
        self.provenance.write_text(json.dumps(provenance))
        self.prepare()
        result = self.fold_audit()
        self.assertTrue(result["integrity_valid"])
        self.assertTrue(result["ready_for_training"])
        self.assertEqual(result["gold_evaluation"], "exploratory")
        self.assertFalse(result["patient_independence_verified"])
        for split in result["per_fold"].values():
            self.assertEqual(split["valid"]["ACL"]["soft"], split["valid_studies"])

    def test_independent_report_check_rejects_fabricated_group_id(self):
        self.prepare()
        header, rows = read_csv(self.manifests / "weak.csv")
        changed = next(row for row in rows if row[ID] == "s03")
        changed["group_id"] = "fabricated"
        changed["fold"] = str((int(changed["fold"]) + 1) % 5)
        write_csv(self.manifests / "weak.csv", header, rows)
        with self.assertRaisesRegex(ValueError, "Report/supplied group"):
            self.fold_audit()

    def test_transitive_gold_link_through_excluded_bridge_is_rejected(self):
        self.rows[4]["Report"] = self.rows[0]["Report"]
        write_csv(self.train, (ID, "Report", *TARGETS), self.rows)
        groups = self.root / "groups.csv"
        write_csv(
            groups,
            (ID, "group_id"),
            [{ID: row[ID], "group_id": "bridge" if i in (3, 4) else f"p{i}"} for i, row in enumerate(self.rows)],
        )
        self.prepare(groups)
        header, rows = read_csv(self.manifests / "weak.csv")
        rows.append(
            {ID: "s03", **{t: "0.7" for t in TARGETS}, "source": "report_weak", "group_id": "fabricated", "fold": "0"}
        )
        write_csv(self.manifests / "weak.csv", header, rows)
        with self.assertRaisesRegex(ValueError, "connected to gold"):
            self.fold_audit(groups)

    def test_source_hash_and_values_cannot_be_changed_silently(self):
        self.prepare()
        header, rows = read_csv(self.manifests / "weak.csv")
        rows[0]["ACL"] = ""
        write_csv(self.manifests / "weak.csv", header, rows)
        with self.assertRaisesRegex(ValueError, "values differ"):
            self.fold_audit()
        write_csv(self.labels, SUBMISSION_COLUMNS, self.label_rows[::-1])
        with self.assertRaisesRegex(ValueError, "hashes"):
            self.fold_audit()

    def test_unobserved_target_blocks_training_readiness(self):
        for row in self.label_rows:
            row["Synovitis"] = ""
        write_csv(self.labels, SUBMISSION_COLUMNS, self.label_rows)
        self.prepare()
        result = self.fold_audit()
        self.assertFalse(result["ready_for_training"])
        self.assertEqual(len(result["warnings"]), 10)
        self.assertTrue(all("Synovitis" in warning for warning in result["warnings"]))

    def vmohitrao_source(self):
        source = self.root / "source"
        source.mkdir()
        keys = [row[ID] for row in self.rows[1:]]
        for filename, value in (("weak_labels.csv", "1"), ("label_weights.csv", "1"), ("label_states.csv", "P")):
            write_csv(source / filename, SUBMISSION_COLUMNS, [{ID: key, **{t: value for t in TARGETS}} for key in keys])
        write_csv(
            source / "label_provenance.csv",
            (ID, "report_group"),
            [{ID: key, "report_group": f"report-{i}"} for i, key in enumerate(keys)],
        )
        manifest = self.root / "source-manifest.json"
        metadata = {
            "source_train_sha256": sha256(self.train),
            "expert_studies_excluded": 1,
            "generated_study_rows": len(keys),
            "states": {t: {"P": len(keys)} for t in TARGETS},
            "api_usage": {"generated_at_utc": "2026-10-03"},
            "model": ["synthetic"],
            "artifact_sha256": {path.name: sha256(path) for path in source.glob("*.csv")},
        }
        manifest.write_text(json.dumps(metadata), encoding="utf-8")
        readme = self.root / "source-readme.txt"
        readme.write_text(
            "All 1 expert-annotated studies were excluded from report extraction and prompt development.",
            encoding="utf-8",
        )
        return source, manifest, readme

    def test_importer_preserves_labels_and_records_author_declaration(self):
        source, manifest, readme = self.vmohitrao_source()
        output = self.root / "imported"
        result = import_vmohitrao(source, manifest, readme, self.train, output, 3)
        self.assertEqual(result["official_studies_present"], 0)
        self.assertEqual(sha256(source / "weak_labels.csv"), sha256(output / "weak_labels.csv"))
        _, groups = read_csv(output / "groups.csv")
        self.assertEqual(len(groups), len(self.rows))
        provenance = json.loads((output / "provenance.json").read_text())
        self.assertIn("Author README", provenance["gold_usage_basis"])
        self.assertIn("not established", provenance["adoption_scope"])

    def test_importer_rejects_hash_and_state_mask_mismatch_before_writing(self):
        source, manifest, readme = self.vmohitrao_source()
        output = self.root / "imported"
        header, rows = read_csv(source / "label_weights.csv")
        rows[0]["ACL"] = "0"
        write_csv(source / "label_weights.csv", header, rows)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            import_vmohitrao(source, manifest, readme, self.train, output, 3)
        metadata = json.loads(manifest.read_text())
        metadata["artifact_sha256"]["label_weights.csv"] = sha256(source / "label_weights.csv")
        manifest.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "state/value"):
            import_vmohitrao(source, manifest, readme, self.train, output, 3)
        self.assertFalse(output.exists())

    def test_importer_rejects_missing_gold_declaration(self):
        source, manifest, readme = self.vmohitrao_source()
        readme.write_text("Gold usage unconfirmed")
        with self.assertRaisesRegex(ValueError, "declaration"):
            import_vmohitrao(source, manifest, readme, self.train, self.root / "imported", 3)


if __name__ == "__main__":
    unittest.main()
