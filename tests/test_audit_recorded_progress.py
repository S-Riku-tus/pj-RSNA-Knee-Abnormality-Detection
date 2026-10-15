"""Artificial metadata only: provenance boundaries and immutable audit outputs."""

import copy
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / "scripts/audit_recorded_progress.py"
SPEC = importlib.util.spec_from_file_location("audit_recorded_progress", MODULE)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class RecordedProgressTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.addCleanup(self.temporary.cleanup)
        self.evidence = b'{"status":"COMPLETE"}\n'
        (self.root / "official.json").write_bytes(self.evidence)
        self.saved = {
            "evidence": {"official": self.descriptor("official.json", self.evidence)},
            "adopted_current_best": {"candidate": "V22R", "submission_ref": 123,
                                     "own_official_Public": "0.956"},
            "own_confirmed_Public_best": "0.956", "goal_target": 0.960, "goal_operator": ">=",
            "official_observation_at_UTC": "2026-10-14T05:33:16+00:00",
            "checkpoint_basis_at_UTC": "2026-10-14T05:59:29+00:00",
            "unresolved_formal_submissions": {"P028": 124, "P024": 125, "F006": 126},
            "own_Public_by_target": {"V22R": "0.956", "P028": None, "P024": None, "F006": None},
            "quota_after_Submit_scope": {"derived_today": 4, "derived_remaining": 1, "derived_total": 27,
                                        "post_Submit_GetSubmissionLimits_performed_for_this_record": False},
        }
        self.write_checkpoint()
        self.now = datetime(2026, 10, 15, 5, 33, 16, tzinfo=timezone.utc)

    @staticmethod
    def descriptor(path, body):
        return {"path": path, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}

    def write_checkpoint(self):
        (self.root / "checkpoint.json").write_text(json.dumps(self.saved), encoding="utf-8")

    def run_audit(self):
        return audit.audit_progress(self.root, "checkpoint.json", now=self.now)

    def test_recorded_score_gap_and_historical_quota_do_not_become_live_facts(self):
        report = self.run_audit()
        self.assertEqual(report["recorded_Public"]["last_recorded_score"], "0.956")
        self.assertEqual(report["recorded_Public"]["observation_age_seconds"], 86400)
        self.assertFalse(report["recorded_Public"]["live_verified"])
        self.assertIsNone(report["recorded_Public"]["current_official_score"])
        self.assertEqual(report["goal"]["gap_from_recorded_score"], "0.004")
        self.assertFalse(report["goal"]["recorded_score_meets_target"])
        self.assertIsNone(report["goal"]["achievement_probability"])
        self.assertEqual(report["historical_quota"]["derived_remaining"], 1)
        self.assertIsNone(report["historical_quota"]["current_remaining"])
        self.assertEqual(len(report["unresolved_in_checkpoint"]), 3)
        self.assertTrue(all(row["current_status"] is None for row in report["unresolved_in_checkpoint"]))
        self.assertEqual(report["evidence_status_counts"], {"matched": 1})

    def test_missing_size_and_hash_mismatches_are_distinct_from_verification(self):
        self.saved["evidence"]["missing"] = self.descriptor("missing.json", b"{}")
        self.saved["evidence"]["wrong_size"] = self.descriptor("official.json", self.evidence)
        self.saved["evidence"]["wrong_size"]["bytes"] += 1
        self.saved["evidence"]["wrong_hash"] = self.descriptor("official.json", self.evidence)
        self.saved["evidence"]["wrong_hash"]["sha256"] = "0" * 64
        self.write_checkpoint()
        report = self.run_audit()
        self.assertEqual(report["evidence_status_counts"], {"matched": 1, "mismatch": 2, "missing": 1})
        rows = {row["location"].split(".")[-1]: row for row in report["evidence"]}
        self.assertFalse(rows["wrong_size"]["size_matches"])
        self.assertTrue(rows["wrong_size"]["sha256_matches"])
        self.assertFalse(rows["wrong_hash"]["sha256_matches"])
        self.assertFalse(report["recorded_Public"]["live_verified"])

    def test_unsafe_paths_and_model_files_are_rejected_before_read(self):
        paths = ("../escape.json", "..\\escape.json", "C:\\outside.json", "//server/share/x.json",
                 "/outside.json", "model.pt", "nested/../../escape.json")
        for name, path in enumerate(paths):
            self.saved["evidence"][str(name)] = self.descriptor(path, b"{}")
        self.write_checkpoint()
        report = self.run_audit()
        self.assertEqual(report["evidence_status_counts"], {"matched": 1, "rejected": len(paths)})
        with self.assertRaises(ValueError):
            audit.audit_progress(self.root, "../checkpoint.json", now=self.now)

    def test_symlink_escape_is_rejected_without_opening_target(self):
        with tempfile.TemporaryDirectory() as other:
            target = Path(other) / "external.json"
            target.write_text("{}", encoding="utf-8")
            link = self.root / "link.json"
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("Creating symbolic links is unavailable on this system")
            self.saved["evidence"]["link"] = self.descriptor("link.json", b"{}")
            self.write_checkpoint()
            original_open = Path.open

            def guarded_open(path, *args, **kwargs):
                if path == target or path == link:
                    self.fail("Escaping evidence must never be opened")
                return original_open(path, *args, **kwargs)

            with patch.object(Path, "open", guarded_open):
                report = self.run_audit()
            self.assertEqual(report["evidence_status_counts"], {"matched": 1, "rejected": 1})

    def test_external_resolution_is_rejected_even_when_link_creation_is_unavailable(self):
        with tempfile.TemporaryDirectory() as other:
            external = Path(other).resolve() / "outside.json"
            unresolved = self.root / "link.json"
            descriptor = self.descriptor("link.json", b"{}")
            original_resolve = Path.resolve

            def resolve(path, *args, **kwargs):
                return external if path == unresolved else original_resolve(path, *args, **kwargs)

            with patch.object(Path, "resolve", resolve), patch.object(Path, "open") as opened:
                result = audit.audit_evidence(self.root, "external_link", descriptor)
            self.assertEqual(result["status"], "rejected")
            opened.assert_not_called()

    def test_bad_descriptors_and_oversized_metadata_fail_closed(self):
        for value in (True, -1, "20"):
            descriptor = self.descriptor("official.json", self.evidence)
            descriptor["bytes"] = value
            self.assertEqual(audit.audit_evidence(self.root, "test", descriptor)["status"], "invalid_descriptor")
        descriptor = self.descriptor("official.json", self.evidence)
        descriptor["sha256"] = "not-a-sha"
        self.assertEqual(audit.audit_evidence(self.root, "test", descriptor)["status"], "invalid_descriptor")
        with patch.object(audit, "MAX_METADATA_BYTES", 1):
            self.assertEqual(audit.audit_evidence(self.root, "test", self.saved["evidence"]["official"])["status"],
                             "rejected")

    def test_credentials_are_rejected_before_any_content_read(self):
        for path in ("kaggle.json", "KAGGLE.JSON", "somewhere/credentials.json", ".aws/config.json",
                     ".codex/settings.json", ".config/app/token.json", ".ssh/metadata.json"):
            with self.subTest(path=path), patch.object(Path, "open") as opened:
                result = audit.audit_evidence(self.root, "credential", self.descriptor(path, b"{}"))
                self.assertEqual(result["status"], "rejected")
                opened.assert_not_called()
        with patch.object(Path, "open") as opened, self.assertRaises(ValueError):
            audit.json_bytes(self.root / "kaggle.json")
        opened.assert_not_called()

    def test_cli_creates_new_output_and_preserves_previous_bytes(self):
        arguments = ["--repo-root", str(self.root), "--checkpoint", "checkpoint.json", "--output-dir", "new/run"]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(audit.main(arguments), 0)
        output = self.root / "new/run/audit.json"
        original = output.read_bytes()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            audit.main(arguments)
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(output.read_bytes(), original)
        arguments[-1] = "../outside"
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            audit.main(arguments)

    def test_disagreeing_scores_or_nonfinite_values_do_not_issue_report(self):
        original = copy.deepcopy(self.saved)
        for score in ("0.954", "NaN", "Infinity", "1.1"):
            self.saved = copy.deepcopy(original)
            self.saved["own_confirmed_Public_best"] = score
            self.write_checkpoint()
            with self.assertRaises(ValueError):
                self.run_audit()


if __name__ == "__main__":
    unittest.main()
