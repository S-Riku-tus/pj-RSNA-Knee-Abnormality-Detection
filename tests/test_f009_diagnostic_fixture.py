"""Synthetic metadata only; the MRI link helper is always mocked."""
import copy
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsna_knee import contracts as C
from rsna_knee import f009_diagnostic_fixture as F


def csv_bytes(fields, rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(fields)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def metadata(root, *, kind=None):
    competition, parent = root / "competition", root / "external-temp"
    competition.mkdir()
    parent.mkdir()
    (competition / "train_series").mkdir()  # Empty artificial MRI directory, never linked or enumerated.
    (competition / "train.csv").write_text("ARTIFICIAL_FORBIDDEN_REPORT_BODY", encoding="utf-8")
    (competition / "gold.csv").write_text("ARTIFICIAL_FORBIDDEN_GOLD_BODY", encoding="utf-8")
    uids = ["ARTIFICIAL_UID_"+str(i).zfill(3) for i in range(F.COUNT)]
    rows = [[uid, "SERIES_"+str(i)+"_"+str(n), "1", "0", "Sagittal"]
            for i, uid in reversed(list(enumerate(uids))) for n in (1, 0)]
    rows.append(["ARTIFICIAL_UNSELECTED", "UNSELECTED_SERIES", "0", "1", "Coronal"])
    fields = F.SERIES_COLUMNS
    if kind == "missing":
        rows = [row for row in rows if row[0] != uids[10]]
    elif kind == "duplicate_series":
        rows.append(list(rows[0]))
    elif kind == "Report_column":
        fields = (*fields, "Report")
        rows = [row+["SYNTHETIC_FORBIDDEN_EXTRA_COLUMN"] for row in rows]
    body = csv_bytes(fields, rows)
    (competition / "train_series.csv").write_bytes(body)
    series_sha = hashlib.sha256(body).hexdigest()
    selection = {"UIDs": uids, "studies": F.COUNT, "train_series_csv_sha256": series_sha,
                 "train_csv_Report_labels_read": False, "Gold_labels_fit_select_or_evaluate": False,
                 "old_native64_UID_overlap": 0}
    if kind == "duplicate_UID":
        selection["UIDs"][-1] = selection["UIDs"][0]
    elif kind == "reordered_UID":
        selection["UIDs"] = list(reversed(uids))
    selection_path = root / "selection.json"
    selection_raw = json.dumps(selection).encode()
    selection_path.write_bytes(selection_raw)
    return {"competition": competition, "parent": parent, "selection": selection_path,
            "selection_sha": hashlib.sha256(selection_raw).hexdigest(), "order_sha": F._order_sha(uids),
            "series_sha": series_sha, "uids": uids, "rows": rows}


def prepare(data, link):
    return F._prepare(data["competition"], data["selection"], data["selection_sha"], data["parent"],
                      expected_order_sha256=data["order_sha"], expected_series_sha256=data["series_sha"],
                      fixture_only=True, link_helper=link)


class FixedFixture(unittest.TestCase):
    def test_default_no_body_and_actual_scope_cannot_use_fake_overrides(self):
        with patch.object(Path, "read_bytes", side_effect=AssertionError("Body forbidden")), \
                patch.object(F.tempfile, "mkdtemp", side_effect=AssertionError("Directory forbidden")):
            with self.assertRaisesRegex(ValueError, "Explicit"):
                F.prepare_fixed192_fixture("absent", "absent.json", F.SELECTION_SHA256, "absent")
            with self.assertRaisesRegex(ValueError, "Fixed saved192"):
                F.prepare_fixed192_fixture("absent", "absent.json", "0"*64, "absent", execute=True)
            with self.assertRaisesRegex(ValueError, "fixed three pins"):
                F._prepare("absent", "absent.json", "0"*64, "absent", expected_order_sha256="0"*64,
                           expected_series_sha256="0"*64, fixture_only=False, link_helper=lambda *args: None)
        self.assertEqual(F.SERIES_COLUMNS, C.SERIES_COLUMNS)
        self.assertEqual(F.TARGETS, C.TARGETS)

    def test_fixed_order_projection_half_baseline_link_mock_and_safe_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = metadata(Path(temporary))
            calls = []

            def link_mock(link, target):
                calls.append((link, target))
                self.assertEqual(target, (data["competition"] / "train_series").resolve())
                link.mkdir()  # No actual symlink, images or decoder.

            original_open, original_iter = Path.open, Path.iterdir

            def no_labels(path, *args, **kwargs):
                self.assertNotIn(path.name, ("train.csv", "gold.csv"))
                return original_open(path, *args, **kwargs)

            def no_MRI_enumeration(path):
                self.assertNotEqual(path.resolve(), (data["competition"] / "train_series").resolve())
                return original_iter(path)

            with patch.object(Path, "open", no_labels), patch.object(Path, "iterdir", no_MRI_enumeration):
                root, baseline, receipt, cleanup = prepare(data, link_mock)
                self.assertEqual(len(calls), 1)
                self.assertTrue(receipt["fixture_complete"])
                self.assertFalse(receipt["actual_fixture_created"])
                self.assertTrue(receipt["artificial_fixture_only"])
                self.assertFalse(receipt["MRI_symlink_created"])
                self.assertEqual(receipt["series_rows"], 384)
                self.assertEqual(receipt["MRI_decode_calls"], 0)
                competition = root / "rsna-knee-abnormality-detection"
                with (competition / "test.csv").open(newline="") as stream:
                    self.assertEqual([row[0] for row in list(csv.reader(stream))[1:]], data["uids"])
                with (competition / "test_series.csv").open(newline="") as stream:
                    projected = list(csv.DictReader(stream))
                self.assertEqual([row[F.ID] for row in projected], [uid for uid in data["uids"] for _ in (0, 1)])
                self.assertEqual([row["SeriesInstanceUID"] for row in projected[:2]], ["SERIES_0_1", "SERIES_0_0"])
                self.assertTrue(all(set(row) == set(F.SERIES_COLUMNS) for row in projected))
                with baseline.open(newline="") as stream:
                    values = list(csv.reader(stream))
                self.assertEqual(values[0], [F.ID, *F.TARGETS])
                self.assertEqual(len(values), 193)
                self.assertTrue(all(row[1:] == ["0.5"]*12 for row in values[1:]))
                self.assertEqual(F._pin(baseline), receipt["generated_csv_pins"]["diagnostic-baseline.csv"])
                self.assertEqual(baseline.read_bytes(), (competition / "sample_submission.csv").read_bytes())
                cleanup()
                cleanup()  # Idempotent only after successful ownership-checked cleanup.
                self.assertFalse(root.exists())
                self.assertTrue(receipt["cleanup_complete"])
                self.assertTrue(receipt["fixture_root_removed"])
                self.assertTrue((data["competition"] / "train_series").is_dir())

    def test_selection_series_tamper_missing_duplicates_and_unsafe_columns_fail_before_directory(self):
        for kind in ("missing", "duplicate_series", "Report_column", "duplicate_UID", "reordered_UID"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                data = metadata(Path(temporary), kind=kind)
                with patch.object(F.tempfile, "mkdtemp", side_effect=AssertionError("No fixture on bad metadata")):
                    with self.assertRaises(ValueError):
                        prepare(data, lambda *args: None)
        with tempfile.TemporaryDirectory() as temporary:
            data = metadata(Path(temporary))
            bad = copy.deepcopy(data)
            bad["selection_sha"] = "0"*64
            with self.assertRaisesRegex(ValueError, "selection source SHA"):
                prepare(bad, lambda *args: None)
            (data["competition"] / "train_series.csv").write_bytes(b"TAMPER")
            with self.assertRaisesRegex(ValueError, "train_series metadata SHA"):
                prepare(data, lambda *args: None)
            self.assertEqual(list(data["parent"].iterdir()), [])

    def test_owner_marker_and_outside_working_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = metadata(Path(temporary))
            with patch.object(F, "WORKING", data["parent"]):
                with self.assertRaisesRegex(ValueError, "outside Kaggle Output"):
                    prepare(data, lambda *args: None)
            root, baseline, receipt, cleanup = prepare(data, lambda link, target: link.mkdir())
            relocated = root.with_name(root.name+"-artificial-relocated")
            self.assertEqual(root.parent, relocated.parent)
            root.rename(relocated)
            receipt["cleanup_complete"] = True  # Caller mutation cannot bypass owned cleanup.
            with self.assertRaisesRegex(ValueError, "disappeared"):
                cleanup()
            self.assertFalse(receipt["cleanup_complete"])
            relocated.rename(root)
            marker = root / ".fixture-owner.json"
            original = marker.read_bytes()
            marker.write_bytes(b"TAMPERED_OWNER")
            with self.assertRaisesRegex(ValueError, "ownership token"):
                cleanup()
            self.assertFalse(receipt["cleanup_complete"])
            self.assertTrue(root.is_dir())
            self.assertTrue((data["competition"] / "train_series").is_dir())
            marker.write_bytes(original)
            cleanup()
            self.assertTrue(receipt["cleanup_complete"])

    def test_partial_link_or_csv_failure_cleans_only_owned_fixture(self):
        for failure in ("link", "CSV"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                data = metadata(Path(temporary))
                seen = []

                def mock(link, target):
                    seen.append(link.parent.parent)
                    if failure == "link":
                        raise RuntimeError("artificial link failure")
                    link.mkdir()

                if failure == "CSV":
                    with patch.object(F, "_write_csv", side_effect=RuntimeError("artificial CSV failure")):
                        with self.assertRaises(RuntimeError):
                            prepare(data, mock)
                else:
                    with self.assertRaises(RuntimeError):
                        prepare(data, mock)
                self.assertEqual(len(seen), 1)
                self.assertFalse(seen[0].exists())
                self.assertEqual(list(data["parent"].iterdir()), [])
                self.assertTrue((data["competition"] / "train_series").is_dir())


if __name__ == "__main__":
    unittest.main()
