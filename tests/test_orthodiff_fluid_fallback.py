"""Bounded stdlib fake-record QA; no real subset/MRI/model/labels/fit."""
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rsna_knee import orthodiff_fluid_fallback as F  # noqa: E402


def candidate(uid, sid, fluid="1", fs="1", plane="Sagittal"):
    return {F.ID: uid, "SeriesInstanceUID": sid, "Fluid_Sensitive": fluid, "Fat_Suppression": fs, "Anatomical_Plane": plane}


def unknown(uid, reasons=("unknown_PD",)):
    rejected = [{"SeriesInstanceUID": "s" + str(i), "reason": reason} for i, reason in enumerate(reasons)]
    return {F.ID: uid, "status": "data_failure", "selected_series": None, "reason_type": "PilotDataError",
            "reason": F.FAMILY_PREFIX + json.dumps(rejected), "feature_forward_calls": 0, "fallback_features": False}


class FluidMetadataPreparation(unittest.TestCase):
    def test_fixed_flags_FS_priority_UID_order_and_no_retry(self):
        for text in F.TRUE_VALUES:
            self.assertTrue(F.metadata_bool(text.upper()))
        for text in F.FALSE_VALUES:
            self.assertFalse(F.metadata_bool(text.upper()))
        for text in (None, "", "unknown", "2", ".5", "nan"):
            with self.assertRaises(ValueError):
                F.metadata_bool(text)
        rows = [candidate("u", "z", fs="false"), candidate("u", "b", fs="YES"), candidate("u", "a", fs="1.0"),
                candidate("u", "axial", plane="Axial"), candidate("u", "invalid", fs="unknown"), candidate("u", "not-fluid", fluid="0")]
        chosen = F.select_fluid_sensitive("u", rows)
        self.assertEqual(chosen["selected_series"]["SeriesInstanceUID"], "a")
        self.assertEqual(chosen["ordered_eligible_series_UIDs"], ["a", "b", "z"])
        self.assertEqual(chosen, F.select_fluid_sensitive("u", list(reversed(rows))))
        self.assertFalse(chosen["confirmed_PD"])
        self.assertFalse(chosen["retry_another_series_after_chosen_failure"])
        self.assertIn("unknown_boolean_flag", [row["reason"] for row in chosen["rejected_series"]])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            F.select_fluid_sensitive("u", [rows[0], rows[0]])
        self.assertEqual(F.select_fluid_sensitive("u", [candidate("u", "none", fluid="0")])["status"], "metadata_unavailable")

    def test_only_unknown_naming_family_and_original_group_fold(self):
        cohort = [{F.ID: uid, "group_id": group, "fold": fold, "source": "report_weak"}
                  for uid, group, fold in (("unknown", "group-A", "2"), ("success", "group-B", "0"), ("geometry", "group-C", "1"))]
        records = [unknown("unknown", ("unknown_PD", "T1_T2_contradiction")), {F.ID: "success", "status": "success"},
                   {**unknown("geometry"), "selected_series": {"SeriesInstanceUID": "chosen"}, "reason": "depth_lt16"}]
        value = F.build_metadata_subset(records, cohort, [candidate("unknown", "fluid")])
        self.assertEqual(value["subset_count"], 1)
        self.assertEqual(value["entries"][0][F.ID], "unknown")
        self.assertEqual((value["entries"][0]["group_id"], value["entries"][0]["fold"]), ("group-A", "2"))
        self.assertFalse(value["fallback_features_qualified"])
        for reasons in ((), ("T1_T2_contradiction",), ("unknown_PD", "no_DICOM_files"), ("unknown_PD", "header_read_error")):
            self.assertFalse(F.unknown_pd_family(unknown("u", reasons))[0])
        for bad in ([*records, records[0]], [{**records[0], F.ID: "new_UID"}, *records[1:]]):
            with self.assertRaises(ValueError):
                F.build_metadata_subset(bad, cohort, [])
        empty_records = copy.deepcopy(records)
        empty_records[0] = unknown("unknown", ("T1_T2_contradiction",))
        self.assertEqual(F.build_metadata_subset(empty_records, cohort, [])["subset_count"], 0)

    def test_incomplete_parent_never_emits_actual_subset(self):
        with self.assertRaisesRegex(ValueError, "Actual complete4207"):
            F.require_qualified_parent({"passed": True, "head_features_qualified": False}, {}, {})
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            audit = temp / "preparation.json"
            audit.write_text('{"passed":false,"head_features_qualified":false}', encoding="utf-8")
            output = temp / "actual-subset.json"
            missing = temp / "must-not-be-read"
            result = subprocess.run([sys.executable, str(ROOT / "scripts/prepare_orthodiffusion_fluid_fallback_subset.py"),
                "--qualification", str(audit), "--expected-qualification-sha256", hashlib.sha256(audit.read_bytes()).hexdigest(),
                "--records", str(missing), "--summary", str(missing), "--status", str(missing), "--output", str(output)],
                capture_output=True, text=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Actual complete parent qualification required", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
