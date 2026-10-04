"""Synthetic export integrity and pixel compatibility; no MRI or saved code runs."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsna_knee.contracts import ID, dump_json, preprocess_fingerprint, sha256, source_hashes, write_csv
from scripts.verify_cache_export import verify

ROOT = Path(__file__).resolve().parents[1]


class CacheCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name) / "export"
        self.config = json.loads((ROOT / "configs/baseline.json").read_text(encoding="utf-8"))
        source_dir = self.root / "code/src/rsna_knee"
        source_dir.mkdir(parents=True)
        for source in (ROOT / "src/rsna_knee").glob("*.py"):
            (source_dir / source.name).write_bytes(source.read_bytes())
        dump_json(self.root / "code/configs/baseline.json", self.config)
        write_csv(self.root / "raw/train.csv", (ID,), [{ID: "synthetic-a"}, {ID: "synthetic-b"}])
        write_csv(self.root / "raw/train_series.csv", (ID,), [{ID: "synthetic-a"}])
        write_csv(self.root / "raw/sample_submission.csv", (ID,), [{ID: "synthetic-a"}])
        self.fingerprint = preprocess_fingerprint(self.config)
        self.cache_metadata = {
            "schema_version": 1,
            "split": "train",
            "preprocess": self.config["preprocess"],
            "fingerprint": self.fingerprint,
            "studies_csv_sha256": sha256(self.root / "raw/train.csv"),
            "series_csv_sha256": sha256(self.root / "raw/train_series.csv"),
        }
        dump_json(self.root / "train-v1/cache.json", self.cache_metadata)
        dump_json(self.root / "train-v1/coverage.json", {})
        dump_json(self.root / "audit.json", {})
        for name in ("synthetic-a", "synthetic-b"):
            (self.root / f"train-v1/{name}.npz").write_bytes(b"synthetic immutable bytes, never decoded")
        self.record = {
            "schema_version": 1,
            "complete": True,
            "studies_total": 2,
            "studies_requested": 2,
            "source_sha256": source_hashes(),
            "config": copy.deepcopy(self.config),
            "cache": {"fingerprint": self.fingerprint},
        }
        self.reseal_files()

    def save_record(self):
        dump_json(self.root / "export.json", self.record)

    def reseal_files(self):
        """Rehash files but preserve independent recorded source/config facts."""
        self.record["files"] = {
            path.relative_to(self.root).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in self.root.rglob("*")
            if path.is_file() and path != self.root / "export.json"
        }
        self.record["output_bytes"] = sum(item["bytes"] for item in self.record["files"].values())
        self.save_record()

    def test_original_one_argument_api_and_strict_match(self):
        result = verify(self.root)
        self.assertTrue(result["valid"])
        self.assertTrue(result["export_integrity"])
        self.assertTrue(result["source_integrity"])
        self.assertTrue(result["preprocess_compatible"])
        self.assertTrue(result["source_matches_repository"])
        self.assertEqual(result["source_differences"], [])
        self.assertEqual(result["studies"], 2)
        self.assertTrue(verify(self.root, require_source_match=True)["valid"])

    def test_current_model_runtime_only_changes_are_reported_and_allowed(self):
        modified = {**source_hashes(), "model.py": "0" * 64, "runtime.py": "1" * 64}
        with patch("scripts.verify_cache_export.source_hashes", return_value=modified):
            result = verify(self.root)
            self.assertTrue(result["valid"])
            self.assertFalse(result["source_matches_repository"])
            self.assertEqual(result["source_differences"], ["model.py", "runtime.py"])
            with self.assertRaisesRegex(ValueError, "strict source matching"):
                verify(self.root, require_source_match=True)

    def test_selected_nonpixel_config_and_config_path_can_reuse_pixels(self):
        modified = copy.deepcopy(self.config)
        modified["train"]["epochs"] += 1
        result = verify(self.root, modified)
        self.assertTrue(result["preprocess_compatible"])
        config_path = Path(self.folder.name) / "current-config.json"
        dump_json(config_path, modified)
        self.assertEqual(verify(self.root, config_path), result)

    def test_selected_pixel_config_or_current_imaging_change_is_rejected(self):
        modified = copy.deepcopy(self.config)
        modified["preprocess"]["image_size"] += 32
        with self.assertRaisesRegex(ValueError, "pixel preprocessing"):
            verify(self.root, modified)
        with patch(
            "scripts.verify_cache_export.source_hashes", return_value={**source_hashes(), "imaging.py": "0" * 64}
        ):
            with self.assertRaisesRegex(ValueError, "pixel preprocessing"):
                verify(self.root)

    def test_stored_code_is_never_executed_and_newline_hash_is_normalized(self):
        runtime = self.root / "code/src/rsna_knee/runtime.py"
        text = runtime.read_text(encoding="utf-8") + "\nraise AssertionError('Never execute stored export code')\n"
        runtime.write_bytes(text.replace("\n", "\r\n").encode())
        self.record["source_sha256"]["runtime.py"] = hashlib.sha256(text.encode()).hexdigest()
        self.reseal_files()
        result = verify(self.root)
        self.assertTrue(result["valid"])
        self.assertEqual(result["source_differences"], ["runtime.py"])

    def test_changed_file_without_rehashing_fails_integrity(self):
        (self.root / "train-v1/synthetic-a.npz").write_bytes(b"tampered bytes")
        with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
            verify(self.root)

    def test_rehashed_stored_source_still_must_match_recorded_source_hash(self):
        source = self.root / "code/src/rsna_knee/model.py"
        source.write_text(source.read_text(encoding="utf-8") + "\n# Changed source\n", encoding="utf-8")
        self.reseal_files()
        with self.assertRaisesRegex(ValueError, "recorded source hashes"):
            verify(self.root)
        self.record["source_sha256"]["model.py"] = "0" * 64
        self.save_record()
        with self.assertRaisesRegex(ValueError, "recorded source hashes"):
            verify(self.root)

    def test_source_manifest_omission_and_unrecorded_source_are_rejected(self):
        del self.record["source_sha256"]["runtime.py"]
        self.save_record()
        with self.assertRaisesRegex(ValueError, "recorded source hashes"):
            verify(self.root)
        self.record["source_sha256"] = source_hashes()
        (self.root / "code/src/rsna_knee/unrecorded.py").write_text("# Only bytes\n", encoding="utf-8")
        self.reseal_files()
        with self.assertRaisesRegex(ValueError, "recorded source hashes"):
            verify(self.root)

    def test_saved_config_is_hashed_and_matches_export_record(self):
        modified = copy.deepcopy(self.config)
        modified["train"]["epochs"] += 1
        dump_json(self.root / "code/configs/baseline.json", modified)
        self.reseal_files()
        with self.assertRaisesRegex(ValueError, "Stored config"):
            verify(self.root)
        self.record["config"] = modified
        self.save_record()
        self.assertTrue(verify(self.root)["valid"])

    def test_config_missing_from_inventory_and_debug_export_are_rejected(self):
        (self.root / "code/configs/baseline.json").unlink()
        self.reseal_files()
        with self.assertRaisesRegex(ValueError, "required metadata"):
            verify(self.root)
        self.record["complete"] = False
        self.save_record()
        with self.assertRaisesRegex(ValueError, "complete schema-1"):
            verify(self.root)

    def test_cache_fingerprint_metadata_and_csv_origin_are_independent_guards(self):
        for field, value in (("fingerprint", "0" * 64), ("preprocess", {}), ("split", "test")):
            with self.subTest(field=field):
                dump_json(self.root / "train-v1/cache.json", {**self.cache_metadata, field: value})
                self.reseal_files()
                with self.assertRaisesRegex(ValueError, "Cache split or preprocessing"):
                    verify(self.root)
        dump_json(self.root / "train-v1/cache.json", self.cache_metadata)
        write_csv(self.root / "raw/train_series.csv", (ID,), [{ID: "synthetic-b"}])
        self.reseal_files()
        with self.assertRaisesRegex(ValueError, "different train_series.csv"):
            verify(self.root)

    def test_cache_uid_and_requested_count_must_cover_full_csv(self):
        (self.root / "train-v1/synthetic-a.npz").rename(self.root / "train-v1/unexpected.npz")
        self.reseal_files()
        with self.assertRaisesRegex(ValueError, "study IDs/count"):
            verify(self.root)
        (self.root / "train-v1/unexpected.npz").rename(self.root / "train-v1/synthetic-a.npz")
        self.record["studies_requested"] = 1
        self.reseal_files()
        with self.assertRaisesRegex(ValueError, "full training set"):
            verify(self.root)

    def test_inventory_and_path_containment_cannot_be_bypassed(self):
        (self.root / "raw/unlisted.txt").write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "inventory"):
            verify(self.root)
        (self.root / "raw/unlisted.txt").unlink()
        (self.root / "train-v1/export.json").write_bytes(b"unexpected nested export")
        with self.assertRaisesRegex(ValueError, "inventory"):
            verify(self.root)
        (self.root / "train-v1/export.json").unlink()
        self.record["files"]["../outside.py"] = {"bytes": 0, "sha256": "0" * 64}
        self.save_record()
        with self.assertRaises(ValueError):
            verify(self.root)


if __name__ == "__main__":
    unittest.main()
