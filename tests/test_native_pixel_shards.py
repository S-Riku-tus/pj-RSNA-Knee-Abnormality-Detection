"""Synthetic group/export/index contracts; no patient pixels, CUDA or fitting."""

import ast
import copy
import importlib.util
import json
import os
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rsna_knee.contracts import ID, SERIES_COLUMNS, SERIES_ID, dump_json, sha256, write_csv
from rsna_knee.feature_contracts import json_hash
from rsna_knee.native_pixel_shards import (
    EXECUTION_ENVIRONMENT,
    SUMMARY_NAME,
    atomic_json,
    build_full_index,
    export_pixel_shard,
    make_plan,
    validate_plan,
)
from rsna_knee.native_training import NativePixelStore

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("native_shard_builder", ROOT / "scripts/build_native392_shards.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
CONFIG = json.loads((ROOT / "configs/experiments/f005-dinov2-native-last2.json").read_text())
SETTINGS = CONFIG["preprocess"]


def rows():
    return [
        {ID: f"synthetic-{index}", "group_id": f"group-{index // 2}", "fold": index // 2, "source": "report_weak"}
        for index in range(6)
    ]


def plan():
    return make_plan(
        rows(),
        ["excluded-gold"],
        ["excluded-gold-group"],
        seed=20261002,
        n_folds=5,
        settings=SETTINGS,
        manifest_sha256={"synthetic": "a" * 64},
        shard_count=3,
    )


def binding():
    return {
        "code_zip_sha256": "b" * 64,
        "code_member_sha256": {"synthetic.py": "c" * 64},
        "environment": EXECUTION_ENVIRONMENT,
    }


def fake_reader(directory, settings):
    windows = np.zeros((1, 3, 392, 392), np.uint8)
    windows[0, :, 3:8, 9:17] = 157
    return (
        windows,
        np.zeros(1, np.float32),
        {
            "native_shape": [1, 11, 13],
            "pixel_spacing_mm": [0.4, 0.7],
            "selected_centers": [0],
            "physical_span_mm": 0.0,
            "order": "physical_position",
            "normalization": "synthetic",
        },
    )


class NativePixelShardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="synthetic-native-shards-")
        cls.root = Path(cls.temp.name)
        cls.plan = plan()
        series = [
            {
                ID: row[ID],
                SERIES_ID: "synthetic-series",
                "Fluid_Sensitive": "1",
                "Fat_Suppression": "true",
                "Anatomical_Plane": "Sagittal",
            }
            for row in rows()
        ]
        write_csv(cls.root / "train_series.csv", SERIES_COLUMNS, series)
        for name in ("train.csv", "weak.csv", "gold.csv"):
            (cls.root / name).write_text("FORBIDDEN TEACHER SENTINEL", encoding="utf-8")
        cls.summaries = []
        from rsna_knee import native_pixel_shards

        opened = []
        original = native_pixel_shards.read_csv

        def series_only(path, required):
            if Path(path).name != "train_series.csv":
                raise AssertionError("Teacher CSV was opened")
            opened.append(Path(path).name)
            return original(path, required)

        cls.binding = binding()
        with patch.object(native_pixel_shards, "read_csv", side_effect=series_only):
            for index in range(3):
                destination = cls.root / f"shard-{index}"
                export_pixel_shard(
                    cls.root,
                    destination,
                    cls.plan,
                    index,
                    reader=fake_reader,
                    qc_reader=fake_reader,
                    execution_binding=cls.binding,
                    finalize_metadata_fn=lambda: {
                        "temporary_files_remaining": 0,
                        "input_csv_reads": ["/kaggle/input/c/train_series.csv"],
                    },
                )
                cls.summaries.append(destination / SUMMARY_NAME)
        cls.opened = opened
        cls.package = {"plan_sha256": json_hash(cls.plan), "execution_binding": cls.binding}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_group_assignment_gold_duplicate_missing_and_source_guards(self):
        self.assertEqual(self.plan, plan())
        self.assertEqual([value["study_count"] for value in self.plan["shards"]], [2, 2, 2])
        assigned = {}
        for shard in self.plan["shards"]:
            for row in shard["studies"]:
                assigned.setdefault(row["group_id"], set()).add(shard["index"])
        self.assertTrue(all(len(value) == 1 for value in assigned.values()))
        variants = []
        duplicate = rows() + [rows()[0]]
        variants.append((duplicate, ["excluded-gold"], ["excluded-gold-group"]))
        gold = rows()
        gold[0][ID] = "excluded-gold"
        variants.append((gold, ["excluded-gold"], ["excluded-gold-group"]))
        group_gold = rows()
        group_gold[0]["group_id"] = "excluded-gold-group"
        variants.append((group_gold, ["excluded-gold"], ["excluded-gold-group"]))
        cross = rows()
        cross[1]["fold"] = 4
        variants.append((cross, [], []))
        teacher = rows()
        teacher[0]["ACL"] = "SENTINEL"
        variants.append((teacher, [], []))
        for value, gold_ids, gold_groups in variants:
            with self.subTest(value=value[0]), self.assertRaises(ValueError):
                make_plan(
                    value,
                    gold_ids,
                    gold_groups,
                    seed=20261002,
                    n_folds=5,
                    settings=SETTINGS,
                    manifest_sha256={},
                    shard_count=3,
                )
        for mutate in ("missing", "source", "assignment"):
            invalid = copy.deepcopy(self.plan)
            if mutate == "missing":
                invalid["shards"][0]["studies"].pop()
            elif mutate == "source":
                invalid["source_sha256"]["triplet_cache.py"] = "0" * 64
            else:
                invalid["shards"][0]["studies"], invalid["shards"][1]["studies"] = (
                    invalid["shards"][1]["studies"],
                    invalid["shards"][0]["studies"],
                )
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                validate_plan(invalid)

    def test_actual_structure24_balance_and_raw_disk_budget_only(self):
        actual = builder.reviewed_plan()
        self.assertEqual(actual["study_count"], 4207)
        self.assertEqual(actual["group_count"], 4102)
        counts = [value["study_count"] for value in actual["shards"]]
        self.assertEqual(min(counts), 175)
        self.assertEqual(max(counts), 176)
        self.assertEqual(sum(counts), 4207)
        self.assertEqual(sum(value["raw_pixel_upper_bytes"] for value in actual["shards"]), 279_272_641_536)
        budget = actual["budgets"]
        self.assertLessEqual(
            budget["archive_bytes"] + budget["native_temp_bytes"] + budget["helper_metadata_reserve_bytes"],
            20_000_000_000,
        )
        for shard in actual["shards"]:
            self.assertLess(
                shard["raw_pixel_upper_bytes"] + shard["container_overhead_budget_bytes"], budget["archive_bytes"]
            )

    def test_streaming_full_index_npz_and_existing_f005_store_compatibility(self):
        index_path = self.root / "full-index.json"
        result = build_full_index(self.plan, self.package, self.summaries, index_path, cache_root=self.root)
        self.assertTrue(result["complete"])
        self.assertEqual(result["verified"], {"studies": 6, "arrays": 30, "valid_windows": 6})
        self.assertEqual(self.opened, ["train_series.csv"] * 3)
        store = NativePixelStore(index_path, CONFIG)
        self.assertEqual(set(store.studies), {row[ID] for row in rows()})
        got = store.load(rows()[0][ID])
        self.assertEqual(got["images"].shape, (3, 48, 3, 392, 392))
        self.assertEqual(int(got["mask"].sum()), 1)
        self.assertFalse(any(".partial" in path.name for parent in self.summaries for path in parent.parent.iterdir()))
        self.assertFalse(index_path.with_name(index_path.name + ".partial").exists())
        with self.assertRaises(ValueError):
            build_full_index(self.plan, self.package, self.summaries, index_path, cache_root=self.root)

    def rejection(self, name, *, mutate=None, summaries=None):
        paths = list(self.summaries) if summaries is None else summaries
        if mutate is not None:
            original = json.loads(paths[0].read_text())
            mutate(original)
            altered = paths[0].with_name(f"altered-{name}.json")
            dump_json(altered, original)
            paths[0] = altered
        with self.assertRaises(ValueError):
            build_full_index(self.plan, self.package, paths, self.root / f"forbidden-{name}.json", cache_root=self.root)

    def test_index_rejects_partial_duplicate_missing_hash_contract_and_metadata_drift(self):
        self.rejection("missing", summaries=self.summaries[:-1])
        self.rejection("duplicate", summaries=[self.summaries[0], self.summaries[0], self.summaries[2]])
        mutations = {
            "partial": lambda value: value.update(complete=False),
            "processed": lambda value: value.update(processed=1),
            "time": lambda value: value.update(elapsed_seconds=self.plan["budgets"]["max_seconds"] + 1),
            "archive-budget": lambda value: value.update(archive_budget_bytes=1),
            "raw-budget": lambda value: value.update(raw_pixel_upper_bytes=1),
            "archive": lambda value: value["archive"].update(sha256="0" * 64),
            "source": lambda value: value["source_sha256"].update({"triplet_cache.py": "0" * 64}),
            "contract": lambda value: value["contract"]["preprocess"].update(image_size=280),
            "cohort": lambda value: value["cohort"]["studies"].pop(),
            "member": lambda value: next(iter(value["files"].values())).update(sha256="0" * 64),
            "array": lambda value: next(iter(value["files"].values()))["array_sha256"].update(mask="0" * 64),
            "shape": lambda value: next(iter(value["files"].values()))["array_metadata"]["images"].update(shape=[1]),
            "teacher": lambda value: value.update(teacher_CSV_read=True),
            "mask": lambda value: next(iter(value["coverage"].values())).update(valid_triplets=4),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                self.rejection(name, mutate=mutate)
        marker = self.summaries[0].parent / "interrupted.partial"
        marker.write_text("partial", encoding="utf-8")
        try:
            self.rejection("remaining-partial")
        finally:
            marker.unlink()

    def test_index_rejects_duplicate_unsafe_or_symlink_zip_entries_even_with_rehashed_outer_zip(self):
        original = json.loads(self.summaries[0].read_text())
        archive_path = self.summaries[0].parent / original["archive"]["name"]
        with zipfile.ZipFile(archive_path) as source:
            members = [(item.filename, source.read(item)) for item in source.infolist()]
        for kind in ("duplicate", "unsafe", "symlink"):
            target = self.root / f"altered-{kind}"
            target.mkdir()
            changed_archive = target / original["archive"]["name"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(changed_archive, "x", compression=zipfile.ZIP_STORED) as archive:
                    for index, (name, payload) in enumerate(members):
                        info = zipfile.ZipInfo("../" + name if kind == "unsafe" and index == 0 else name)
                        if kind == "symlink" and index == 0:
                            info.create_system = 3
                            info.external_attr = 0o120777 << 16
                        archive.writestr(info, payload)
                    if kind == "duplicate":
                        archive.writestr(members[0][0], members[0][1])
            summary = copy.deepcopy(original)
            summary["archive"].update(bytes=changed_archive.stat().st_size, sha256=sha256(changed_archive))
            altered = target / SUMMARY_NAME
            dump_json(altered, summary)
            self.rejection(kind, summaries=[altered, *self.summaries[1:]])

    def test_incomplete_byteQC_or_cleanup_cannot_publish_complete_archive(self):
        for kind in ("QC", "cleanup"):

            def no_reference(directory, settings):
                values = fake_reader(directory, settings)
                values[2]["native_shape"] = [500, 512, 512]
                return values

            destination = self.root / f"failed-{kind}"
            with self.assertRaises(ValueError):
                export_pixel_shard(
                    self.root,
                    destination,
                    self.plan,
                    0,
                    reader=no_reference if kind == "QC" else fake_reader,
                    qc_reader=fake_reader,
                    execution_binding=self.binding,
                    finalize_metadata_fn=lambda: {
                        "temporary_files_remaining": 1,
                        "input_csv_reads": ["/kaggle/input/c/train_series.csv"],
                    },
                )
            summary = json.loads((destination / SUMMARY_NAME).read_text())
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["status"], "failed")
            self.assertTrue(list(destination.glob("*.zip.partial")))
            self.assertFalse(list(destination.glob("*.zip")))

    def test_final_study_time_overrun_cannot_publish_complete_archive(self):
        from rsna_knee import native_pixel_shards

        clock = {"seconds": 0.0}
        last = self.plan["shards"][0]["studies"][-1][ID]

        def delayed_final_study(directory, settings):
            if Path(directory).parent.name == last:
                clock["seconds"] = float(self.plan["budgets"]["max_seconds"] + 1)
            return fake_reader(directory, settings)

        destination = self.root / "failed-final-time"
        with patch.object(native_pixel_shards.time, "perf_counter", side_effect=lambda: clock["seconds"]):
            with self.assertRaisesRegex(ValueError, "Final study/ZIP/hash/QC"):
                export_pixel_shard(
                    self.root,
                    destination,
                    self.plan,
                    0,
                    reader=delayed_final_study,
                    qc_reader=delayed_final_study,
                    execution_binding=self.binding,
                    finalize_metadata_fn=lambda: {
                        "temporary_files_remaining": 0,
                        "input_csv_reads": ["/kaggle/input/c/train_series.csv"],
                    },
                )
        summary = json.loads((destination / SUMMARY_NAME).read_text())
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["processed"], 2)
        self.assertGreater(summary["elapsed_seconds"], self.plan["budgets"]["max_seconds"])
        self.assertTrue(list(destination.glob("*.zip.partial")))
        self.assertFalse(list(destination.glob("*.zip")))

    def test_index_stage_exclusive_and_oversized_npz_rejected_before_payload_read(self):
        stage_target = self.root / "interrupted-index.json"
        stage = stage_target.with_name(stage_target.name + ".partial")
        stage.write_text("PRESERVE EXISTING INCOMPLETE INDEX", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            atomic_json(stage_target, {"must_not_replace": True}, new=True)
        self.assertEqual(stage.read_text(), "PRESERVE EXISTING INCOMPLETE INDEX")
        with self.assertRaises(ValueError):
            build_full_index(self.plan, self.package, self.summaries, stage_target, cache_root=self.root)
        self.assertFalse(stage_target.exists())
        original = zipfile.ZipFile.infolist
        cap = 66_382_848 + self.plan["budgets"]["per_study_container_overhead_bytes"]

        def oversized_inventory(archive):
            values = copy.deepcopy(original(archive))
            values[0].file_size = cap + 1
            return values

        with (
            patch.object(zipfile.ZipFile, "infolist", oversized_inventory),
            patch.object(zipfile.ZipFile, "read", side_effect=AssertionError("Large payload was read")),
        ):
            with self.assertRaisesRegex(ValueError, "memory budget before read"):
                build_full_index(
                    self.plan, self.package, self.summaries, self.root / "oversize-index.json", cache_root=self.root
                )

    def test_notebook_teacher_guard_native_disk_cap_and_rerun_prohibition(self):
        book = builder.notebook(self.plan, 0, "SGVsbG8=", self.binding, "d" * 64, "e" * 64)
        codes = ["".join(value["source"]) for value in book["cells"] if value["cell_type"] == "code"]
        self.assertEqual(len(codes), 2)
        self.assertIn("KAGGLE_IS_COMPETITION_RERUN", codes[0])
        self.assertFalse(book["metadata"]["rsna_native_pixel_shard"]["for_submission"])
        tree = ast.parse(codes[0])
        guard = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "prohibit_teacher_csv"
        )
        namespace = {"os": os, "Path": Path, "CSV_OPENED": []}
        exec(compile(ast.Module(body=[guard], type_ignores=[]), "guard", "exec"), namespace)
        for name in ("train.csv", "weak.csv", "gold.csv", "test.csv", "sample_submission.csv"):
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                namespace["prohibit_teacher_csv"]("open", (f"/kaggle/input/c/{name}",))
        namespace["prohibit_teacher_csv"]("open", ("/kaggle/input/c/train_series.csv",))
        self.assertEqual(namespace["CSV_OPENED"], ["/kaggle/input/c/train_series.csv"])
        tree = ast.parse(codes[1])
        guard = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "native_budget_observer"
        )
        namespace = {"PLAN": self.plan}
        exec(compile(ast.Module(body=[guard], type_ignores=[]), "native-budget", "exec"), namespace)
        with self.assertRaises(ValueError):
            namespace["native_budget_observer"]("allocation_plan", {"minimum_free_disk_bytes": 5 * 1024**3})
        namespace["native_budget_observer"]("allocation_plan", {"minimum_free_disk_bytes": 1024**3})


if __name__ == "__main__":
    unittest.main()
