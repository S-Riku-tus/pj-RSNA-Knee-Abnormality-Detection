"""Stress contracts and scaled original-cache boundaries, without real MRI/model."""

import ast
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_p003_stress import build, sized_profile_runtime  # noqa: E402
from p003_speed_guard import P003SpeedProfileGuard  # noqa: E402
from p003_stress_runtime import (  # noqa: E402
    P003StressProfileGuard,
    ResourceRecorder,
    array_fingerprint,
    event_category,
    read_cgroup,
)


class StressTests(unittest.TestCase):
    def test_cgroup_v2_observations_and_oom_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, text in {
                "memory.current": "123",
                "memory.max": "max",
                "memory.peak": "789",
                "memory.events": "oom 0\noom_kill 1\n",
            }.items():
                (root / name).write_text(text)
            report = read_cgroup(root)
            self.assertTrue(report["available"])
            self.assertEqual(report["version"], "v2")
            self.assertEqual(report["memory.current"], 123)
            self.assertEqual(report["memory.max"], "max")
            self.assertEqual(report["memory.events"]["oom_kill"], 1)

    def test_cgroup_v1_nested_memory_counters(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "memory").mkdir()
            for name, text in {
                "memory.usage_in_bytes": "234",
                "memory.limit_in_bytes": "9000",
                "memory.max_usage_in_bytes": "678",
                "memory.failcnt": "3",
            }.items():
                (root / "memory" / name).write_text(text)
            report = read_cgroup(root)
            self.assertTrue(report["available"])
            self.assertEqual(report["version"], "v1")
            self.assertEqual((report["memory.current"], report["memory.max"], report["memory.peak"]), (234, 9000, 678))
            self.assertEqual(report["memory.failcnt"], 3)

    def test_missing_cgroup_observations_are_explicitly_unavailable(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = read_cgroup(Path(temporary))
            self.assertFalse(report["available"])
            self.assertEqual(report["version"], "unavailable")
            self.assertNotIn("memory.current", report)

    def test_completed_model_profile_is_not_failed_for_missing_measurements(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "profile_predictions.csv").write_text("synthetic-file")
            (root / "P003_SPEED_SUMMARY.json").write_text("{}")
            (root / "P003_SPEED_CONTRACT.json").write_text('{"mode":"stress_profile"}')
            guard = P003StressProfileGuard({"speed": {"mode": "stress_profile"}}, work=root)
            guard.ids = ["synthetic"]
            guard.recorder = ResourceRecorder(root)
            guard.recorder.errors.append("Artificial measurement unavailable")
            parent = {
                "parent_completion": {"selection": {"studies": 1}, "elapsed_seconds_including_preflight": 2},
                "raptor": {},
                "phase_events": [],
                "coat_receipts": {},
            }
            schedule = type("ArtificialSchedule", (), {"profile": False})()
            with patch.object(P003SpeedProfileGuard, "finish", return_value=parent):
                with patch("p003_stress_runtime.process_rss", return_value={}):
                    with patch(
                        "p003_stress_runtime.read_cgroup", return_value={"available": False, "version": "unavailable"}
                    ):
                        result = guard.finish({"_p003_speed_schedule": schedule})
            self.assertEqual(result["status"], "profile_complete_not_for_submission")
            self.assertEqual(result["measurement_status"], "incomplete")
            self.assertIsNone(result["resources"]["parent_peak_rss_bytes"])
            self.assertIsNone(result["resources"]["sampled_cgroup_memory_current_peak_bytes"])
            self.assertFalse(result["memmap_path_observed"])

    def test_original_allocator_boundary_and_zero_masked_pixels(self):
        """Artificial 132/133 bytes exercise the exact original <= threshold."""
        vendor = json.loads((ROOT / "notebooks/public/vendor/haideptry-speedy-v2.ipynb").read_text(encoding="utf-8"))
        parsed = ast.parse("".join(vendor["cells"][4]["source"]))
        function = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "rsna_array")
        namespace = {
            "np": np,
            "os": os,
            "Path": Path,
            "tempfile": tempfile,
            "rsna_event": lambda *a, **k: None,
            "_RSNA_CACHE_FILES": [],
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<frozen-allocator>", "exec"), namespace)
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, {"RSNA_CACHE_RAM_GIB": str(132 / (1 << 30)), "RSNA_PIXEL_SCRATCH": temporary}):
                ram = namespace["rsna_array"]((132,), "scaled-132")
                mapped = namespace["rsna_array"]((133,), "scaled-133")
                self.assertNotIsInstance(ram, np.memmap)
                self.assertIsInstance(mapped, np.memmap)
                self.assertTrue(np.array_equal(ram, np.zeros(132, np.uint8)))
                self.assertTrue(np.array_equal(mapped, np.zeros(133, np.uint8)))
                mapped[:3] = [9, 8, 7]
                self.assertTrue(np.array_equal(mapped[3:], np.zeros(130, np.uint8)))
                mapped.flush()
                mapped._mmap.close()

    def test_ram_memmap_cache_fingerprints_agree_without_modifying_pixels(self):
        pixels = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        expected = hashlib.sha256(pixels.tobytes()).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            mapped = np.lib.format.open_memmap(
                Path(temporary) / "cache.npy", mode="w+", dtype=np.uint8, shape=pixels.shape
            )
            mapped[:] = pixels
            ram_report, mapped_report = array_fingerprint(pixels), array_fingerprint(mapped)
            self.assertEqual(ram_report["sha256"], expected)
            self.assertEqual(mapped_report["sha256"], expected)
            self.assertEqual(mapped_report["storage"], "memmap")
            self.assertEqual((mapped_report["minimum"], mapped_report["maximum"]), (0, 23))
            self.assertTrue(np.array_equal(mapped, pixels))
            mapped._mmap.close()

    def test_child_observer_preserves_return_and_original_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = ResourceRecorder(temporary)
            self.assertEqual(recorder.wrap_child(lambda *args: 0)([], {}, "child.log"), 0)
            error = RuntimeError("Required model branch failed (-9)")

            def failing(*args):
                raise error

            with self.assertRaises(RuntimeError) as observed:
                recorder.wrap_child(failing)([], {}, "failure.log")
            self.assertIs(observed.exception, error)
            self.assertEqual([item["status"] for item in recorder.summary()["child_processes"]], ["complete", "failed"])
            self.assertIn("(-9)", recorder.children[-1]["error"])

    def test_cache_observer_preserves_tuple_array_and_mask_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = ResourceRecorder(temporary)
            value = (["uid"], np.arange(12, dtype=np.uint8).reshape(1, 3, 2, 2), np.ones((1, 3)))
            output = recorder.wrap_cache(lambda *args: value)({}, {}, {}, "test")
            self.assertIs(output, value)
            self.assertIs(output[1], value[1])
            self.assertIs(output[2], value[2])
            self.assertEqual(recorder.cache_fingerprints[0]["study_count"], 1)

    def test_typed_event_observation_preserves_existing_gate_event(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = ResourceRecorder(temporary)
            observed = []

            def original(kind, **details):
                observed.append((kind, details))
                return "original-result"

            wrapped = recorder.wrap_event(original)
            result = wrapped("partial_slice_decode", series="synthetic", errors=[{"error": "decoder missing"}])
            self.assertEqual(result, "original-result")
            self.assertEqual(observed[0][0], "partial_slice_decode")
            self.assertEqual(observed[0][1]["errors"], [{"error": "decoder missing"}])
            self.assertEqual(event_category("a5_no_metadata"), "layout_absence_observed_check_other_errors")
            self.assertEqual(event_category("raptor_acquisition_empty"), "selected_source_missing")
            self.assertEqual(event_category("coat_global96_child_failed"), "model_or_required_component_failure")
            self.assertEqual(recorder.summary()["source_event_category_counts"], {"decoding_or_pixel_preparation": 1})

    def test_larger_notebook_pins_same_inference_without_publication_or_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "stress"
            manifest = build(destination)
            notebook = json.loads((destination / manifest["notebook"]).read_text(encoding="utf-8"))
            parent = json.loads((ROOT / "notebooks/public/05_profile_p003_speed_64.ipynb").read_text(encoding="utf-8"))
            before = {
                cell["id"]: "".join(cell["source"]) for cell in parent["cells"] if cell["id"].startswith("p003-source-")
            }
            after = {
                cell["id"]: "".join(cell["source"])
                for cell in notebook["cells"]
                if cell["id"].startswith("p003-source-")
            }
            self.assertEqual(before, after)
            self.assertNotIn("p003-source-29", after)
            bootstrap = "".join(notebook["cells"][1]["source"])
            self.assertIn("profile=False", bootstrap)
            tree = ast.parse(bootstrap)
            call = next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "P003StressProfileGuard"
            )
            contract = ast.literal_eval(call.args[0])
            self.assertEqual((contract["profile"]["count"], contract["profile"]["seed"]), (256, 20261005))
            self.assertEqual(contract["speed"]["mode"], "stress_profile")
            self.assertEqual(len(contract["inputs"]), 14)
            self.assertEqual(contract["speed"]["reference"], None)
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), "<stress-cell>", "exec")
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])
            with self.assertRaises(FileExistsError):
                build(destination)

    def test_profile_runtime_changes_only_count_and_audit_messages(self):
        original = (ROOT / "scripts/p003_profile.py").read_text(encoding="utf-8")
        derived = sized_profile_runtime(original, 256)
        self.assertEqual(
            derived.replace("PROFILE_COUNT = 256\n", "PROFILE_COUNT = 64\n")
            .replace("256-study", "64-study")
            .replace("256 studies", "64 studies"),
            original,
        )
        with self.assertRaises(ValueError):
            build(Path("unused"), 132)


if __name__ == "__main__":
    unittest.main()
