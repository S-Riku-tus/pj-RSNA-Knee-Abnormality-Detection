"""Stdlib-only journal, source-identity and metadata tests; no MRI/model imports."""
import ast
import base64
import copy
import hashlib
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


runtime = module("ortho_full_test_runtime", "scripts/orthodiffusion_fullweak_runtime.py")
builder = module("ortho_full_test_builder", "scripts/prepare_orthodiffusion_fullweak.py")


def success(uid):
    values = [float(x) / 128 for x in range(256)]
    raw = struct.pack("<256f", *values)
    return {"StudyInstanceUID": uid, "status": "success", "metadata": {"stratum": "PD_nonFS"},
            "pooled_float32_values": values, "pooled": {"shape": [1, 256], "dtype": "torch.float32",
                "bytes": 1024, "sha256": hashlib.sha256(raw).hexdigest()},
            "forward_counts": {"denoiser": 1, "decoder_blocks": 0, "final_out": 0}, "all_RNG_restored": True}


def failure(uid):
    return {"StudyInstanceUID": uid, "status": "data_failure", "reason": "unknown_PD", "reason_type": "PilotDataError",
            "feature_forward_calls": 0, "fallback_features": False}


class FullweakTransport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.pack = Path(cls.temp.name) / "pack"
        builder.prepare(cls.pack)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_append_is_exact_record_stream_with_incremental_durable_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "records.jsonl"
            journal = runtime.RecordJournal(path, ["a", "b", "c"])
            records = [success("a"), failure("b"), success("c")]
            for index, record in enumerate(records, 1):
                previous = path.read_bytes()
                journal.append(record)
                raw = path.read_bytes()
                self.assertTrue(raw.startswith(previous))
                self.assertEqual(journal.pin(), {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
                self.assertEqual(journal.count, index)
            self.assertEqual([json.loads(line) for line in path.read_bytes().splitlines()], records)
            self.assertEqual(journal.finalize(), builder.pin(path))
            self.assertEqual(journal.coverage(), {"requested": 3, "recorded": 3, "success": 2, "failure": 1,
                "success_strata": {"PD_nonFS": 2}, "failure_reason_types": {"PilotDataError": 1}})

    def test_missing_never_zero_filled_and_outoforder_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            journal = runtime.RecordJournal(Path(temp) / "records.jsonl", ["a", "b"])
            try:
                for bad in (failure("b"), {**failure("a"), "pooled_float32_values": [0.0] * 256},
                            {**failure("a"), "fallback_features": True}, {**failure("a"), "feature_forward_calls": 1}):
                    with self.assertRaises(ValueError):
                        journal.append(bad)
                    self.assertEqual(journal.count, 0)
                    self.assertEqual(journal.path.stat().st_size, 0)
                journal.append(failure("a"))
                with self.assertRaises(ValueError):
                    journal.append(failure("a"))
                journal.append(success("b"))
                rows = [json.loads(line) for line in journal.path.read_bytes().splitlines()]
                self.assertNotIn("pooled_float32_values", rows[0])
                self.assertEqual(sum("pooled_float32_values" in row for row in rows), 1)
            finally:
                journal.close()

    def test_nonfinite_nonfp32_wrongpin_and_bad_forward_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            journal = runtime.RecordJournal(Path(temp) / "records.jsonl", ["a"])
            bad_rows = []
            for value in (float("nan"), float("inf"), 0.1, True):
                row = success("a")
                row["pooled_float32_values"][0] = value
                bad_rows.append(row)
            row = success("a")
            row["pooled"]["sha256"] = "0" * 64
            bad_rows.append(row)
            for key, value in (("all_RNG_restored", False), ("forward_counts", {"denoiser": 1, "decoder_blocks": 1, "final_out": 0})):
                bad_rows.append({**success("a"), key: value})
            try:
                for row in bad_rows:
                    with self.assertRaises(ValueError):
                        journal.append(row)
                self.assertEqual(journal.count, 0)
            finally:
                journal.close()

    def test_small_status_does_not_grow_with_old_records_or_all_UIDs(self):
        with tempfile.TemporaryDirectory() as temp:
            journal = runtime.RecordJournal(Path(temp) / "records.jsonl", [str(i) for i in range(4207)])
            state = {"phase": "start", "complete": False, "requested_UIDs": list(range(4207)), "records": ["ignored"] * 4207}
            try:
                journal.append(failure("0"))
                snapshot = runtime.status_snapshot(state, journal, runtime.time.monotonic())
                raw = json.dumps(snapshot).encode()
                self.assertLess(len(raw), 2000)
                self.assertNotIn("requested_UIDs", snapshot)
                self.assertNotIn("records", snapshot)
                self.assertEqual(snapshot["coverage"]["recorded"], 1)
                self.assertEqual(snapshot["journal"]["sha256"], hashlib.sha256(journal.path.read_bytes()).hexdigest())
            finally:
                journal.close()

    def test_final_rehash_rejects_modified_journal(self):
        with tempfile.TemporaryDirectory() as temp:
            journal = runtime.RecordJournal(Path(temp) / "records.jsonl", ["a"])
            journal.append(failure("a"))
            journal.close()
            with journal.path.open("ab") as handle:
                handle.write(b"\n")
            with self.assertRaises(ValueError):
                journal.finalize()

    def test_all4207_UID_map_groups_folds_and_frozen_science_transport(self):
        manifest = json.loads((self.pack / "manifest.json").read_bytes())
        notebook = json.loads((self.pack / "source.ipynb").read_bytes())
        payload = json.loads((self.pack / "save-payload.json").read_bytes())
        tree = ast.parse(notebook["cells"][1]["source"])
        assign = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "_of_raw" for target in node.targets))
        raw = zlib.decompress(base64.b85decode(ast.literal_eval(assign.value.args[0].args[0])))
        bundle = json.loads(raw)
        self.assertEqual(builder.pin(self.pack / "source.ipynb"), manifest["source"])
        self.assertEqual(payload["text"].encode(), (self.pack / "source.ipynb").read_bytes())
        self.assertEqual({"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}, manifest["bundle_pin"])
        for name, text in bundle["runtimes"].items():
            self.assertEqual(text.encode(), (self.pack / (name + "-runtime.py")).read_bytes())
            self.assertEqual({"bytes": len(text.encode()), "sha256": hashlib.sha256(text.encode()).hexdigest()}, manifest["runtime_pins"][name])
        self.assertEqual(bundle["config"]["preprocess"], json.loads((builder.PILOT / "config.json").read_bytes())["preprocess"])
        self.assertEqual(bundle["vendor"], json.loads((builder.PILOT / "vendor.json").read_bytes()))
        self.assertEqual(len(bundle["config"]["uids"]), 4207)
        self.assertEqual(manifest["cohort"]["fold_counts"], {"0": 821, "1": 891, "2": 888, "3": 778, "4": 829})
        self.assertEqual(manifest["cohort"]["non_singleton_study_count"], 140)
        self.assertTrue(manifest["science_parity"]["numerical_forward_block_byte_identical"])
        self.assertEqual(len(payload["dataset_data_sources"] + payload["competition_data_sources"] + payload["kernel_data_sources"]), 3)
        self.assertTrue(payload["is_private"] and payload["enable_gpu"] and not payload["enable_internet"])
        for field in ("requested", "group_id", "fold", "Report", "gold_ids"):
            self.assertNotIn(field, bundle["config"])
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("pydicom", sys.modules)

    def test_scientific_block_mutation_is_refused(self):
        old = (builder.PILOT / "pilot-runtime.py").read_bytes().decode()
        full = (ROOT / "scripts/orthodiffusion_fullweak_runtime.py").read_bytes().decode()
        builder.science_parity(old, full)
        with self.assertRaises(ValueError):
            builder.science_parity(old, full.replace("torch.tensor([100]", "torch.tensor([101]"))
        config = copy.deepcopy(json.loads((self.pack / "config.json").read_bytes()))
        config["uids"] = ["one"]
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                runtime.run(config, {}, {"require": builder.require}, {}, Path(temp) / "new")
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
