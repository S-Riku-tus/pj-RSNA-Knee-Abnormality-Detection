"""Kaggle-only preparation and completion checks around unchanged public cell sources.

This module does not download, submit, decode MRI, load weights, or train models.
The generated Notebook calls run_cell only when its owner starts the saved run.
"""

import csv
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import linecache
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

LABELS = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)
HEADER = ("StudyInstanceUID", *LABELS)
CELL_ORDER = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12)
WEIGHTS = {"ACL": 0.8, "Lateral OA": 0.35, "PF OA": 0.35, "Synovitis": 0.35, "Baker's": 0.375}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_table(path, expected_ids=None):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        if tuple(next(reader, ())) != HEADER:
            raise ValueError(f"Unexpected submission columns: {Path(path).name}")
        rows = list(reader)
    if not rows or any(len(row) != 13 for row in rows):
        raise ValueError("Empty or malformed predictions")
    ids = [row[0] for row in rows]
    if len(set(ids)) != len(ids) or any(not uid for uid in ids):
        raise ValueError("Empty or duplicated prediction UID")
    if expected_ids is not None and ids != expected_ids:
        raise ValueError("Prediction UID order differs from test.csv")
    for row in rows:
        for value in row[1:]:
            score = float(value)
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Nonfinite or out-of-range prediction")
    return rows


class Tee:
    """Forward original output unchanged; inspect only bounded complete log lines."""

    def __init__(self, stream, observer):
        self.stream, self.observer, self.pending = stream, observer, ""

    def write(self, text):
        result = self.stream.write(text)
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            self.observer(line)
        if len(self.pending) > 65536:
            self.pending = self.pending[-65536:]
        return result

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class CandidateGuard:
    def __init__(self, contract, work="/kaggle/working", input_root="/kaggle/input"):
        self.contract = contract
        self.work, self.input_root = Path(work), Path(input_root)
        self.started = time.monotonic()
        self.completed, self.failures, self.observations = [], [], {}
        self.active_cell, self.filename = None, None
        self.test_ids, self.mounts = [], {}
        self.owns_outputs = False
        self.ready = False

    def output_paths(self):
        paths = set(self.work.glob("submission*.csv"))
        paths.update(self.work.glob("*receipt*.json"))
        paths.update(self.work.glob("public0033*.csv"))
        return paths

    def prepare_files(self):
        self.work.mkdir(parents=True, exist_ok=True)
        if self.output_paths() or (self.work / "P002_READY.json").exists():
            raise RuntimeError("Use a fresh Kaggle saved run; existing prediction/receipt files are present")
        records = []
        for entry in self.contract["inputs"]:
            candidates = [self.input_root / path for path in entry["mounts"]]
            for path in candidates:
                path.resolve().relative_to(self.input_root.resolve())
            found = {path.resolve() for path in candidates if path.is_dir()}
            if len(found) != 1:
                raise RuntimeError(f"Attach exactly one pinned Input for {entry['key']}: {entry['ref']}")
            mount = found.pop()
            self.mounts[entry["key"]] = mount
            for item in entry["files"]:
                file = (mount / item["path"]).resolve()
                file.relative_to(mount)
                if not file.is_file() or file.stat().st_size != item["bytes"]:
                    raise RuntimeError(f"Missing/wrong-size file: {entry['key']}/{item['path']}")
                actual = sha256(file)
                if actual != item["sha256"]:
                    raise RuntimeError(f"SHA-256 mismatch: {entry['key']}/{item['path']}")
                records.append({"input": entry["key"], "path": item["path"], "sha256": actual})
        if self.contract.get("strict_input_mounts"):
            # Prune each declared mount: do not walk any MRI or model payload tree.
            # Extra mounts can shadow the public Raptor resolver's audited weights.
            declared = set(self.mounts.values())
            pending = [self.input_root.resolve()]
            while pending:
                for child in pending.pop().iterdir():
                    if not child.is_dir():
                        continue
                    resolved = child.resolve()
                    if resolved in declared:
                        continue
                    if any(mount.is_relative_to(resolved) for mount in declared):
                        pending.append(resolved)
                    else:
                        raise RuntimeError(f"Remove unlisted Input mount before running: {child}")
        comp = self.mounts["competition"]
        for name in ("train.csv", "test.csv", "test_series.csv", "sample_submission.csv"):
            if not (comp / name).is_file():
                raise RuntimeError(f"Missing competition file {name}")
        if not (comp / "test_series").is_dir():
            raise RuntimeError("test_series directory missing; refuse training-data fallback")
        with (comp / "test.csv").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            self.test_ids = [row["StudyInstanceUID"] for row in reader]
        if not self.test_ids or len(set(self.test_ids)) != len(self.test_ids):
            raise RuntimeError("test.csv has empty/duplicated study set")
        read_table(comp / "sample_submission.csv", self.test_ids)
        self.owns_outputs = True
        self.file_records = records
        return records

    def prepare_environment(self):
        if Path.cwd().resolve() != self.work.resolve():
            raise RuntimeError("Run this Notebook in /kaggle/working")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        # Install only the hash-checked local wheel, without resolving/downloading dependencies.
        wheel = self.contract["timm_wheel"]
        try:
            installed = importlib.metadata.version("timm")
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed != wheel["version"]:
            if "timm" in sys.modules:
                raise RuntimeError("Restart the session before changing timm")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-index",
                    "--no-deps",
                    str(self.mounts[wheel["input"]] / wheel["path"]),
                ],
                check=True,
            )
        modules = ("numpy", "pandas", "pydicom", "torch", "torchvision", "timm", "cv2", "transformers", "safetensors")
        loaded = {name: importlib.import_module(name) for name in modules}
        torch = loaded["torch"]
        if torch.cuda.device_count() != 2 or any("T4" not in torch.cuda.get_device_name(i) for i in range(2)):
            raise RuntimeError("This pinned public recipe requires GPU T4 x2")
        if not hasattr(loaded["transformers"], "Dinov2Model"):
            raise RuntimeError("transformers Dinov2Model is unavailable")
        if not loaded["timm"].is_model("coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"):
            raise RuntimeError("Required CoAtNet model is absent from timm")
        # Import/API checks only: no MRI, network, model construction or forward here.
        self.environment = {
            "python": sys.version,
            "packages": {name: getattr(mod, "__version__", "unknown") for name, mod in loaded.items()},
            "gpu": [torch.cuda.get_device_name(i) for i in range(2)],
            "torch_cuda": torch.version.cuda,
        }

    def prepare(self):
        try:
            self.prepare_files()
            self.prepare_environment()
            self.ready = True
            save_json(
                self.work / "p002-preflight.json",
                {
                    "status": "passed",
                    "candidate_id": self.contract["candidate_id"],
                    "studies": len(self.test_ids),
                    "input_files": self.file_records,
                    "environment": self.environment,
                },
            )
            print(f"P002 PREFLIGHT PASSED | studies={len(self.test_ids)} | pinned files={len(self.file_records)}")
        except BaseException:
            self.reject("Preflight failed")
            raise

    def reject(self, reason):
        self.ready = False
        if self.owns_outputs:
            # Only fresh-run outputs belong to this guard; never delete prior experiments.
            for path in self.output_paths():
                target = path.with_name("p002-rejected-" + path.name + ".disabled")
                if target.exists():
                    raise RuntimeError("Rejected-output destination already exists")
                path.rename(target)
            ready = self.work / "P002_READY.json"
            if ready.exists():
                ready.rename(self.work / "P002_READY.rejected.json")
            save_json(
                self.work / "P002_FAILED.json",
                {
                    "status": "failed",
                    "reason": reason,
                    "completed_cells": self.completed,
                    "active_source_cell": self.active_cell,
                },
            )

    def _array_valid(self, value, shape):
        import numpy as np

        if tuple(value.shape) != tuple(shape) or not np.isfinite(value).all() or (value < 0).any() or (value > 1).any():
            raise ValueError("Raw member prediction shape/range/finite check failed")

    def _inspect_frame(self, mode):
        frame = inspect.currentframe()
        try:
            frame = frame.f_back
            while frame is not None:
                if frame.f_code.co_filename == self.filename:
                    values = frame.f_locals
                    n = len(self.test_ids)
                    if mode == "raptor" and frame.f_code.co_name == "main":
                        self._array_valid(__import__("numpy").asarray(values["arm_probs"]), (4, n, 12))
                        if list(map(str, values["test_ids"])) != self.test_ids:
                            raise ValueError("Raptor raw prediction UID order drift")
                        self.observations["raptor_raw_valid"] = True
                        self.observations["raptor_observations"] = self.observations.get("raptor_observations", 0) + 1
                        return
                    if mode == "dino" and frame.f_code.co_name == "infer_from_package":
                        expected = set(self.contract["dino_member_ids"])
                        for key in ("per_member", "public_frontier_members"):
                            members = values[key]
                            if len(members) != 20 or {m["id"] for m in members} != expected:
                                raise ValueError("DINO needs all 20 unique pinned members and full frontier")
                            for member in members:
                                if len(member["ids"]) != n or set(map(str, member["ids"])) != set(self.test_ids):
                                    raise ValueError("DINO raw prediction study set drift")
                                self._array_valid(member["pred"], (n, 12))
                                if "soft_pred" in member:
                                    self._array_valid(member["soft_pred"], (n, 12))
                        self.observations["dino_raw_valid"] = True
                        self.observations["dino_observations"] = self.observations.get("dino_observations", 0) + 1
                        return
                    if mode == "a5" and frame.f_code.co_name == "<module>":
                        self._array_valid(values["preds"], (5, n, 12))
                        if len(values["studies"]) != n or set(map(str, values["studies"])) != set(self.test_ids):
                            raise ValueError("DINOv3 raw prediction study set drift")
                        self.observations["a5_raw_valid"] = True
                        self.observations["a5_observations"] = self.observations.get("a5_observations", 0) + 1
                        return
                frame = frame.f_back
            raise ValueError(f"Missing source frame for {mode} observation")
        finally:
            del frame

    def observe(self, line):
        # Never raise inside public print/catch blocks; reject immediately after the cell.
        try:
            if self.active_cell == 3 and "final submission.csv = weighted rank mean of" in line:
                self._inspect_frame("dino")
            if self.active_cell == 5:
                if "study failed:" in line:
                    self.failures.append("DINOv3 study failure")
                if "inference done in" in line:
                    self._inspect_frame("a5")
            if self.active_cell == 7:
                if "FALLBACK (" in line or "CoAtNet branch failed;" in line or "CoAtNet output unavailable;" in line:
                    self.failures.append("Raptor fallback or branch failure")
                done = re.search(r"\[arm ([0-3])\] done \+ freed", line)
                if done:
                    self.observations.setdefault("raptor_done", []).append(int(done[1]))
                progress = re.search(r"\[arm ([0-3])\] (\d+)/(\d+) \|", line)
                if progress and int(progress[2]) == int(progress[3]) == len(self.test_ids):
                    self.observations.setdefault("raptor_full_progress", []).append(int(progress[1]))
                if line.startswith("wrote /kaggle/working/submission_coatnet.csv |"):
                    self._inspect_frame("raptor")
        except Exception as exc:
            self.failures.append(f"Member observation failed: {type(exc).__name__}: {exc}")

    def run_cell(self, number, source, namespace):
        if not self.ready or self.completed != list(CELL_ORDER[: len(self.completed)]):
            raise RuntimeError("Prepare first and run cells exactly once in order")
        if len(self.completed) >= len(CELL_ORDER) or number != CELL_ORDER[len(self.completed)]:
            self.reject("Source cell execution order drift")
            raise RuntimeError("Source cell execution order drift")
        self.active_cell = number
        self.filename = f"<p002-public-source-cell-{number}>"
        try:
            if hashlib.sha256(source.encode()).hexdigest() != self.contract["source_cells"][str(number)]:
                raise ValueError("Public inference source changed")
            linecache.cache[self.filename] = (len(source), None, source.splitlines(keepends=True), self.filename)
            old_out, old_err = sys.stdout, sys.stderr
            try:
                sys.stdout, sys.stderr = Tee(old_out, self.observe), Tee(old_err, self.observe)
                exec(compile(source, self.filename, "exec", dont_inherit=True), namespace)
            finally:
                sys.stdout, sys.stderr = old_out, old_err
            if self.failures:
                raise RuntimeError("; ".join(self.failures))
            required = {3: "dino_raw_valid", 5: "a5_raw_valid", 7: "raptor_raw_valid"}.get(number)
            if required and not self.observations.get(required):
                raise RuntimeError(f"Full member completion was not observed: {required}")
            mode = {3: "dino", 5: "a5", 7: "raptor"}.get(number)
            if mode and self.observations.get(mode + "_observations") != 1:
                raise RuntimeError(f"Repeated or missing {mode} completion event")
            if number == 3:
                read_table(self.work / "submission.csv", self.test_ids)
            if number == 6 and namespace.get("V18_CALIBRATOR_APPLIED") is not True:
                raise RuntimeError("Fixed transformer calibration was skipped")
            if number == 7:
                for key in ("raptor_done", "raptor_full_progress"):
                    if self.observations.get(key) != [0, 1, 2, 3]:
                        raise RuntimeError("Raptor did not complete all four full-study arms")
            self.completed.append(number)
        except BaseException as exc:
            self.reject(f"Source cell {number} failed: {type(exc).__name__}")
            raise

    def finish(self):
        try:
            if not self.ready or self.completed != list(CELL_ORDER):
                raise RuntimeError("Incomplete source execution")
            if self.failures:
                raise RuntimeError("Unresolved member failures")
            for mode in ("dino", "a5", "raptor"):
                if not self.observations.get(mode + "_raw_valid") or self.observations.get(mode + "_observations") != 1:
                    raise RuntimeError(f"Missing or repeated {mode} completion")
            for key in ("raptor_done", "raptor_full_progress"):
                if self.observations.get(key) != [0, 1, 2, 3]:
                    raise RuntimeError("Raptor completion drift")
            receipt_files = {
                "public0033_cached_inference_receipt.json": "public0033_cached_inference_receipt_v1",
                "infra0021_runtime_receipt.json": "infra0021_runtime_receipt_v1",
                "public0033_overlay_receipt.json": "public0033_medial_t30_r60_b10_overlay_v1",
                "dinosaur_v4_v6_receipt.json": "dinosaur_v4_v6_outer_weights_v1",
            }
            receipts = {}
            for name, schema in receipt_files.items():
                receipt = json.loads((self.work / name).read_text(encoding="utf-8"))
                if receipt.get("status") != "passed" or (schema and receipt.get("schema_version") != schema):
                    raise ValueError(f"Missing/failed/wrong-schema receipt: {name}")
                for field, expected in self.contract.get("receipt_values", {}).get(name, {}).items():
                    actual = receipt
                    for part in field.split("."):
                        actual = actual[part]
                    if actual != expected or (isinstance(expected, bool) and actual is not expected):
                        raise ValueError(f"Receipt setting mismatch: {name}/{field}")
                for field, shape in self.contract.get("receipt_shapes", {}).get(name, {}).items():
                    actual = receipt
                    for part in field.split("."):
                        actual = actual[part]
                    if actual != [len(self.test_ids) if v == "N" else v for v in shape]:
                        raise ValueError(f"Receipt shape mismatch: {name}/{field}")
                receipts[name] = receipt
            final_path = self.work / "submission.csv"
            final_rows = read_table(final_path, self.test_ids)
            control = read_table(self.work / "submission_dinosaur_v4_0937_control.csv", self.test_ids)
            digest = sha256(final_path)
            if digest != sha256(self.work / "submission_dinosaur_v4_v6_candidate.csv"):
                raise ValueError("Wrong final candidate CSV")
            overlay = receipts["dinosaur_v4_v6_receipt.json"]
            if overlay.get("output_sha256") != digest or overlay.get("fixed_outer_weights") != WEIGHTS:
                raise ValueError("V6 output hash/weights mismatch")
            if overlay.get("medial_meniscus_route_preserved") is not True:
                raise ValueError("Medial Meniscus preservation gate failed")
            for index, target in enumerate(LABELS, 1):
                if target not in WEIGHTS and any(a[index] != b[index] for a, b in zip(final_rows, control)):
                    raise ValueError("Untouched target changed")
            bag_overlay = receipts["public0033_overlay_receipt.json"]
            if bag_overlay.get("output_sha256") != sha256(self.work / "submission_dinosaur_v4_0937_control.csv"):
                raise ValueError("Control does not match specialist overlay receipt")
            if bag_overlay.get("parent_submission_sha256") != sha256(self.work / "submission_parent_exact.csv"):
                raise ValueError("Parent identity mismatch")
            if bag_overlay.get("bag_raw_sha256") != sha256(self.work / "public0033_bag_raw.csv"):
                raise ValueError("Specialist output identity mismatch")
            parent_rows = read_table(self.work / "submission_parent_exact.csv", self.test_ids)
            for index, target in enumerate(LABELS, 1):
                if target != "Medial Meniscus" and any(a[index] != b[index] for a, b in zip(parent_rows, control)):
                    raise ValueError("Specialist overlay changed a non-target column")
            if self.contract.get("check_receipt_inputs"):
                audit = receipts["infra0021_runtime_receipt.json"]
                output = audit["output"]
                parent_sha = sha256(self.work / "submission_parent_exact.csv")
                if output["submission_sha256"] != parent_sha or output["backup_sha256"] != parent_sha:
                    raise ValueError("Audit parent hash mismatch")
                for name in ("test.csv", "test_series.csv"):
                    if audit["input"][name.replace(".", "_") + "_sha256"] != sha256(self.mounts["competition"] / name):
                        raise ValueError("Audit competition CSV hash mismatch")
                uid_hash = hashlib.sha256("".join(uid + "\n" for uid in self.test_ids).encode()).hexdigest()
                if audit["input"]["uid_sha256"] != uid_hash or audit["input"]["study_count"] != len(self.test_ids):
                    raise ValueError("Audit study identity mismatch")
                if bag_overlay["parent_uid_sha256"] != uid_hash:
                    raise ValueError("Overlay study identity mismatch")
                specialist = receipts["public0033_cached_inference_receipt.json"]
                if specialist["bundle_manifest_sha256"] != sha256(self.mounts["specialist"] / "bundle_manifest.json"):
                    raise ValueError("Specialist bundle mismatch")
                with (self.work / "public0033_bag_raw.csv").open(encoding="utf-8", newline="") as stream:
                    reader = csv.reader(stream)
                    bag_header, bag_rows = next(reader), list(reader)
                if bag_header != ["StudyInstanceUID", "Medial Meniscus", "Lateral Meniscus"]:
                    raise ValueError("Specialist CSV columns drift")
                if any(len(row) != 3 for row in bag_rows):
                    raise ValueError("Malformed specialist CSV")
                for row in bag_rows:
                    for value in row[1:]:
                        score = float(value)
                        if not math.isfinite(score) or not 0 <= score <= 1:
                            raise ValueError("Nonfinite or out-of-range specialist prediction")
                bag_ids = [row[0] for row in bag_rows]
                if len(bag_ids) != len(self.test_ids) or set(bag_ids) != set(self.test_ids):
                    raise ValueError("Specialist CSV study set drift")
                bag_uid_hash = hashlib.sha256("".join(uid + "\n" for uid in bag_ids).encode()).hexdigest()
                if specialist["study_uid_sha256"] != bag_uid_hash or specialist["study_count"] != len(bag_ids):
                    raise ValueError("Specialist UID receipt mismatch")
                if specialist["raw_output"]["sha256"] != sha256(self.work / "public0033_bag_raw.csv"):
                    raise ValueError("Specialist raw output hash mismatch")
            elapsed = time.monotonic() - self.started
            if elapsed >= 9 * 3600:
                raise RuntimeError("Full Notebook exceeded nine hours")
            result = {
                "status": "passed",
                "candidate_id": self.contract["candidate_id"],
                "studies": len(self.test_ids),
                "source_cells": self.completed,
                "output": "submission.csv",
                "sha256": digest,
                "elapsed_seconds": elapsed,
                "observed_members": self.observations,
                "public_score": None,
                "scoring_performed": False,
                "note": "This saved-run check is not evidence of hidden-test scoring or author-score reproduction.",
            }
            save_json(self.work / "P002_READY.json", result)
            print(f"P002 READY FOR MANUAL SUBMISSION | studies={len(self.test_ids)} | sha256={digest}")
            return result
        except BaseException as exc:
            self.reject(f"Final contract failed: {type(exc).__name__}")
            raise
