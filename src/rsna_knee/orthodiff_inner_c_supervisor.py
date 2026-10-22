"""Bound innerC runtime supervision; no parent selection, upload or overlay.

Actual POSIX launch is a separate, still unqualified integration. Import and
the fake-clock/process tests are standard-library only. The supervised source
is fixed, not an arbitrary argv runner. Parent source/Input wiring is external.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import math
import os
import signal
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

ID = "StudyInstanceUID"
TARGETS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")
RUNTIME_SHA = "c949ddebf6b571172193d10529d84c64dbaf2255d4e626b4175cfd145e5441a4"
MODEL_SHA = "c9b5ae26d0ad5895903002ff6bfd2c4947bed152c070c749588be8c76710a714"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def pin(path):
    h, size = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
            size += len(block)
    return {"bytes": size, "sha256": h.hexdigest()}


def load_json(path, expected_sha=None, maximum=64 * 1024**2):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= maximum,
            "Bounded regular JSON required")
    raw = path.read_bytes()
    require(expected_sha is None or digest(expected_sha) and hashlib.sha256(raw).hexdigest() == expected_sha,
            "JSON SHA256 differs")

    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON token")))


def save_json(path, value):
    path = Path(path)
    path.write_bytes((json.dumps(value, allow_nan=False, indent=2) + "\n").encode())


def safe_uid(uid):
    require(isinstance(uid, str) and uid and uid.strip() == uid and uid not in {".", ".."}
            and not any(c in uid for c in "/\\:\x00\r\n"), "Unsafe UID")
    return uid


def uid_hash(uids):
    return hashlib.sha256("".join(uid + "\n" for uid in uids).encode()).hexdigest()


def test_uids(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        require(header and len(set(header)) == len(header) and ID in header, "Test UID header required")
        index, uids = header.index(ID), []
        for row in reader:
            require(len(row) == len(header), "Test row width differs")
            uids.append(safe_uid(row[index]))
    require(uids and len(set(uids)) == len(uids), "Unique requested test UIDs required")
    return uids


@dataclass(frozen=True)
class TimeBudget:
    notebook_started: float
    notebook_seconds: float
    head_reserve_seconds: float
    csv_reserve_seconds: float
    term_grace_seconds: float
    reap_timeout_seconds: float
    poll_seconds: float

    def validate(self):
        values = tuple(vars(self).values())
        require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values), "Finite time parameters required")
        require(0 < self.notebook_seconds <= 9 * 3600 and self.head_reserve_seconds > 0
                and self.csv_reserve_seconds > 0 and 0 < self.term_grace_seconds <= 10
                and 0 < self.reap_timeout_seconds <= 10 and 0 < self.poll_seconds <= 1,
                "Explicit whole-notebook/reserve/short cleanup parameters required")

    @property
    def hard_deadline(self):
        return self.notebook_started + self.notebook_seconds

    @property
    def cleanup_reserve(self):
        return self.term_grace_seconds + self.reap_timeout_seconds

    def allowance(self, mode, now):
        self.validate()
        require(now >= self.notebook_started, "Monotonic clock precedes Notebook start")
        remaining = self.hard_deadline - now - self.csv_reserve_seconds - self.cleanup_reserve
        value = remaining - self.head_reserve_seconds if mode == "features" else min(remaining, self.head_reserve_seconds)
        require(value > 0, "Insufficient whole-notebook time/reserve; do not launch")
        return value


@dataclass(frozen=True)
class BoundRun:
    bundle: Path
    bundle_sha256: str
    model: Path
    qualification: Path
    qualification_sha256: str
    competition_root: Path
    parent_csv: Path
    expected_parent_sha256: str
    output_dir: Path
    budget: TimeBudget
    max_artifact_bytes: int

    @property
    def runtime(self):
        return Path(self.bundle).parent / "runtime.py"


class ParentBytes:
    def __init__(self, path, expected_sha, uids, maximum):
        self.path = Path(path).absolute()
        require(self.path.is_file() and not self.path.is_symlink() and self.path.stat().st_size <= maximum,
                "Bounded regular owned parent CSV required")
        self.raw = self.path.read_bytes()
        require(digest(expected_sha) and hashlib.sha256(self.raw).hexdigest() == expected_sha, "Parent SHA differs")
        reader = csv.reader(io.StringIO(self.raw.decode("utf-8-sig"), newline=""))
        require(next(reader, None) == [ID, *TARGETS], "Exact canonical parent header required")
        parent_uids = []
        for row in reader:
            require(len(row) == 13, "Parent CSV width differs")
            parent_uids.append(safe_uid(row[0]))
            require(all(math.isfinite(float(cell)) and 0 <= float(cell) <= 1 for cell in row[1:]),
                    "Finite parent probability required")
        require(len(parent_uids) == len(set(parent_uids)) == len(uids) and set(parent_uids) == set(uids),
                "Exact parent/test UID join required")

    def unchanged(self):
        return self.path.is_file() and not self.path.is_symlink() and self.path.read_bytes() == self.raw

    def restore(self):
        if self.unchanged():
            return False
        # Replace only this explicitly SHA-bound parent path. No recursive cleanup.
        fd, name = tempfile.mkstemp(prefix=".innerC-parent-restore-", dir=self.path.parent)
        with os.fdopen(fd, "wb") as stream:
            stream.write(self.raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, self.path)
        require(self.unchanged(), "Parent byte restoration failed")
        return True


def validate_source_binding(spec):
    require(isinstance(spec, BoundRun) and type(spec.max_artifact_bytes) is int
            and spec.max_artifact_bytes > 0, "Explicit bound source/resource spec required")
    spec.budget.validate()
    require(spec.runtime.is_file() and not spec.runtime.is_symlink() and pin(spec.runtime)["sha256"] == RUNTIME_SHA,
            "Only the frozen innerC runtime.py may be supervised")
    bundle = load_json(spec.bundle, spec.bundle_sha256, maximum=4 * 1024**2)
    require(bundle.get("schema_version") == "orthodiff_innerC_dynamic_test_transport_v1"
            and bundle.get("actual_entry_enabled") is True and bundle.get("model_sha256") == MODEL_SHA
            and bundle.get("transport_sources", {}).get("runtime") == pin(spec.runtime),
            "Actual-enabled pinned runtime bundle required")
    q = load_json(spec.qualification, spec.qualification_sha256, maximum=2 * 1024**2)
    require(q.get("schema_version") == "orthodiff_innerC_numpy_transport_qualification_v1"
            and q.get("passed") is True and q.get("scope") == "actual_sealed_single_model"
            and q.get("feature_transport_bundle_sha256") == spec.bundle_sha256
            and q.get("bindings", {}).get("model_sha256") == MODEL_SHA
            and q.get("actual_model_qualified") is True and q.get("runtime_numerical_audit_passed") is True
            and q.get("qualified_numpy_version") == "2.1.3"
            and q.get("outer2_engineering_and_fixed10_positive") is True
            and q.get("root_candidate_approved") is True and digest(q.get("outer2_result_sha256"))
            and digest(q.get("root_candidate_review_sha256")), "Actual outer2/root/source qualification required")
    require(Path(spec.model).is_file() and not Path(spec.model).is_symlink(), "Regular staged model required")
    # The immutable child separately hashes/validates c9b and all N33af metadata.
    # Do not deserialize a model or import scientific dependencies here.
    return bundle


def _npy(raw):
    require(raw[:6] == b"\x93NUMPY" and raw[6:8] in (b"\x01\x00", b"\x02\x00", b"\x03\x00"), "NPY header required")
    width = 2 if raw[6] == 1 else 4
    size = int.from_bytes(raw[8:8 + width], "little")
    require(0 < size <= 8192 and len(raw) >= 8 + width + size, "Bounded complete NPY header required")
    header = ast.literal_eval(raw[8 + width:8 + width + size].decode("utf-8" if raw[6] == 3 else "latin1"))
    require(isinstance(header, dict) and set(header) == {"descr", "fortran_order", "shape"}
            and header["fortran_order"] is False, "Simple C-order numeric/Unicode NPY required")
    return header, raw[8 + width + size:]


def validate_features(folder, spec, expected_uids, allowance):
    folder = Path(folder)
    execution = load_json(folder / "execution-receipt.json", maximum=spec.max_artifact_bytes)
    path = folder / "feature-receipt.json"
    receipt = load_json(path, maximum=spec.max_artifact_bytes)
    require(execution.get("schema_version") == "orthodiff_innerC_test_feature_execution_v1"
            and execution.get("complete") is True and "error" not in execution
            and execution.get("model_load_calls") == 1 and execution.get("owned_scientific_temp_removed") is True,
            "Normal-exit complete/clean single-backbone execution required")
    resources = execution["resources"]
    require(0 <= resources["elapsed_seconds"] <= allowance and 0 <= resources["CUDA_reserved_bytes"] <= 14 * 1024**3
            and 0 <= resources["RSS_bytes"] <= 8 * 1024**3, "Feature execution resource receipt differs")
    require(receipt.get("schema_version") == "orthodiff_innerC_test_features_complete_v1"
            and receipt.get("complete") is True and receipt.get("feature_transport_bundle_sha256") == spec.bundle_sha256
            and receipt.get("feature_dimension") == 1024 and receipt.get("zero_filled_rows") == 0
            and receipt.get("requested_uids") == expected_uids and receipt.get("UID_order_sha256") == uid_hash(expected_uids)
            and execution.get("handoff", {}).get("receipt") == pin(path), "Complete bound UID/receipt required")
    matrix_path = folder / "success-features.npz"
    require(matrix_path.is_file() and not matrix_path.is_symlink() and matrix_path.stat().st_size <= spec.max_artifact_bytes
            and receipt["matrix"]["filename"] == matrix_path.name
            and {key: receipt["matrix"][key] for key in ("bytes", "sha256")} == pin(matrix_path), "Matrix artifact SHA differs")
    success = receipt["success_uids"]
    require(isinstance(success, list) and len(success) == len(set(success)) and set(success) <= set(expected_uids), "Unique success subset required")
    with zipfile.ZipFile(matrix_path) as archive:
        entries = archive.infolist()
        require(len(entries) == 2 and {item.filename for item in entries} == {"success_uids.npy", "features.npy"}
                and all(item.compress_type == zipfile.ZIP_STORED and item.file_size <= spec.max_artifact_bytes for item in entries),
                "Only the fixed uncompressed numeric handoff arrays are allowed")
        uid_header, uid_raw = _npy(archive.read("success_uids.npy"))
        matrix_header, matrix = _npy(archive.read("features.npy"))
    descriptor = uid_header["descr"]
    require(isinstance(descriptor, str) and descriptor.startswith("<U") and descriptor[2:].isdigit(), "Unicode UID dtype required")
    chars = int(descriptor[2:])
    require(0 < chars <= 512 and uid_header["shape"] == (len(success),)
            and len(uid_raw) == len(success) * chars * 4, "Bounded exact UID array required")
    actual_uids = [uid_raw[i * chars * 4:(i + 1) * chars * 4].decode("utf-32-le").rstrip("\0") for i in range(len(success))]
    require(actual_uids == success and matrix_header["descr"] == "<f4"
            and matrix_header["shape"] == (len(success), 1024) and len(matrix) == len(success) * 4096,
            "Exact success UID and FP32 matrix dimensions required")
    require(all(math.isfinite(value[0]) for value in struct.iter_unpack("<f", matrix)), "Nonfinite successful features")
    require(len(receipt["records"]) == len(expected_uids), "Exactly one record per test UID required")
    index, failures = 0, []
    for uid, record in zip(expected_uids, receipt["records"]):
        require(record.get(ID) == uid, "Ordered UID record differs")
        if record.get("status") == "success":
            row = matrix[index * 4096:(index + 1) * 4096]
            require(index < len(success) and success[index] == uid and record.get("regional_2x2") == {
                "shape": [1, 1024], "dtype": "torch.float32", "bytes": 4096, "sha256": hashlib.sha256(row).hexdigest()}
                and record.get("forward_counts") == {"denoiser": 1, "decoder_blocks": 0, "final_out": 0}
                and record.get("all_RNG_restored") is True, "Success row SHA/forward/RNG differs")
            index += 1
        else:
            require(record.get("status") == "data_failure" and record.get("reason")
                    and record.get("feature_forward_calls") == 0 and record.get("fallback_features") is False
                    and not any(key in record for key in ("regional_2x2", "regional_2x2_float32_values", "mid_2")),
                    "No-forward typed failure without fabricated features required")
            failures.append(uid)
    require(index == len(success) and failures == receipt["failure_uids"]
            and execution["handoff"]["success"] == len(success) and execution["handoff"]["failure"] == len(failures),
            "Complete success/failure partition differs")
    return receipt


def validate_head(folder, spec, features, feature_folder):
    folder = Path(folder)
    receipt = load_json(folder / "head-receipt.json", maximum=spec.max_artifact_bytes)
    require(receipt.get("schema_version") == "orthodiff_innerC_test_head_predictions_v1"
            and receipt.get("complete") is True and receipt.get("bundle_sha256") == spec.bundle_sha256
            and receipt.get("model_sha256") == MODEL_SHA and receipt.get("qualification_sha256") == spec.qualification_sha256
            and receipt.get("feature_receipt") == pin(Path(feature_folder) / "feature-receipt.json")
            and receipt.get("UID_order_sha256") == features["UID_order_sha256"]
            and receipt.get("success_uids") == features["success_uids"] and receipt.get("failure_uids") == features["failure_uids"]
            and receipt.get("numpy") == "2.1.3" and receipt.get("parent_selected_or_overlaid") is False
            and receipt.get("Public_submission_contract_qualified") is False, "Complete source-bound CPU head required")
    path = folder / "head-success.csv"
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= spec.max_artifact_bytes
            and receipt["probabilities"] == pin(path), "Head CSV artifact SHA differs")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        require(next(reader, None) == [ID, *TARGETS], "Exact canonical head header required")
        seen = []
        for row in reader:
            require(len(row) == 13 and all(math.isfinite(float(x)) and 0 <= float(x) <= 1 for x in row[1:]),
                    "Twelve finite ordered head probabilities required")
            seen.append(row[0])
    require(seen == features["success_uids"], "Head success UID order differs")
    return receipt


class PosixOps:
    """Keep exited leader waitable until failure group cleanup has finished."""
    def ids(self, pid):
        return os.getpgid(pid), os.getsid(pid)

    def own_group(self):
        return os.getpgrp()

    def peek(self, process):
        state = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        if state is None:
            return None
        return state.si_status if state.si_code == os.CLD_EXITED else -state.si_status

    def group_signal(self, pgid, value):
        os.killpg(pgid, value)


def _stop_owned(process, ops, clock, sleep, budget, trace):
    require(ops.ids(process.pid) == (process.pid, process.pid) and process.pid != ops.own_group(),
            "Refusing signal outside the fresh owned session/group")
    for value in (signal.SIGTERM, getattr(signal, "SIGKILL", 9)):
        try:
            ops.group_signal(process.pid, value)
            trace["signals"].append(int(value))
        except ProcessLookupError:
            trace["group_already_absent"] = True
        if value == signal.SIGTERM:
            deadline = clock() + budget.term_grace_seconds
            while clock() < deadline:
                sleep(min(budget.poll_seconds, deadline - clock()))
    trace["wait_returncode"] = process.wait(timeout=budget.reap_timeout_seconds)
    trace["leader_reaped"] = True


def _run_child(spec, mode, deadline, workspace, spawn, ops, clock, sleep, traces, validate_artifacts,
               artifact_folder, allowance, prior_feature_folder=None):
    """Private monitor constructs only the fixed runtime/mode argv itself."""
    require(pin(spec.runtime)["sha256"] == RUNTIME_SHA, "Runtime changed before launch")
    argv = _argv(spec, mode, artifact_folder, allowance, prior_feature_folder)
    workspace.mkdir()
    trace = {"mode": mode, "launches": 1, "signals": [], "leader_reaped": False,
             "start_new_session": True, "deadline_monotonic": deadline}
    traces.append(trace)
    with (workspace / "stdout.log").open("xb") as stdout, (workspace / "stderr.log").open("xb") as stderr:
        process = spawn(argv, cwd=str(workspace), stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                        shell=False, close_fds=True, start_new_session=True)
        trace["pid"] = process.pid
        try:
            require(type(process.pid) is int and process.pid > 1 and ops.ids(process.pid) == (process.pid, process.pid)
                    and process.pid != ops.own_group(), "Fresh owned POSIX session/group required")
        except BaseException:
            # Only this Popen child handle, never an unverified group.
            process.kill()
            trace["unverified_group_direct_child_kill"] = True
            trace["wait_returncode"] = process.wait(timeout=spec.budget.reap_timeout_seconds)
            trace["leader_reaped"] = True
            raise
        try:
            while True:
                status = ops.peek(process)
                if status is not None:
                    require(status == 0, "Child exited nonzero: " + str(status))
                    # Keep the exited leader unreaped while checking artifacts.
                    # Incomplete output still gets safe cleanup of its owned PG.
                    validated = validate_artifacts()
                    require(clock() < deadline, "Artifact validation exceeded reserved stage deadline")
                    trace["wait_returncode"] = process.wait(timeout=spec.budget.reap_timeout_seconds)
                    trace["leader_reaped"] = True
                    require(trace["wait_returncode"] == 0, "Child wait status differs")
                    return validated
                require(clock() < deadline, "Owned child exceeded reserved Notebook deadline")
                sleep(min(spec.budget.poll_seconds, deadline - clock()))
        except BaseException:
            if not trace["leader_reaped"]:
                _stop_owned(process, ops, clock, sleep, spec.budget, trace)
            raise


def _argv(spec, mode, output, allowance, feature_folder=None):
    common = [sys.executable, "-I", str(spec.runtime.absolute()), mode,
              "--bundle", str(Path(spec.bundle).absolute()), "--bundle-sha", spec.bundle_sha256,
              "--model", str(Path(spec.model).absolute()), "--model-sha", MODEL_SHA,
              "--qualification", str(Path(spec.qualification).absolute()),
              "--qualification-sha", spec.qualification_sha256, "--output-dir", str(output.absolute())]
    if mode == "features":
        return common + ["--competition-root", str(Path(spec.competition_root).absolute()),
                         "--seconds-budget", repr(allowance)]
    require(mode == "head" and feature_folder is not None, "Only the fixed feature/head modes are permitted")
    return common + ["--feature-dir", str(feature_folder.absolute()), "--feature-receipt-sha",
                     pin(feature_folder / "feature-receipt.json")["sha256"]]


def _pipeline(spec, *, spawn, ops, clock, sleep):
    """Private dependency injection for CPU fake probes; public entry checks OS."""
    validate_source_binding(spec)
    requested = test_uids(Path(spec.competition_root) / "test.csv")
    parent = ParentBytes(spec.parent_csv, spec.expected_parent_sha256, requested, spec.max_artifact_bytes)
    output = Path(spec.output_dir).absolute()
    require(not output.exists() and not output.is_symlink(), "Fresh dedicated supervisor workspace required")
    require(parent.path != output and output not in parent.path.parents, "Parent must be outside child workspace")
    output.mkdir(parents=True)
    result = {"schema_version": "orthodiff_innerC_bound_child_supervision_v1", "complete": False,
              "head_artifacts_verified": False, "Public_submission_contract_qualified": False,
              "parent_sha256": spec.expected_parent_sha256, "bundle_sha256": spec.bundle_sha256,
              "source_sha256": RUNTIME_SHA, "requested_UID_order_sha256": uid_hash(requested),
              "children": [], "automatic_retry_count": 0, "parent_restored": False,
              "budget": vars(spec.budget), "parent_source_Input_version_or_overlay_selected": False,
              "upper_driver_must_fail_when_incomplete": True}
    try:
        feature_folder, head_folder = output / "features", output / "head"
        now = clock()
        allowance = spec.budget.allowance("features", now)

        def feature_validation():
            value = validate_features(feature_folder, spec, requested, allowance)
            require(parent.unchanged(), "Parent changed during feature stage")
            return value

        features = _run_child(spec, "features", now + allowance,
                   output / "feature-process", spawn, ops, clock, sleep, result["children"],
                   feature_validation, feature_folder, allowance)
        validate_source_binding(spec)
        now = clock()
        allowance = spec.budget.allowance("head", now)

        def head_validation():
            value = validate_head(head_folder, spec, features, feature_folder)
            require(parent.unchanged(), "Parent changed during head stage")
            return value

        _run_child(spec, "head", now + allowance,
                   output / "head-process", spawn, ops, clock, sleep, result["children"],
                   head_validation, head_folder, allowance, feature_folder)
        require(parent.unchanged() and clock() < spec.budget.hard_deadline - spec.budget.csv_reserve_seconds,
                "Parent/remaining CSV completion budget differs")
        result.update(complete=True, head_artifacts_verified=True,
                      feature_receipt=pin(feature_folder / "feature-receipt.json"),
                      head_receipt=pin(head_folder / "head-receipt.json"),
                      head_csv=pin(head_folder / "head-success.csv"),
                      remaining_seconds=spec.budget.hard_deadline - clock())
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        result["parent_restored"] = parent.restore()
    finally:
        result["parent_bytes_unchanged"] = parent.unchanged()
        save_json(output / "supervisor-receipt.json", result)
    return result


class SupervisionFailure(RuntimeError):
    """An invalid extension is a failed candidate, never a successful parent-only run."""
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__("innerC extension incomplete; upper candidate driver must stop")


def supervise(spec):
    """Future actual entry, POSIX only; Windows refuses before any launch/read."""
    require(os.name == "posix", "Actual owned-session supervision requires POSIX; Windows is not qualified")
    require(all(hasattr(os, name) for name in ("waitid", "WNOWAIT", "WEXITED", "WNOHANG", "P_PID", "getpgid", "getsid", "killpg")),
            "Non-reaping POSIX child observation required")
    result = _pipeline(spec, spawn=subprocess.Popen, ops=PosixOps(), clock=time.monotonic, sleep=time.sleep)
    if result["complete"] is not True:
        raise SupervisionFailure(result)
    return result
