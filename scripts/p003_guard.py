"""Strict execution envelope for the frozen haideptry V2 candidate.

Only the manually executed Kaggle notebook runs inference. This module does not
download data, fit coefficients, train, upload, or submit anything.
"""

import csv
import hashlib
import importlib.metadata
import json
import linecache
import os
import subprocess
import sys
import time
from pathlib import Path

LABELS = [
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
]
MEMBERS = ("resgated_top3", "global96_top3", "d4_swa3", "repairv1_top3")
PROB_FILES = (
    "coat_resgated_ep10_top3_predictions.npz",
    "g96_input/coatnet_global96_top3_predictions.npz",
    "d4_input/coatnet_d4_depthzone_swa3_predictions.npz",
    "coatnet_repairv1_top3_predictions.npz",
)
RECEIPT_FILES = (
    "_coat_arm_receipt.json",
    "_coat_g96_receipt.json",
    "d4_input/d4.receipt.json",
    "_coat_repairv1_receipt.json",
)
RECEIPT_STATUS = (
    "VALID_COAT_RESGATED_EP10_TOP3_RANK_SUBMISSION",
    "VALID_COATNET_GLOBAL96_TOP3_T4X2_SUBMISSION",
    "VALID_COATNET_D4_DEPTHZONE_SWA3_T4X2_SUBMISSION",
    "VALID_COATNET_REPAIRV1_TOP3_PROBABILITY_RANK_SUBMISSION",
)
OUTER = dict.fromkeys(LABELS, 0.60)
OUTER.update({"ACL": 0.75, "Medial Meniscus": 0.80, "Lateral Meniscus": 1.0, "Lateral OA": 0.75, "Fracture": 0.75})
ALLOWED_EVENTS = {
    "dino_shared_path_parity",
    "raptor_checkpoint",
    "dino_fp16_nonfinite_retry_fp32",
    "dino_legacy_fp16_nonfinite_retry_fp32",
    "raptor_fp16_nonfinite_retry_fp32",
    "a5_fp16_nonfinite_retry_fp32",
    "rad_fp16_nonfinite_retry_fp32",
    "rad_head_fp16_nonfinite_retry_fp32",
    "cache_complete",
    "scratch_allocation",
    "scratch_fallback",
    "a5_no_metadata",
    "a5_empty_study",
    "empty_study_rows",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_scores(path, ids):
    import numpy as np

    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows or rows[0] != ["StudyInstanceUID", *LABELS]:
        raise ValueError(f"Wrong label order: {path}")
    if any(len(row) != 13 for row in rows[1:]) or [row[0] for row in rows[1:]] != ids:
        raise ValueError(f"Wrong UID order/coverage: {path}")
    return probabilities(np.asarray([row[1:] for row in rows[1:]], dtype=np.float64), (len(ids), 12))


def probabilities(value, shape):
    import numpy as np

    value = np.asarray(value)
    if value.shape != shape or not np.isfinite(value).all() or (value < 0).any() or (value > 1).any():
        raise ValueError(f"Invalid raw predictions: expected {shape}, got {value.shape}")
    return value


def aligned_npz(path, ids, key="values", labels_required=True):
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        uids = data["study_uids"].astype(str).tolist()
        if len(uids) != len(ids) or len(set(uids)) != len(uids) or set(uids) != set(ids):
            raise ValueError(f"Raw UID coverage mismatch: {path}")
        if (labels_required or "labels" in data) and data["labels"].astype(str).tolist() != LABELS:
            raise ValueError(f"Raw label order mismatch: {path}")
        if key == "probability_mean" and key not in data:
            raw = np.asarray(data["raw_probabilities"], dtype=np.float64)
            probabilities(raw, (3, len(ids), 12))
            value = raw.mean(axis=0)
        else:
            value = np.asarray(data[key], dtype=np.float64)
    value = probabilities(value, (len(ids), 12))
    order = {uid: index for index, uid in enumerate(uids)}
    return value[[order[uid] for uid in ids]]


def rank_pct(values):
    """Average ties over the complete cohort, never separately by GPU/chunk."""
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Rank input must be a nonempty finite matrix")
    result = np.empty_like(values)
    n = len(values)
    for col in range(values.shape[1]):
        order = np.argsort(values[:, col], kind="stable")
        start = 0
        while start < n:
            end = start + 1
            while end < n and values[order[end], col] == values[order[start], col]:
                end += 1
            result[order[start:end], col] = ((start + 1 + end) / 2) / n
            start = end
    return result


def replay_final(parent, raptor, coats):
    """Frozen V2 coefficients; callers must align the full cohort first."""
    import numpy as np

    if len(coats) != 4:
        raise ValueError("All four CoAt families are required")
    shape = parent.shape
    for value in (parent, raptor, *coats):
        probabilities(value, shape)
    family = rank_pct(sum(coats) / 4.0)
    hybrid = 0.6 * rank_pct(raptor) + 0.4 * family
    weights = np.asarray([OUTER[label] for label in LABELS])
    return rank_pct((1 - weights) * rank_pct(parent) + weights * hybrid), family


def check_blend_receipt(receipt, n):
    expected = {
        "members": list(MEMBERS),
        "study_count": n,
        "inner_rerank": False,
        "private_alpha": 0.4,
        "public_raptor_alpha": 0.6,
        "within_coat": dict.fromkeys(MEMBERS, 0.25),
        "finding_specific_weights": False,
        "family_reduction": "rank_of_member_probability_mean",
    }
    for key, value in expected.items():
        if receipt.get(key) != value or (isinstance(value, bool) and receipt.get(key) is not value):
            raise ValueError(f"CoAt composition changed: {key}")


class P003Guard:
    def __init__(self, contract, work="/kaggle/working", input_root="/kaggle/input"):
        self.contract = contract
        self.work, self.input_root = Path(work), Path(input_root)
        self.started = time.monotonic()
        self.completed, self.timings = [], []
        self.ids, self.mounts, self.input_files = [], {}, []
        self.ready = False
        self.owns_outputs = False

    def prepare_files(self):
        self.work.mkdir(parents=True, exist_ok=True)
        # Source rsna_initialize removes its old output. Refuse before it can do so.
        notebook_runtime_files = {".virtual_documents", "__notebook__.ipynb", "__notebook_source__.ipynb"}
        if any(path.name not in notebook_runtime_files for path in self.work.iterdir()):
            raise RuntimeError("Use a fresh Kaggle saved session with an empty /kaggle/working")
        for entry in self.contract["inputs"]:
            roots = set()
            for relative in entry["mounts"]:
                path = (self.input_root / relative).resolve()
                path.relative_to(self.input_root.resolve())
                if path.is_dir():
                    roots.add(path)
            if len(roots) != 1:
                raise RuntimeError(f"Attach one pinned Input: {entry['ref']} version {entry['version']}")
            root = roots.pop()
            self.mounts[entry["key"]] = root
            for item in entry["files"]:
                path = (root / item["path"]).resolve()
                path.relative_to(root)
                if not path.is_file() or (item.get("bytes") is not None and path.stat().st_size != item["bytes"]):
                    raise ValueError(f"Missing/wrong-size asset: {entry['key']}/{item['path']}")
                digest = sha256(path)
                if item.get("sha256") and digest != item["sha256"]:
                    raise ValueError(f"Input SHA-256 mismatch: {entry['key']}/{item['path']}")
                self.input_files.append(
                    {
                        "input": entry["key"],
                        "path": item["path"],
                        "sha256": digest,
                        "expected_hash_verified": bool(item.get("sha256")),
                    }
                )
        # Reject undeclared mounts that could shadow the source's filename resolver.
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
                    raise ValueError(f"Remove undeclared Input: {child}")
        competition = self.mounts["competition"]
        for name in ("train.csv", "test.csv", "test_series.csv", "sample_submission.csv"):
            if not (competition / name).is_file():
                raise FileNotFoundError(name)
        if not (competition / "test_series").is_dir():
            raise RuntimeError("Missing test_series; refuse train fallback")
        with (competition / "test.csv").open(encoding="utf-8-sig", newline="") as f:
            self.ids = [row["StudyInstanceUID"] for row in csv.DictReader(f)]
        if not self.ids or any(not uid for uid in self.ids) or len(set(self.ids)) != len(self.ids):
            raise ValueError("Invalid test UID set")
        read_scores(competition / "sample_submission.csv", self.ids)
        self.owns_outputs = True

    def prepare(self):
        self.prepare_files()
        if Path.cwd().resolve() != self.work.resolve():
            raise RuntimeError("Execute in /kaggle/working")
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "PIP_NO_INDEX"):
            os.environ[name] = "1"
        # Preserve the source's cohort-size scheduling branch; reject inherited overrides.
        if "RSNA_PARALLEL_COAT_READERS" in os.environ or "SLOT_SCHEME" in os.environ:
            raise RuntimeError("Remove scheduling/slot overrides; this is the frozen V2 recipe")
        wheel = self.mounts["knee_mri_fold_weights"] / "timm-1.0.22-py3-none-any.whl"
        try:
            installed = importlib.metadata.version("timm")
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed != "1.0.22":
            if "timm" in sys.modules:
                raise RuntimeError("Restart before pinning timm")
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)], check=True)
        import torch

        if torch.cuda.device_count() != 2 or any("T4" not in torch.cuda.get_device_name(i) for i in range(2)):
            raise RuntimeError("Select GPU T4 x2")
        self.ready = True
        save_json(
            self.work / "P003_PREFLIGHT.json",
            {
                "status": "passed",
                "inputs": self.input_files,
                "candidate_id": self.contract["candidate_id"],
                "studies": len(self.ids),
                "environment": {
                    "python": sys.version,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "timm": importlib.metadata.version("timm"),
                    "gpu": [torch.cuda.get_device_name(i) for i in range(2)],
                },
                "historical_author_inputs_verified": False,
            },
        )

    def check_events(self, namespace):
        bad = [
            event
            for event in namespace.get("_RSNA_AUDIT", {}).get("events", [])
            if event.get("kind") not in ALLOWED_EVENTS
        ]
        if bad:
            raise ValueError(f"Source reported degraded/unknown events: {bad[:8]}")

    def check_dino(self, namespace):
        if namespace.get("_DINOV2_MATCHED_MEMBERS") != 20:
            raise ValueError("DINO must complete 20 members")
        parity = [
            event["member"]
            for event in namespace["_RSNA_AUDIT"]["events"]
            if event["kind"] == "dino_shared_path_parity"
        ]
        expected = self.contract["dino_member_ids"]
        if len(parity) != 20 or set(parity) != set(expected):
            raise ValueError("DINO shared/full forward parity did not cover all 20 members")
        for member in expected:
            aligned_npz(self.work / "diagnostics" / f"dino_{member}.npz", self.ids)

    def check_a5(self):
        import numpy as np

        with np.load(self.work / "speed_a5_raw.npz", allow_pickle=False) as data:
            if data["study_uids"].astype(str).tolist() != self.ids:
                raise ValueError("A5 UID order changed")
            raw = data["raw_probabilities"]
            applicable = data["applicable"]
            if raw.shape != (5, len(self.ids), 12) or applicable.shape != (len(self.ids),) or applicable.dtype != bool:
                raise ValueError("A5 shape/applicability contract failed")
            probabilities(raw[:, applicable], (5, int(applicable.sum()), 12))
            if not np.isnan(raw[:, ~applicable]).all():
                raise ValueError("A5 unavailable rows must remain missing")
        aligned_npz(self.work / "diagnostics/a5_rank_mean.npz", self.ids)

    def check_coats(self, namespace):
        import numpy as np

        logs = ("resgated_runtime.log", "global96_runtime.log", "d4_input/d4.log", "repairv1_runtime.log")
        for name in logs:
            path = self.work / name
            with path.open(encoding="utf-8", errors="replace") as stream:
                if any("[dense-fallback]" in line for line in stream):
                    raise ValueError(f"CoAt input preparation changed after failure: {name}")
        check_blend_receipt(read_json(self.work / "_coat_raptor_blend_receipt.json"), len(self.ids))
        for name, status, epochs, models in zip(
            RECEIPT_FILES, RECEIPT_STATUS, ([4, 6, 8], [16, 23, 18], None, [12, 7, 11]), (3, 3, 1, 3)
        ):
            receipt = read_json(self.work / name)
            if (
                receipt.get("status") != status
                or receipt.get("fallback_studies") != 0
                or receipt.get("models") != models
            ):
                raise ValueError(f"Incomplete CoAt family: {name}")
            if epochs and [item["epoch"] for item in receipt["checkpoints"]] != epochs:
                raise ValueError(f"Wrong checkpoint epochs: {name}")
            for shard in receipt.get("shards", []):
                if shard.get("preparation_warnings", 0) != 0 or shard.get("fallback_studies", 0) != 0:
                    raise ValueError(f"Degraded preparation in CoAt shard: {name}")
        expected = {(name, uid) for name in ("maxspan-v5", "native384dense-v10", "native384-v8") for uid in self.ids}
        if namespace.get("_ke_input_ids") != expected:
            raise ValueError("Incomplete Raptor preparation UID coverage")
        if namespace.get("_coatnet_weight") != OUTER:
            raise ValueError("Outer weights changed")
        parent = aligned_npz(self.work / "btk_v32_rest_input.npz", self.ids)
        raptor = read_scores(self.work / "raptor_input_before_coat.csv", self.ids)
        coats = [aligned_npz(self.work / name, self.ids, "probability_mean", False) for name in PROB_FILES]
        _, family = replay_final(parent, raptor, coats)
        expected_hybrid = 0.6 * rank_pct(raptor) + 0.4 * family
        # Source serializes the hybrid CSV and reads it with pandas before ranking.
        # Validate that roundtrip, then use its actual floating values for ties.
        actual_hybrid = probabilities(np.asarray(namespace["_blend_cr"]), (len(self.ids), 12))
        actual_parent = probabilities(np.asarray(namespace["_blend_tr"]), (len(self.ids), 12))
        if not np.allclose(actual_hybrid, expected_hybrid, rtol=0, atol=1e-12):
            raise ValueError("Inner hybrid changed beyond CSV roundtrip precision")
        if not np.allclose(actual_parent, rank_pct(parent), rtol=0, atol=1e-12):
            raise ValueError("Parent full-cohort ranks changed")
        weights = np.asarray([OUTER[label] for label in LABELS])
        final = rank_pct((1 - weights) * actual_parent + weights * actual_hybrid)
        if not np.allclose(read_scores(self.work / "_coat_family_rank.csv", self.ids), family, rtol=0, atol=1e-12):
            raise ValueError("CoAt reduction is not rank of the full-cohort probability mean")
        if not np.allclose(read_scores(self.work / "_pipeline_stage.csv", self.ids), final, rtol=0, atol=1e-12):
            raise ValueError("Final predictions do not match the frozen V2 recipe")

    def reject(self, reason):
        self.ready = False
        if self.owns_outputs:
            final = self.work / "submission.csv"
            if final.exists():
                final.rename(self.work / "P003_REJECTED_submission.csv.disabled")
            save_json(
                self.work / "P003_FAILED.json",
                {"status": "failed", "reason": reason, "completed_cells": self.completed, "timings": self.timings},
            )

    def run_cell(self, number, source, namespace):
        order = self.contract["cell_order"]
        started = time.monotonic()
        try:
            if not self.ready or self.completed != order[: len(self.completed)] or number != order[len(self.completed)]:
                raise RuntimeError("Run every cell once, in order, in a fresh session")
            if hashlib.sha256(source.encode()).hexdigest() != self.contract["source_cells"][str(number)]:
                raise ValueError("Frozen source changed")
            filename = f"<p003-source-{number}>"
            linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
            exec(compile(source, filename, "exec", dont_inherit=True), namespace)
            self.check_events(namespace)
            if number == 17:
                self.check_dino(namespace)
            if number == 22:
                self.check_a5()
            if number == 25 and namespace.get("V18_CALIBRATOR_APPLIED") is not True:
                raise ValueError("Fixed Rad calibrator missing")
            if number == 27:
                self.check_coats(namespace)
            self.completed.append(number)
        except BaseException as exc:
            self.reject(f"Cell {number}: {type(exc).__name__}: {exc}")
            raise
        finally:
            self.timings.append({"cell": number, "seconds": time.monotonic() - started})
            if self.owns_outputs:
                save_json(self.work / "P003_TIMINGS.json", self.timings)

    def finish(self, namespace):
        try:
            if not self.ready or self.completed != self.contract["cell_order"]:
                raise ValueError("Incomplete notebook")
            self.check_events(namespace)
            read_scores(self.work / "submission.csv", self.ids)
            receipt = read_json(self.work / "btkd_v559_complete.json")
            digest = sha256(self.work / "submission.csv")
            if receipt.get("status") != "COMPLETE" or receipt.get("submission_sha256") != digest:
                raise ValueError("Final receipt/hash mismatch")
            if (
                receipt.get("coat_family_members") != 4
                or receipt.get("family_reduction") != "rank_of_member_probability_mean"
            ):
                raise ValueError("Final CoAt receipt incomplete")
            elapsed = time.monotonic() - self.started
            if elapsed >= 9 * 3600:
                raise ValueError("Full notebook exceeded nine hours")
            result = {
                "status": "passed",
                "candidate_id": self.contract["candidate_id"],
                "studies": len(self.ids),
                "submission_sha256": digest,
                "elapsed_seconds": elapsed,
                "timings": self.timings,
                "source_notebook_sha256": self.contract["source_notebook_sha256"],
                "public_score": None,
                "score_reproduction_claimed": False,
                "note": "Saved-run completion does not establish hidden-test runtime or score.",
            }
            save_json(self.work / "P003_READY.json", result)
            print(f"P003 READY FOR MANUAL SUBMISSION | {len(self.ids)} studies | {digest}")
            return result
        except BaseException as exc:
            self.reject(f"Final gate: {type(exc).__name__}: {exc}")
            raise
