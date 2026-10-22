"""F015 model-specific guards with the unchanged native data and numeric loop.

The original DINO module is never patched. New functions are compiled into this
module's globals, using the CNN validator/factory/checkpoint functions below.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from . import convnext_small_model as M
from . import native_training_fast16 as core
from .checkpoints import OptimizerStepCounter
from .contracts import ID, SUBMISSION_COLUMNS, TARGETS, dump_json, index_studies, read_csv, sha256, write_csv
from .feature_contracts import json_hash, new_directory, selection_structure_hash, validate_fold_audit
from .metrics import evaluate_rows
from .runtime import check_split, ensure_cuda

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "artifacts/research/20261021-F015-ConvNeXtSmall-prospective-v1/policy.json"
POLICY_SHA = "45d4cf42149064f2205f061223fa297f602ca64f64927b5bf26f3fa362b95dfb"
CORE_SHA = "9f5c3b490374201f72cc73ebf11a26f8a8163945387d4d493e7a04c08e729d57"
ENGINEERING_LINEAGE = {"version": "F015_separate_CNN_encoder_v1", "unchanged_numeric_core_SHA": CORE_SHA,
                       "scope": "Encoder choice includes init/architecture/width/suffix; no isolated architecture claim"}
ACTIVE = {}
# These names are referenced by the compiled, unchanged numerical/data bodies.
SHARED_BINDINGS = {"random": random, "np": np, "OptimizerStepCounter": OptimizerStepCounter,
                   "SUBMISSION_COLUMNS": SUBMISSION_COLUMNS, "index_studies": index_studies,
                   "read_csv": read_csv, "write_csv": write_csv, "new_directory": new_directory,
                   "selection_structure_hash": selection_structure_hash, "validate_fold_audit": validate_fold_audit,
                   "evaluate_rows": evaluate_rows, "check_split": check_split, "ensure_cuda": ensure_cuda}

# Explicitly reused architecture-independent functions, not old model guards.
NativePixelStore = core.NativePixelStore
read_manifest_structure = core.read_manifest_structure
label_tensors = core.label_tensors
observed_bce_sum = core.observed_bce_sum
backward_observed_micro = core.backward_observed_micro
effective_observation_count = core.effective_observation_count
tensor_study = core.tensor_study
native_epoch_learning_rates = core.native_epoch_learning_rates
validate_native_learning_rate_schedule = core.validate_native_learning_rate_schedule
retain_epoch_checkpoint = core.retain_epoch_checkpoint
prefix_digest = M.prefix_digest
native_source_hashes = M.source_hashes
make_native_checkpoint = M.make_checkpoint


class TrainingStopRequested(Exception):
    pass


def validate_native_config(config):
    baseline = json.loads((ROOT / "configs/experiments/f006-dinov2-native-last2-late-cosine.json").read_bytes())
    require = M.require
    require(config["schema_version"] == "f015_convnext_native392_v1"
            and config["online"]["schema_version"] == "f015_convnext_native392_last2_fast16_v1", "Distinct CNN config schema")
    a, b = copy.deepcopy(baseline), copy.deepcopy(config)
    for value in (a, b):
        value.pop("reason", None)
        value.pop("schema_version")
        value["online"].pop("schema_version")
        value["encoder"].pop("model_name")
        value["encoder"].pop("feature_dim")
    require(a == b and config["encoder"]["model_name"] == M.MODEL and config["encoder"]["feature_dim"] == 768,
            "Only the declared encoder choice may differ from F006 controls")
    require(sha256(POLICY) == POLICY_SHA and sha256(Path(core.__file__)) == CORE_SHA, "Frozen prospective policy and numeric core")
    return config


def validate_resource_gate(path, config=None):
    value = json.loads(Path(path).read_bytes())
    M.require(value["schema_version"] == "F015_native392_CNN_resource_gate_v1" and value["complete"] is True
              and value["all_pass"] is True and value["checks"] and all(v is True for v in value["checks"].values()), "Actual nonempty CNN GPU qualification")
    M.require(value["source_sha256"] == M.source_hashes() and value["config_hash"] == json_hash(config)
              and value["generic_weights_sha256"] == M.WEIGHTS_SHA and value["policy_SHA"] == POLICY_SHA,
              "Exact qualified CNN/config/generic source")
    M.require(value["GPU"] == torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 4090"
              and value["full144_peak_reserved_bytes"] < 16 * 1024**3 and value["optimizer_updates"] == 0
              and value["MRI_Report_Gold_labels_reads"] == 0, "Current artificial GPU resource scope")
    return value


def new_native_model(repo, weights, provenance, config, device="cuda"):
    del repo  # No remote hub/factory or alternate initialization is permitted.
    validate_native_config(config)
    return M.create_model(weights, config, provenance, device=device)


def check_running_budget(output, started):
    ACTIVE.update(output=Path(output), started=started)
    if (Path(output) / "STOP_REQUESTED").is_file():
        raise TrainingStopRequested("Root requested a stop of this experiment")
    if time.perf_counter() - started > 43200:
        raise TimeoutError("F015 prespecified 12-hour fold budget exhausted")


def record_training_progress(output, epoch, offset, total, counter, started):
    if offset % 256 == 0:
        value = {"epoch": epoch, "train_studies_completed": offset, "train_studies_total": total,
                 "optimizer_steps": counter.steps, "optimizer_opportunities": counter.opportunities,
                 "elapsed_seconds": time.perf_counter() - started}
        dump_json(Path(output) / "progress.json", value)
        print("F015_PROGRESS", json.dumps(value), flush=True)


def exception_status(error):
    return "stopped_by_root_request" if isinstance(error, TrainingStopRequested) else "failed"


@torch.no_grad()
def evaluate_native(model, rows, store):
    model.eval()
    predictions, numerator, denominator = [], 0.0, 0
    for row in rows:
        if ACTIVE:
            check_running_budget(ACTIVE["output"], ACTIVE["started"])
        logits = model(**tensor_study(store.load(row[ID]))).float()
        targets, observed = label_tensors(row, logits.device)
        numerator += float(observed_bce_sum(logits, targets, observed))
        denominator += int(observed.sum())
        M.require(bool(torch.isfinite(logits).all()), "Finite CNN validation output")
        predictions.append({ID: row[ID], **dict(zip(TARGETS, logits.sigmoid()[0].cpu().tolist()))})
    return predictions, numerator / denominator


def training_core_ast():
    """Preserve shuffle, accumulation, AMP accounting, clipping, LR and retention."""
    M.require(sha256(Path(core.__file__)) == CORE_SHA, "Pinned original native core")
    preflight = ast.parse(inspect.getsource(core.preflight_native_training)).body[0]
    loop = ast.parse(inspect.getsource(core.train_native)).body[0]
    loop.name = "train_convnext_small"
    changed = {"schema": 0, "progress": 0, "status": 0}

    class Adapter(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == "native392_training_run_fast16_v1":
                changed["schema"] += 1
                return ast.copy_location(ast.Constant(value="F015_convnext_native392_training_run_v1"), node)
            return node

        def visit_For(self, node):
            node = self.generic_visit(node)
            if isinstance(node.target, ast.Name) and node.target.id == "offset":
                changed["progress"] += 1
                node.body[:0] = ast.parse("check_running_budget(out, start)\nrecord_training_progress(out, epoch, offset, len(ordered), counter, start)").body
            return node

        def visit_keyword(self, node):
            if node.arg == "status" and isinstance(node.value, ast.Constant) and node.value.value == "failed":
                changed["status"] += 1
                node.value = ast.parse("exception_status(error)", mode="eval").body
            return self.generic_visit(node)

    loop = Adapter().visit(loop)
    M.require(changed == {"schema": 1, "progress": 1, "status": 1}, "Only declared metadata/budget/progress loop boundaries")
    return ast.fix_missing_locations(ast.Module(body=[preflight, loop], type_ignores=[]))


# No workload is invoked by defining these functions. The copied preflight uses
# the new CNN validators in this module, never a fabricated DINO config.
globals().update(SHARED_BINDINGS)
exec(compile(training_core_ast(), "F015-shared-native-data-and-numeric-core", "exec"), globals())


def loop_signature():
    return hashlib.sha256(ast.dump(training_core_ast()).encode()).hexdigest()
