"""Dedicated regional4x4 last2 training with the standard dense backward graph.

Importing this module never loads an encoder or starts fitting. The original CLS
producer stays immutable. Head attention is evaluated once over every regional
descriptor, retaining the existing suffix checkpoint and encoder window chunk.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .checkpoints import OptimizerStepCounter
from .contracts import (
    ID,
    SERIES_COLUMNS,
    SUBMISSION_COLUMNS,
    TARGETS,
    dump_json,
    index_studies,
    read_csv,
    sha256,
    validate_submission,
    write_csv,
)
from .feature_contracts import (
    head_implementation_hash,
    json_hash,
    new_directory,
    selection_structure_hash,
    validate_encoder_provenance,
    validate_feature_config,
    validate_fold_audit,
)
from .feature_imaging import load_frozen_encoder
from .metrics import evaluate_rows
from .native_training_fast16 import (
    NativePixelStore,
    effective_observation_count,
    label_tensors,
    native_source_hashes,
    observed_bce_sum,
    prefix_digest,
    read_manifest_structure,
    read_prediction_ids,
    retain_epoch_checkpoint,
    suffix_keys,
    tensor_study,
)
from .regional_triplet_model import RegionalTripletMIL
from .runtime import check_split, ensure_cuda
from .triplet_cache import extract_pixel_study, pixel_contract

ONLINE_SCHEMA = "native392_regional4x4_last2_deterministic_dense_training_fast16_v3"
CHECKPOINT_SCHEMA = "native392_generic_dino_regional4x4_suffix_head_deterministic_dense_fast16_v3"
RUN_SCHEMA = "native392_regional4x4_deterministic_dense_training_run_fast16_v3"
RESOURCE_GATE_SCHEMA = "regional392_deterministic_dense_QA_v3"
PIXEL_INDEX_SHA256 = "7d373ae85098758a8bc322b4e65079de07e998f7b7cd42a5a2d26cc4577eae31"
REPRESENTATION = {
    "name": "normalized_CLS_plus_fixed4x4_patch_means_v1",
    "model_class": "RegionalTripletMIL",
    "normalized_patch_grid": [28, 28],
    "regional_grid": [4, 4],
    "patches_per_region": [7, 7],
    "order": "CLS first, then16 disjoint regional means row-major within each original window",
    "pooling": "FP32 means of normalized patch tokens",
    "instances_per_window": 17,
    "instances_per_series": 816,
    "feature_dim": 384,
    "quantization": "all17 FP16 thenFP32",
    "positions_and_masks": "repeat_interleave17 per original window",
    "learned_regional_coordinates": False,
}
EXECUTION = {
    "version": "regional_standard_deterministic_dense_suffix_checkpoint_backward_v3",
    "encoder_window_chunk": 16,
    "head_forward_per_microstudy": 1,
    "backward": "one standard autograd backward through the full regional study; no streamed VJP or fallback",
    "checkpoint_last2": True,
    "AMP": "scale observed group loss once; unscale/clip/AdamW/update once after the actual effective group",
    "determinism": {
        "cublas_workspace_config": ":4096:8",
        "torch_deterministic_algorithms": True,
        "torch_deterministic_warn_only": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "configured_before_cuda": True,
        "SDPA_policy": "default dispatch under strict deterministic algorithms; no forced math backend",
    },
}
_REGIONAL_DETERMINISM_PREPARED = False


def regional_determinism_state():
    return {
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "configured_before_cuda": _REGIONAL_DETERMINISM_PREPARED,
        "SDPA_policy": EXECUTION["determinism"]["SDPA_policy"],
    }


def configure_regional_determinism():
    """Set the shared strict execution before CUDA setup/model operations."""
    global _REGIONAL_DETERMINISM_PREPARED
    if torch.cuda.is_initialized() and not _REGIONAL_DETERMINISM_PREPARED:
        raise ValueError("Regional deterministic setup must run before CUDA initialization in this process")
    required = EXECUTION["determinism"]["cublas_workspace_config"]
    configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = required
    elif configured != required:
        raise ValueError("Dedicated regional execution requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    _REGIONAL_DETERMINISM_PREPARED = True
    state = regional_determinism_state()
    if state != EXECUTION["determinism"]:
        raise ValueError("Strict regional deterministic execution was not enabled")
    return state


def regional_source_hashes():
    """Hash the dedicated implementation and every immutable shared CLS helper."""
    here = Path(__file__).parent
    return {
        **native_source_hashes(),
        **{
            name: hashlib.sha256((here / name).read_text(encoding="utf-8").replace("\r\n", "\n").encode()).hexdigest()
            for name in (
                "regional_training_fast16.py",
                "regional_triplet_model.py",
                "checkpoints.py",
                "runtime.py",
                "metrics.py",
            )
        },
    }


def cli_hashes():
    workspace = Path(__file__).resolve().parents[2]
    return {
        name: sha256(workspace / "scripts" / name)
        for name in ("train_regional_dinov2_fast16.py", "check_regional_training_fast16.py")
    }


def validate_regional_config(config):
    validate_feature_config(config)
    expected = {
        "schema_version": ONLINE_SCHEMA,
        "trainable_blocks": 2,
        "microbatch_studies": 1,
        "effective_batch_studies": 16,
        "window_chunk": 16,
        "checkpoint_last2": True,
        "encoder_learning_rate": 1e-5,
        "accumulation_denominator": "total_observed_labels_in_effective_batch",
        "folds": [0, 1],
        "amp_grad_scaler": True,
        "execution": EXECUTION["version"],
    }
    if config.get("online") != expected or config.get("representation") != REPRESENTATION:
        raise ValueError("Require the dedicated regional4x4 last2 representation/config")
    if config.get("representation_sha256") != json_hash(REPRESENTATION):
        raise ValueError("Regional representation fingerprint differs")
    if config.get("execution") != EXECUTION:
        raise ValueError("Require explicit standard dense regional backward; rejected VJP is not a fallback")
    if config["seed"] != 20261002 or config["n_folds"] != 5:
        raise ValueError("Preserve seed20261002 and the fixed five group folds")
    if config["preprocess"] != {
        "version": "physical_letterbox_triplet_v1",
        "image_size": 392,
        "max_series": 3,
        "slices_per_series": 48,
        "percentiles": [1, 99],
    }:
        raise ValueError("Preserve the existing native392 pixel cache exactly")
    pixel_contract(config["preprocess"])
    if config["encoder"] != {"model_name": "dinov2_vits14", "feature_dim": 384, "batch_size": 16, "precision": "fp16"}:
        raise ValueError("Preserve the generic DINOv2 ViT-S/14 and FP16 descriptor boundary")
    if config["head"] != {"hidden_dim": 192, "dropout": 0.2, "pooling": "attention"}:
        raise ValueError("Preserve the existing 83352-parameter finding-specific head")
    if config["train"] != {
        "epochs": 12,
        "batch_size": 16,
        "learning_rate": 0.0003,
        "weight_decay": 0.01,
        "num_workers": 0,
        "selection": "bce",
    }:
        raise ValueError("Preserve twelve epochs, old head LR and global weak-BCE selection")
    return config


def load_regional_config(path):
    return validate_regional_config(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def _check_model(model):
    if type(model) is not RegionalTripletMIL:
        raise ValueError("Dedicated regional weights cannot be applied to a CLS model")
    validate_regional_config(model.config)
    if (model.trainable_blocks, model.window_chunk, model.checkpoint_last2) != (2, 16, True):
        raise ValueError("Runtime last2/window16/checkpoint settings differ")
    contract = model.parameter_contract()
    if (
        contract["trainable_encoder_parameters"] != 3550464
        or contract["trainable_head_parameters"] != 83352
        or contract["representation"] != REPRESENTATION["name"]
        or not contract["encoder_eval"]
        or not contract["final_norm_frozen"]
    ):
        raise ValueError("Regional runtime encoder/head/representation contract differs")


def new_regional_model(repo, weights, provenance, config, device="cuda"):
    configure_regional_determinism()
    validate_regional_config(config)
    encoder = load_frozen_encoder(repo, weights, provenance, device)
    model = RegionalTripletMIL(encoder, config, trainable_blocks=2, window_chunk=16, checkpoint_last2=True).to(device)
    _check_model(model)
    return model


def _validate_checkpoint_identity(value, *, allow_artificial=False):
    if not isinstance(value, dict) or value.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("Dedicated regional checkpoint schema required; CLS checkpoints are incompatible")
    validate_regional_config(value.get("config", {}))
    if (
        value.get("representation") != REPRESENTATION
        or value.get("representation_sha256") != json_hash(REPRESENTATION)
        or value.get("execution") != EXECUTION
        or value.get("targets") != list(TARGETS)
        or value.get("source_sha256") != regional_source_hashes()
        or value.get("CLI_sha256") != cli_hashes()
        or value.get("head_implementation_sha256") != head_implementation_hash()
        or value.get("pixel_contract") != pixel_contract(value["config"]["preprocess"])
    ):
        raise ValueError("Regional representation/source/targets/head/pixel identity differs")
    validate_encoder_provenance(value.get("encoder_provenance", {}))
    artificial = value.get("artificial")
    if artificial is True:
        if not allow_artificial or value.get("selection_data") != "artificial_contract":
            raise ValueError("Artificial regional checkpoints are not serving weights")
    elif artificial is not False or value.get("selection_data") != "weak_validation":
        raise ValueError("Real regional checkpoints require weak-only validation selection")
    if value.get("gold_used_for_training_or_selection") is not False or value.get("resume_supported") is not False:
        raise ValueError("Gold-tuned/resumable regional checkpoints are forbidden")
    epoch = value.get("epoch")
    if (
        type(value.get("fold")) is not int
        or value["fold"] not in (0, 1)
        or type(epoch) is not int
        or epoch < (0 if artificial else 1)
        or epoch > 12
        or not isinstance(value.get("weak_bce"), (float, int))
        or not math.isfinite(value.get("weak_bce", float("nan")))
        or value["weak_bce"] < 0
    ):
        raise ValueError("Invalid regional fold/epoch/weak BCE")
    return value


def make_regional_checkpoint(model, provenance, *, fold, epoch, weak_bce, prefix_sha256, audit, artificial=False):
    _check_model(model)
    if prefix_digest(model) != prefix_sha256:
        raise ValueError("Frozen generic prefix/norm changed")
    state = model.encoder.state_dict()
    value = {
        "schema_version": CHECKPOINT_SCHEMA,
        "config": model.config,
        "representation": REPRESENTATION,
        "representation_sha256": json_hash(REPRESENTATION),
        "execution": EXECUTION,
        "targets": list(TARGETS),
        "encoder_provenance": provenance,
        "source_sha256": regional_source_hashes(),
        "CLI_sha256": cli_hashes(),
        "head_implementation_sha256": head_implementation_hash(),
        "pixel_contract": pixel_contract(model.config["preprocess"]),
        "encoder_suffix": {name: state[name].detach().cpu().clone() for name in sorted(suffix_keys(model))},
        "head": {name: tensor.detach().cpu().clone() for name, tensor in model.head.state_dict().items()},
        "frozen_prefix_sha256": prefix_sha256,
        "fold": fold,
        "epoch": epoch,
        "weak_bce": weak_bce,
        "selection_data": "artificial_contract" if artificial else "weak_validation",
        "artificial": artificial,
        "gold_used_for_training_or_selection": False,
        "audit": audit,
        "resume_supported": False,
        "optimizer_scaler_RNG_saved": False,
        "note": "Regional inference weights only; fresh generic initialization is required for every new run.",
    }
    _validate_checkpoint_identity(value, allow_artificial=artificial)
    return value


def load_regional_checkpoint(path, *, allow_artificial=False):
    return _validate_checkpoint_identity(
        torch.load(path, map_location="cpu", weights_only=True), allow_artificial=allow_artificial
    )


def apply_regional_checkpoint(model, checkpoint, provenance, *, allow_artificial=False):
    # Validate all semantic identities and all tensor metadata before copying any tensor.
    _check_model(model)
    _validate_checkpoint_identity(checkpoint, allow_artificial=allow_artificial)
    if checkpoint["config"] != model.config or checkpoint["encoder_provenance"] != provenance:
        raise ValueError("Generic initialization or regional config differs")
    if prefix_digest(model) != checkpoint["frozen_prefix_sha256"]:
        raise ValueError("Frozen prefix/norm cannot reconstruct regional checkpoint")
    merged = model.encoder.state_dict()
    expected = suffix_keys(model)
    for label, supplied, reference in (
        ("suffix", checkpoint.get("encoder_suffix", {}), {name: merged[name] for name in expected}),
        ("head", checkpoint.get("head", {}), model.head.state_dict()),
    ):
        if set(supplied) != set(reference):
            raise ValueError(f"Strict regional {label} keys differ")
        for name, tensor in supplied.items():
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != reference[name].shape
                or tensor.dtype != reference[name].dtype
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError(f"Malformed/nonfinite regional {label} tensor: {name}")
    merged.update(checkpoint["encoder_suffix"])
    model.encoder.load_state_dict(merged, strict=True)
    model.head.load_state_dict(checkpoint["head"], strict=True)
    return model


def _regional_input_layout(inputs):
    images, mask = inputs["images"], inputs["mask"]
    expected = (1, 3, 48)
    if (
        images.shape != (*expected, 3, 392, 392)
        or images.dtype != torch.uint8
        or mask.shape != expected
        or mask.dtype != torch.bool
        or not mask.any()
        or inputs["positions"].shape != expected
        or inputs["planes"].shape != (1, 3)
        or inputs["flags"].shape != (1, 3, 2)
    ):
        raise ValueError("Dense regional backward requires one native392 study and unchanged3x48 metadata")
    return images.reshape(-1, 3, 392, 392), mask.reshape(-1).nonzero(as_tuple=False).flatten()


def dense_regional_backward(model, inputs, targets, observed, effective_denominator, *, scaler=None):
    """Accumulate one standard dense backward with the actual observed denominator.

    The returned logits are detached. Head gradients remain accumulated across
    microstudies; this function never unscales, clips or updates an optimizer.
    Encoder activation checkpointing remains inside RegionalTripletMIL.forward.
    """
    if type(model) is not RegionalTripletMIL or model.encoder.training or model.trainable_blocks != 2:
        raise ValueError("Use the dedicated RegionalTripletMIL last2 encoder in eval mode")
    _, indices = _regional_input_layout(inputs)
    if (
        targets.shape != (1, 12)
        or observed.shape != targets.shape
        or observed.dtype != torch.bool
        or type(effective_denominator) is not int
        or effective_denominator < int(observed.sum())
        or effective_denominator < 1
    ):
        raise ValueError("Invalid actual effective-group observed-label denominator")
    logits = model(**inputs)
    if logits.shape != (1, 12) or not torch.isfinite(logits).all():
        raise ValueError("Malformed/nonfinite dense regional logits")
    numerator = observed_bce_sum(logits, targets, observed)
    loss = numerator / effective_denominator
    (scaler.scale(loss) if scaler is not None else loss).backward()
    # Nonfinite scaled gradients are intentionally left to GradScaler's group-level skip.
    result = {
        "logits": logits.detach(),
        "observed_bce_sum": float(numerator.detach()),
        "observed": int(observed.sum()),
        "valid_windows": len(indices),
    }
    del logits, numerator, loss
    return result


def make_regional_optimizer(model):
    _check_model(model)
    return torch.optim.AdamW(
        [
            {"params": list(model.head.parameters()), "lr": model.config["train"]["learning_rate"], "name": "head"},
            {
                "params": [p for p in model.encoder.parameters() if p.requires_grad],
                "lr": model.config["online"]["encoder_learning_rate"],
                "name": "last2",
            },
        ],
        weight_decay=model.config["train"]["weight_decay"],
    )


def finish_regional_group(model, optimizer, scaler, counter=None):
    """Unscale/clip/step/update once after all actual group microstudies."""
    temporary = counter is None
    if temporary:
        counter = OptimizerStepCounter(optimizer)
    before = counter.steps
    try:
        scaler.unscale_(optimizer)
        parameters = [p for p in model.parameters() if p.requires_grad]
        finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters)
        grad_norm = None
        if finite:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
        counter.opportunity()
        scaler.step(optimizer)
        scaler.update()
        return {"finite": finite, "updated": counter.steps > before, "grad_norm": grad_norm}
    finally:
        if temporary:
            counter.close()


def validate_regional_resource_gate(path, config, provenance=None):
    """Require dedicated dense artificial QA, never the rejected streamed QA."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    checks = value.get("checks", {})
    required = {
        "shared_deterministic_execution_enabled",
        "encoder_eval_final_norm_frozen",
        "head_FP32_and_fixed_chunk16",
        "full144_all_trainable_gradient_repeatability",
        "full144_logit_repeatability",
        "full144_masked_BCE_repeatability",
        "full144_dropout_RNG_once_and_scale_repeatability",
        "sparse_masks_padding_NaN_positions_and_missing_teacher_excluded",
        "prefix_unchanged_after_six_artificial_updates",
        "dedicated_artificial_roundtrip_exact",
        "artificial_checkpoint_real_load_rejected",
        "artificial_checkpoint_default_apply_rejected",
        "regional_checkpoint_old_CLS_load_rejected",
        "CLS_checkpoint_regional_load_rejected",
        "CLS_checkpoint_regional_apply_rejected",
        "CLS_rejected_before_copy",
        "tampered_representation_load_rejected",
        "tampered_representation_apply_rejected",
        "tampered_representation_rejected_before_copy",
        *{
            f"full144_{path}_{check}"
            for path in ("main", "replay")
            for check in ("frozen_prefix_no_grad", "memory_within_limit")
        },
        *{
            f"effective_group{size}_{check}"
            for size in (16, 10, 4)
            for check in (
                "all_gradient_repeatability",
                "post_AdamW_parameter_repeatability",
                "AdamW_state_repeatability",
                "AMP_scaler_and_dropout_RNG_repeatability",
                "logits_and_masked_BCE_repeatability",
            )
        },
        *{
            f"group{size}_{path}_{check}"
            for size in (16, 10, 4)
            for path in ("main", "replay")
            for check in ("frozen_prefix_no_grad", "AMP_clip_AdamW_updated")
        },
    }
    if (
        value.get("schema_version") != RESOURCE_GATE_SCHEMA
        or value.get("all_passed") is not True
        or not isinstance(checks, dict)
        or not checks
        or not required.issubset(checks)
        or not all(check is True for check in checks.values())
        or value.get("full144_valid_windows") != 144
        or value.get("group_sizes") != [16, 10, 4]
        or type(value.get("optimizer_steps_artificial")) is not int
        or value["optimizer_steps_artificial"] != 6
        or value.get("trainable_parameters_compared") != 3633816
        or value.get("real_teacher_labels_read") is not False
        or value.get("real_MRI_pixels_read") is not False
        or value.get("real_fit") is not False
        or value.get("raw_DICOM_decoded_or_downloaded") is not False
    ):
        raise ValueError("Actual full144 regional dense/short-group artificial QA must pass")
    if (
        value.get("source_sha256") != regional_source_hashes()
        or value.get("CLI_sha256") != cli_hashes()
        or value.get("config_semantic_sha256") != json_hash(validate_regional_config(config))
        or value.get("representation") != REPRESENTATION
        or value.get("representation_sha256") != json_hash(REPRESENTATION)
        or value.get("execution") != EXECUTION
    ):
        raise ValueError("Regional QA source/config/representation pins differ")
    validate_encoder_provenance(value.get("encoder_provenance", {}))
    if provenance is not None and value["encoder_provenance"] != provenance:
        raise ValueError("Regional artificial QA used a different generic encoder")
    if value.get("environment") != {
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "determinism": regional_determinism_state(),
    }:
        raise ValueError("Regional artificial QA belongs to a different local GPU/runtime")
    return value


def preflight_regional_training(manifest_dir, index_path, config, fold, gate_path, provenance=None):
    configure_regional_determinism()
    ensure_cuda()
    validate_regional_config(config)
    if fold not in (0, 1):
        raise ValueError("Use the fixed regional folds0/1")
    gate = validate_regional_resource_gate(gate_path, config, provenance)
    manifest_dir = Path(manifest_dir)
    manifest = json.loads((manifest_dir / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["seed"] != config["seed"]
        or manifest["n_folds"] != config["n_folds"]
        or manifest["label_provenance"].get("gold_used_for_tuning") is not False
    ):
        raise ValueError("Reviewed weak-only group split/provenance differs")
    fold_audit = validate_fold_audit(manifest_dir, manifest_dir / "fold-audit-v2.json", manifest)
    weak_structure = read_manifest_structure(manifest_dir / "weak.csv", (ID, "source", "group_id", "fold"))
    gold = read_manifest_structure(manifest_dir / "gold.csv", (ID, "source", "group_id"))
    train_structure, valid_structure = check_split(weak_structure, gold, fold, config["n_folds"])
    grouped = {}
    for row in weak_structure:
        if row["group_id"] in grouped and grouped[row["group_id"]] != int(row["fold"]):
            raise ValueError("One supplied/report group crosses the fixed folds")
        grouped[row["group_id"]] = int(row["fold"])
    _, weak = read_csv(manifest_dir / "weak.csv", (ID, *TARGETS, "source", "group_id", "fold"))
    keyed = index_studies(weak)
    if set(keyed) != set(index_studies(weak_structure)) or any(
        any(keyed[row[ID]][field] != row[field] for field in ("source", "group_id", "fold")) for row in weak_structure
    ):
        raise ValueError("Teacher pass changed the audited study/group/fold structure")
    training = [keyed[row[ID]] for row in train_structure]
    validation = [keyed[row[ID]] for row in valid_structure]
    for row in training + validation:
        effective_observation_count([row])
    if sha256(index_path) != PIXEL_INDEX_SHA256:
        raise ValueError("Use the accepted complete4207 native392 pixel index")
    store = NativePixelStore(index_path, config)
    hashes = {
        name: sha256(manifest_dir / name) for name in ("manifest.json", "weak.csv", "gold.csv", "fold-audit-v2.json")
    }
    if (
        store.index.get("split") != "train"
        or store.index.get("limited") is not False
        or store.index.get("verified", {}).get("studies") != 4207
        or store.index.get("verified", {}).get("arrays") != 21035
        or set(store.studies) != set(index_studies(weak))
        or store.index.get("manifest_sha256") != hashes
        or store.index.get("selection_structure_sha256") != selection_structure_hash(weak)
    ):
        raise ValueError("Complete pixel cache and weak group/fold/Gold exclusion manifest are not bound")
    return (
        training,
        validation,
        store,
        {
            "manifest": manifest,
            "manifest_sha256": hashes,
            "fold_audit": fold_audit,
            "pixel_index_sha256": sha256(index_path),
            "resource_gate_sha256": sha256(gate_path),
            "resource_gate": gate,
        },
    )


@torch.inference_mode()
def evaluate_regional(model, rows, store):
    model.eval()
    predictions, numerator, denominator = [], 0.0, 0
    for row in rows:
        logits = model(**tensor_study(store.load(row[ID]))).float()
        if logits.shape != (1, 12) or not torch.isfinite(logits).all():
            raise ValueError("Malformed/nonfinite regional validation logits")
        targets, observed = label_tensors(row, logits.device)
        numerator += float(observed_bce_sum(logits, targets, observed))
        denominator += int(observed.sum())
        predictions.append({ID: row[ID], **dict(zip(TARGETS, logits.sigmoid()[0].cpu().tolist()))})
    if denominator < 1:
        raise ValueError("Regional validation has no observed weak cells")
    return predictions, numerator / denominator


def train_regional(manifest_dir, index_path, output, config, fold, repo, weights, provenance, gate_path):
    configure_regional_determinism()
    ensure_cuda()
    if Path(output).exists():
        raise ValueError("Use a fresh regional run directory; existing runs cannot be resumed or overwritten")
    training, validation, store, audit = preflight_regional_training(
        manifest_dir, index_path, config, fold, gate_path, provenance
    )
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    model = new_regional_model(repo, weights, provenance, config)
    initial_prefix = prefix_digest(model)
    optimizer = make_regional_optimizer(model)
    counter = OptimizerStepCounter(optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    out = new_directory(output)
    dump_json(out / "config.json", config)
    record = {
        "schema_version": RUN_SCHEMA,
        "status": "running",
        "config": config,
        "seed": config["seed"],
        "fold": fold,
        "representation": REPRESENTATION,
        "representation_sha256": json_hash(REPRESENTATION),
        "execution": EXECUTION,
        "train_studies": len(training),
        "valid_studies": len(validation),
        "audit": audit,
        "source_sha256": regional_source_hashes(),
        "CLI_sha256": cli_hashes(),
        "encoder_provenance": provenance,
        "parameters": model.parameter_contract(),
        "targets": list(TARGETS),
        "gpu": torch.cuda.get_device_name(0),
        "environment": {
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "determinism": regional_determinism_state(),
        },
        "selection_data": "weak_validation",
        "selection_rule": "minimum global observed-cell weak BCE; earliest exact tie",
        "gold_used_for_training_or_selection": False,
        "gold_evaluated": False,
        "patient_independence_verified": False,
        "history": [],
        "public_lb": None,
        "resume_supported": False,
        "denominator": "total observed weak cells per actual effective group, including final short group",
        "epoch_checkpoint_index": "epoch_checkpoint_index.json",
    }
    dump_json(out / "run.json", record)
    generator = torch.Generator().manual_seed(config["seed"])
    selected, start = None, time.perf_counter()
    try:
        for epoch in range(1, 13):
            model.train()
            ordered = [training[index] for index in torch.randperm(len(training), generator=generator).tolist()]
            loss_sum, count = 0.0, 0
            steps_start, opportunities_start = counter.steps, counter.opportunities
            group_sizes, group_observations = [], []
            for offset in range(0, len(ordered), 16):
                group = ordered[offset : offset + 16]
                denominator = effective_observation_count(group)
                group_sizes.append(len(group))
                group_observations.append(denominator)
                optimizer.zero_grad(set_to_none=True)
                for row in group:
                    inputs = tensor_study(store.load(row[ID]))
                    targets, observed = label_tensors(row, "cuda")
                    micro = dense_regional_backward(model, inputs, targets, observed, denominator, scaler=scaler)
                    loss_sum += micro["observed_bce_sum"]
                    count += micro["observed"]
                    del inputs, targets, observed, micro
                finish_regional_group(model, optimizer, scaler, counter)
            if counter.steps == steps_start:
                raise ValueError("No regional optimizer update completed in this epoch")
            predictions, bce = evaluate_regional(model, validation, store)
            metrics = evaluate_rows(index_studies(validation), index_studies(predictions))
            predictions_path = out / f"epoch_{epoch:03d}_weak.csv"
            write_csv(predictions_path, SUBMISSION_COLUMNS, predictions)
            item = {
                "epoch": epoch,
                "train_bce": loss_sum / count,
                "weak_bce": bce,
                "weak_macro_auc_12": metrics["macro_auc_12"],
                "metrics": metrics,
                "optimizer_steps": counter.steps - steps_start,
                "optimizer_step_opportunities": counter.opportunities - opportunities_start,
                "amp_skipped_steps": counter.opportunities - opportunities_start - (counter.steps - steps_start),
                "actual_group_sizes": group_sizes,
                "actual_group_observed_counts": group_observations,
                "observed_cells": count,
                "elapsed_seconds": time.perf_counter() - start,
                "prediction_sha256": sha256(predictions_path),
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            record["history"].append(item)
            checkpoint = make_regional_checkpoint(
                model, provenance, fold=fold, epoch=epoch, weak_bce=bce, prefix_sha256=initial_prefix, audit=audit
            )
            torch.save(checkpoint, out / "last.pt")
            if selected is None or bce < selected:
                selected = bce
                torch.save(checkpoint, out / "best.pt")
                write_csv(out / "weak_valid_predictions.csv", SUBMISSION_COLUMNS, predictions)
                record["selected"] = {**item, "checkpoint_sha256": sha256(out / "best.pt")}
            retain_epoch_checkpoint(out, checkpoint, item, predictions_path)
            dump_json(out / "run.json", record)
            print(f"Regional epoch{epoch}/12: weak BCE={bce:.6f}, AUC12={metrics['macro_auc_12']}", flush=True)
        record.update(status="complete", elapsed_seconds=time.perf_counter() - start)
    except Exception as error:
        record.update(
            status="failed", error=f"{type(error).__name__}: {error}", elapsed_seconds=time.perf_counter() - start
        )
        raise
    finally:
        counter.close()
        dump_json(out / "run.json", record)
    return record


def predict_regional(root, checkpoint_path, output_csv, repo, weights, provenance, *, reader):
    configure_regional_determinism()
    ensure_cuda()
    if not Path("/kaggle/input").is_dir():
        raise ValueError("MRI inference belongs on Kaggle; local training accepts processed pixel caches only")
    root = Path(root)
    if Path(output_csv).exists():
        raise ValueError("Use a fresh regional prediction path")
    checkpoint = load_regional_checkpoint(checkpoint_path)
    model = new_regional_model(repo, weights, provenance, checkpoint["config"])
    apply_regional_checkpoint(model, checkpoint, provenance).eval()
    ids = read_prediction_ids(root / "test.csv")
    _, series = read_csv(root / "test_series.csv", SERIES_COLUMNS)
    by_study = {}
    for row in series:
        by_study.setdefault(row[ID], []).append(row)

    def test_reader(train_shaped_path, settings):
        relative = Path(train_shaped_path).relative_to(root / "train_series")
        return reader(root / "test_series" / relative, settings)

    predictions = []
    with torch.inference_mode():
        for row in ids:
            arrays, _ = extract_pixel_study(
                root, row[ID], by_study.get(row[ID], []), model.config["preprocess"], reader=test_reader
            )
            logits = model(**tensor_study(arrays))
            if logits.shape != (1, 12) or not torch.isfinite(logits).all():
                raise ValueError("Malformed/nonfinite regional prediction")
            predictions.append({ID: row[ID], **dict(zip(TARGETS, logits.sigmoid()[0].cpu().tolist()))})
    write_csv(output_csv, SUBMISSION_COLUMNS, predictions)
    receipt = validate_submission(output_csv, root / "test.csv", root / "sample_submission.csv")
    dump_json(
        Path(output_csv).with_suffix(".receipt.json"),
        {
            **receipt,
            "checkpoint_sha256": sha256(checkpoint_path),
            "submission_sha256": sha256(output_csv),
            "representation": REPRESENTATION,
            "representation_sha256": json_hash(REPRESENTATION),
            "Report_required": False,
            "public_score": None,
        },
    )
    return receipt
